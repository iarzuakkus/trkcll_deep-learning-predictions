import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path


# __file__ --> .../question-1/question1.py
root = Path(__file__).resolve().parents[1]   # iki seviye yukarı, projeni̇n kökü
sys.path.insert(0, str(root))
from db import Database


# ===============================
# A. Veritabanından Veriyi Al ve CSV'ye Kaydet
# ===============================

# Sınıf örneğini oluştur
db = Database()

# Veriyi al
data = db.user_product_interactions()

# DataFrame'e çevir
df = pd.DataFrame(data, columns=["user_id", "product_id", "rating"])

# CSV dosyası olarak kaydet
os.makedirs("data", exist_ok=True)
df.to_csv("data/user_product_interactions.csv", index=False)
print("Veri başarıyla 'data/user_product_interactions.csv' dosyasına kaydedildi.")

# Bağlantıyı kapat
db.close()

# ===============================
# B. Model Eğitimi - Neural Collaborative Filtering
# ===============================

# Veriyi yükle
df = pd.read_csv("data/user_product_interactions.csv")

# Encode user ve product (ID'den index'e)
user_ids = df['user_id'].unique().tolist()
product_ids = df['product_id'].unique().tolist()

user_id_to_index = {x: i for i, x in enumerate(user_ids)}
product_id_to_index = {x: i for i, x in enumerate(product_ids)}

df['user'] = df['user_id'].map(user_id_to_index)
df['product'] = df['product_id'].map(product_id_to_index)

# Giriş ve hedef verisi
X = df[['user', 'product']].values
y = df['rating'].values.astype('float32')  # MSE için float olmalı

# Eğitim/test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Embed boyutu
embedding_dim = 50

# Giriş katmanları
user_input = Input(shape=(1,))
product_input = Input(shape=(1,))

# Embedding katmanları
user_embedding = Embedding(input_dim=len(user_ids), output_dim=embedding_dim)(user_input)
product_embedding = Embedding(input_dim=len(product_ids), output_dim=embedding_dim)(product_input)

# Vektörleri düzleştirme
user_vec = Flatten()(user_embedding)
product_vec = Flatten()(product_embedding)

# Birleştir
concat = Concatenate()([user_vec, product_vec])
dense = Dense(128, activation='relu')(concat)
output = Dense(1, activation='linear')(dense)  # rating tahmini

# Model
model = Model(inputs=[user_input, product_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Eğitim
model.fit([X_train[:, 0], X_train[:, 1]], y_train,
          epochs=10,
          batch_size=64,
          validation_split=0.1)

# Modeli kaydet
os.makedirs("model", exist_ok=True)
model.save("model/ncf_recommender.keras")
print("Model başarıyla kaydedildi.")

user_id = "HILAA"  # Öneri almak istediğin kullanıcı
user_index = user_id_to_index[user_id]

# Bu kullanıcının puan verdiği ürünleri bul
seen_products = df[df['user_id'] == user_id]['product'].values

# Daha önce görmediği ürünleri seç
all_product_indices = np.array(range(len(product_ids)))
unseen_products = np.setdiff1d(all_product_indices, seen_products)

# Bu ürünler için puan tahmini yap
user_array = np.full(len(unseen_products), user_index)
predicted_ratings = model.predict([user_array, unseen_products])

# En yüksek puanlı ürünleri sırala
top_indices = np.argsort(predicted_ratings.flatten())[::-1]
top_product_indices = unseen_products[top_indices]

# Orijinal ürün ID'lerine çevir
recommended_product_ids = [product_ids[i] for i in top_product_indices[:10]]
print(f"{user_id} kullanıcısı için önerilen ürün ID'leri:", recommended_product_ids)
