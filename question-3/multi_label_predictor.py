import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Veriyi yükle
df = pd.read_csv("data/new_product.csv")

# Pivot tablo (her müşteri için kategori bazlı harcama)
pivot_df = df.pivot_table(index='customer_id',
                          columns='category_name',
                          values='total_spending',
                          aggfunc='sum',
                          fill_value=0)

# X ve y (hepsi label olacak çünkü multi-label)
X = pivot_df.copy()
y = (pivot_df > 0).astype(int)  # 0: hiç almamış, 1: almış

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='sigmoid')  # Multi-label için sigmoid
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitim
model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

# Kaydet
model.save("model/multi_label_predictor.h5")


# ===============================
# C. Müşteri ID'si ile Öneri Yapma Fonksiyonu
# ===============================

def recommend_categories_by_customer_id(model, customer_id, pivot_df, category_names, threshold=0.5):
    """
    Belirli bir müşteri ID'sine göre önerilecek kategorileri döndürür.
    
    :param model: Eğitilmiş Keras modeli
    :param customer_id: Müşteri ID'si
    :param pivot_df: Kategoriler bazında harcama verilerini içeren pivot tablo (DataFrame)
    :param category_names: Kategori isimlerinin listesi (pivot_df.columns)
    :param threshold: Öneri için eşik değeri (default: 0.5)
    :return: Önerilen kategoriler listesi
    """
    # Müşteri ID'sine ait veriyi pivot_df'den al
    customer_vector = pivot_df.loc[customer_id].values
    
    # Model ile tahmin yap
    probs = model.predict(customer_vector.reshape(1, -1))[0]
    
    # Eşik değerini geçen kategorileri öner
    recommended = [cat for cat, prob in zip(category_names, probs) if prob >= threshold]
    return recommended

# Kullanım örneği:
category_names = pivot_df.columns.tolist()

# Belirli bir müşteri ID'sine göre öneri yapalım (örneğin müşteri ID'si 'MORGK')
customer_id = 'MORGK'
recommended = recommend_categories_by_customer_id(model, customer_id, pivot_df, category_names, threshold=0.5)
print(f"{customer_id} için önerilen kategoriler:", recommended)

