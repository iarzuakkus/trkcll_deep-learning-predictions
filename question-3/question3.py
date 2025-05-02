import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import os

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
data = db.new_product()

# DataFrame'e çevir
df = pd.DataFrame(data, columns=["customer_id", "category_id", "category_name", "total_spending", "product_id", "product_name"])

# CSV dosyası olarak kaydet
os.makedirs("data", exist_ok=True)
df.to_csv("data/new_product.csv", index=False)
print("Veri başarıyla 'data/new_product.csv' dosyasına kaydedildi.")

# Bağlantıyı kapat
db.close()

# ===============================
# B. Derin Öğrenme Modeli Kurulumu ve Eğitimi
# ===============================

# 1. Veriyi Yükle
df = pd.read_csv("data/new_product.csv")

# 2. Pivotlama (kategori ve ürün bazlı harcamalar)
category_pivot = df.pivot_table(index='customer_id',
                                 columns='category_name',
                                 values='total_spending',
                                 aggfunc='sum',
                                 fill_value=0)

# 3. Tüm özellikleri birleştir
pivot_df = category_pivot.copy()

# 4. Yeni ürünün kategorisini belirleme (manuel olarak giriliyor)
NEW_PRODUCT = "Chocolade"  # Yeni ürünün adı
NEW_PRODUCT_CATEGORY = "Confections"  # Yeni ürünün kategorisi

# Mevcut kategorilerden seçilen kategori
existing_categories = df['category_name'].unique()

if NEW_PRODUCT_CATEGORY not in existing_categories:
    print(f"{NEW_PRODUCT_CATEGORY} mevcut kategoriler arasında bulunmuyor. Lütfen geçerli bir kategori seçin.")
    exit()

# Kategoriyi pivot tablosuna ekle
if NEW_PRODUCT_CATEGORY not in pivot_df.columns:
    pivot_df[NEW_PRODUCT_CATEGORY] = 0  # Kategori eklenmemişse, sıfır olarak ekleriz

# 5. Hedef değişken: Yeni ürünün alınma durumu
pivot_df["target"] = (pivot_df[NEW_PRODUCT_CATEGORY] > 0).astype(int)

# 6. Giriş ve hedef ayrımı
X = pivot_df.drop(columns=["target"])
y = pivot_df["target"]

# 7. Normalizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Eğitim ve test ayırımı
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 9. Model tanımı
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# 10. Derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 11. Eğitim
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 12. Performans değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {accuracy:.2f}")

# ===============================
# C. Model ve Tahmin Sonuçlarını Kaydet
# ===============================

# Modeli kaydet
# Model ve scaler'ı birlikte kaydet
import joblib

# Modeli kaydet
model.save("model/new_product_model.h5")
# Scaler'ı kaydet
joblib.dump(scaler, "model/scaler.joblib")
print("Model ve scaler başarıyla kaydedildi.")


# Tahminleri yap ve sonuçları kaydet
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

results_df = pd.DataFrame(X_test, columns=X.columns)
results_df["actual"] = y_test.values
results_df["predicted"] = predicted_labels
results_df.to_csv("data/predictions.csv", index=False)
print("Tahmin sonuçları 'data/predictions.csv' dosyasına kaydedildi.")

# ===============================
# D. Yeni Ürün Tahmini (Toplu)
# ===============================

all_customers_df = pivot_df.drop(columns=["target"])
all_customers_scaled = scaler.transform(all_customers_df)

new_product_predictions = model.predict(all_customers_scaled)
new_product_labels = (new_product_predictions > 0.5).astype(int)

output_df = pd.DataFrame(all_customers_df, columns=X.columns)
output_df["predicted_to_buy_new_product"] = new_product_labels
output_df.to_csv("data/new_product_purchase_predictions.csv", index=False)
print("Yeni ürün için satın alma tahminleri 'data/new_product_purchase_predictions.csv' dosyasına kaydedildi.")

# ===============================
# E. Müşteri Bazlı Tahmin Fonksiyonu
# ===============================

def predict_customer_purchase(customer_id, model, scaler, pivot_df, new_product_category):
    if customer_id not in pivot_df.index:
        print(f"Müşteri ID {customer_id} veride bulunamadı.")
        return None

    customer_data = pivot_df.drop(columns=["target"]).loc[[customer_id]]
    customer_scaled = scaler.transform(customer_data)

    prediction = model.predict(customer_scaled)[0][0]
    predicted_label = int(prediction > 0.5)

    print(f"\nMüşteri ID: {customer_id}")
    print(f"Tahmin Skoru (0-1): {prediction:.4f}")
    print(f"Sonuç: {'Alır' if predicted_label == 1 else 'Almaz'}")

    return predicted_label

# ÖRNEK KULLANIM
customer_id = "ALFKI"
predict_customer_purchase(customer_id, model, scaler, pivot_df, NEW_PRODUCT_CATEGORY)

