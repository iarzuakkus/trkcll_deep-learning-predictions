import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
import sys
from pathlib import Path
from db import Database 
# ============ A. VERİ YÜKLEME ============

# Proje kök dizini ayarla ve veritabanı sınıfını import et
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from db import Database

# Veriyi al
db = Database()
data = db.new_product()
db.close()

# DataFrame'e çevir
df = pd.DataFrame(data, columns=["customer_id", "category_id", "category_name", "total_spending", "product_id", "product_name"])
os.makedirs("data", exist_ok=True)
df.to_csv("data/new_product.csv", index=False)
print("Veri başarıyla 'data/new_product.csv' dosyasına kaydedildi.")

# ============ B. VERİ ÖN İŞLEME ============

# 1. Veriyi oku
df = pd.read_csv("data/new_product.csv")

# 2. Pivotla: müşteri başına kategori harcamaları
category_pivot = df.pivot_table(index='customer_id',
                                 columns='category_name',
                                 values='total_spending',
                                 aggfunc='sum',
                                 fill_value=0)

# 3. Yeni ürün tanımı (örnek olarak)
NEW_PRODUCT = "Chocolade"
NEW_PRODUCT_CATEGORY = "Confections"

if NEW_PRODUCT_CATEGORY not in category_pivot.columns:
    print(f"{NEW_PRODUCT_CATEGORY} kategorisi veride yok. Yeni kategori olarak eklendi.")
    category_pivot[NEW_PRODUCT_CATEGORY] = 0

# 4. Hedef değişken oluştur (müşteri bu kategoriden harcama yapmış mı?)
category_pivot["target"] = (category_pivot[NEW_PRODUCT_CATEGORY] > 0).astype(int)

# 5. Etiket kaçağını önlemek için hedef kategori girişlerden çıkar
X = category_pivot.drop(columns=["target", NEW_PRODUCT_CATEGORY])
y = category_pivot["target"]

# 6. Normalizasyon
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Eğitim/test ayırımı
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 8. Sınıf ağırlıkları (class imbalance çözümü)
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}
print("Sınıf ağırlıkları:", class_weights)

# ============ C. MODEL ============

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# ============ D. KAYDETME ============

# Modeli kaydet
os.makedirs("model", exist_ok=True)
model.save("model/new_product_model.h5")
joblib.dump(scaler, "model/scaler.joblib")
print("Model ve scaler kaydedildi.")

# ============ E. PERFORMANS ============

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {accuracy:.4f}")

# Tahmin çıktıları
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

results_df = pd.DataFrame(X_test, columns=X.columns)
results_df["actual"] = y_test.values
results_df["predicted"] = predicted_labels
results_df.to_csv("data/predictions.csv", index=False)
print("Tahminler 'data/predictions.csv' dosyasına yazıldı.")

# ============ F. TÜM MÜŞTERİLERE TAHMİN ============

all_customers_df = X
all_customers_scaled = scaler.transform(all_customers_df)
new_product_predictions = model.predict(all_customers_scaled)
new_product_labels = (new_product_predictions > 0.5).astype(int)

output_df = pd.DataFrame(all_customers_df, columns=X.columns)
output_df["predicted_to_buy_new_product"] = new_product_labels
output_df.to_csv("data/new_product_purchase_predictions.csv", index=False)
print("Tüm müşteriler için tahminler kaydedildi.")

# ============ G. TEK MÜŞTERİ TAHMİNİ ============

def predict_customer_purchase(customer_id, model, scaler, pivot_df, new_product_category):
    if customer_id not in pivot_df.index:
        print(f"Müşteri ID {customer_id} veride yok.")
        return None

    # Hedef ve kategori çıkarılarak input hazırlanır
    customer_data = pivot_df.drop(columns=["target", new_product_category]).loc[[customer_id]]
    customer_scaled = scaler.transform(customer_data)
    prediction = model.predict(customer_scaled)[0][0]
    predicted_label = int(prediction > 0.5)

    print(f"\nMüşteri ID: {customer_id}")
    print(f"Tahmin Skoru: {prediction:.4f}")
    print(f"Sonuç: {'Alır' if predicted_label == 1 else 'Almaz'}")

    return predicted_label

# ÖRNEK KULLANIM
customer_id = "ALFKI"
predict_customer_purchase(customer_id, model, scaler, category_pivot, NEW_PRODUCT_CATEGORY)

