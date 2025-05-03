import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# ========================
# A. Veriyi yükle ve pivot yapalım
# ========================
df = pd.read_csv("data/new_product.csv")

# Pivot tablo (her müşteri için kategori bazlı harcama)
pivot_df = df.pivot_table(index='customer_id',
                          columns='category_name',
                          values='total_spending',
                          aggfunc='sum',
                          fill_value=0)

# ========================
# B. Dengeli etiketleme stratejimiz
# ========================
# Her kategori için ortalama harcama üstündekilere 1
mean_spending = pivot_df.mean(axis=0)
y = (pivot_df > mean_spending).astype(int)

X = pivot_df.copy()  # Giriş verisi

# ========================
# C. Ölçekleme ve ayırma
# ========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========================
# D. Model
# ========================
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

# ========================
# E. Model ve scaler kaydet
# ========================
model.save("model/multi_label_predictor.keras")
joblib.dump(scaler, "model/scaler.joblib")

# ========================
# F. Öneri Fonksiyonları
# ========================

def recommend_categories_and_products_by_customer_id(model, customer_id, pivot_df, original_df, threshold=0.5):
    """
    Belirli bir müşteri ID'sine göre kategori ve ürün önerisi döner.
    """
    # Eğer müşteri ID yoksa None döner
    if customer_id not in pivot_df.index:
        return None

    customer_vector = pivot_df.loc[customer_id].values.reshape(1, -1)
    probs = model.predict(customer_vector)[0]

    category_names = pivot_df.columns
    recommended_categories = [cat for cat, prob in zip(category_names, probs) if prob >= threshold]

    # Her kategori için en çok harcanan ürünü bulan:
    recommended_products = []
    for category in recommended_categories:
        products_in_cat = original_df[original_df['category_name'] == category]
        most_common_product = (
            products_in_cat.groupby('product_name')['total_spending']
            .sum()
            .sort_values(ascending=False)
            .head(1)
            .index[0]
        )
        recommended_products.append({"category": category, "product": most_common_product})

    return recommended_products

# ========================
# G. Kullanım Örneği
# ========================

# Örneğin 'MORGK' müşterisi için öneri alalım
category_names = pivot_df.columns.tolist()
customer_id = 'CENTC'

recommendations = recommend_categories_and_products_by_customer_id(
    model,
    customer_id,
    pivot_df,
    df,
    threshold=0.5
)

print(f"{customer_id} için öneriler:")
for rec in recommendations:
    print(f"Kategori: {rec['category']} -> Ürün: {rec['product']}")

