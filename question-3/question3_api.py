from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from db import Database
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os
import joblib
import logging

# FastAPI uygulaması
app = FastAPI()

# Logger ayarı
logging.basicConfig(level=logging.INFO)

# İstek modeli
class PredictionRequest(BaseModel):
    product_name: str
    category_name: str
    customer_id: str

# Model ve scaler yükleniyor
MODEL_PATH = "model/new_product_model.h5"
SCALER_PATH = "model/scaler.joblib"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Veritabanı bağlantısı
db = Database()

# Veriyi alıp pivotlayan fonksiyon
def load_and_prepare_data(new_category):
    data = db.new_product()
    df = pd.DataFrame(data, columns=["customer_id", "category_id", "category_name", "total_spending", "product_id", "product_name"])

    # Pivot tablo
    pivot = df.pivot_table(index='customer_id',
                           columns='category_name',
                           values='total_spending',
                           aggfunc='sum',
                           fill_value=0)

    # Eğer yeni kategori yoksa sıfırdan ekle
    if new_category not in pivot.columns:
        pivot[new_category] = 0

    return df, pivot

# Tahmin fonksiyonu
def predict_customer_purchase(customer_id: str, category_name: str):
    df, pivot_df = load_and_prepare_data(category_name)

    if customer_id not in pivot_df.index:
        return None, None

    # Giriş verisini hazırla (etiket veya hedef kategori yok!)
    X_input = pivot_df.drop(columns=[category_name], errors="ignore").loc[[customer_id]]
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0][0]
    label = int(prediction > 0.5)

    return label, prediction

# Tahmin endpoint'i
@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        product_name = request.product_name
        category_name = request.category_name
        customer_id = request.customer_id

        logging.info(f"İstek alındı: customer_id={customer_id}, category={category_name}, product={product_name}")

        label, score = predict_customer_purchase(customer_id, category_name)

        if label is None:
            raise HTTPException(status_code=404, detail="Müşteri ID veride bulunamadı.")

        return {
            "customer_id": customer_id,
            "product_name": product_name,
            "category_name": category_name,
            "prediction_score": float(score),
            "prediction_label": "Alır" if label == 1 else "Almaz"
        }

    except Exception as e:
        logging.error(f"Hata oluştu: {str(e)}")
        raise HTTPException(status_code=500, detail="Bir hata oluştu.")

# Uvicorn ile çalıştırmak istersen
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




# Uygulama çalıştırmak için bu komut gereklidir
# uvicorn question3_api:app --reload



