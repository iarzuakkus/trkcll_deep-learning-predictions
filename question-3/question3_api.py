from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os
import joblib  # scaler'ı yüklemek için joblib'i ekliyoruz
import logging

import sys
from pathlib import Path


# __file__ --> .../question-1/question1.py
root = Path(__file__).resolve().parents[1]   # iki seviye yukarı, projeni̇n kökü
sys.path.insert(0, str(root))
from db import Database


# FastAPI uygulamasını başlat
app = FastAPI()

# Pydantic modelini oluştur
class PredictionRequest(BaseModel):
    product_name: str
    category_name: str
    customer_id: str

# Veritabanı bağlantısı (aynı şekilde kullanılacak)
db = Database()

# Model ve scaler'ı yükle
model = load_model("model/new_product_model.h5")
scaler = joblib.load("model/scaler.joblib")  # scaler'ı yükle

# Veriyi yükle ve pivot yap
def load_and_prepare_data():
    # Veriyi al
    data = db.new_product()

    # DataFrame'e çevir
    df = pd.DataFrame(data, columns=["customer_id", "category_id", "category_name", "total_spending", "product_id", "product_name"])

    # Pivotlama (kategori ve ürün bazlı harcamalar)
    category_pivot = df.pivot_table(index='customer_id',
                                    columns='category_name',
                                    values='total_spending',
                                    aggfunc='sum',
                                    fill_value=0)

    return df, category_pivot

# Kategoriyi pivot tablosuna ekle (Yeni ürün kategorisi)
def add_new_category_to_pivot(category_pivot, NEW_PRODUCT_CATEGORY):
    if NEW_PRODUCT_CATEGORY not in category_pivot.columns:
        category_pivot[NEW_PRODUCT_CATEGORY] = 0
    return category_pivot

# Tahmin fonksiyonu
def predict_customer_purchase(customer_id: str, product_name: str, category_name: str):
    df, category_pivot = load_and_prepare_data()

    # Kategoriyi pivot tablosuna ekle
    NEW_PRODUCT_CATEGORY = category_name
    category_pivot = add_new_category_to_pivot(category_pivot, NEW_PRODUCT_CATEGORY)

    if customer_id not in category_pivot.index:
        return None

    # "target" sütunu kaldırıldı, bu satır düzeltildi
    customer_data = category_pivot.loc[[customer_id]]
    customer_scaled = scaler.transform(customer_data)  # scaler'ı burada kullanıyoruz

    # Modeli kullanarak tahmin yap
    prediction = model.predict(customer_scaled)[0][0]
    predicted_label = int(prediction > 0.5)

    return predicted_label, prediction

# API POST isteği için endpoint

# Hata ayıklama için logging ayarlarını yapalım
logging.basicConfig(level=logging.DEBUG)

@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        # Parametreleri al
        product_name = request.product_name
        category_name = request.category_name
        customer_id = request.customer_id
        logging.debug(f"Received request: product_name={product_name}, category_name={category_name}, customer_id={customer_id}")

        # Müşteri için tahmin yap
        predicted_label, prediction = predict_customer_purchase(customer_id, product_name, category_name)
        logging.debug(f"Prediction result: predicted_label={predicted_label}, prediction={prediction}")

        if predicted_label is None:
            raise HTTPException(status_code=404, detail="Müşteri ID veritabanında bulunamadı.")

        # Tahmin sonucunu döndür
        result = {
            "customer_id": customer_id,
            "product_name": product_name,
            "category_name": category_name,
            "prediction_score": float(prediction),  # NumPy float -> Python float
            "prediction_label": "Alır" if predicted_label == 1 else "Almaz"
        }

        return result

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Uygulama başlatılacak
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)









# Uygulama çalıştırmak için bu komut gereklidir
# uvicorn question3_api:app --reload

