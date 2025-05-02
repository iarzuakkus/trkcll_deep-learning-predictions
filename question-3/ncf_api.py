from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
import uvicorn
from tensorflow.keras.losses import MeanSquaredError

app = FastAPI()

# Modeli ve ID eşlemelerini yükle
model_path = "model/ncf_recommender.keras"

# MeanSquaredError özel tanımı
def custom_mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

# Modeli yükle
model = load_model(model_path, custom_objects={'MeanSquaredError': custom_mse})

# Veriyi yükle ve user_id/product_id'leri string yap
df = pd.read_csv("data/user_product_interactions.csv")
df['user_id'] = df['user_id'].astype(str)
df['product_id'] = df['product_id'].astype(str)

# Benzersiz kullanıcı ve ürün ID'lerini al
user_ids = df['user_id'].unique().tolist()
product_ids = df['product_id'].unique().tolist()

# ID -> index ve index -> ID map'leri
user_id_to_index = {x: i for i, x in enumerate(user_ids)}
product_id_to_index = {x: i for i, x in enumerate(product_ids)}
index_to_product_id = {i: x for i, x in enumerate(product_ids)}


# API input modeli
class UserRequest(BaseModel):
    user_id: str
    top_k: int = 10

@app.post("/recommend/")
def recommend_products(request: UserRequest):
    user_id = request.user_id
    top_k = request.top_k

    # Debug: kullanıcı ID'leri ve eşleşme kontrolü
    print("Tüm user_id'ler (veri içinden):", df['user_id'].unique())
    matched_rows = df[df['user_id'] == user_id]
    print(f"Girilen user_id: {user_id}, eşleşen satır sayısı: {len(matched_rows)}")

    if user_id not in user_id_to_index:
        return {"error": f"Kullanıcı ID'si bulunamadı: {user_id}"}

    user_index = user_id_to_index[user_id]

    # Kullanıcının daha önce gördüğü ürünleri al
    seen_products = matched_rows['product_id'].values
    seen_product_indices = [product_id_to_index[pid] for pid in seen_products if pid in product_id_to_index]

    # Görülmeyen ürün indekslerini bul
    all_product_indices = np.array(range(len(product_ids)))
    unseen_product_indices = np.setdiff1d(all_product_indices, seen_product_indices)

    if len(unseen_product_indices) == 0:
        return {"message": "Kullanıcının görmediği başka ürün yok."}

    # Model input'u: kullanıcı indeksleri ve ürün indeksleri
    user_array = np.full(len(unseen_product_indices), user_index)

    # Puan tahminlerini al
    predicted_ratings = model.predict([user_array, unseen_product_indices], verbose=0)

    # En iyi top_k tahmini bul
    top_indices = np.argsort(predicted_ratings.flatten())[::-1]
    top_product_indices = unseen_product_indices[top_indices][:top_k]
    recommended_ids = [product_ids[i] for i in top_product_indices]

    return {
        "user_id": user_id,
        "recommended_product_ids": recommended_ids
    }


# Opsiyonel: localde çalıştırmak istersen
if __name__ == "__main__":
    uvicorn.run("ncf_api:app", host="127.0.0.1", port=8000, reload=True)




