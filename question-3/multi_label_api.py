from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np

# FastAPI uygulamasını başlatalım
app = FastAPI()

# Modeli yükle
model = load_model("model/multi_label_predictor.h5")

# Veriyi yükle ve hazırlık yapalım
df = pd.read_csv("data/new_product.csv")
pivot_df = df.pivot_table(index='customer_id',
                          columns='category_name',
                          values='total_spending',
                          aggfunc='sum',
                          fill_value=0)
category_names = pivot_df.columns.tolist()

# Öneri fonksiyonumuz
def recommend_categories_by_customer_id(model, customer_id, pivot_df, category_names, threshold=0.5):
    customer_vector = pivot_df.loc[customer_id].values
    probs = model.predict(customer_vector.reshape(1, -1))[0]
    recommended = [cat for cat, prob in zip(category_names, probs) if prob >= threshold]
    return recommended

# Pydantic ile giriş parametreleri
class CustomerRequest(BaseModel):
    customer_id: str
    threshold: float = 0.5

@app.post("/recommend_categories/")
def recommend_categories(request: CustomerRequest):
    recommended_categories = recommend_categories_by_customer_id(
        model=model,
        customer_id=request.customer_id,
        pivot_df=pivot_df,
        category_names=category_names,
        threshold=request.threshold
    )
    return {"customer_id": request.customer_id, "recommended_categories": recommended_categories}

# API'yi çalıştırmak için terminalde şu komutla başlatabiliriz:
# uvicorn multi_label_api:app --reload
