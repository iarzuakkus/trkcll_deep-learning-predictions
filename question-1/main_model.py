import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import joblib
from question1 import prepare_dataset
from tensorflow.keras.models import load_model


X , y = prepare_dataset()
df = pd.read_csv('data/extracted_data.csv')

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Sequential([
        Dense(16, input_dim=X.shape[1], activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs= 9, batch_size=16)

    # Model ve scaler'ı kaydet
    model.save("model/customer_model.h5")
    joblib.dump(scaler, "model/scaler.pkl")

    print("Model ve scaler kaydedildi.")
    return model, scaler


def predict_by_customer_id(customer_id, df):  
    # 2. İlgili müşteri satırını al
    customer_row = df[df['customer_id'] == customer_id]
    if customer_row.empty:
        return f"Müşteri bulunamadı: {customer_id}"

    # 3. Özellikleri al
    features = customer_row[['total_order', 'total_amount', 'avg_order_value']].values

    # 4. Scaler ve model'i yükle
    scaler = joblib.load("model/scaler.pkl")
    model = load_model("model/customer_model.h5")

    # 5. Tahmin yap
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0][0]

    return f"{customer_id} müşterisinin tekrar sipariş verme olasılığı: {prediction:.2f}"


if __name__ == "__main__":
   model, scaler = train_model(X,y)
   result = predict_by_customer_id("ALFKI",df)
   print(result)