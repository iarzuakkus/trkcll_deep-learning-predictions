import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
from pathlib import Path


# __file__ --> .../question-1/question1.py
root = Path(__file__).resolve().parents[1]   # iki seviye yukarı, projeni̇n kökü
sys.path.insert(0, str(root))
from db import Database

def extract_and_save_data():
    db = Database()
    summary_data = db.get_customer_order_summary()
    df_summary = pd.DataFrame(summary_data, columns=["customer_id", "total_order", "total_amount", "avg_order_value", "last_order_date"])
    df_summary.to_csv("data/extracted_data.csv", index=False)
    
    order_data = db.orders()
    df_orders = pd.DataFrame(order_data, columns=["customer_id", "order_date"])
    df_orders.to_csv("data/orders.csv", index=False)
    db.close()
    print("Veriler kaydedildi.")

import pandas as pd

def prepare_dataset(summary_path="data/extracted_data.csv", orders_path="data/orders.csv"):
    df = pd.read_csv(summary_path)
    orders = pd.read_csv(orders_path)

    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    orders['order_date'] = pd.to_datetime(orders['order_date'])

    # Her müşteri için sipariş tarihlerini sırala ve farklara bak
    def customer_has_quick_repeat(customer_id):
        dates = orders[orders['customer_id'] == customer_id]['order_date'].sort_values()
        diffs = dates.diff().dropna().dt.days
        return int((diffs < 180).any())

    df['reordered_within_6_months'] = df['customer_id'].apply(customer_has_quick_repeat)

    print(df)
    # Silmek istediğin sütunlar
    drop_cols = ['reordered_within_6_months', 'last_order_date', 'customer_id']

    # Mevcut olanları filtrele
    existing_drop = [col for col in drop_cols if col in df.columns]

   # Silinmesini istediğiniz sütunlar
    drop_cols = ['reordered_within_6_months', 'last_order_date', 'customer_id']

    # Var olanları hata fırlatmadan düş
    X = df.drop(columns=drop_cols, errors='ignore')


    y = df['reordered_within_6_months']
    return X, y


def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(16, input_dim=X.shape[1], activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=9, batch_size=16, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")
    return model, scaler

def evaluate_model_metrics(model, scaler, X, y,threshold):
    """
    Model ve scaler ile X üzerinde tahmin yapar, ardından
    accuracy, precision, recall ve F1-score'u hesaplar ve ekrana yazdırır.
    Returns a dict with all metrik değerleri.
    """
    # Özellikleri ölçekle
    X_scaled = scaler.transform(X)
    # Olasılık tahminleri al
    y_pred_prob = model.predict(X_scaled).ravel()
    # 0.5 eşik değeriyle sınıfa dönüştür
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Metrikleri hesapla
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    # Ekrana yazdır
    print(f"Accuracy : {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall   : {rec:.2f}")
    print(f"F1 Score : {f1:.2f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

if __name__ == "__main__":
    extract_and_save_data()
    X, y = prepare_dataset()
    model, scaler = train_model(X, y)
    # Tüm veri seti üzerinde metrikleri hesapla
    metrics = evaluate_model_metrics(model, scaler, X, y,0.5)
    sum_reordered_within_6_months = y.sum()
    print(sum_reordered_within_6_months)