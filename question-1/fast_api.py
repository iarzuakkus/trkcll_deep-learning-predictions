from fastapi import FastAPI, Query, HTTPException
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

# Question1 pipeline
from question1 import prepare_dataset, train_model, evaluate_model_metrics
# Temporal features pipeline
from temporalfeatures import build_and_train_temporal_model
# Data augmentation pipeline
from data_augmentation import (
    label_within_6_months,
    generate_customer_ids,
    generate_synthetic_data,
    generate_summary_with_labels,
    combine_with_real_data
)
#Class Imblance
from class_imbalance import preprocess_data, train_and_evaluate

#predict
from main_model import predict_by_customer_id

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})

# ——— Global initialization ———
# 1) Question1 data & model
X_q, y_q = prepare_dataset("data/extracted_data.csv", "data/orders.csv")
model_q, scaler_q = train_model(X_q, y_q)

# 2) Temporal‐features data & model
#    build_and_train_temporal_model returns: model, scaler, X, y
model_t, scaler_t, X_t, y_t = build_and_train_temporal_model()

# 3) Combined‐augmentation data & model
customer_ids = generate_customer_ids()
df_fake_orders = generate_synthetic_data(customer_ids)
df_fake_summary = generate_summary_with_labels(df_fake_orders)

# combine_with_real_data artık X, y döndürüyor
X_c, y_c = combine_with_real_data(df_fake_summary=df_fake_summary, df_fake_orders=df_fake_orders)
model_c, scaler_c = train_model(X_c, y_c)

#main df

# Veri çerçevesini global olarak yükleyelim:
df_main = pd.read_csv("data/extracted_data.csv")

@app.get("/metrics")
def get_metrics(
    pipeline: str = Query(
        "question1",
        enum=["question1", "temporal", "combined"],
        description="Which metrics pipeline to use"
    ),
    threshold: float = Query(
        0.5, ge=0.0, le=1.0,
        description="Probability threshold for positive class"
    )
):
    """
    Returns accuracy, precision, recall & F1 for the chosen pipeline.
    `/metrics?pipeline=combined&threshold=0.6` 
    """
    if pipeline == "question1":
        metrics = evaluate_model_metrics(model_q, scaler_q, X_q, y_q, threshold=threshold)
    elif pipeline == "temporal":
        metrics = evaluate_model_metrics(model_t, scaler_t, X_t, y_t, threshold=threshold)
    elif pipeline == "combined":
        metrics = evaluate_model_metrics(model_c, scaler_c, X_c, y_c, threshold=threshold)
    else:
        raise HTTPException(status_code=400, detail="Unknown pipeline")

    return { name: round(val, 4) for name, val in metrics.items() }

@app.get("/class_imbalance_full")
def class_imbalance_full(
    smote: bool = Query(True, description="SMOTE uygula mı?"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Pozitif sınıf eşiği")
):
    """
    Sentetik veri oluşturup LogisticRegression ile eğitir ve test seti performansını döner.
    """
    # 1) Sentetik müşteri & sipariş + özet + etiket
    customer_ids   = generate_customer_ids()
    df_orders      = generate_synthetic_data(customer_ids)
    df_summary     = generate_summary_with_labels(df_orders)
    df_labeled     = label_within_6_months(df_summary, df_orders)

    # 2) Özellikler & hedef
    X = df_labeled[['total_order', 'total_amount', 'avg_order_value']]
    y = df_labeled['reordered_within_6_months']

    # 3) Ön işleme: split/scale/SMOTE
    X_train, X_test, y_train, y_test = preprocess_data(X, y, smote=smote)

    # 4) Model eğit
    model = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 5) Test seti üzerinde tahmin & metrikler
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred)

    return {
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "confusion_matrix": cm.tolist(),
        "labels": ["negative", "positive"]
    }

@app.get("/predict_customer")

def predict_customer(
    customer_id: str = Query(..., description="Tahmin için müşteri ID'si")
):
    """
    Belirtilen customer_id için 'tekrar sipariş verme olasılığı'nı döner.
    """
    result = predict_by_customer_id(customer_id, df_main)
    if result.startswith("Müşteri bulunamadı"):
        raise HTTPException(status_code=404, detail=result)
    return {"prediction": result}
#python -m uvicorn fast_api:app --reload
#http://127.0.0.1:8000/docs