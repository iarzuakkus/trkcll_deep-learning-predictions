from fastapi import FastAPI
import pandas as pd
import numpy as np
from cost_sensitive import preprocess_and_train_model
from evaluator_xai import evaluate_model, explain_model_with_shap, plot_training_history
from main_model import train_and_save_model, predict
app = FastAPI(title="Return Risk Evaluation API")

CSV_PATH = "data/labeled_data.csv"  # CSV dosyasının yolu

@app.get("/train-model")
def train_model():
    df = pd.read_csv(CSV_PATH)
    model, history, X_train, X_val, y_train, y_val = preprocess_and_train_model(df)
    return {
    "status": "Model başarıyla eğitildi.",
    "final_val_accuracy": float(history.history['val_accuracy'][-1]),
    "final_val_loss": float(history.history['val_loss'][-1])
}


@app.get("/evaluate")
def evaluate():
    df = pd.read_csv(CSV_PATH)

    model, history, X_train, X_val, y_train, y_val = preprocess_and_train_model(df)
    y_pred_prob, y_pred = evaluate_model(model, X_val, y_val)

    return {
        "roc_auc": float(np.round(np.mean(y_pred_prob), 4)),
        "y_pred_sample": y_pred[:10].tolist()
    }

@app.get("/plot-history")
def plot_history():
    df = pd.read_csv(CSV_PATH)
    model, history, *_ = preprocess_and_train_model(df)

    # Grafik çizimi (dosyaya kaydeder)
    plot_training_history(history)

    return {"status": "Eğitim geçmişi grafikleri çizildi"}

@app.get("/predict/{customer_id}")
def predict_endpoint(customer_id: str):
    """
    Tek bir Customer ID alır, modelden iade riskini tahmin eder.
    """
    try:
        result = predict(customer_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))



#python -m uvicorn fast_api:app --reload
#http://127.0.0.1:8000/docs