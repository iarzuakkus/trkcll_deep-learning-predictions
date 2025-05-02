# main_model.py

import joblib
import numpy as np
import tensorflow as tf
from data_loader import load_or_generate_labeled_data

MODEL_PATH = 'model.h5'
SCALER_PATH = 'scaler.pkl'

def train_and_save_model():
    """
    Etiketlenmiş veriyi yükler, tüm veriyle modeli eğitir,
    ardından scaler ve modeli kaydeder.
    """
    # Veri yükle
    df = load_or_generate_labeled_data()
    X = df[['avg_discount', 'total_quantity', 'total_spending']].values
    y = df['return_risk'].values

    # Ölçekleyici oluştur ve kaydet
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    joblib.dump(scaler, SCALER_PATH)
    X_scaled = scaler.transform(X)

    # Model mimarisi
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Modeli tüm veriyle eğit
    model.fit(
        X_scaled, y,
        epochs=50,
        batch_size=32,
        class_weight={0: 1.0, 1: 5.0},
        verbose=1
    )

    # Modeli kaydet
    model.save(MODEL_PATH)


def predict(customer_id: str):
    """
    Customer ID'ye göre labeled_data.csv içinden avg_discount, total_quantity ve
    total_spending değerlerini alır, scaler ve model ile risk tahmini yapar.

    Returns:
        dict: {'customer_id': str, 'probability': float, 'prediction': int}
    """
    # Veriyi yükle ve müşteri değerlerini çek
    df = load_or_generate_labeled_data()
    row = df[df['customer_id'] == customer_id]
    if row.empty:
        raise ValueError(f"Customer ID {customer_id} bulunamadı.")
    values = row[['avg_discount', 'total_quantity', 'total_spending']].iloc[0].values

    # Model ve scaler yükle
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Ölçekle ve tahmin et
    X = np.array([values])
    X_scaled = scaler.transform(X)
    prob = float(model.predict(X_scaled)[0][0])

    return {
        'customer_id': customer_id,
        'probability': prob,
        'prediction': int(prob >= 0.5)
    }


if __name__ == '__main__':
    # Modeli eğit ve kaydet
    print("Model eğitiliyor ve kaydediliyor...")
    train_and_save_model()
    print("Model kaydedildi.")

    # ALFKI için tahmin yap
    cid = 'OCEAN'
    print(f"{cid} için tahmin yapılıyor...")
    result = predict(cid)
    print(f"Customer ID: {result['customer_id']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Prediction: {result['prediction']}")
