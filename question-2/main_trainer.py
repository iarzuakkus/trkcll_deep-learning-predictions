# train_evaluate_model.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from data_loader import load_or_generate_labeled_data


def load_data():
    """
    Etiketlenmiş veriyi yükler ve özellik (X) ile hedef (y) dizilerini döner.
    """
    df = load_or_generate_labeled_data()
    X = df[['avg_discount', 'total_quantity', 'total_spending']].values
    y = df['return_risk'].values
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Veriyi train/test olarak böler, StandardScaler ile ölçekler ve
    ölçeklenmiş dizileri ile scaler objesini döner.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def build_model(input_dim):
    """
    Sinir ağı mimarisini oluşturur ve derler.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, X_train, y_train,
                validation_split=0.1, epochs=9,
                batch_size=32, class_weight={0:1.0,1:5.0},
                patience=5):
    """
    Modeli eğitir ve eğitim geçmişini döner.
    """
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[es]
    )
    return history


def evaluate_model(model, X_test, y_test):
    """
    Test verisi üzerinde tahmin yapar, metrikleri hesaplar ve döner.
    """
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    scores = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, digits=4)
    }
    return scores


def main():
    # 1. Veri yükle
    X, y = load_data()

    # 2. Böl ve ölçekle
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    # 3. Modeli oluştur
    model = build_model(X_train.shape[1])

    # 4. Eğit
    _ = train_model(model, X_train, y_train)

    # 5. Değerlendir
    scores = evaluate_model(model, X_test, y_test)
    
    print('--- Evaluation Results ---')
    print(f"Accuracy: {scores['accuracy']:.4f}")
    print(f"Precision: {scores['precision']:.4f}")
    print(f"Recall: {scores['recall']:.4f}")
    print(f"F1-score: {scores['f1']:.4f}")
    print(f"ROC AUC: {scores['roc_auc']:.4f}")
    print('--- Confusion Matrix ---')
    print(scores['confusion_matrix'])
    print('\n--- Classification Report ---')
    print(scores['classification_report'])

if __name__ == '__main__':
    main()
