from data_augmentation import (
    label_within_6_months,
    generate_customer_ids,
    generate_synthetic_data,
    generate_summary_with_labels
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def preprocess_data(X, y, smote=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if smote:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    return X_train_res, X_test_scaled, y_train_res, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return model

if __name__ == "__main__":
    # 1. Sahte veri üret
    customer_ids = generate_customer_ids()
    df_fake_orders = generate_synthetic_data(customer_ids)
    df_fake_summary = generate_summary_with_labels(df_fake_orders)

    # 2. Etiketle
    df_fake_summary = label_within_6_months(df_fake_summary, df_fake_orders)

    # 3. Özellikler ve hedef değişken
    X = df_fake_summary[['total_order', 'total_amount', 'avg_order_value']]
    y = df_fake_summary['reordered_within_6_months']

    # 4. Dengesiz veri sorununa çözüm: SMOTE + class_weight
    X_train, X_test, y_train, y_test = preprocess_data(X, y, smote=True)
    model = train_and_evaluate(X_train, X_test, y_train, y_test)
