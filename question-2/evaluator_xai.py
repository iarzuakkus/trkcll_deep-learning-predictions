# evaluator_xai.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report, confusion_matrix

def evaluate_model(model, X_val, y_val):
    y_pred_prob = model.predict(X_val).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_val, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred, digits=4))
    print(f"ROC AUC:    {roc_auc_score(y_val, y_pred_prob):.4f}")
    print(f"Precision:  {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:     {recall_score(y_val, y_pred):.4f}")

    return y_pred_prob, y_pred

def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Kayıp Eğrisi')
    plt.show()

    plt.figure()
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Doğruluk Eğrisi')
    plt.show()

def explain_model_with_shap(model, X_train, X_val):
    try:
        import shap
        explainer = shap.GradientExplainer(model, X_train[:100])
        shap_values = explainer.shap_values(X_val[:100])[0]

        shap.summary_plot(shap_values, X_val[:100], feature_names=['avg_discount', 'total_quantity', 'total_spending'], show=False)
        plt.show()

        shap.dependence_plot('avg_discount', shap_values, X_val[:100], feature_names=['avg_discount', 'total_quantity', 'total_spending'], show=False)
        plt.show()
    except ImportError:
        print("shap kütüphanesi yüklenmemiş, XAI atlanıyor.")
    except Exception as e:
        print(f"XAI adımında hata oluştu: {e}")