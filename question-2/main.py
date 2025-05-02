# main.py
from data_loader import load_or_generate_labeled_data
from cost_sensitive import preprocess_and_train_model
from evaluator_xai import evaluate_model, plot_training_history, explain_model_with_shap

if __name__ == "__main__":
    df = load_or_generate_labeled_data()
    model, history, X_train, X_val, y_train, y_val = preprocess_and_train_model(df)
    y_pred_prob, y_pred = evaluate_model(model, X_val, y_val)
    explain_model_with_shap(model, X_train, X_val)
    plot_training_history(history)