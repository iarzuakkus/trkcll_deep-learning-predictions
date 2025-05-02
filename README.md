# TURKCELL Geleceği Yazan Kadınlar Deep Learning Predictions

## Collaborators PAIR-8

* İlayda Arzu Akkuş
* Elif Erdal
* Aybüke Altuntaş
* Seda Mürütsoy
* Zeynep Melike Işık
# trkcll\_deep-learning-predictions

---

##  Project Overview

This repository contains three deep learning use cases built on the Northwind sample database, each exposed via a FastAPI-based REST endpoint:

1. **Order Habit Prediction**

   * Predict whether a customer will place another order within the next six months, using features such as total spend, order count, and average order value.

2. **Return Risk Scoring**

   * Estimate the likelihood of an order being returned, based on discount rate, quantity, and total spend in the order details.

3. **New Product Purchase Potential**

   * Forecast the chance that a customer will invest in a newly launched product category, leveraging their past category-level spending patterns.

Each use case includes data preparation scripts, model definitions, training pipelines, and a `/predict` endpoint to serve real-time predictions.

---

##  Project Structure

```text
trkcll_deep-learning-predictions/
├── __pycache__/
├── question-1/             # Order Habit Prediction
│   ├── __pycache__/
│   ├── data/
│   ├── model/
│   ├── class_imbalance.py
│   ├── data_augmentation.py
│   ├── fast_api.py
│   ├── main_model.py
│   ├── question1.py
│   └── temporalfeatures.py
├── question-2/             # Return Risk Scoring
│   ├── __pycache__/
│   ├── data/
│   ├── cost_sensitive.py
│   ├── data_loader.py
│   ├── db.py
│   ├── evaluator_xai.py
│   ├── fast_api.py
│   ├── main.py
│   ├── main_model.py
│   ├── main_trainer.py
│   ├── model.h5
│   └── scaler.pkl
├── question-3/             # New Product Purchase Potential
│   ├── __pycache__/
│   ├── data/
│   ├── model/
│   ├── multi_label_api.py
│   ├── multi_label_predictor.py
│   ├── ncf_api.py
│   ├── ncf_recommender.py
│   ├── question3.py
│   └── question3_api.py
├── db.py
├── requirements.txt
└── .gitignore
```

---

##  Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/iarzuakkus/trkcll_deep-learning-predictions.git
   cd trkcll_deep-learning-predictions
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

##  API Usage

Start the FastAPI server and access the interactive docs:

```bash
python -m uvicorn fast_api:app --reload
```

Open your browser to:

```
http://127.0.0.1:8000/docs
```

---

##  R\&D Topics

* **Temporal Features:** Incorporate seasonality effects (e.g., summer vs. winter patterns).
* **Data Augmentation:** Generate synthetic samples to boost model generalization.
* **Class Imbalance:** Apply class weights or SMOTE to handle skewed labels.
* **Cost-sensitive Learning:** Penalize misclassifications based on business costs.
* **Explainable AI (XAI):** Use SHAP or LIME for model interpretability.
* **Recommendation Systems:** Explore Neural Collaborative Filtering and AutoEncoders.
* **Multi-label Prediction:** Support simultaneous predictions for multiple categories.

---
