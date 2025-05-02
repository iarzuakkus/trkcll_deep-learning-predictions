# TRKCLL Deep Learning Predictions

## Collaborators

* İlayda Arzu Akkuş
* Elif Erdal
* Aybüke Altuntaş
* Seda Mürütsoy
* Zeynep Melike Işık

A comprehensive repository containing deep learning and machine learning solutions for customer reorder prediction, product return risk evaluation, and product recommendation. Built with Python, TensorFlow, scikit-learn, SQLAlchemy, and FastAPI.

## Table of Contents

* [Project Structure](#project-structure)
* [Setup](#setup)
* [Database Configuration](#database-configuration)
* [Dependencies](#dependencies)
* [Question 1: Reorder Prediction](#question-1-reorder-prediction)
* [Question 2: Return Risk Evaluation](#question-2-return-risk-evaluation)
* [Question 3: Product Recommendation and Purchase Prediction](#question-3-product-recommendation-and-purchase-prediction)
* [License](#license)

## Project Structure

```
├── db.py                     # Database helper using SQLAlchemy
├── question-1/              # Reorder prediction pipeline
│   ├── data/                # Sample datasets (orders.csv, extracted_data.csv, combined_summary.csv)
│   ├── class_imbalance.py   # Handles class-weighting and imbalance techniques
│   ├── data_augmentation.py # Implements data augmentation strategies
│   ├── temporalfeatures.py  # Extracts temporal features from order history
│   ├── main_model.py        # Defines and trains the neural network model
│   ├── fast_api.py          # FastAPI app for serving reorder predictions
│   └── question1.py         # Orchestrator script for end-to-end workflow
├── question-2/              # Return risk evaluation with XAI and cost-sensitive learning
│   ├── data/                # Labeled data and raw extracts
│   ├── data_loader.py       # Loads or generates labeled return-risk dataset
│   ├── cost_sensitive.py    # Preprocesses data and trains a cost-sensitive model
│   ├── evaluator_xai.py     # Evaluation metrics, training history plots, SHAP explanations
│   ├── main_model.py        # Full retraining and model persistence logic
│   ├── main_trainer.py      # Alternative training entry point
│   └── fast_api.py          # FastAPI app for return-risk endpoints
└── question-3/              # Product recommendation and purchase prediction
    ├── data/                # Interaction logs, product catalogs, prediction outputs
    ├── question3.py         # Script for single-customer purchase prediction
    ├── ncf_recommender.py   # Implements Neural Collaborative Filtering model
    ├── multi_label_predictor.py # Multi-label classification for category predictions
    └── question3_api.py     # FastAPI app exposing recommendation APIs
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/trkcll_deep-learning-predictions.git
   cd trkcll_deep-learning-predictions
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # on Windows: venv\\Scripts\\activate
   ```

3. Install required packages (example):

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn tensorflow shap fastapi uvicorn sqlalchemy psycopg2-binary joblib
   ```

## Database Configuration

* Ensure PostgreSQL is running and accessible at:

  * Host: localhost
  * Port: 5432
  * Database: gyk1
  * User: postgres
  * Password: 12345
* Tables required:

  * `customers`
  * `orders`
  * `order_details`
  * Custom query in `db.py` for product returns

Adjust the connection string in `db.py` if your settings differ.

## Dependencies

* Python 3.7 or higher
* pandas, numpy
* scikit-learn
* imbalanced-learn
* TensorFlow (>=2.x)
* shap (optional, for XAI)
* FastAPI, uvicorn
* SQLAlchemy, psycopg2-binary
* joblib

## Question 1: Reorder Prediction

Predict whether a customer will place a new order within the next 6 months.

**Usage**:

1. Navigate to `question-1/`:

   ```bash
   cd question-1
   ```
2. Run the end-to-end script:

   ```bash
   python question1.py
   ```
3. Train and save the model:

   ```bash
   python main_model.py
   ```
4. Launch the API:

   ```bash
   uvicorn fast_api:app --reload
   ```

   Access Swagger UI at `http://127.0.0.1:8000/docs`.

## Question 2: Return Risk Evaluation

Estimate the risk of a product being returned using cost-sensitive learning and interpretability.

**Usage**:

1. Navigate to `question-2/`:

   ```bash
   cd question-2
   ```
2. Train and evaluate:

   ```bash
   python main.py
   ```
3. Persist full model:

   ```bash
   python main_model.py
   ```
4. Serve endpoints:

   ```bash
   uvicorn fast_api:app --reload
   ```

## Question 3: Product Recommendation and Purchase Prediction

Combine collaborative filtering and multi-label classification to recommend products and predict new-item purchases.

**Usage**:

1. Navigate to `question-3/`:

   ```bash
   cd question-3
   ```
2. Train NCF recommender:

   ```bash
   python ncf_recommender.py
   ```
3. Train multi-label predictor:

   ```bash
   python multi_label_predictor.py
   ```
4. Make single-customer predictions:

   ```bash
   python question3.py
   ```
5. Run the API:

   ```bash
   uvicorn question3_api:app --reload
   ```

