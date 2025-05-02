# data_loader.py
import pandas as pd

import sys
from pathlib import Path


# __file__ --> .../question-1/question1.py
root = Path(__file__).resolve().parents[1]   # iki seviye yukarı, projeni̇n kökü
sys.path.insert(0, str(root))
from db import Database

def load_or_generate_labeled_data(csv_path='data/labeled_data.csv'):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        db = Database()
        data = db.product_return()
        df = pd.DataFrame(data, columns=["customer_id", "avg_discount", "total_quantity", "total_spending"])
        db.close()

        median_discount = df['avg_discount'].median()
        median_spending = df['total_spending'].median()
        df['return_risk'] = ((df['avg_discount'] > median_discount) & (df['total_spending'] < median_spending)).astype(int)
        df.to_csv(csv_path, index=False)
    return df