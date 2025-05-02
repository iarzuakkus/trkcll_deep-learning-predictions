import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
from question1 import train_model, prepare_dataset

# Sabit rastgelelik
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def generate_random_id():
    return ''.join(random.choices(string.ascii_uppercase, k=5))

def generate_customer_ids(n_customers=200):
    unique_customers = set()
    while len(unique_customers) < n_customers:
        unique_customers.add(generate_random_id())
    return list(unique_customers)

def generate_synthetic_data(customer_ids, n_rows=4000):
    rows = []
    customer_last_dates = {}
    customer_order_counts = {}

    start_date = datetime(1998, 1, 1)

    # 15 müşteri 6 ay içinde tekrar sipariş verebilir
    early_repeat_customers = set(random.sample(customer_ids, 15))

    for _ in range(n_rows):
        cust_id = random.choice(customer_ids)
        total_order = np.random.randint(1, 100)
        avg_order_value = np.round(np.random.uniform(50, 500), 2)
        total_amount = np.round(total_order * avg_order_value * np.random.uniform(0.9, 1.1), 2)

        if cust_id not in customer_last_dates:
            order_date = start_date + timedelta(days=random.randint(0, 365))
            customer_order_counts[cust_id] = 1
        else:
            last_date = customer_last_dates[cust_id]
            order_count = customer_order_counts.get(cust_id, 1)

            if cust_id in early_repeat_customers:
                # Bu 15 müşteri istediği zaman sipariş verebilir (örnek çeşitliliği için)
                order_date = last_date + timedelta(days=random.randint(30, 400))
            else:
                # Diğer müşteriler her zaman 6 aydan (180 gün) sonra sipariş verebilir
                order_date = last_date + timedelta(days=random.randint(180, 400))

            customer_order_counts[cust_id] = order_count + 1

        customer_last_dates[cust_id] = order_date

        rows.append({
            'customer_id': cust_id,
            'total_order': total_order,
            'total_amount': total_amount,
            'avg_order_value': avg_order_value,
            'order_date': order_date.strftime('%Y-%m-%d')
        })

    return pd.DataFrame(rows)

def generate_summary_with_labels(df_fake):
    df_fake['order_date'] = pd.to_datetime(df_fake['order_date'])
    df_summary = df_fake.sort_values('order_date').groupby('customer_id').nth(-2).reset_index()
    df_summary = df_summary.rename(columns={'order_date': 'last_order_date'})

    def has_reorder_within_6_months(row):
        cid = row['customer_id']
        last_date = row['last_order_date']
        future_orders = df_fake[
            (df_fake['customer_id'] == cid) &
            (df_fake['order_date'] > last_date) &
            (df_fake['order_date'] <= last_date + pd.DateOffset(days=180))
        ]
        return 1 if not future_orders.empty else 0

    df_summary['reordered_within_6_months'] = df_summary.apply(has_reorder_within_6_months, axis=1)
    return df_summary

def label_within_6_months(df_summary, df_orders):
    df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
    df_summary['last_order_date'] = pd.to_datetime(df_summary['last_order_date'])

    def gave_order_within_6_months(row):
        customer_id = row['customer_id']
        last_date = row['last_order_date']
        six_months_later = last_date + pd.DateOffset(months=6)
        future_orders = df_orders[
            (df_orders['customer_id'] == customer_id) &
            (df_orders['order_date'] > last_date) &
            (df_orders['order_date'] <= six_months_later)
        ]
        return 1 if not future_orders.empty else 0

    df_summary['reordered_within_6_months'] = df_summary.apply(gave_order_within_6_months, axis=1)
    return df_summary
    
def combine_with_real_data(df_fake_summary=None, df_fake_orders=None):
    X_real, y_real = prepare_dataset("data/extracted_data.csv", "data/orders.csv")

    if df_fake_summary is not None and df_fake_orders is not None:
        df_fake_summary = label_within_6_months(df_fake_summary, df_fake_orders)
        X_fake = df_fake_summary[['total_order', 'total_amount', 'avg_order_value']]
        y_fake = df_fake_summary['reordered_within_6_months']

        X_combined = pd.concat([X_real, X_fake], ignore_index=True)
        y_combined = pd.concat([y_real, y_fake], ignore_index=True)
    else:
        X_combined, y_combined = X_real, y_real

    print("Toplam örnek sayısı:", len(X_combined))
    print("Pozitif sınıf oranı:", y_combined.mean())
    return X_combined, y_combined

def train_on_combined_summary(df_combined_summary):
    df_combined_summary['last_order_date'] = pd.to_datetime(df_combined_summary['last_order_date'])

    X = df_combined_summary[['total_order', 'total_amount', 'avg_order_value']]
    y = df_combined_summary['reordered_within_6_months']

    model, scaler = train_model(X, y)
    print(df_combined_summary)
    return model, scaler, df_combined_summary

if __name__ == "__main__":
    customer_ids = generate_customer_ids()
    df_fake_orders = generate_synthetic_data(customer_ids)
    df_fake_summary = generate_summary_with_labels(df_fake_orders)

    # combine_with_real_data artık X, y döndürüyor
    X, y = combine_with_real_data(df_fake_summary=df_fake_summary, df_fake_orders=df_fake_orders)

    # Model eğitimi
    from question1 import train_model
    model, scaler = train_model(X, y)

    print("Model eğitim tamamlandı.")
    print("Pozitif etiketli örnek sayısı:", y.sum())
    #pd.concat([X, y], axis=1).to_csv("data/combined_summary.csv", index=False)
