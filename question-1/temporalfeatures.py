import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from question1 import train_model, prepare_dataset  # Yeni fonksiyon içe aktarılıyor

def add_temporal_features(data):
    data['month'] = data['last_order_date'].dt.month
    data['season'] = data['month'].apply(lambda x: 'Spring' if x in [3, 4, 5] else
                                                 'Summer' if x in [6, 7, 8] else
                                                 'Fall' if x in [9, 10, 11] else 'Winter')

    # One-hot encoding for season
    data = pd.get_dummies(data, columns=['season'])
    for col in ['season_Spring', 'season_Summer', 'season_Fall', 'season_Winter']:
        if col not in data.columns:
            data[col] = 0

    return data

def build_and_train_temporal_model():
    # Etiketlenmiş veriyi hazırla
    X, y = prepare_dataset()  

    # Temporal feature'ları eklemek için özet veriyi oku
    data = pd.read_csv("data/extracted_data.csv")
    data['last_order_date'] = pd.to_datetime(data['last_order_date'])
    data = add_temporal_features(data)

    # Yeni feature'ları X'e ekle
    X['month']          = data['month']
    X['season_Spring']  = data['season_Spring']
    X['season_Summer']  = data['season_Summer']
    X['season_Fall']    = data['season_Fall']
    X['season_Winter']  = data['season_Winter']

    # Modeli eğit
    model, scaler = train_model(X, y)

    # Artık X ve y'yi de dönüyoruz
    return model, scaler, X, y


if __name__ == "__main__":
    model, scaler, X_temporal, y_temporal = build_and_train_temporal_model()

