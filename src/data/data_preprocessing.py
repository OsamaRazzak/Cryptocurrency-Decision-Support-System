import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def data_cleaning_and_feature_engineering(data):
    """Fill missing values and add technical indicators."""
    try:
        scaler = MinMaxScaler()
        for crypto, crypto_data in data.items():
            crypto_data.ffill(inplace=True)
            crypto_data["MA_7"] = crypto_data["Close"].rolling(window=7).mean()
            crypto_data["Volatility"] = crypto_data["Close"].rolling(window=30).std()
            crypto_data.bfill(inplace=True)

            data[crypto][['Open', 'High', 'Low', 'Close', 'Volume', 'MA_7', 'Volatility']] = \
                scaler.fit_transform(data[crypto][['Open', 'High', 'Low', 'Close', 'Volume', 'MA_7', 'Volatility']])
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")

