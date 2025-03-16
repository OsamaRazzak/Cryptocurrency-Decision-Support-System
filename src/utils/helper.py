import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_data(crypto_data, target_col="Close", time_steps=60):
    """
    Prepare time series data for training.
    """
    try:
        X, y = [], []
        for i in range(time_steps, len(crypto_data)):
            X.append(crypto_data.iloc[i - time_steps:i].values)
            y.append(crypto_data[target_col].iloc[i])
        return np.array(X), np.array(y)
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        return None, None
    
def split_data(data, time_steps=60, test_size=0.2):
    """
    Prepares and splits data into training and testing sets.
    """
    train_data, test_data = {}, {}
    
    try:
        prepared_data = {crypto: prepare_data(crypto_data, time_steps=time_steps) for crypto, crypto_data in data.items()}
        
        for crypto, (X, y) in prepared_data.items():
            if X is None or y is None:
                continue  # Skip if data preparation failed
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            train_data[crypto] = (X_train, y_train)
            test_data[crypto] = (X_test, y_test)
        
        return train_data, test_data
    except Exception as e:
        print(f"Error in split_data: {e}")
        return {}, {}

def data_values(raw_data, crypto_choice):
    try:
        crypto_df = raw_data[crypto_choice]
        return crypto_df[["Close",  "High", "Low", "Open"]]
    except Exception as e:
        logging.error(f"Error in data_values: {e}")
        return None
    
def moving_average(raw_data , crypto_choice):
    try:
        crypto_df = raw_data[crypto_choice]
        crypto_df["MA_7"] = crypto_df["Close"].rolling(window=7).mean()
        crypto_df["30-day MA"] = crypto_df["Close"].rolling(window=30).mean()
        return crypto_df[["Close", "MA_7", "30-day MA"]]
    except Exception as e:
        logging.error(f"Error in moving_average: {e}")
        return None
