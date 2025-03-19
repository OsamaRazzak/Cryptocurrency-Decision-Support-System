import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def arima_forecast(raw_data, crypto_choice, select_time):

    '''Perform ARIMA forecasting on the selected cryptocurrency.'''
    try: 
        crypto_df = raw_data[crypto_choice].reset_index()
        crypto_df['Date'] = pd.to_datetime(crypto_df['Date'])  
        train_crypto_data = crypto_df['Close']
        
        model = ARIMA(train_crypto_data, order=(30,0,0))
        model_fit = model.fit()
        arima_prediction = model_fit.forecast(steps=select_time)

        last_date = crypto_df['Date'].iloc[-1]  
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=select_time)

        arima_predictions = pd.DataFrame({'Date': future_dates, 'ARIMA_Prediction': arima_prediction})
        arima_predictions['Date'] = pd.to_datetime(arima_predictions['Date'])

        arima_predictions.set_index('Date', inplace=True)
        return arima_predictions
    except Exception as e:
        logging.error(f"Error in ARIMA forecast: {e}")
        return None

def prophet_forecast(raw_data, crypto_choice, select_time):

    '''Perform Prophet forecasting on the selected cryptocurrency.'''
    try:
        crypto_data = raw_data[crypto_choice].reset_index()      
        prophet_crypto_data = crypto_data.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        prophet_model = Prophet()
        prophet_model.fit(prophet_crypto_data)
        future = prophet_model.make_future_dataframe(periods=select_time)         
        prophet_forecast = prophet_model.predict(future)

        prophet_pred = prophet_forecast.set_index('ds').loc[future['ds'], 'yhat']        
        return prophet_pred
    
    except Exception as e:
        logging.error(f"Error in Prophet forecast: {e}")
        return None


def lstm_forecast(crypto_choice, train_data, test_data):
    """
    Forecast cryptocurrency prices using LSTM.
    """
    try:
        X_train, y_train = train_data[crypto_choice]
        X_test, _ = test_data[crypto_choice]
        
        def build_lstm_model(input_shape):
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mse")
            return model

        lstm_model = build_lstm_model(X_train.shape[1:])
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        y_pred = lstm_model.predict(X_test).flatten()
        return y_pred
    except Exception as e:
        logging.error(f"Error in lstm_forecast: {e}")
        return None
    

def xgboost_forecast(crypto_choice, train_data, test_data):
    """
    Forecast cryptocurrency prices using XGBoost.
    """
    try:
        X_train, y_train = train_data[crypto_choice]
        X_test, _ = test_data[crypto_choice]
        
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgboost_model.fit(X_train_2d, y_train)
        y_pred = xgboost_model.predict(X_test_2d).flatten()
        return y_pred
    except Exception as e:
        logging.error(f"Error in xgboost_forecast: {e}")
        return None
    
