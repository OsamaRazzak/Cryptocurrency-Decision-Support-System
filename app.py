import streamlit as st
from src.data.data_fetcher import fetch_crypto_data
from src.data.data_preprocessing import data_cleaning_and_feature_engineering
from src.data.pca_kmean import pca_and_kmean_clustering
from src.data.correlation import perform_correlation
from src.utils.helper import split_data, data_values, moving_average
from src.utils.visualization import plot_closing_price, plot_correlation_heatmap, plot_clusters
from src.models.forcasting_models import arima_forecast,prophet_forecast,lstm_forecast,xgboost_forecast
import logging
import copy
import warnings

# Suppress warnings
warnings.simplefilter("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("crypto_pipeline.log"), logging.StreamHandler()]
)
tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
          "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "AVAX-USD",
          "SHIB-USD", "TRX-USD", "LTC-USD", "UNI-USD", "ATOM-USD",
          "LINK-USD", "ETC-USD", "XLM-USD", "FTT-USD", "ALGO-USD",
          "VET-USD", "MANA-USD", "AXS-USD", "SAND-USD", "EGLD-USD",
          "THETA-USD", "XTZ-USD", "FIL-USD", "HBAR-USD", "ICP-USD"]

period = [1, 2, 3, 4, 5, 6, 7]

logging.info("Fetching cryptocurrency data...")
data = fetch_crypto_data(tickers, '2023-01-01', '2025-01-01')

logging.info("Creating a deep copy of raw data...")
raw_data = copy.deepcopy(data)

logging.info("Performing data cleaning and feature engineering...")
data_cleaning_and_feature_engineering(data)

logging.info("Executing PCA and KMeans clustering...")
cluster_df , _ = pca_and_kmean_clustering(raw_data, tickers)

logging.info("Performing correlation analysis...")
correlation_matrix, positive_correlations, negative_correlations = perform_correlation(raw_data, tickers)

logging.info("Splitting data into training and testing sets...")
train_data, test_data = split_data(data, time_steps=60, test_size=0.2)

st.title("📈 Cryptocurrency Decision Support System")

st.write("### PCA and KMeans Clustering")
plot_clusters(cluster_df)

# Display Clustered Data
st.write("### Clustered Cryptocurrencies")
st.dataframe(cluster_df)

# Heatmap of Correlation Matrix
st.write("### Correlation Heatmap")
plot_correlation_heatmap(correlation_matrix)

# Display Top Positive Correlations in DataFrames
st.write("### Top Positive Correlations")
st.dataframe(positive_correlations)

# Display Top Negative Correlations in DataFrames
st.write("### Top Negative Correlations")
st.dataframe(negative_correlations)

# Select Cryptocurrency
st.sidebar.header("Select Forecasting Options")
crypto_choice = st.sidebar.selectbox("Select Cryptocurrency", tickers)

# Display Closing price for Selected Cryptocurrency
st.write(f"###  Closing Price for {crypto_choice}")
plot_closing_price(raw_data, crypto_choice)

# Display Data Values for Selected Cryptocurrency
st.write(f"### Data Values for {crypto_choice}")
st.line_chart(data_values(raw_data, crypto_choice))

# Display Moving Average for Selected Cryptocurrency
st.write(f"### Moving Averages for {crypto_choice}")
st.line_chart(moving_average(raw_data, crypto_choice))

# Selectbox for forecasting period (only affects ARIMA & Prophet)
select_time = int(st.selectbox('Select Forecasting Period', period))

# Arima Forcast
st.write(f"### ARIMA Forecast for {crypto_choice}")
st.line_chart(arima_forecast(raw_data, crypto_choice, select_time))

# Prophet Forcast
st.write(f"### Prophet Forecast for {crypto_choice}")
st.line_chart(prophet_forecast(raw_data, crypto_choice, select_time))

# LSTM forcast
st.write(f"### LSTM Forecast for {crypto_choice}")
st.line_chart(lstm_forecast(crypto_choice, train_data, test_data))

# xgboost forcast
st.write(f"### xgboost Forecast for {crypto_choice}")
st.line_chart(xgboost_forecast(crypto_choice, train_data, test_data))