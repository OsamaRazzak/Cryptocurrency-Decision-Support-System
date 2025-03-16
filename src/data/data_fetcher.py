import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_crypto_data(tickers, start_date, end_date):
    """Fetch historical cryptocurrency data using Yahoo Finance API."""
    data = {}
    for ticker in tickers:
        try:
            crypto_data = yf.download(ticker, start=start_date, end=end_date)
            crypto_data.reset_index(inplace=True)
            crypto_data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
            crypto_data.set_index("Date", inplace=True)
            data[ticker] = crypto_data
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
    return data
