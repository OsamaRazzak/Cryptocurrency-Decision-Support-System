# Cryptocurrency Decision Support System

## Overview

A Streamlit-based application for cryptocurrency investment analysis, featuring data fetching, preprocessing, clustering, correlation analysis, and forecasting using ML models.

## Features

- **Data Fetching**: Retrieves historical price data for 30 cryptocurrencies.
- **Data Preprocessing**: Cleans and prepares data for analysis.
- **Clustering & Correlation**: Uses PCA and KMeans for clustering, with correlation heatmaps for analysis.
- **Forecasting**: Supports ARIMA, Prophet, LSTM, and XGBoost models for price prediction.
- **Visualization**: Displays price trends, cluster analysis, correlation heatmaps, and moving averages.

## Installation

Install dependencies from the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Usage

```sh
streamlit run app.py
```
