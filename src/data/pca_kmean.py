from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pca_and_kmean_clustering(raw_data,tickers):
    """Perform PCA and KMeans clustering on cryptocurrencies."""
    dataframe = {}
    scaler = StandardScaler()
    try:
        for key, item in raw_data.items():
            dataframe[key] = pd.DataFrame(item['Close'])

        crypto_prices = pd.DataFrame({symbol: df["Close"] for symbol, df in dataframe.items()})
        returns_scaled = scaler.fit_transform(crypto_prices.T)

        pca = PCA(n_components=2)  
        pca_result = pca.fit_transform(returns_scaled)

        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(pca_result)

        cluster_df = pd.DataFrame({"Crypto": tickers, "Cluster": clusters, "PC1": pca_result[:, 0], "PC2": pca_result[:, 1]})
        return cluster_df, crypto_prices
    
    except Exception as e:
        logging(f"Error in PCA and KMeans clustering: {e}")
        return None, None