from src.data.pca_kmean import pca_and_kmean_clustering
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_correlation(raw_data, tickers):
    """
    Find top positive and negative correlations.
    """
    try:
        cluster_df, crypto_prices = pca_and_kmean_clustering(raw_data, tickers)
        selected_cryptos = [cluster_df[cluster_df["Cluster"] == cluster]["Crypto"].iloc[0] for cluster in range(4)]
        returns = crypto_prices.pct_change().dropna()
        selected_returns = returns[selected_cryptos]
        correlation_matrix = selected_returns.corr()

        correlation_values = correlation_matrix.unstack().sort_values().reset_index()
        correlation_values.columns = ["Crypto_1", "Crypto_2", "Correlation"]
        correlation_pairs = correlation_values[correlation_values['Crypto_1'] != correlation_values['Crypto_2']]
        correlation_pairs['Sorted'] = correlation_pairs.apply(lambda x: tuple(sorted([x['Crypto_1'], x['Crypto_2']])), axis=1)
        correlation_pairs = correlation_pairs.drop_duplicates(subset=['Sorted']).drop(columns=['Sorted'])

        positive_correlations = correlation_pairs.nlargest(4, "Correlation")
        negative_correlations = correlation_pairs.nsmallest(4, "Correlation")

        return correlation_matrix, positive_correlations, negative_correlations
    except Exception as e:
        logging.error(f"Error in correlation analysis: {e}")
        return None, None, None
