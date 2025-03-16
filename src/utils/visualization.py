import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

def plot_clusters(cluster_df):
    """Plot PCA & KMeans Clusters."""
    fig = px.scatter(cluster_df, x="PC1", y="PC2", color=cluster_df["Cluster"].astype(str),
                     hover_data=["Crypto"], labels={"Cluster": "Cluster Group"})
    st.plotly_chart(fig)

def plot_correlation_heatmap(correlation_matrix):
    """Display correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)


def plot_closing_price(raw_data, crypto_choice):
    """
    Plots the closing price of a selected cryptocurrency.
"""
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=raw_data[crypto_choice].index, 
            y=raw_data[crypto_choice]['Close'], 
            mode='lines', 
            name='Close Price'
        ))
        st.plotly_chart(fig)
    except KeyError:
        st.error(f"Error: Data for {crypto_choice} not found.")
    except Exception as e:
        st.error(f"An error occurred while plotting closing prices: {str(e)}")
