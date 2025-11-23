import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from utils.paths import data_path
from models.sentiment_model import enrich_sentiment, dealer_sentiment

@st.cache_data
def load_data():
    dealer = pd.read_csv(data_path("dealer_master.csv"))
    sales = pd.read_csv(data_path("sales_transactions.csv"))
    inv = pd.read_csv(data_path("inventory_stock.csv"))
    claims = pd.read_csv(data_path("warranty_claims.csv"))
    crm = pd.read_csv(data_path("crm_engagement.csv"))
    feedback = pd.read_csv(data_path("feedback_forms.csv"))
    return dealer, sales, inv, claims, crm, feedback


def main():
    st.title("Segmentation")

    dealer, sales, inv, claims, crm, feedback = load_data()

    # Sentiment must be enriched first
    feedback = enrich_sentiment(feedback)
    sent = dealer_sentiment(feedback)

    # Aggregate sales volume
    vol = sales.groupby("dealer_id")["units_sold"].sum().reset_index()

    df = (
        dealer
        .merge(vol, on="dealer_id")
        .merge(sent, on="dealer_id", how="left")
    )

    df.fillna(0, inplace=True)

    # Choose variables to cluster on
    X = df[["units_sold", "sentiment_avg"]]

    k = st.slider("Number of clusters", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, n_init="auto").fit(X)

    df["cluster"] = kmeans.labels_

    st.subheader("Clusters")
    st.dataframe(df)

    st.subheader("Visualization")
    st.scatter_chart(df, x="units_sold", y="sentiment_avg", color="cluster")


if __name__ == "__main__":
    main()
