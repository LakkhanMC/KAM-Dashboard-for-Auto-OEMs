import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

from models.health_score import compute_health_score
from models.churn_model import compute_churn_probability
from models.sentiment_model import enrich_feedback_sentiment, aggregate_dealer_sentiment


@st.cache_data
def load_data():
    dealer = pd.read_csv("data/dealer_master.csv")
    sales = pd.read_csv("data/sales_transactions.csv")
    inv = pd.read_csv("data/inventory_stock.csv")
    claims = pd.read_csv("data/warranty_claims.csv")
    crm = pd.read_csv("data/crm_engagement.csv")
    feedback = pd.read_csv("data/feedback_forms.csv")
    return dealer, sales, inv, claims, crm, feedback


def main():
    st.title("Segmentation & Portfolio View")

    dealer, sales, inv, claims, crm, feedback = load_data()

    health = compute_health_score(sales, claims, crm, inv)
    churn = compute_churn_probability(sales, claims, crm, inv)
    feedback_enriched = enrich_feedback_sentiment(feedback)
    sentiment_agg = aggregate_dealer_sentiment(feedback_enriched)

    # Aggregate sales volume
    sales_agg = (
        sales.groupby("dealer_id")["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"units_sold": "total_units_sold"})
    )

    df = dealer.merge(health, on="dealer_id", how="left") \
               .merge(churn, on="dealer_id", how="left") \
               .merge(sales_agg, on="dealer_id", how="left") \
               .merge(sentiment_agg, on="dealer_id", how="left")

    df["total_units_sold"] = df["total_units_sold"].fillna(0)
    df["avg_sentiment_score"] = df["avg_sentiment_score"].fillna(0)

    st.subheader("Clustering Input Features")
    st.write("We cluster dealers using: total units sold, health score, churn probability, sentiment.")

    features = df[["total_units_sold", "health_score", "churn_probability", "avg_sentiment_score"]].copy()
    features = (features - features.mean()) / (features.std() + 1e-6)

    n_clusters = st.slider("Number of clusters", min_value=2, max_value=6, value=3)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(features)

    st.subheader("Cluster Summary")
    st.dataframe(
        df.groupby("cluster")
        .agg(
            dealers=("dealer_id", "count"),
            avg_health=("health_score", "mean"),
            avg_churn=("churn_probability", "mean"),
            avg_units=("total_units_sold", "mean"),
            avg_sentiment=("avg_sentiment_score", "mean")
        )
        .reset_index()
    )

    st.subheader("Health vs Volume by Cluster")
    st.caption("Use this to identify 'large but risky' vs 'small but healthy' accounts.")
    chart_df = df[["dealer_id", "cluster", "health_score", "total_units_sold"]]
    st.scatter_chart(
        chart_df,
        x="health_score",
        y="total_units_sold",
    )

    st.subheader("Dealer List")
    cluster_filter = st.selectbox("Filter by cluster", options=["All"] + sorted(df["cluster"].unique().tolist()))
    if cluster_filter != "All":
        filtered = df[df["cluster"] == cluster_filter]
    else:
        filtered = df

    st.dataframe(
        filtered[[
            "dealer_id", "dealer_name", "region", "tier",
            "cluster", "health_score", "churn_probability", "total_units_sold", "avg_sentiment_score"
        ]]
    )


if __name__ == "__main__":
    main()
