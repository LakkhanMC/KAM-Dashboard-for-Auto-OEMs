from utils.paths import data_path
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    dealer = pd.read_csv(data_path("dealer_master.csv"))
    sales = pd.read_csv(data_path("sales_transactions.csv"))
    inv = pd.read_csv(data_path("inventory_stock.csv"))
    claims = pd.read_csv(data_path("warranty_claims.csv"))
    crm = pd.read_csv(data_path("crm_engagement.csv"))
    feedback = pd.read_csv(data_path("feedback_forms.csv"))
    return dealer, sales, inv, claims, crm, feedback

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from utils.paths import data_path
from models.health_score import compute_health_score
from models.churn_model import compute_churn
from models.sentiment_model import dealer_sentiment

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
    health = compute_health_score(sales, claims, crm, inv)
    churn = compute_churn(sales, claims, crm, inv)
    sent = dealer_sentiment(feedback)

    vol = sales.groupby("dealer_id")["units_sold"].sum().reset_index()

    df = dealer.merge(health, on="dealer_id").merge(churn, on="dealer_id").merge(vol, on="dealer_id").merge(sent, on="dealer_id", how="left")
    df.fillna(0, inplace=True)

    X = df[["units_sold", "health_score", "churn_prob", "sentiment_avg"]]
    k_val = st.slider("Clusters", 2, 6, 3)
    df["cluster"] = KMeans(k_val, random_state=42).fit_predict(X)

    st.dataframe(df)

    st.subheader("Cluster Heatmap")
    st.scatter_chart(df, x="health_score", y="units_sold", color="cluster")


if __name__ == "__main__":
    main()
