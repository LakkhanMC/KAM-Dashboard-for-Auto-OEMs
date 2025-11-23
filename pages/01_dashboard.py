import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="KAM AI Dashboard – Executive Overview", layout="wide")

@st.cache_data
def load_data():
    # Debug prints – will show in the app
    st.write("Current working directory:", os.getcwd())
    st.write("Contents of repo root:", os.listdir())
    if os.path.exists("data"):
        st.write("Contents of /data:", os.listdir("data"))
    else:
        st.write("⚠️ 'data' folder NOT found at repo root")

    # Main data loads
    dealer = pd.read_csv("data/dealer_master.csv")
    sales = pd.read_csv("data/sales_transactions.csv")
    inv = pd.read_csv("data/inventory_stock.csv")
    claims = pd.read_csv("data/warranty_claims.csv")
    crm = pd.read_csv("data/crm_engagement.csv")
    feedback = pd.read_csv("data/feedback_forms.csv")
    return dealer, sales, inv, claims, crm, feedback


def main():
    st.title("KAM AI Dashboard – Executive Overview (Debug Mode)")

    dealer, sales, inv, claims, crm, feedback = load_data()

    st.subheader("✅ Data loaded successfully")
    st.write("Dealers:", dealer.shape)
    st.write("Sales:", sales.shape)
    st.write("Inventory:", inv.shape)
    st.write("Claims:", claims.shape)
    st.write("CRM:", crm.shape)
    st.write("Feedback:", feedback.shape)

    st.subheader("Sample – dealer_master")
    st.dataframe(dealer.head())


if __name__ == "__main__":
    main()
