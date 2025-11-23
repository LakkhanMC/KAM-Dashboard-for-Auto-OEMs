import streamlit as st

st.set_page_config(
    page_title="AI-Powered KAM Platform",
    layout="wide"
)

st.title("AI-Powered Key Account Management (KAM) Platform – Automobile OEMs")

st.markdown("""
This app demonstrates an **AI-assisted KAM dashboard** for an Automobile OEM.

Use the sidebar to navigate:
- **01_dashboard** → Portfolio overview, health & churn
- **02_account_explorer** → Drill-down per dealer
- **03_segmentation** → Cluster dealers by risk/value
- **04_forecast** → Simple demand forecasts & opportunities
""")

st.info("Select a page from the left sidebar.")
