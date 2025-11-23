import pandas as pd
from .health_score import compute_health_score


def compute_churn(sales, claims, crm, inv):
    health = compute_health_score(sales, claims, crm, inv)
    df = health.copy()

    df["churn_prob"] = 1 - (df["health_score"] / 100)

    df["risk_bucket"] = pd.cut(
        df["churn_prob"],
        bins=[-0.01, 0.33, 0.66, 1.1],
        labels=["Low", "Medium", "High"]
    )

    return df[["dealer_id", "churn_prob", "risk_bucket"]]
