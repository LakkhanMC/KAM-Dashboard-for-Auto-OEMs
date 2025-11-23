import pandas as pd

POS = {"good", "satisfied", "appreciated", "timely", "support", "excellent", "happy"}
NEG = {"delay", "delays", "insufficient", "bad", "poor", "complaint", "issue", "problem", "pressure", "trust"}


def sentiment_score(text):
    if not isinstance(text, str):
        return 0
    t = text.lower().split()
    p = sum(w in POS for w in t)
    n = sum(w in NEG for w in t)
    if p + n == 0:
        return 0
    return (p - n) / (p + n)


def enrich_sentiment(feedback):
    df = feedback.copy()
    df["sentiment_val"] = df["comments"].apply(sentiment_score)
    df["sentiment"] = df["sentiment_val"].apply(
        lambda x: "Positive" if x > 0.2 else ("Negative" if x < -0.2 else "Neutral")
    )
    return df


def dealer_sentiment(feedback):
    df = feedback.copy()
    if df.empty:
        return pd.DataFrame(columns=["dealer_id", "sentiment_avg"])
    return df.groupby("dealer_id")["sentiment_val"].mean().reset_index().rename(columns={"sentiment_val": "sentiment_avg"})
