"""
generate_insights.py
---------------------
Precomputes churn-rate breakdowns (by geography, gender, age band, product
count, activity status, balance band) from data/Churn_Modelling.csv and
saves them to models/insights.json so the Flask app can render charts
without touching pandas at request time.
"""

import json
import os

import pandas as pd

DATA_PATH = "data/Churn_Modelling.csv"
OUT_PATH = "models/insights.json"

df = pd.read_csv(DATA_PATH)


def churn_rate_by(col):
    grp = df.groupby(col)["Exited"].agg(["mean", "count"]).reset_index()
    grp = grp.sort_values(col)
    return {
        "labels": grp[col].astype(str).tolist(),
        "churn_rate": (grp["mean"] * 100).round(1).tolist(),
        "count": grp["count"].tolist(),
    }


age_bins = [18, 30, 40, 50, 60, 70, 93]
age_labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
df["AgeBand"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

balance_bins = [-0.01, 0.01, 50000, 100000, 150000, 1e9]
balance_labels = ["Zero", "1-50k", "50-100k", "100-150k", "150k+"]
df["BalanceBand"] = pd.cut(df["Balance"], bins=balance_bins, labels=balance_labels)

insights = {
    "overall_churn_rate": round(float(df["Exited"].mean()) * 100, 1),
    "total_customers": int(len(df)),
    "churned_customers": int(df["Exited"].sum()),
    "avg_age": round(float(df["Age"].mean()), 1),
    "avg_credit_score": round(float(df["CreditScore"].mean()), 0),
    "by_geography": churn_rate_by("Geography"),
    "by_gender": churn_rate_by("Gender"),
    "by_age_band": churn_rate_by("AgeBand"),
    "by_num_products": churn_rate_by("NumOfProducts"),
    "by_activity": {
        "labels": ["Inactive", "Active"],
        "churn_rate": [
            round(float(df[df.IsActiveMember == 0]["Exited"].mean()) * 100, 1),
            round(float(df[df.IsActiveMember == 1]["Exited"].mean()) * 100, 1),
        ],
    },
    "by_balance_band": churn_rate_by("BalanceBand"),
}

os.makedirs("models", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(insights, f, indent=2)

print(f"Wrote insights to {OUT_PATH}")
