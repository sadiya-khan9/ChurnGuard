"""
train_model.py
---------------
Loads data/Churn_Modelling.csv, preprocesses it, trains a RandomForestClassifier
to predict the `Exited` (churn) label, evaluates it, and saves:
  - models/churn_model.pkl   (trained RandomForestClassifier)
  - models/scaler.pkl        (StandardScaler fit on training features)
  - models/metadata.json     (feature order, geography/gender categories,
                               metrics, feature importances — used by the app)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/Churn_Modelling.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Drop identifier columns that carry no predictive signal
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

# One-hot encode categoricals (drop_first to avoid collinearity)
geo_dummies = pd.get_dummies(df["Geography"], prefix="Geography", drop_first=True)
gender_dummies = pd.get_dummies(df["Gender"], prefix="Gender", drop_first=True)

X = pd.concat(
    [df.drop(columns=["Geography", "Gender", "Exited"]), geo_dummies, gender_dummies],
    axis=1,
)
y = df["Exited"]

feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=6,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "churn_rate_overall": round(float(y.mean()), 4),
}

cm = confusion_matrix(y_test, y_pred).tolist()
metrics["confusion_matrix"] = cm  # [[TN, FP], [FN, TP]]

importances = dict(zip(feature_columns, model.feature_importances_.round(4).tolist()))
importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

metadata = {
    "feature_columns": feature_columns,
    "geography_categories": sorted(df["Geography"].unique().tolist()),
    "gender_categories": sorted(df["Gender"].unique().tolist()),
    "metrics": metrics,
    "feature_importances": importances,
}

joblib.dump(model, os.path.join(MODELS_DIR, "churn_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Model trained and saved to models/")
print(json.dumps(metrics, indent=2))
