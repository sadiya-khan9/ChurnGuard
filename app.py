"""
app.py
------
Flask app for the Bank Customer Churn Prediction project.

Pages:
  /            Dashboard  — headline KPIs + quick charts
  /predict     Predict    — form to score a single customer, GET + POST
  /insights    Insights   — deeper EDA breakdowns (geography, age, products...)
  /about       About      — project & model info, feature importances

Run:
  python -m venv .venv && source .venv/bin/activate   (or venv\\Scripts\\activate on Windows)
  pip install -r requirements.txt
  python src/generate_data.py        # only needed once, or swap in the real Kaggle CSV
  python src/train_model.py
  python src/generate_insights.py
  python app.py
  -> open http://127.0.0.1:5000
"""

import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODELS_DIR = "models"
model = None
scaler = None
metadata = None
insights = None


def load_artifacts():
    global model, scaler, metadata, insights
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    insights_path = os.path.join(MODELS_DIR, "insights.json")
    if os.path.exists(insights_path):
        with open(insights_path) as f:
            insights = json.load(f)


try:
    load_artifacts()
except FileNotFoundError:
    # Artifacts not trained yet — pages that need them will show a friendly notice.
    pass


def build_feature_vector(form):
    """Turn the predict-form input into a row matching metadata['feature_columns']."""
    row = {
        "CreditScore": float(form["credit_score"]),
        "Age": float(form["age"]),
        "Tenure": float(form["tenure"]),
        "Balance": float(int(float(form["balance"]))),
        "NumOfProducts": float(form["num_products"]),
        "HasCrCard": float(form.get("has_cr_card", 0)),
        "IsActiveMember": float(form.get("is_active_member", 0)),
        "EstimatedSalary": float(int(float(form["estimated_salary"]))),
    }
    for geo in metadata["geography_categories"]:
        if geo == metadata["geography_categories"][0]:
            continue  # this is the dropped baseline category
        row[f"Geography_{geo}"] = 1.0 if form["geography"] == geo else 0.0
    for gender in metadata["gender_categories"]:
        if gender == metadata["gender_categories"][0]:
            continue
        row[f"Gender_{gender}"] = 1.0 if form["gender"] == gender else 0.0

    ordered = [row.get(col, 0.0) for col in metadata["feature_columns"]]
    return np.array(ordered).reshape(1, -1)


@app.route("/")
def dashboard():
    return render_template(
        "index.html",
        active="dashboard",
        metadata=metadata,
        insights=insights,
        ready=model is not None,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form_values = request.form if request.method == "POST" else {}

    if request.method == "POST" and model is not None:
        try:
            X = build_feature_vector(request.form)
            X_scaled = scaler.transform(X)
            proba = float(model.predict_proba(X_scaled)[0, 1])
            pred = int(proba >= 0.5)
            if proba >= 0.66:
                risk_level, risk_class = "High risk", "high"
            elif proba >= 0.33:
                risk_level, risk_class = "Medium risk", "medium"
            else:
                risk_level, risk_class = "Low risk", "low"
            result = {
                "prediction": pred,
                "probability": round(proba * 100, 1),
                "risk_level": risk_level,
                "risk_class": risk_class,
            }
        except (KeyError, ValueError) as e:
            result = {"error": f"Invalid input: {e}"}

    return render_template(
        "predict.html",
        active="predict",
        metadata=metadata,
        result=result,
        form_values=form_values,
        ready=model is not None,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model not trained yet"}), 503
    try:
        payload = request.get_json(force=True)
        X = build_feature_vector(payload)
        X_scaled = scaler.transform(X)
        proba = float(model.predict_proba(X_scaled)[0, 1])
        return jsonify({"prediction": int(proba >= 0.5), "probability": round(proba, 4)})
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/insights")
def insights_page():
    return render_template(
        "insights.html",
        active="insights",
        insights=insights,
        ready=insights is not None,
    )


if __name__ == "__main__":
    app.run(debug=True)