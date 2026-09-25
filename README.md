# ChurnGuard — Bank Customer Churn Prediction

A multi-page Flask app that predicts whether a bank customer will churn, using a
Random Forest classifier trained on the schema of the Kaggle **"Bank Customer
Churn Prediction"** dataset (Saurabh Badole).


Frontend

Tailwind CSS (via CDN)
Chart.js — dashboard/insights/about charts
Jinja2 templates (HTML rendered server-side, no React)
Vanilla JS — small script for mobile nav toggle

Backend

Flask — app server, routing, template rendering
Python 3

Data / ML

pandas, numpy — data generation, cleaning, feature engineering
scikit-learn — RandomForestClassifier + StandardScaler
joblib — persisting the trained model/scaler to disk

Data storage

CSV file (Churn_Modelling.csv) — no database, data loaded and processed at training time; predictions computed on the fly, not stored

Deployment/tooling

requirements.txt for dependencies, run locally with python app.py (Flask dev server) — no Docker or cloud deploy config included, so that's worth adding if you want it on your resume as "deployed."

## Pages
- **Dashboard** (`/`) — headline KPIs (churn rate, model accuracy, ROC-AUC) + quick charts
- **Predict** (`/predict`) — form to score a single customer, with a live risk gauge
- **Insights** (`/insights`) — churn-rate breakdowns by geography, gender, age, products, balance
- **About** (`/about`) — project + model details, metrics, feature importance chart

There's also a small JSON API at `POST /api/predict` for programmatic use.

## Using the real Kaggle dataset

This repo ships with a **synthetic** `data/Churn_Modelling.csv` that matches the real
dataset's column schema (`RowNumber, CustomerId, Surname, CreditScore, Geography,
Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
EstimatedSalary, Exited`), so the whole pipeline runs immediately.

To use the **real data**:
1. Download "Bank Customer Churn Prediction" by Saurabh Badole from Kaggle.
2. Save the CSV as `data/Churn_Modelling.csv` (same column names as above).
3. Re-run the training steps below — no code changes needed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Only needed if you don't already have data/Churn_Modelling.csv:
python src/generate_data.py

python src/train_model.py
python src/generate_insights.py

python app.py
```

Then open http://127.0.0.1:5000

## Project structure

```
churn-predictor/
├── app.py                   # Flask app — routes for all pages
├── requirements.txt
├── data/
│   └── Churn_Modelling.csv  # dataset (synthetic by default)
├── models/                  # created by the scripts below
│   ├── churn_model.pkl      # trained RandomForestClassifier
│   ├── scaler.pkl           # StandardScaler
│   ├── metadata.json        # feature order, categories, metrics, importances
│   └── insights.json        # precomputed EDA aggregates
├── src/
│   ├── generate_data.py     # builds the synthetic dataset
│   ├── train_model.py       # preprocess + train + evaluate + save model
│   └── generate_insights.py # precompute churn-rate breakdowns for /insights
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── insights.html
│   └── about.html
└── static/
    ├── css/style.css
    └── js/script.js
```

## Model

- **Algorithm:** `RandomForestClassifier` (300 trees, max depth 10, `class_weight="balanced"`)
- **Features:** credit score, age, tenure, balance, number of products, credit-card
  flag, active-member flag, estimated salary, one-hot encoded geography and gender
- **Preprocessing:** `StandardScaler` on all numeric features (after one-hot encoding)
- **Split:** 80/20 train/test, stratified on the target

Re-run `python src/train_model.py` any time you change the data or want to retrain.



