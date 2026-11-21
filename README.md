# ChurnGuard — Bank Customer Churn Prediction
 
**ChurnGuard** is a multi-page Flask web application that predicts whether a bank customer is likely to churn using a Random Forest machine learning classifier. It provides interactive dashboards, customer-level churn predictions, data-driven insights, and model performance visualizations.
 
🚀 **Live Demo:** [https://churnguard-7bmz.onrender.com/](https://churnguard-7bmz.onrender.com/)
 
---
 
## Features
 
- **Interactive Dashboard:** Displays key performance indicators including churn rate, model accuracy, and ROC-AUC, along with visual summaries.
- **Customer Churn Prediction:** Accepts individual customer details and predicts churn probability with a live risk gauge.
- **Customer Insights:** Explores churn patterns across geography, gender, age, number of products, and account balance.
- **Prediction API:** Provides a JSON-based REST endpoint for programmatic customer churn predictions.
## Tech Stack
 
### Frontend
- **Tailwind CSS** — Responsive UI styling via CDN.
- **Chart.js** — Interactive dashboard, insights, and model visualizations.
- **Jinja2** — Server-side HTML templating.
- **Vanilla JavaScript** — Client-side interactions and mobile navigation.
### Backend
- **Python 3**
- **Flask** — Application server, routing, and template rendering.
### Data Science & Machine Learning
- **Pandas & NumPy** — Data processing, cleaning, and feature engineering.
- **Scikit-learn** — Random Forest classification and StandardScaler preprocessing.
- **Joblib** — Model and scaler serialization.
### Data Storage
- **Churn_Modelling.csv** — Banking Customer Churn Prediction dataset (Saurabh Badole, Kaggle).
- No database required. Customer predictions are generated on the fly and are not persisted.
### Deployment & Tools
- **Render** — Cloud hosting and deployment.
- **Gunicorn** — Production WSGI server.
- **Git & GitHub** — Version control and source code management.
- **requirements.txt** — Python dependency management.
---
 
## Application Pages
 
| Page | Route | Description |
|---|---|---|
| Dashboard | `/` | Churn KPIs, model metrics, and overview charts. |
| Predict | `/predict` | Customer input form with churn probability and risk visualization. |
| Insights | `/insights` | Churn analysis by geography, gender, age, products, and balance. |
 
### Prediction API
 
**Endpoint:** `POST /api/predict`
 
Accepts customer information in JSON format and returns a churn prediction with associated risk information.
 
---
 
## Model Details
 
- **Algorithm:** `RandomForestClassifier`
- **Estimators:** 300 trees
- **Maximum Depth:** 10
- **Class Weight:** `balanced`
- **Preprocessing:** `StandardScaler` applied to numeric features after one-hot encoding.
- **Train-Test Split:** 80:20, stratified on the target variable.
### Input Features
 
- Credit Score
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card Status
- Active Member Status
- Estimated Salary
- Geography
- Gender
The model is trained to predict the customer's churn status (`Exited`).
 
To retrain the model after modifying the dataset, run:
 
```bash
python src/train_model.py
```
 
---
 
## Getting Started
 
### 1. Clone the repository
 
```bash
git clone https://github.com/sadiya-khan9/ChurnGuard
cd churn-predictor
```
 
### 2. Create and activate a virtual environment
 
```bash
python -m venv .venv
```
 
**macOS / Linux:**
```bash
source .venv/bin/activate
```
 
**Windows:**
```bash
.venv\Scripts\activate
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Train the model
 
```bash
python src/train_model.py
```
 
### 5. Generate customer insights
 
```bash
python src/generate_insights.py
```
 
### 6. Run the application
 
```bash
python app.py
```
 
Open your browser and visit:
 
```text
http://127.0.0.1:5000
```
 
---
 
## Project Structure
 
```text
churn-predictor/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── Churn_Modelling.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── metadata.json
│   └── insights.json
│
├── src/
│   ├── train_model.py
│   └── generate_insights.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   └── insights.html
│
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```
 
---
 
## Deployment
 
ChurnGuard is deployed on **Render** using Gunicorn as the production WSGI server.
 
**Live Application:** [https://churnguard-7bmz.onrender.com/](https://churnguard-7bmz.onrender.com/)
 
**Production Start Command:**
 
```bash
gunicorn app:app
```
 
---
 
## Future Improvements
 
- Add customer prediction history and database integration.
- Implement batch customer churn prediction.
- Add model comparison and hyperparameter tuning.
- Introduce explainable AI techniques for individual predictions.
- Add user authentication and role-based access.
---
 
## Author
 
Developed as a machine learning and web development project to demonstrate customer churn prediction, model integration, data visualization, and Flask application deployment.
 