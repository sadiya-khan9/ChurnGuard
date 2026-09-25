"""
generate_data.py
-----------------
Generates a synthetic bank-customer dataset that matches the schema of the
popular Kaggle "Bank Customer Churn Prediction" dataset (Saurabh Badole /
Churn_Modelling.csv): RowNumber, CustomerId, Surname, CreditScore, Geography,
Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
EstimatedSalary, Exited.

IMPORTANT: This is a synthetic stand-in so the whole pipeline (train -> model
-> Flask app) runs out of the box. For the real project, download the actual
CSV from Kaggle (search "Bank Customer Churn Prediction" by Saurabh Badole)
and drop it into data/Churn_Modelling.csv with the SAME column names — the
rest of the pipeline needs no changes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

geographies = np.random.choice(["France", "Germany", "Spain"], size=N, p=[0.5, 0.25, 0.25])
genders = np.random.choice(["Male", "Female"], size=N, p=[0.545, 0.455])

age = np.random.gamma(shape=9, scale=4, size=N).astype(int) + 18
age = np.clip(age, 18, 92)

credit_score = np.random.normal(650, 96, size=N).astype(int)
credit_score = np.clip(credit_score, 350, 850)

tenure = np.random.randint(0, 11, size=N)

# Balance: a chunk of customers keep a zero balance
has_balance = np.random.rand(N) > 0.36
balance = np.where(
    has_balance,
    np.random.normal(97000, 62000, size=N).clip(0, None),
    0.0,
)

num_products = np.random.choice([1, 2, 3, 4], size=N, p=[0.51, 0.46, 0.02, 0.01])
has_cr_card = np.random.choice([0, 1], size=N, p=[0.29, 0.71])
is_active_member = np.random.choice([0, 1], size=N, p=[0.48, 0.52])
estimated_salary = np.random.uniform(11, 200000, size=N)

# --- Build churn probability from a mix of realistic risk factors ---
# Weights are tuned so the resulting dataset is learnable (RF gets ~85% accuracy,
# similar to what's typically reported on the real Kaggle dataset) while keeping
# an ~20% overall churn rate, matching the real dataset's class balance.
risk = np.zeros(N)
risk += (age - 18) / 74 * 5.5                         # older customers churn noticeably more
risk += (geographies == "Germany") * 1.8               # Germany has notably higher churn
risk += (num_products >= 3) * 3.8                      # 3-4 products -> very strong churn signal
risk += (num_products == 1) * 0.3
risk += (is_active_member == 0) * 1.7                  # inactive members churn more
risk += (genders == "Female") * 0.55
risk += (balance == 0) * -0.4
risk += (credit_score < 500) * 1.0
risk += np.random.normal(0, 0.2, size=N)               # light noise

prob = 1 / (1 + np.exp(-(risk - 5.7)))
exited = (np.random.rand(N) < prob).astype(int)

surnames = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
]

df = pd.DataFrame({
    "RowNumber": np.arange(1, N + 1),
    "CustomerId": np.arange(15600000, 15600000 + N),
    "Surname": np.random.choice(surnames, size=N),
    "CreditScore": credit_score,
    "Geography": geographies,
    "Gender": genders,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance.round(2),
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary.round(2),
    "Exited": exited,
})

out_path = "data/Churn_Modelling.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(f"Churn rate: {df['Exited'].mean():.2%}")
