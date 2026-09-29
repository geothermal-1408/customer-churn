"""
generate_sample_data.py
-----------------------
Generates a synthetic Telco-style customer churn dataset and saves it to
data/customer_churn.csv.

This script is a FALLBACK for when the real Telco dataset cannot be downloaded.
The synthetic data mirrors the column schema and approximate distributions of the
real dataset so the entire ML pipeline works end-to-end.

Usage:
    python generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N = 7043  # match real dataset size

rng = np.random.default_rng(RANDOM_STATE)


def generate_dataset(n: int = N) -> pd.DataFrame:
    tenure = rng.integers(0, 72, size=n)
    monthly_charges = rng.uniform(18, 119, size=n).round(2)
    total_charges = (monthly_charges * tenure + rng.uniform(0, 50, size=n)).round(2)
    # Make some TotalCharges blank (as in real dataset)
    blank_idx = rng.choice(n, size=11, replace=False)
    total_charges_str = total_charges.astype(str)
    total_charges_str[blank_idx] = " "

    gender = rng.choice(["Male", "Female"], size=n)
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = rng.choice(["Yes", "No", "No phone service"], size=n, p=[0.42, 0.48, 0.10])
    internet_service = rng.choice(["Fiber optic", "DSL", "No"], size=n, p=[0.44, 0.34, 0.22])
    online_security = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.29, 0.49, 0.22])
    online_backup = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.34, 0.44, 0.22])
    device_protection = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.34, 0.44, 0.22])
    tech_support = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.29, 0.49, 0.22])
    streaming_tv = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.38, 0.40, 0.22])
    streaming_movies = rng.choice(["Yes", "No", "No internet service"], size=n, p=[0.39, 0.39, 0.22])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n, p=[0.34, 0.23, 0.22, 0.21],
    )

    # Logistic churn model (approximate real distributions)
    log_odds = (
        -3.5
        + 0.04 * monthly_charges
        - 0.05 * tenure
        + 1.5 * (contract == "Month-to-month").astype(float)
        + 0.8 * (internet_service == "Fiber optic").astype(float)
        + 0.6 * (tech_support == "No").astype(float)
        + 0.4 * (online_security == "No").astype(float)
        + 0.3 * (payment == "Electronic check").astype(float)
        + rng.normal(0, 0.5, size=n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    churn = (rng.uniform(size=n) < prob).astype(int)
    churn_str = np.where(churn == 1, "Yes", "No")

    customer_ids = [f"0000-{str(i).zfill(5)}" for i in range(n)]

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_str,
        "Churn": churn_str,
    })
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dest = "data/customer_churn.csv"
    df = generate_dataset()
    df.to_csv(dest, index=False)
    print(f"Generated synthetic dataset: {df.shape[0]} rows → {dest}")
    print(f"Churn distribution: {df['Churn'].value_counts().to_dict()}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
