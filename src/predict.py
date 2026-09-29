"""
predict.py
----------
Prediction interface for the saved churn model pipeline.

Usage:
    # Programmatic
    from src.predict import predict_customer

    customer = {
        "tenure": 3,
        "MonthlyCharges": 95.0,
        "TotalCharges": 285.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        ...
    }
    result = predict_customer(customer)
    print(result)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "models/churn_model.pkl"


def load_model(model_path: str = MODEL_PATH) -> dict:
    """Load the saved model bundle from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at '{model_path}'. "
            "Run `python -m src.train` first to train and save the model."
        )
    return joblib.load(model_path)


def predict_customer(customer: dict, model_path: str = MODEL_PATH) -> dict:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer   : dict mapping feature names to values.
    model_path : path to the saved joblib bundle.

    Returns
    -------
    dict with keys:
      - prediction      : int (1 = Churn, 0 = Stay)
      - label           : str ("Likely to Churn" | "Likely to Stay")
      - probability     : float (churn probability 0–1)
      - top_factors     : list of (feature_name, importance) tuples
    """
    bundle = load_model(model_path)
    pipeline = bundle["pipeline"]
    feature_names = bundle.get("feature_names", [])
    threshold = float(bundle.get("threshold", 0.5))  # use tuned threshold

    # Apply the same feature engineering as training
    from src.preprocess import engineer_features
    X = pd.DataFrame([customer])
    X = engineer_features(X)

    prob = float(pipeline.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)
    label = "Likely to Churn" if pred == 1 else "Likely to Stay"

    # --- Feature importance (if available) ---
    top_factors: list[tuple[str, float]] = []
    model_step = pipeline.named_steps["model"]

    if hasattr(model_step, "feature_importances_") and feature_names:
        importances = model_step.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        top_factors = pairs[:5]
    elif hasattr(model_step, "coef_") and feature_names:
        importances = np.abs(model_step.coef_[0])
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        top_factors = pairs[:5]

    return {
        "prediction": pred,
        "label": label,
        "probability": prob,
        "top_factors": top_factors,
    }


def print_prediction(customer: dict, model_path: str = MODEL_PATH) -> None:
    """Pretty-print a prediction for a customer."""
    result = predict_customer(customer, model_path)

    print("\n" + "=" * 50)
    print("  CHURN PREDICTION REPORT")
    print("=" * 50)
    print("  Customer profile:")
    for k, v in customer.items():
        print(f"    {k}: {v}")
    print()
    print(f"  Prediction  : {result['label']}")
    print(f"  Probability : {result['probability']:.1%}")
    if result["top_factors"]:
        print("\n  Contributing factors:")
        for feat, imp in result["top_factors"]:
            print(f"    - {feat}: {imp:.4f}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    # Demo prediction
    sample_customer = {
        "tenure": 3,
        "MonthlyCharges": 95.0,
        "TotalCharges": 285.0,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    print_prediction(sample_customer)
