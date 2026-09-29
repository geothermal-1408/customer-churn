"""
preprocess.py
-------------
Builds a reproducible scikit-learn preprocessing pipeline for the
Telco Customer Churn dataset.

The pipeline:
  1. Identifies numerical and categorical columns dynamically.
  2. Imputes, scales, and encodes features using ColumnTransformer.
  3. Is designed to be fitted only on training data to avoid leakage.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Column definitions (derived from the Telco dataset; override if needed)
# ---------------------------------------------------------------------------
TARGET_COL = "Churn"
DROP_COLS = ["customerID"]  # non-informative identifier

NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# All other object columns after dropping TARGET and DROP_COLS are treated as
# categorical; computed dynamically inside `build_preprocessor`.


def load_raw(filepath: str) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean the raw DataFrame and separate features from the target.

    Steps:
      - Drop non-informative columns.
      - Convert TotalCharges to numeric (it may contain whitespace strings).
      - Encode the binary target (Yes → 1, No → 0).
      - Drop rows where the target is missing.

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — binary target
    """
    df = df.copy()

    # ---- Drop ID-like columns ----
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # ---- Fix TotalCharges: may be stored as string with spaces ----
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # ---- Encode target ----
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    df.dropna(subset=[TARGET_COL], inplace=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # ---- Separate X and y ----
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    return X, y


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that improve signal for churn prediction.

    New columns (only added if source columns are present):
      - ChargeRatio       : MonthlyCharges / (AvgMonthlyCharge + 1)
                            — captures whether recent charges are rising vs history.
      - AvgMonthlyCharge  : TotalCharges / (tenure + 1)
                            — smoothed historical monthly rate.
      - TenureGroup       : Ordinal bucket (0-5) for tenure ranges.
      - IsNewCustomer     : 1 if tenure <= 6 months.
      - IsLongTerm        : 1 if tenure >= 48 months.
      - NumServices       : count of add-on services subscribed.

    All features are computed purely from existing columns — no leakage risk
    as long as this function is called before fitting the preprocessor.
    """
    X = X.copy()

    if "tenure" in X.columns and "MonthlyCharges" in X.columns and "TotalCharges" in X.columns:
        avg_charge = X["TotalCharges"] / (X["tenure"] + 1)
        X["AvgMonthlyCharge"] = avg_charge.fillna(X["MonthlyCharges"])
        X["ChargeRatio"] = X["MonthlyCharges"] / (X["AvgMonthlyCharge"] + 1)
        X["IsNewCustomer"] = (X["tenure"] <= 6).astype(int)
        X["IsLongTerm"] = (X["tenure"] >= 48).astype(int)

    if "tenure" in X.columns:
        bins = [-1, 6, 12, 24, 36, 48, 72]
        labels = [0, 1, 2, 3, 4, 5]
        X["TenureGroup"] = pd.cut(X["tenure"], bins=bins, labels=labels).astype(float)

    # Count binary "Yes" service columns
    service_cols = [
        c for c in [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        ] if c in X.columns
    ]
    if service_cols:
        X["NumServices"] = sum(
            (X[c] == "Yes").astype(int) for c in service_cols
        )

    # HighRisk: month-to-month contract AND fiber optic — highest churn combo
    if "Contract" in X.columns and "InternetService" in X.columns:
        X["HighRisk"] = (
            (X["Contract"] == "Month-to-month") &
            (X["InternetService"] == "Fiber optic")
        ).astype(int)

    # ChargeTier: bucket monthly charges into Low / Mid / High
    if "MonthlyCharges" in X.columns:
        X["ChargeTier"] = pd.cut(
            X["MonthlyCharges"], bins=[0, 35, 65, 120], labels=[0, 1, 2]
        ).astype(float)

    # NoSupport: no security, no backup, no tech support at all
    support_flags = [c for c in ["OnlineSecurity", "OnlineBackup", "TechSupport"] if c in X.columns]
    if support_flags:
        X["NoSupport"] = (
            (X[support_flags] == "No").all(axis=1)
        ).astype(int)

    return X


def get_column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Derive numerical and categorical column lists from the feature matrix.
    Uses NUMERICAL_COLS as a whitelist and treats remaining object/bool columns
    as categorical.
    """
    engineered_num = [
        "AvgMonthlyCharge", "ChargeRatio", "IsNewCustomer",
        "IsLongTerm", "TenureGroup", "NumServices",
        "HighRisk", "ChargeTier", "NoSupport",
    ]
    num_cols = [c for c in NUMERICAL_COLS + engineered_num if c in X.columns]
    cat_cols = [
        c for c in X.columns
        if c not in num_cols and X[c].dtype in (object, "category", bool)
    ]
    # Also include integer binary columns (e.g. SeniorCitizen) as numerical
    remaining = [c for c in X.columns if c not in num_cols and c not in cat_cols]
    num_cols += remaining
    return num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Imputes + scales numerical features.
      - Imputes + one-hot encodes categorical features.
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, num_cols: list[str], cat_cols: list[str]) -> list[str]:
    """Extract feature names after the ColumnTransformer has been fitted."""
    cat_encoder = preprocessor.named_transformers_["cat"]["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    return num_cols + cat_feature_names
