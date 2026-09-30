"""
train.py
--------
End-to-end training script for Customer Churn Prediction.

Improvements over baseline:
  1. Feature engineering (ChargeRatio, TenureGroup, NumServices, etc.)
  2. RandomizedSearchCV hyperparameter tuning per model
  3. Optimal decision-threshold tuning on a validation split (maximises F1)
  4. Voting Ensemble of tuned base models
  5. No hard-coded class_weight="balanced" — threshold tuning handles the tradeoff

Pipeline:
  1. Load & clean data
  2. Feature engineering
  3. EDA (visualisations saved to models/figures/)
  4. Train / Val / Test split  (60 / 20 / 20)
  5. Build preprocessing + model pipelines
  6. Hyperparameter search (RandomizedSearchCV on train set, 5-fold CV)
  7. Threshold tuning on val set
  8. Evaluate on test set
  9. Save the best pipeline

Run:
    python -m src.train
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — safe for no-display environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# LightGBM is optional — requires libomp on macOS (same as XGBoost)
HAS_LGBM = True

# try:
#     from lightgbm import LGBMClassifier
#     HAS_LGBM = True
# except Exception as _lgbm_err:
#     HAS_LGBM = False
#     print(f"[INFO] LightGBM unavailable ({_lgbm_err.__class__.__name__}) — skipping.")
#     print("       Fix: sudo mkdir -p /opt/homebrew/opt/libomp/lib && sudo ln -s /usr/local/opt/libomp/lib/libomp.dylib /opt/homebrew/opt/libomp/lib/libomp.dylib")

# XGBoost is optional — handle macOS libomp absence gracefully

HAS_XGB = True

# try:
#     import xgboost as _xgb  # noqa: F401
#     from xgboost import XGBClassifier
#     HAS_XGB = True
# except Exception as _xgb_err:
#     HAS_XGB = False   # must reset — initial value is False
#     print(f"[INFO] XGBoost unavailable ({_xgb_err.__class__.__name__}) — skipping.")
#     print("       To enable XGBoost on macOS: brew install libomp")

from src.preprocess import (
    load_raw, clean, engineer_features,
    get_column_groups, build_preprocessor, get_feature_names,
)
from src.evaluate import print_report, plot_confusion_matrix, plot_roc_curves, compare_models

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)   # suppress sklearn deprecation noise

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_STATE = 42
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
FIGURES_DIR = os.path.join(MODEL_DIR, "figures")

N_ITER_SEARCH = 30   # RandomizedSearchCV iterations per model
CV_FOLDS = 5         # cross-validation folds during tuning


def ensure_dirs() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, y: pd.Series) -> None:
    """Lightweight EDA — saves figures to models/figures/."""
    print("\n--- EDA ---")
    print(f"Dataset shape : {df.shape}")
    print(f"Churn rate    : {y.mean():.2%}")
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    if not mv.empty:
        print(f"Missing values:\n{mv}")

    palette = {"Stay": "#4ECDC4", "Churn": "#FF6B6B"}

    # 1. Churn distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = pd.Series(["Churn" if v == 1 else "Stay" for v in y]).value_counts()
    ax.bar(counts.index, counts.values, color=[palette["Stay"], palette["Churn"]])
    ax.set_title("Churn Distribution", fontweight="bold")
    ax.set_ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/churn_distribution.png", dpi=150)
    plt.close()

    # 2. Tenure vs Churn
    if "tenure" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        df_plot = df[["tenure"]].copy()
        df_plot["Status"] = y.map({1: "Churn", 0: "Stay"}).values
        sns.histplot(data=df_plot, x="tenure", hue="Status",
                     palette=palette, kde=True, bins=30, ax=ax)
        ax.set_title("Tenure Distribution by Churn Status", fontweight="bold")
        plt.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/tenure_vs_churn.png", dpi=150)
        plt.close()

    # 3. Monthly Charges vs Churn
    if "MonthlyCharges" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        df_plot = df[["MonthlyCharges"]].copy()
        df_plot["Status"] = y.map({1: "Churn", 0: "Stay"}).values
        sns.boxplot(data=df_plot, x="Status", y="MonthlyCharges",
                    palette=palette, ax=ax)
        ax.set_title("Monthly Charges by Churn Status", fontweight="bold")
        plt.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/monthly_charges_vs_churn.png", dpi=150)
        plt.close()

    # 4. Contract type vs Churn
    if "Contract" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        contract_churn = (
            df[["Contract"]].assign(Churn=y.values)
            .groupby("Contract")["Churn"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        bars = ax.bar(contract_churn["Contract"], contract_churn["Churn"],
                      color=["#FF6B6B", "#FFD93D", "#4ECDC4"])
        ax.set_title("Churn Rate by Contract Type", fontweight="bold")
        ax.set_ylabel("Churn Rate")
        for bar, val in zip(bars, contract_churn["Churn"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.1%}", ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/contract_vs_churn.png", dpi=150)
        plt.close()

    print("  EDA figures saved to models/figures/")


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Search thresholds in [0.2, 0.8] and return the one that maximises F1
    on the provided (validation) labels.
    """
    thresholds = np.linspace(0.2, 0.8, 61)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred_t, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return float(best_t)


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------
def plot_feature_importance(
    model_pipeline: Pipeline,
    feature_names: list[str],
    model_name: str,
    top_n: int = 20,
) -> None:
    """Plot and save feature importance for tree-based models or LR coefficients."""
    step = model_pipeline.named_steps["model"]

    # Handle StackingClassifier — use the final_estimator or first base estimator
    if isinstance(step, StackingClassifier):
        # Try final_estimator first (it's a LR trained on OOF predictions)
        fe = step.final_estimator_
        if hasattr(fe, "coef_"):
            importances = np.abs(fe.coef_[0])
        else:
            # Fall back to first base estimator with importances
            for est in step.estimators_:
                if hasattr(est, "feature_importances_"):
                    importances = est.feature_importances_
                    break
                elif hasattr(est, "coef_"):
                    importances = np.abs(est.coef_[0])
                    break
            else:
                print(f"  [WARN] {model_name}: no feature importance available.")
                return
    elif hasattr(step, "feature_importances_"):
        importances = step.feature_importances_
    elif hasattr(step, "coef_"):
        importances = np.abs(step.coef_[0])
    else:
        print(f"  [WARN] {model_name} has no feature importance attribute.")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
    ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
    ax.set_title(f"Feature Importance — {model_name}", fontweight="bold", fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fname = f"{FIGURES_DIR}/feature_importance_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved feature importance → {fname}")


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------
def get_param_grids() -> dict[str, dict]:
    """Return RandomizedSearchCV param distributions keyed by model name."""
    lr_params = {
        "model__C": np.logspace(-3, 2, 30),
        "model__solver": ["lbfgs", "saga"],
        "model__penalty": ["l2"],
        "model__max_iter": [1000],
    }
    rf_params = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [6, 8, 10, 12, None],
        "model__min_samples_leaf": [1, 2, 4, 6],
        "model__max_features": ["sqrt", "log2", 0.5],
    }
    xgb_params = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "model__gamma": [0, 0.1, 0.2],
    }
    lgbm_params = {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63],
        "model__min_child_samples": [10, 20, 40],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
    }
    return {
        "Logistic Regression": lr_params,
        "Random Forest": rf_params,
        "XGBoost": xgb_params,
        "LightGBM": lgbm_params,
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train(data_path: str = DATA_PATH) -> None:
    ensure_dirs()

    # ---- Load & clean ----
    print(f"\n[1/7] Loading data from '{data_path}' ...")
    raw_df = load_raw(data_path)
    print(f"      Raw shape: {raw_df.shape}")

    X, y = clean(raw_df)
    print(f"      After cleaning — X: {X.shape}, Churn rate: {y.mean():.2%}")

    # ---- Feature engineering ----
    print("\n[2/7] Engineering features ...")
    X = engineer_features(X)
    print(f"      After engineering — X: {X.shape} ({X.shape[1] - 19} new features)")

    # ---- EDA ----
    print("\n[3/7] Running EDA ...")
    run_eda(X, y)

    # ---- 3-way split: train / val / test ----
    print("\n[4/7] Splitting data  (60% train | 20% val | 20% test) ...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ---- Preprocessor ----
    num_cols, cat_cols = get_column_groups(X_train)
    print(f"\n      Numerical cols  ({len(num_cols)}): {num_cols}")
    print(f"      Categorical cols ({len(cat_cols)}): {cat_cols}")
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # ---- Define base models ----
    base_models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    if HAS_LGBM:
        base_models["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    if HAS_XGB:
        base_models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )


    param_grids = get_param_grids()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ---- Train & tune ----
    print(f"\n[5/7] Hyperparameter search ({N_ITER_SEARCH} iterations × {CV_FOLDS}-fold CV) ...")
    results: dict[str, dict] = {}
    roc_data: list[dict] = []
    fitted_pipelines: dict[str, Pipeline] = {}
    best_thresholds: dict[str, float] = {}
    feature_names: list[str] | None = None

    for name, clf in base_models.items():
        print(f"\n  → {name}")

        # Use ImbPipeline so SMOTE is applied only inside CV folds (no leakage)
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", clf),
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grids.get(name, {}),
            n_iter=N_ITER_SEARCH,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        print(f"     Best CV ROC-AUC: {search.best_score_:.4f}  |  params: {search.best_params_}")

        # ---- Threshold tuning on validation set ----
        val_prob = best_pipe.predict_proba(X_val)[:, 1]
        best_t = find_best_threshold(y_val.values, val_prob)
        best_thresholds[name] = best_t
        print(f"     Optimal threshold (val F1): {best_t:.2f}")

        # ---- Test evaluation using tuned threshold ----
        test_prob = best_pipe.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= best_t).astype(int)

        metrics = print_report(name, y_test.values, test_pred, test_prob)
        results[name] = metrics
        roc_data.append({"name": name, "y_true": y_test.values, "y_prob": test_prob})
        fitted_pipelines[name] = best_pipe

        plot_confusion_matrix(
            y_test.values, test_pred, name,
            save_path=f"{FIGURES_DIR}/cm_{name.lower().replace(' ', '_')}.png",
        )

        if feature_names is None:
            fitted_preprocessor = best_pipe.named_steps["preprocessor"]
            feature_names = get_feature_names(fitted_preprocessor, num_cols, cat_cols)

        plot_feature_importance(best_pipe, feature_names, name)

    # ---- Stacking Ensemble (stronger than simple voting) ----
    print("\n  → Stacking Ensemble")
    # Use tuned base models' classifiers as level-0 estimators
    stack_estimators = [
        (n.lower().replace(" ", "_"), fitted_pipelines[n].named_steps["model"])
        for n in fitted_pipelines
    ]
    # Meta-learner: simple Logistic Regression on out-of-fold predictions
    stack_clf = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        passthrough=False,   # only use base-model OOF predictions as meta-features
        n_jobs=-1,
    )
    # Fit stacker on full train+val set (base models already tuned, meta-learner trains fresh)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    ensemble_pipe = ImbPipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", stack_clf),
    ])
    ensemble_pipe.fit(X_trainval, y_trainval)
    fitted_pipelines["Stacking Ensemble"] = ensemble_pipe

    ens_prob = ensemble_pipe.predict_proba(X_test)[:, 1]
    ens_val_prob = ensemble_pipe.predict_proba(X_val)[:, 1]
    ens_t = find_best_threshold(y_val.values, ens_val_prob)
    best_thresholds["Stacking Ensemble"] = ens_t
    ens_pred = (ens_prob >= ens_t).astype(int)

    ens_metrics = print_report("Stacking Ensemble", y_test.values, ens_pred, ens_prob)
    results["Stacking Ensemble"] = ens_metrics
    roc_data.append({"name": "Stacking Ensemble", "y_true": y_test.values, "y_prob": ens_prob})

    plot_confusion_matrix(
        y_test.values, ens_pred, "Stacking Ensemble",
        save_path=f"{FIGURES_DIR}/cm_stacking_ensemble.png",
    )
    plot_feature_importance(ensemble_pipe, feature_names, "Stacking Ensemble")

    # ---- Compare ----
    print("\n[6/7] Model comparison:")
    comparison_df = compare_models(results)
    print(comparison_df.to_string(index=False))

    plot_roc_curves(roc_data, save_path=f"{FIGURES_DIR}/roc_curves.png")

    # ---- Save best ----
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_threshold = best_thresholds.get(best_model_name, 0.5)

    print(f"\n[7/7] Saving best model ({best_model_name}, threshold={best_threshold:.2f}) → {MODEL_PATH}")
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_model_name,
            "feature_names": feature_names,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "threshold": best_threshold,
            "metrics": results[best_model_name],
            "all_results": results,
        },
        MODEL_PATH,
    )
    print(f"      Done! Pipeline saved to {MODEL_PATH}")
    print("\n✅ Training complete.")


if __name__ == "__main__":
    train()
