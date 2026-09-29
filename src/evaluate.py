"""
evaluate.py
-----------
Evaluation utilities for binary classification models.

Computes and prints:
  - Accuracy
  - Precision, Recall, F1-score
  - ROC-AUC
  - Confusion matrix (as a labelled DataFrame)

All metrics are explained in the context of churn prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true  : Ground-truth labels (0 / 1).
    y_pred  : Predicted labels (0 / 1).
    y_prob  : Predicted probabilities for the positive class (optional, needed for AUC).

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc (if y_prob provided).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def print_report(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    """
    Print a full evaluation report and return the metrics dict.

    Includes:
      - Per-class classification report
      - Summary metrics
      - Churn-context interpretation
    """
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=["Stay (0)", "Churn (1)"]))
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}  (of predicted churners, how many actually churn)")
    print(f"  Recall   : {metrics['recall']:.4f}  (of actual churners, how many we caught)")
    print(f"  F1       : {metrics['f1']:.4f}  (harmonic mean of precision & recall)")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}  (probability that model ranks a churner above non-churner)")
    print(f"{'='*60}\n")
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str | None = None,
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Stay", "Churn"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        cbar=False,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confusion matrix → {save_path}")
    plt.show()


def plot_roc_curves(models_data: list[dict], save_path: str | None = None) -> None:
    """
    Overlay ROC curves for multiple models.

    Parameters
    ----------
    models_data : list of dicts with keys:
                    'name'   : model label (str)
                    'y_true' : ground truth (array)
                    'y_prob' : positive-class probabilities (array)
    save_path   : if provided, saves the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for entry in models_data:
        RocCurveDisplay.from_predictions(
            entry["y_true"],
            entry["y_prob"],
            name=entry["name"],
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved ROC curve → {save_path}")
    plt.show()


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a dict of {model_name: metrics_dict}.
    Returns a sorted DataFrame (by ROC-AUC descending).
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Accuracy": round(m.get("accuracy", 0), 4),
            "Precision": round(m.get("precision", 0), 4),
            "Recall": round(m.get("recall", 0), 4),
            "F1": round(m.get("f1", 0), 4),
            "ROC-AUC": round(m.get("roc_auc", 0), 4),
        })
    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return df
