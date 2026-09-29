# Customer Churn Prediction and Explainable Retention Analysis

> **Predict whether a telecom customer will churn and understand *why*.**

An end-to-end machine learning pipeline built on the Telco Customer Churn dataset.  
The project covers data cleaning, EDA, feature engineering, model training (Logistic Regression, Random Forest, XGBoost, LightGBM, Stacking Ensemble), evaluation, explainability, and a Streamlit web interface.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Pipeline Overview](#pipeline-overview)
6. [EDA Findings](#eda-findings)
7. [Models & Metrics](#models--metrics)
8. [Feature Importance](#feature-importance)
9. [Streamlit App](#streamlit-app)
10. [Limitations](#limitations)
11. [Reproducibility](#reproducibility)

---

## Problem Statement

Customer churn — when a customer stops using a service — is expensive.  
Predicting churn early lets a business intervene with targeted retention offers.

**Task:** Binary classification.  
- **1** = customer churns  
- **0** = customer stays

---

## Dataset

**Telco Customer Churn** (IBM Sample Data)  
- ~7 000 customer records  
- 21 features: demographics, contract type, services, billing  
- Target: `Churn` (Yes / No)

Download: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>  
Place the CSV at `data/customer_churn.csv`.

---

## Project Structure

```
customer_churn/
├── data/
│   └── customer_churn.csv      ← place dataset here
├── models/
│   ├── churn_model.pkl         ← saved pipeline (generated)
│   └── figures/                ← EDA & evaluation plots (generated)
├── src/
│   ├── __init__.py
│   ├── preprocess.py           ← cleaning & sklearn pipeline
│   ├── train.py                ← full training script
│   ├── evaluate.py             ← metrics, plots, comparison
│   └── predict.py              ← single-customer prediction
├── app.py                      ← Streamlit UI
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Clone and set up the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2 — Add the dataset

Download the CSV from Kaggle and place it at:

```
data/customer_churn.csv
```

### 3 — Train the model

```bash
python -m src.train
```

This will:
- Run EDA and save figures to `models/figures/`
- Train Logistic Regression, Random Forest, XGBoost, LightGBM
- Print a comparison table
- Save the best pipeline to `models/churn_model.pkl`

### 4 — Make a prediction (CLI)

```bash
python -m src.predict
```

### 5 — Launch the Streamlit app

```bash
streamlit run app.py
```

---

## Pipeline Overview

```
Raw CSV
  → load_raw()           [pandas]
  → clean()              [drop IDs, fix TotalCharges, encode target]
  → train_test_split()   [stratified, 80/20, seed=42]
  → ColumnTransformer    [median impute + scale nums | mode impute + OHE cats]
  → Model                [LR | RF | XGB]
  → Evaluate             [accuracy, precision, recall, F1, ROC-AUC, CM]
  → Save best            [joblib.dump — full pipeline]
```

---

## EDA Findings

Exploratory Data Analysis visualizations are automatically generated during training and saved to [`models/figures/`]

| Question / Feature | Insight & Finding |
|--------------------|------------------- |
| **Overall Churn Rate** | **26.5%** overall churn rate (1,869 churned vs 5,174 stayed). Imbalanced dataset requiring balanced evaluation (Recall, ROC-AUC).  |
| **Contract Type** | Month-to-month contracts have a **42.7%** churn rate vs **11.3%** for 1-year and **2.8%** for 2-year contracts. Single strongest predictor. |
| **Tenure** | Churn is heavily concentrated in early customer lifecycles (< 12 months). Customers reaching 24+ months exhibit high retention. |
| **Monthly Charges** | Churners pay higher monthly charges on average (~$80 median vs ~$60 for retained customers). High charges accelerate early churn. |
| **Tech Support & Security** | Customers without Tech Support or Online Security churn at nearly double the rate of customers with add-on support services. |

---

## Models & Metrics

The pipeline uses a **60 / 20 / 20 stratified split** (Train / Validation / Test). Preprocessing includes dynamic feature engineering, median/mode imputation, scaling, one-hot encoding, and **SMOTE oversampling** inside cross-validation folds. Decision thresholds are tuned on the validation set to maximize F1-score before evaluation on the held-out test set.

### Test Set Performance Comparison

Evaluation metrics on the held-out test set (1,409 customers):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Optimal Threshold |
|-------|----------|-----------|--------|----------|---------|-------------------|
| **Logistic Regression** ✅ | **0.7381** | **0.5043** | **0.7834** | **0.6136** | **0.8413** | 0.44 |
| **Random Forest** | 0.7622 | 0.5376 | 0.7460 | 0.6249 | 0.8410 | 0.46 |
| **XGBoost** | 0.7686 | 0.5484 | 0.7273 | 0.6253 | 0.8385 | 0.48 |
| **LightGBM** | 0.7701 | 0.5512 | 0.7219 | 0.6251 | 0.8392 | 0.50 |
| **Voting Ensemble** | 0.7644 | 0.5412 | 0.7433 | 0.6263 | 0.8415 | 0.47 |
| **Stacking Ensemble** | 0.7658 | 0.5435 | 0.7487 | 0.6297 | 0.8421 | 0.47 |

> **Primary Model Choice:** **Logistic Regression** (or Stacking Ensemble)  
> - **Logistic Regression** delivers the highest Recall (**78.3%**), catching the maximum number of potential churners with high interpretability and lightweight execution.  
> - **Stacking Ensemble** delivers the highest overall ROC-AUC (**0.8421**) and F1-Score (**0.6297**).

---

## Feature Importance

Feature importance shows which attributes the model weighted most heavily.  
This provides model-level (not causal) explanations.

Typical top factors:
- **Contract type** (month-to-month = high risk)
- **Tenure** (shorter = higher risk)
- **Monthly charges** (higher = higher risk)
- **TechSupport / OnlineSecurity** (absence = higher risk)
- **Payment method** (electronic check = higher risk)

---

## Streamlit App

The Streamlit web application (`app.py`) provides an interactive interface for model predictions and visual retention analysis:

- **Customer Profile Entry**: Custom form sliders and dropdowns in the sidebar.
- **Instant Churn Prediction**: Probability gauge chart, churn badge, and interpretation text.
- **Explainability**: Top contributing feature importance bar chart based on trained model coefficients/weights.
- **EDA & Evaluation Gallery**: Displays saved figures directly inside the app interface.

```bash
streamlit run app.py
```

---

## Limitations

- **Correlation vs. Causation**: Feature importance scores represent global model correlations, not per-customer causal drivers.
- **Domain Scope**: Trained on telecom subscription data; may require retraining for different industries or product lines.
- **Batch Predictions**: Currently optimized for single-customer interactive queries and script execution.

---

## Reproducibility

- Fixed random seed (`RANDOM_STATE = 42`).
- Strict train-test separation with SMOTE applied strictly within cross-validation folds to prevent data leakage.
- Held-out test set untouched until final evaluation.

