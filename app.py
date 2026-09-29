"""
app.py — Customer Churn Prediction & Explainable Retention Analysis
====================================================================
Streamlit front-end that loads the saved pipeline and lets users
enter customer details to receive a churn prediction with probability
and top contributing factors.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/churn_model.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model …")
def load_bundle(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    return joblib.load(path)


def gauge_chart(prob: float) -> plt.Figure:
    """Create a minimal gauge / arc chart for churn probability."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.2)
    ax.axis("off")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#2d3436", lw=20, solid_capstyle="round")

    # Coloured fill
    fill_theta = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    color = "#e17055" if prob > 0.5 else "#00b894"
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, lw=20, solid_capstyle="round")

    # Needle
    angle = np.pi - prob * np.pi
    ax.annotate(
        "", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=15),
    )
    ax.text(0, -0.3, f"{prob:.0%}", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold")
    ax.text(0, -0.05, "Churn Probability", ha="center", va="top",
            fontsize=9, color="#b2bec3")
    return fig


def factor_bar(top_factors: list[tuple[str, float]]) -> plt.Figure:
    """Horizontal bar chart of top contributing factors."""
    labels = [f[0] for f in top_factors][::-1]
    values = [f[1] for f in top_factors][::-1]

    cmap = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(labels)))

    fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.55)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.barh(labels, values, color=cmap, edgecolor="none", height=0.6)
    ax.set_xlabel("Importance", color="#b2bec3", fontsize=10)
    ax.set_title("Top Contributing Factors", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#b2bec3")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#2d3456")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="white", fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar — customer inputs
# ---------------------------------------------------------------------------
def sidebar_inputs() -> dict:
    st.sidebar.header("🧑 Customer Profile")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Service Details")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("💳 Billing")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges = round(monthly_charges * tenure, 2)
    st.sidebar.metric(
    label="Total Charges",
    value=f"${total_charges:,.2f}",
    help="auto-calculated (tenure × monthly)",
)

    return {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
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
        "TotalCharges": total_charges,
    }


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    # Custom CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .hero-title {
            font-size: 2.6rem; font-weight: 800; color: #ffffff;
            background: linear-gradient(135deg, #a29bfe 0%, #fd79a8 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero-sub { color: #b2bec3; font-size: 1rem; margin-top: -0.5rem; }

        .metric-card {
            background: linear-gradient(145deg, #1e1e2e, #16213e);
            border: 1px solid #2d3456;
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.8rem;
        }
        .metric-card .metric-label { color: #b2bec3; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; }
        .metric-card .metric-value { color: #ffffff; font-size: 1.7rem; font-weight: 700; }

        .churn-badge-red {
            display: inline-block; padding: 0.5rem 1.5rem;
            background: linear-gradient(135deg, #e17055, #d63031);
            border-radius: 50px; color: white; font-weight: 700; font-size: 1.2rem;
        }
        .churn-badge-green {
            display: inline-block; padding: 0.5rem 1.5rem;
            background: linear-gradient(135deg, #00b894, #00cec9);
            border-radius: 50px; color: white; font-weight: 700; font-size: 1.2rem;
        }

        div[data-testid="stSidebar"] { background: #0f0f1a; border-right: 1px solid #2d3456; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown('<p class="hero-title">Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Explainable AI-powered retention analysis · Telco Customer Churn Dataset</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Load model
    bundle = load_bundle(MODEL_PATH)
    if not bundle:
        st.error(
            "⚠️ No trained model found. Please run the training script first:\n\n"
            "```bash\npython -m src.train\n```"
        )
        st.stop()

    model_name = bundle.get("model_name", "Unknown")
    metrics = bundle.get("metrics", {})
    pipeline = bundle["pipeline"]
    feature_names = bundle.get("feature_names", [])

    # Sidebar inputs
    customer = sidebar_inputs()
    predict_btn = st.sidebar.button("🔮 Predict Churn", use_container_width=True, type="primary")

    # Model info strip
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    for col, label, key in [
        (col_m1, "Active Model", None),
        (col_m2, "Test Accuracy", "accuracy"),
        (col_m3, "ROC-AUC", "roc_auc"),
        (col_m4, "F1-Score", "f1"),
    ]:
        with col:
            val = model_name if label == "Active Model" else f"{metrics.get(key, 0):.4f}" if key else "—"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Prediction section
    if predict_btn:
        with st.spinner("Analysing customer profile …"):
            from src.preprocess import engineer_features
            X = pd.DataFrame([customer])
            X = engineer_features(X)
            threshold = float(bundle.get("threshold", 0.5))
            prob = float(pipeline.predict_proba(X)[0, 1])
            pred = int(prob >= threshold)
            label_str = "Likely to Churn" if pred else "Likely to Stay"
            badge_class = "churn-badge-red" if pred else "churn-badge-green"

            # ---- Layout ----
            left, right = st.columns([1.2, 1], gap="large")

            with left:
                st.markdown("### 📊 Prediction")
                st.markdown(
                    f'<span class="{badge_class}">{label_str}</span>',
                    unsafe_allow_html=True,
                )
                st.write("")
                fig_gauge = gauge_chart(prob)
                st.pyplot(fig_gauge, width='content')

                st.markdown("**Interpretation**")
                if pred:
                    st.warning(
                        f"This customer has a **{prob:.0%}** probability of churning. "
                        "Consider proactive retention actions such as a targeted offer or personal outreach."
                    )
                else:
                    st.success(
                        f"This customer has only a **{prob:.0%}** probability of churning. "
                        "They appear to be a low-risk customer."
                    )

            with right:
                st.markdown("### 🔍 Contributing Factors")
                # Feature importance
                model_step = pipeline.named_steps["model"]
                if hasattr(model_step, "feature_importances_") and feature_names:
                    importances = model_step.feature_importances_
                elif hasattr(model_step, "coef_") and feature_names:
                    importances = np.abs(model_step.coef_[0])
                else:
                    importances = None

                if importances is not None and feature_names:
                    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:8]
                    fig_bar = factor_bar(pairs)
                    st.pyplot(fig_bar, width='stretch')
                    st.caption(
                        "ℹ️ Feature importance reflects the model's global learned weights, "
                        "not per-customer causal attributions."
                    )
                else:
                    st.info("Feature importance not available for this model.")

            # Customer profile expander
            with st.expander("📋 Full Customer Profile"):
                profile_df = pd.DataFrame(customer.items(), columns=["Feature", "Value"])
                profile_df["Value"] = profile_df["Value"].astype(str)  # ensure Arrow-compatible uniform type
                st.dataframe(profile_df, width='stretch', hide_index=True)

    else:
        st.info("👈  Fill in the customer profile in the sidebar and click **Predict Churn**.")

    # EDA figures (if available)
    fig_dir = "models/figures"
    eda_figs = {
        "Churn Distribution": f"{fig_dir}/churn_distribution.png",
        "Tenure vs Churn": f"{fig_dir}/tenure_vs_churn.png",
        "Monthly Charges vs Churn": f"{fig_dir}/monthly_charges_vs_churn.png",
        "Contract vs Churn Rate": f"{fig_dir}/contract_vs_churn.png",
        "ROC Curves": f"{fig_dir}/roc_curves.png",
    }
    available = {k: v for k, v in eda_figs.items() if os.path.exists(v)}
    if available:
        st.markdown("---")
        st.markdown("### 📈 Exploratory Data Analysis")
        cols = st.columns(min(len(available), 3))
        for i, (title, path) in enumerate(available.items()):
            with cols[i % len(cols)]:
                st.image(path, caption=title, width='stretch')


if __name__ == "__main__":
    main()
