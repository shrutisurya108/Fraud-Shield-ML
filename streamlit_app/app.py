"""
fraud-shield-ml | streamlit_app/app.py
────────────────────────────────────────
Phase 8: Interactive Streamlit Dashboard

Four tabs:
  1. Overview       — project summary, model metrics, PR/ROC curves
  2. EDA Explorer   — all 8 EDA figures with captions
  3. Live Predict   — enter transaction features → fraud score + SHAP explanation
  4. XAI Deep Dive  — SHAP beeswarm, waterfall, LIME false positive

Run locally:
    streamlit run streamlit_app/app.py

Deploy:
    Push to GitHub → connect Streamlit Community Cloud
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR  = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Shield ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import Syne for headers, Source Serif for body */
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Source+Serif+4:ital,wght@0,300;0,400;1,300&display=swap');

  html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
  }

  h1, h2, h3, .metric-label {
    font-family: 'Syne', sans-serif !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1B2A4A 0%, #0f1d35 100%);
    border-right: 1px solid #2d4a7a;
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #1B2A4A, #253d6b);
    border: 1px solid #2d4a7a;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
  }
  [data-testid="metric-container"] label {
    color: #93c5fd !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700;
  }

  /* Tab styling */
  button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: #64748b;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #1B2A4A;
    border-bottom: 3px solid #E63946;
  }

  /* Prediction verdict boxes */
  .verdict-fraud {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 2px solid #E63946;
    border-radius: 16px;
    padding: 24px 32px;
    text-align: center;
    margin: 16px 0;
  }
  .verdict-legit {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 2px solid #10b981;
    border-radius: 16px;
    padding: 24px 32px;
    text-align: center;
    margin: 16px 0;
  }
  .verdict-text {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: white;
  }
  .verdict-sub {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.75);
    margin-top: 6px;
  }

  /* Section headers */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #1B2A4A;
    border-left: 4px solid #E63946;
    padding-left: 12px;
    margin: 20px 0 12px 0;
  }

  /* Footer */
  .footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    padding: 20px 0 8px 0;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
  }

  /* Divider */
  hr { border-color: #e2e8f0; }

  /* Info / warning boxes */
  .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model_and_config():
    """
    Load model wrapper and config.

    Priority order:
      1. Full model  (xgboost_model.joblib)   — best performance, too large for git
      2. Surrogate   (surrogate_model.joblib)  — ~3-5 MB, committed to git, Streamlit-friendly
      3. No model    — show informative warning in Live Predict tab
    """
    import __main__
    try:
        from src.models.train_models import XGBWrapper, LGBMWrapper
        __main__.XGBWrapper  = XGBWrapper
        __main__.LGBMWrapper = LGBMWrapper
    except Exception:
        pass

    import joblib

    # ── Try full model first ──────────────────────────────────────────────────
    config_path = MODELS_DIR / "best_model_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        full_model_path = MODELS_DIR / config["best_model_file"]
        if full_model_path.exists():
            wrapper = joblib.load(full_model_path)
            config["_model_source"] = "full"
            return wrapper, config

    # ── Fall back to surrogate model ──────────────────────────────────────────
    surrogate_config_path = MODELS_DIR / "surrogate_config.json"
    surrogate_model_path  = MODELS_DIR / "surrogate_model.joblib"

    if surrogate_config_path.exists() and surrogate_model_path.exists():
        with open(surrogate_config_path) as f:
            config = json.load(f)
        # Map surrogate config keys to match full config structure
        if "feature_cols" not in config:
            config["feature_cols"] = {"numeric": [], "categorical": []}
        config["_model_source"]       = "surrogate"
        config["best_model"]          = "XGBoost (Surrogate)"
        config["best_model_file"]     = "surrogate_model.joblib"
        config["hard_block_threshold"] = config.get("hard_block_threshold", 0.90)
        wrapper = joblib.load(surrogate_model_path)
        return wrapper, config

    # ── No model available ────────────────────────────────────────────────────
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config["_model_source"] = "none"
        return None, config

    return None, None


@st.cache_data(show_spinner=False)
def load_model_results():
    path = MODELS_DIR / "model_results.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_stats():
    """Load precomputed feature medians/modes for live prediction."""
    path = MODELS_DIR / "feature_stats.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_shap_values():
    path = MODELS_DIR / "shap_values.npy"
    if not path.exists():
        return None
    return np.load(path)


def fig_path(name: str) -> Path:
    return FIGURES_DIR / name


def fig_exists(name: str) -> bool:
    return (FIGURES_DIR / name).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(config):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 8px 0 20px 0;'>
          <div style='font-family:Syne,sans-serif; font-size:1.6rem;
                      font-weight:800; color:#f8fafc; letter-spacing:-0.02em;'>
            🛡️ Fraud Shield
          </div>
          <div style='font-size:0.75rem; color:#93c5fd; margin-top:4px;
                      letter-spacing:0.1em; text-transform:uppercase;'>
            ML Detection System
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        tab = st.radio(
            "Navigation",
            ["📊 Overview", "🔍 EDA Explorer",
             "⚡ Live Predict", "🧠 XAI Deep Dive"],
            label_visibility="collapsed",
        )

        st.divider()

        if config:
            m = config.get("metrics", {})
            model_source = config.get("_model_source", "full")
            label = "Deployed Model" if model_source == "surrogate" else "Best Model"
            st.markdown(
                f"<div style='font-family:Syne,sans-serif; font-size:0.7rem;"
                f"color:#93c5fd; letter-spacing:0.1em; text-transform:uppercase;"
                f"margin-bottom:8px;'>{label}</div>",
                unsafe_allow_html=True,
            )
            display_name = config.get("best_model", "—")
            st.markdown(
                f"<div style='font-family:Syne,sans-serif; font-size:1rem;"
                f"font-weight:700; color:#f8fafc;'>"
                f"{display_name}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"PR-AUC: **{m.get('pr_auc', 0):.4f}**")
            st.caption(f"F2 Score: **{m.get('f2_score', 0):.4f}**")
            st.caption(f"Recall: **{m.get('recall', 0):.1%}**")

        st.divider()
        st.markdown(
            "<div style='font-size:0.72rem; color:#64748b; text-align:center;'>"
            "IEEE-CIS Fraud Detection<br>590,540 transactions · 3.50% fraud rate"
            "</div>",
            unsafe_allow_html=True,
        )

    return tab


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────

def render_overview(config, results):
    st.markdown(
        "<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; "
        "font-weight:800; color:#1B2A4A; margin-bottom:4px;'>"
        "🛡️ Fraud Shield ML</h1>"
        "<p style='color:#64748b; font-size:1rem; margin-top:0;'>"
        "End-to-end credit card fraud detection · IEEE-CIS Dataset · XGBoost</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Key metrics ──────────────────────────────────────────────────────────
    if config and config.get("metrics"):
        m = config["metrics"]
        st.markdown('<div class="section-header">Model Performance</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PR-AUC",    f"{m['pr_auc']:.3f}")
        c2.metric("ROC-AUC",   f"{m['roc_auc']:.3f}")
        c3.metric("F2 Score",  f"{m['f2_score']:.3f}")
        c4.metric("Recall",    f"{m['recall']:.1%}")
        c5.metric("Precision", f"{m['precision']:.1%}")

        st.caption(
            f"Threshold: **{config['f2_threshold']:.3f}** (F2-optimal) · "
            f"Hard-block: **{config['hard_block_threshold']:.3f}** (≥80% precision) · "
            f"Val set: 118,108 transactions"
        )

    # ── Model comparison table ────────────────────────────────────────────────
    if results:
        st.markdown('<div class="section-header">Model Comparison</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame(results)[
            ["model", "pr_auc", "roc_auc", "f2_score",
             "recall", "precision", "threshold"]
        ].rename(columns={
            "model": "Model", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
            "f2_score": "F2 Score", "recall": "Recall",
            "precision": "Precision", "threshold": "Threshold",
        })
        best_idx = df["PR-AUC"].idxmax()

        def highlight_best(row):
            return ["background-color: #d1fae5; font-weight:bold"
                    if row.name == best_idx else "" for _ in row]

        st.dataframe(
            df.style
              .apply(highlight_best, axis=1)
              .format({"PR-AUC": "{:.4f}", "ROC-AUC": "{:.4f}",
                       "F2 Score": "{:.4f}", "Recall": "{:.4f}",
                       "Precision": "{:.4f}", "Threshold": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # ── Evaluation curves ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Evaluation Curves</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if fig_exists("14_pr_curves.png"):
            st.image(str(fig_path("14_pr_curves.png")),
                     caption="Precision-Recall Curves — All Models",
                     use_container_width=True)
    with col2:
        if fig_exists("13_roc_curves.png"):
            st.image(str(fig_path("13_roc_curves.png")),
                     caption="ROC Curves — All Models",
                     use_container_width=True)

    # ── Threshold + confusion matrix ──────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        if fig_exists("15_threshold_analysis.png"):
            st.image(str(fig_path("15_threshold_analysis.png")),
                     caption="Threshold Analysis",
                     use_container_width=True)
    with col4:
        if fig_exists("12_confusion_matrix.png"):
            st.image(str(fig_path("12_confusion_matrix.png")),
                     caption="Confusion Matrix at F2-Optimal Threshold",
                     use_container_width=True)

    # ── Pipeline summary ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Project Pipeline</div>',
                unsafe_allow_html=True)

    pipeline = [
        ("📥", "Data Ingestion",       "590K transactions, left-join on TransactionID, "
                                        "timestamp engineering, >50% missing drop → 221 features"),
        ("🔍", "EDA",                  "8 publication-quality figures: class imbalance, "
                                        "temporal patterns, missing audit, correlation heatmap"),
        ("⚖️", "Imbalance Handling",   "SMOTE vs ADASYN vs class_weight vs threshold_tune "
                                        "on PR-AUC; tree models use scale_pos_weight"),
        ("🤖", "Model Training",       "LR baseline → XGBoost + Optuna (50 trials) "
                                        "→ LightGBM + Optuna. XGBoost wins: PR-AUC 0.824"),
        ("🧠", "Explainability",       "SHAP TreeExplainer (global + per-transaction), "
                                        "LIME for false positive audit"),
        ("📄", "DS Report",            "6-section PDF memo: exec summary, methodology, "
                                        "results, model card, XAI, limitations"),
    ]
    for icon, title, desc in pipeline:
        with st.container():
            st.markdown(
                f"<div style='display:flex; align-items:flex-start; "
                f"margin-bottom:8px; padding:10px 14px; "
                f"background:#f8fafc; border-radius:10px; "
                f"border-left:3px solid #E63946;'>"
                f"<span style='font-size:1.4rem; margin-right:12px;'>{icon}</span>"
                f"<div><b style='font-family:Syne,sans-serif; color:#1B2A4A;'>"
                f"{title}</b><br>"
                f"<span style='font-size:0.88rem; color:#64748b;'>{desc}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ── Download report ───────────────────────────────────────────────────────
    report_path = REPORTS_DIR / "fraud_detection_report.pdf"
    if report_path.exists():
        st.markdown('<div class="section-header">Internal DS Report</div>',
                    unsafe_allow_html=True)
        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Download Internal DS Report (PDF)",
                data=f,
                file_name="fraud_detection_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — EDA Explorer
# ─────────────────────────────────────────────────────────────────────────────

def render_eda():
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif; color:#1B2A4A;'>"
        "🔍 EDA Explorer</h2>"
        "<p style='color:#64748b;'>8 publication-quality figures from the "
        "exploratory data analysis pipeline.</p>",
        unsafe_allow_html=True,
    )

    figures = [
        ("01_class_imbalance.png",
         "Class Imbalance",
         "590,540 transactions · 3.50% fraud · 27.6:1 imbalance ratio. "
         "Justifies PR-AUC over accuracy as primary metric."),
        ("02_amount_distribution.png",
         "Transaction Amount Distribution",
         "Log-scale comparison of fraud vs legitimate transaction amounts. "
         "Fraud skews slightly higher ($149 avg vs $134) but amount alone is a weak signal."),
        ("03_fraud_by_hour.png",
         "Fraud Rate by Hour of Day",
         "Dual-axis: transaction volume (bars) vs fraud rate (line). "
         "Peak fraud at 07:00 — typical of overnight batch fraud exploiting card-holder sleep windows."),
        ("04_fraud_by_dow.png",
         "Fraud Rate by Day of Week",
         "Fraud rate and volume by weekday. "
         "Subtle but consistent weekday/weekend pattern present in the data."),
        ("05_fraud_over_time.png",
         "Fraud Volume Over Time",
         "Weekly fraud count and rate over the 6-month dataset window "
         "(December 2017 – May 2018). Fraud patterns are stable with slight elevation mid-period."),
        ("06_missing_values.png",
         "Missing Value Audit",
         "Top 40 columns by missing rate, colour-coded by feature family. "
         "V-columns (blue) dominate — their structured missingness correlates with product type."),
        ("07_correlation_heatmap.png",
         "Correlation Heatmap",
         "Top 30 features by |correlation| with isFraud. "
         "V-features cluster in correlated groups, reflecting Vesta's engineered signal families."),
        ("08_categorical_fraud_rates.png",
         "Categorical Fraud Rates",
         "Fraud rate by ProductCD, card brand, card type, and email domain. "
         "Product code 'C' shows significantly elevated fraud risk."),
    ]

    # Grid: 2 cols
    for i in range(0, len(figures), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(figures):
                fname, title, caption = figures[i + j]
                with col:
                    st.markdown(
                        f"<div class='section-header'>{title}</div>",
                        unsafe_allow_html=True,
                    )
                    if fig_exists(fname):
                        st.image(str(fig_path(fname)),
                                 use_container_width=True)
                    else:
                        st.info(f"Figure not found: {fname}")
                    st.caption(caption)
        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Live Predict
# ─────────────────────────────────────────────────────────────────────────────

def render_live_predict(wrapper, config, feature_stats):
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif; color:#1B2A4A;'>"
        "⚡ Live Prediction</h2>"
        "<p style='color:#64748b;'>Enter transaction features to get a real-time "
        "fraud probability score and feature explanation.</p>",
        unsafe_allow_html=True,
    )

    if wrapper is None:
        st.error(
            "**No model available for live prediction.** "
            "To enable this tab locally, run: "
            "`python3 src/models/train_models.py` (full model, ~50MB) or "
            "`python3 src/reporting/train_surrogate_model.py` (surrogate, ~4MB). "
            "Then commit `outputs/models/surrogate_model.joblib` to enable "
            "live prediction on Streamlit Cloud.",
            icon="🚫",
        )
        return

    # ── Surrogate model notice ────────────────────────────────────────────────
    model_source = config.get("_model_source", "full") if config else "none"
    if model_source == "surrogate":
        st.info(
            "**Running surrogate model** (150-estimator XGBoost, trained on 80K rows). "
            "PR-AUC ≈ 0.76 vs full model's 0.824. "
            "Predictions are indicative — the full model runs locally with "
            "`python3 src/models/train_models.py`.",
            icon="🔬",
        )

    if not feature_stats:
        st.warning(
            "Feature statistics not found. "
            "Run `python3 src/reporting/generate_feature_stats.py` to generate them.",
            icon="⚠️",
        )
        return

    # Use feature_cols from feature_stats if config doesn't have them (surrogate path)
    num_cols  = (config.get("feature_cols") or {}).get("numeric") or                 feature_stats.get("numeric_cols", [])
    cat_cols  = (config.get("feature_cols") or {}).get("categorical") or                 feature_stats.get("cat_cols", [])
    medians   = feature_stats.get("medians", {})
    modes     = feature_stats.get("modes", {})
    threshold  = config.get("f2_threshold", 0.45)
    hard_block = config.get("hard_block_threshold", 0.90)

    st.info(
        f"The model uses 221 features. You control the key interpretable ones below. "
        f"Remaining V-features are filled with their validation-set medians. "
        f"Operating threshold: **{threshold:.3f}** (F2-optimal) · "
        f"Hard-block: **{hard_block:.3f}**",
        icon="ℹ️",
    )

    # ── Input widgets ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Transaction Details</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        tx_amt = st.number_input(
            "Transaction Amount (USD)", min_value=0.01,
            max_value=20_000.0, value=125.0, step=0.01,
            help="Raw transaction value in USD",
        )
        product_cd = st.selectbox(
            "Product Code", ["W", "H", "C", "S", "R"],
            help="W=web, H=hotel, C=card, S=services, R=retail",
        )

    with col2:
        card4 = st.selectbox(
            "Card Brand", ["visa", "mastercard", "discover", "amex"],
        )
        card6 = st.selectbox(
            "Card Type", ["debit", "credit"],
        )

    with col3:
        tx_hour = st.slider(
            "Transaction Hour (0=midnight)", 0, 23, 14,
            help="Hour of day when transaction occurred",
        )
        tx_dow = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Sunday"],
        )
        p_email = st.selectbox(
            "Purchaser Email Domain",
            ["gmail.com", "yahoo.com", "hotmail.com",
             "outlook.com", "anonymous.com", "other"],
        )

    # ── Build feature vector ──────────────────────────────────────────────────
    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    def build_feature_row():
        """Fill all model features: user inputs + medians for unknowns."""
        row = {}

        # Fill numeric with medians
        for col in num_cols:
            row[col] = medians.get(col, 0.0)

        # Fill categorical with modes
        for col in cat_cols:
            row[col] = modes.get(col, "")

        # Override with user inputs
        if "TransactionAmt" in row:
            row["TransactionAmt"] = float(tx_amt)
        if "tx_hour" in row:
            row["tx_hour"] = int(tx_hour)
        if "tx_day_of_week" in row:
            row["tx_day_of_week"] = dow_map[tx_dow]
        if "tx_day_of_month" in row:
            row["tx_day_of_month"] = medians.get("tx_day_of_month", 15)
        if "tx_month" in row:
            row["tx_month"] = medians.get("tx_month", 3)
        if "ProductCD" in row:
            row["ProductCD"] = product_cd
        if "card4" in row:
            row["card4"] = card4
        if "card6" in row:
            row["card6"] = card6
        if "P_emaildomain" in row:
            row["P_emaildomain"] = p_email

        return row

    # ── Predict ───────────────────────────────────────────────────────────────
    if st.button("🔍 Run Fraud Analysis", use_container_width=True, type="primary"):
        with st.spinner("Scoring transaction…"):
            row     = build_feature_row()
            X_input = pd.DataFrame([row])

            # Cast categoricals to object (pandas 2.3 safety)
            for col in cat_cols:
                if col in X_input.columns:
                    X_input[col] = X_input[col].astype(object)

            try:
                X_proc   = wrapper.pp.transform(X_input)
                y_prob   = wrapper.model.predict_proba(X_proc)[0, 1]
                is_fraud = y_prob >= threshold
                hard     = y_prob >= hard_block

                # ── Verdict ───────────────────────────────────────────────────
                st.divider()
                if hard:
                    st.markdown(
                        "<div class='verdict-fraud'>"
                        "<div class='verdict-text'>🚨 AUTO-BLOCKED</div>"
                        "<div class='verdict-sub'>High-confidence fraud · automatic decline</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                elif is_fraud:
                    st.markdown(
                        "<div class='verdict-fraud'>"
                        "<div class='verdict-text'>⚠️ FRAUD FLAGGED</div>"
                        "<div class='verdict-sub'>Sent to analyst review queue</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='verdict-legit'>"
                        "<div class='verdict-text'>✅ LEGITIMATE</div>"
                        "<div class='verdict-sub'>Transaction approved</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                # ── Score display ─────────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud Probability",  f"{y_prob:.1%}")
                m2.metric("F2 Threshold",        f"{threshold:.3f}")
                m3.metric("Hard-Block Threshold", f"{hard_block:.3f}")

                # ── SHAP explanation ──────────────────────────────────────────
                st.markdown(
                    '<div class="section-header">Why This Decision?</div>',
                    unsafe_allow_html=True,
                )
                try:
                    import shap
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    explainer  = shap.TreeExplainer(wrapper.model)
                    shap_out   = explainer(X_proc)
                    sv = shap_out.values
                    if sv.ndim == 3:
                        sv = sv[:, :, 1]

                    # Get feature names
                    try:
                        feat_names = list(wrapper.pp.get_feature_names_out())
                        feat_names = [n.replace("numeric__", "").replace(
                            "categorical__", "") for n in feat_names]
                    except Exception:
                        feat_names = [f"f{i}" for i in range(sv.shape[1])]

                    sv_1d   = sv[0]
                    top_n   = 12
                    top_idx = np.argsort(np.abs(sv_1d))[::-1][:top_n]
                    top_sv  = sv_1d[top_idx][::-1]
                    top_nm  = [feat_names[i] if i < len(feat_names)
                               else f"f{i}" for i in top_idx][::-1]

                    colors = ["#E63946" if v > 0 else "#457B9D" for v in top_sv]

                    fig, ax = plt.subplots(figsize=(9, 5))
                    fig.patch.set_facecolor("#F8F9FA")
                    ax.set_facecolor("#F8F9FA")
                    ax.barh(range(len(top_sv)), top_sv, color=colors,
                            edgecolor="white", linewidth=0.5)
                    ax.set_yticks(range(len(top_nm)))
                    ax.set_yticklabels(top_nm, fontsize=9)
                    ax.axvline(0, color="#6B7280", linewidth=1,
                               linestyle="--", alpha=0.6)
                    ax.set_xlabel("SHAP Value  (red = toward fraud, blue = toward legit)",
                                  fontsize=9)
                    ax.set_title(f"Top {top_n} Features Driving This Prediction",
                                 fontsize=11, fontweight="bold", color="#1B2A4A")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                except Exception as shap_err:
                    st.caption(f"SHAP explanation unavailable: {shap_err}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — XAI Deep Dive
# ─────────────────────────────────────────────────────────────────────────────

def render_xai():
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif; color:#1B2A4A;'>"
        "🧠 XAI Deep Dive</h2>"
        "<p style='color:#64748b;'>SHAP global importance, single-transaction "
        "waterfall explanation, and LIME false positive audit.</p>",
        unsafe_allow_html=True,
    )

    # ── SHAP Beeswarm ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Global Feature Importance — SHAP Beeswarm</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Each dot represents one transaction (1,000 sampled). "
        "**Colour** = feature value (red=high, blue=low). "
        "**Position** = SHAP value — how strongly that feature pushed the prediction "
        "toward fraud (right) or legitimate (left). "
        "Features at the top dominate the model's decisions.",
    )
    if fig_exists("17_shap_beeswarm.png"):
        st.image(str(fig_path("17_shap_beeswarm.png")),
                 use_container_width=True)
    else:
        st.info("Run `python3 src/explainability/run_xai.py` to generate this figure.")

    st.divider()

    # ── SHAP Waterfall ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">'
                '"Why Did We Flag This Transaction?" — SHAP Waterfall</div>',
                unsafe_allow_html=True)
    st.markdown(
        "One real fraud transaction from the validation set, flagged with **100% confidence**. "
        "Each bar shows how one feature shifted the prediction from the baseline "
        "(average fraud rate) up toward fraud (red) or down toward legitimate (blue). "
        "The final prediction is the sum of all contributions.",
    )
    if fig_exists("18_shap_waterfall_fraud.png"):
        st.image(str(fig_path("18_shap_waterfall_fraud.png")),
                 use_container_width=True)
    else:
        st.info("Run `python3 src/explainability/run_xai.py` to generate this figure.")

    st.divider()

    # ── LIME False Positive ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">'
                '"Why Did We Incorrectly Flag This Customer?" — LIME</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            "One **legitimate** transaction that was incorrectly flagged as fraud "
            "(a false positive). LIME fits a local linear model around this specific "
            "transaction to explain which features caused the wrong classification. "
            "\n\n"
            "**Red bars** = features that pushed the model toward a fraud flag. "
            "**Blue bars** = features that pushed toward legitimate. "
            "The transaction cleared the decision threshold despite being legitimate, "
            "resulting in one customer friction event (declined card or manual review call).",
        )
    with col2:
        st.info(
            "**Business impact:** Each false positive costs ~$8–15 in "
            "customer service handling. At 1,413 false positives on the "
            "validation set, the total friction cost is approximately "
            "**$11,000–21,000** per validation cycle.",
            icon="💡",
        )

    if fig_exists("19_lime_false_positive.png"):
        st.image(str(fig_path("19_lime_false_positive.png")),
                 use_container_width=True)
    else:
        st.info("Run `python3 src/explainability/run_xai.py` to generate this figure.")

    st.divider()

    # ── SHAP values stats ─────────────────────────────────────────────────────
    shap_vals = load_shap_values()
    if shap_vals is not None:
        st.markdown('<div class="section-header">SHAP Value Statistics</div>',
                    unsafe_allow_html=True)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top10    = np.argsort(mean_abs)[::-1][:10]
        st.caption(
            f"SHAP values computed on 1,000 sampled validation rows · "
            f"Shape: {shap_vals.shape[0]} samples × {shap_vals.shape[1]} features"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    wrapper, config = load_model_and_config()
    results         = load_model_results()
    feature_stats   = load_feature_stats()

    tab = render_sidebar(config)

    if tab == "📊 Overview":
        render_overview(config, results)

    elif tab == "🔍 EDA Explorer":
        render_eda()

    elif tab == "⚡ Live Predict":
        render_live_predict(wrapper, config, feature_stats)

    elif tab == "🧠 XAI Deep Dive":
        render_xai()

    # Footer
    st.markdown(
        "<div class='footer'>fraud-shield-ml · IEEE-CIS Fraud Detection · "
        "Built with XGBoost, SHAP, LIME, Streamlit · "
        "<a href='https://github.com/shrutisurya108/Fraud-Shield-ML' "
        "style='color:#457B9D;'>GitHub</a></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
