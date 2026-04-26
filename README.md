# 🛡️ Fraud Shield ML

End-to-end credit card fraud detection — from raw transaction data to a deployed
interactive dashboard. Built on the IEEE-CIS dataset (590,540 transactions, Vesta Corporation).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-shield-ml.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-PR--AUC%200.824-brightgreen)](https://xgboost.readthedocs.io)
[![Tests](https://img.shields.io/badge/tests-215%20passing-success)](tests/)

---

## Live Demo

**→ [fraud-shield-ml.streamlit.app](https://fraud-shield-ml.streamlit.app)**

Four tabs: project overview · EDA figures · live transaction scoring · SHAP + LIME explainability.

---

## Results

| Model | PR-AUC | ROC-AUC | F2 Score | Recall | Precision |
|-------|--------|---------|----------|--------|-----------|
| Logistic Regression | 0.3062 | 0.8372 | 0.3958 | 0.5437 | 0.1896 |
| LightGBM + Optuna | 0.7861 | 0.9622 | 0.7364 | 0.7687 | 0.6305 |
| **XGBoost + Optuna** | **0.8242** | **0.9670** | **0.7669** | **0.7866** | **0.6970** |

XGBoost catches **78.7% of all fraud** at **69.7% precision** on 118,108
held-out transactions — a **169% PR-AUC improvement** over logistic regression.
Operating threshold: **0.446** (F2-optimal). Hard-block threshold: **0.990**.

---

## Why This Problem Is Hard

The dataset has a **27.6:1 class imbalance** — 3.50% fraud rate.
A model predicting every transaction as legitimate scores 96.5% accuracy
while catching zero fraud. Accuracy is useless here.

The primary metric is **F2 score** (recall weighted 2× over precision),
reflecting the real cost asymmetry: a missed fraud costs the full transaction
value plus chargeback fees; a false alarm causes recoverable customer friction.
**PR-AUC** is used for threshold-independent model and strategy comparison.

---

## Project Structure

```
Fraud-Shield-ML/
├── config/config.yaml           # all parameters and paths
├── data/raw/                    # place Kaggle CSVs here (git-ignored)
├── docs/business_framing.md     # problem definition, metrics rationale
├── src/
│   ├── ingestion/load_data.py   # merge, timestamp engineering → parquet
│   ├── eda/run_eda.py           # 8 EDA figures
│   ├── imbalance/               # SMOTE vs ADASYN vs class_weight comparison
│   ├── models/                  # LR → XGBoost → LightGBM + Optuna tuning
│   ├── explainability/          # SHAP + LIME
│   ├── reporting/               # PDF report + surrogate model
│   └── utils/                   # logger, config loader
├── outputs/figures/             # 19 PNG figures (committed)
├── outputs/models/              # JSON configs + surrogate model (committed)
├── streamlit_app/app.py         # 4-tab dashboard
├── tests/                       # 215-test pytest suite
├── requirements.txt             # full local dependencies
└── requirements_streamlit.txt   # minimal Streamlit Cloud dependencies
```

---

## Quick Start

```bash
git clone https://github.com/shrutisurya108/Fraud-Shield-ML.git
cd Fraud-Shield-ML
pip3 install -r requirements.txt && pip3 install -e .

# Place train_transaction.csv and train_identity.csv in data/raw/
# Download from: kaggle.com/competitions/ieee-fraud-detection/data

python3 src/ingestion/load_data.py
python3 src/eda/run_eda.py
python3 src/imbalance/compare_strategies.py   # ~10-15 min
python3 src/models/train_models.py             # ~45-65 min
python3 src/explainability/run_xai.py
python3 src/reporting/generate_report.py
python3 src/reporting/generate_feature_stats.py

streamlit run streamlit_app/app.py
pytest tests/ -v   # 215 tests
```

---

## Pipeline Overview

| Phase | What it does |
|-------|-------------|
| Data Ingestion | Left-join transaction + identity on TransactionID, derive temporal features, drop >50% missing columns → 221 features |
| EDA | Class imbalance, amount distributions, hourly/weekly fraud patterns, missing audit, correlation heatmap, categorical fraud rates |
| Imbalance Handling | SMOTE vs ADASYN vs class_weight vs threshold_tune on 50K stratified subsample, evaluated on PR-AUC |
| Model Training | LR baseline → XGBoost + Optuna (50 trials) → LightGBM + Optuna, F2-optimal threshold search |
| Explainability | SHAP beeswarm (global), SHAP waterfall (single fraud), LIME false positive ("why did we flag this customer?") |
| DS Report | 6-section ReportLab PDF: exec summary, methodology, results, model card, XAI, limitations |
| Dashboard | Streamlit with live scoring, SHAP explanation per prediction, EDA explorer, XAI deep dive |

---

## Key Engineering Decisions

**Pandas 2.3 dtype handling** — `future.infer_string=True` silently converts
object columns to `StringDtype` on slice operations, breaking sklearn's
median imputer. Fixed by using `df.select_dtypes(exclude='number')` and
explicit object casts — immune to pandas version changes.

**Joblib pickling scope** — model wrapper classes must be at module level,
not defined inside functions, for pickle to resolve them across execution
contexts.

**Surrogate for deployment** — the full XGBoost model is ~50MB, exceeding
GitHub's practical commit limits. A lightweight surrogate (150 estimators,
80K training rows, ~4MB) enables live prediction on Streamlit Cloud's free tier.

---

## Explainability

**SHAP global** — V-features (Vesta behavioural signals) dominate.
TransactionAmt ranks lower than V12, V87, V258, confirming the model
learned velocity and behavioural patterns — not just amount thresholds.

**SHAP waterfall** — the highest-confidence fraud (probability = 1.000)
shows multiple V-features compounding in the same direction, a convergence
of behavioural anomalies rather than a single trigger.

**LIME false positive** — selected at fraud probability 0.595
against threshold 0.446, close enough to the boundary to reveal exactly
which features caused the incorrect flag on a legitimate transaction.

---

## Tech Stack

pandas · numpy · scikit-learn · XGBoost · LightGBM · imbalanced-learn ·
Optuna · SHAP · LIME · matplotlib · seaborn · ReportLab · Streamlit ·
loguru · pytest · PyArrow

---

## Limitations

- Trained on 2017–2018 data; fraud tactics evolve and the model should be retrained quarterly
- Vesta V-features are proprietary — exact definitions are not publicly documented
- No real-time velocity features (transactions per card per hour), which are among the strongest production signals
- Subgroup fairness across card issuers and geographies was not evaluated
- Live prediction on Streamlit Cloud uses a surrogate model (PR-AUC ≈ 0.76); full model runs locally

---

## Collaboration and Acknowledgements
This project was built and developed in collaboration with [Harshith Bhattaram](https://github.com/maniharshith68).


## 👤 Authors
- [Harshith Bhattaram](https://github.com/maniharshith68)
- [Shruti Kumari](https://github.com/shrutisurya108)

