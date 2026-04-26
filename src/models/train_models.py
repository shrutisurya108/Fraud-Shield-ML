"""
fraud-shield-ml | src/models/train_models.py
─────────────────────────────────────────────
Phase 5: Model Training & Selection

Pipeline:
  1. Load full processed dataset (590K rows)
  2. Train/val split (80/20 stratified)
  3. Logistic Regression baseline  → evaluate
  4. XGBoost + Optuna tuning       → evaluate
  5. LightGBM + Optuna tuning      → evaluate
  6. Threshold optimisation on best model
  7. Save all models + metrics + figures

Outputs:
  outputs/models/lr_baseline.joblib
  outputs/models/xgboost_model.joblib
  outputs/models/lightgbm_model.joblib
  outputs/models/model_results.json
  outputs/models/best_model_config.json
  outputs/figures/12_confusion_matrix.png
  outputs/figures/13_roc_curves.png
  outputs/figures/14_pr_curves.png
  outputs/figures/15_threshold_analysis.png
  outputs/figures/16_model_comparison.png

Run:
    python3 src/models/train_models.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import cfg
from src.eda.plot_style import (
    apply_style, save_fig,
    FRAUD_COLOR, LEGIT_COLOR, ACCENT, NEUTRAL, BG, FIGSIZE, DPI
)
from src.imbalance.preprocessor import prepare_xy, build_preprocessor
from src.models.evaluate import (
    compute_all_metrics,
    find_best_f2_threshold,
    find_precision_threshold,
    get_curve_data,
)

log = get_logger("model-training")

RANDOM_STATE = cfg.models.random_state
TEST_SIZE    = cfg.imbalance.test_size

MODEL_COLORS = {
    "Logistic Regression": NEUTRAL,
    "XGBoost":             "#E63946",
    "LightGBM":            "#2ECC71",
}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level wrappers — must be at module scope so joblib can pickle them
# ─────────────────────────────────────────────────────────────────────────────

class XGBWrapper:
    """Wraps preprocessor + XGBClassifier for a unified predict_proba."""
    def __init__(self, pp, model):
        self.pp    = pp
        self.model = model
    def predict_proba(self, X):
        return self.model.predict_proba(self.pp.transform(X))


class LGBMWrapper:
    """Wraps preprocessor + LGBMClassifier for a unified predict_proba."""
    def __init__(self, pp, model):
        self.pp    = pp
        self.model = model
    def predict_proba(self, X):
        return self.model.predict_proba(self.pp.transform(X))



# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split() -> tuple:
    """Load full parquet, split, return raw X/y splits and column lists."""
    from src.ingestion.load_data import load_processed
    log.info("Loading full processed dataset …")
    df = load_processed()
    log.info(f"  Shape: {df.shape[0]:,} × {df.shape[1]:,}")

    X, y, numeric_cols, cat_cols = prepare_xy(df, scale_numeric=False)
    log.info(f"  Features: {len(numeric_cols)} numeric + {len(cat_cols)} categorical")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    log.info(
        f"  Train: {len(X_train):,} rows "
        f"({y_train.sum():,} fraud / {(y_train==0).sum():,} legit)"
    )
    log.info(
        f"  Val:   {len(X_val):,} rows "
        f"({y_val.sum():,} fraud / {(y_val==0).sum():,} legit)"
    )
    return X_train, X_val, y_train.values, y_val.values, numeric_cols, cat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Logistic Regression baseline
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    numeric_cols: list,
    cat_cols: list,
) -> tuple[Pipeline, np.ndarray, dict]:
    """
    Logistic Regression with StandardScaler preprocessing.
    Uses class_weight='balanced' (Phase 4 winner).
    Returns (pipeline, val_probabilities, metrics_dict).
    """
    log.info("─" * 50)
    log.info("Training Logistic Regression baseline …")
    t0 = time.time()

    # LR needs scaling — use full preprocessor
    pp = build_preprocessor(numeric_cols, cat_cols, scale_numeric=True)
    lr = LogisticRegression(
        max_iter=cfg.models.logistic_regression.max_iter,
        solver=cfg.models.logistic_regression.solver,
        class_weight=cfg.models.logistic_regression.class_weight,
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline([("preprocessor", pp), ("model", lr)])
    pipeline.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info(f"  Fitted in {elapsed:.1f}s")

    y_prob = pipeline.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_f2_threshold(y_val, y_prob)
    metrics = compute_all_metrics(y_val, y_prob, best_t, "Logistic Regression")

    log.info(
        f"  PR-AUC={metrics['pr_auc']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F2={metrics['f2_score']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f}"
    )
    return pipeline, y_prob, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — XGBoost + Optuna
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    numeric_cols: list,
    cat_cols: list,
) -> tuple:
    """
    XGBoost with Optuna hyperparameter tuning.
    Uses scale_pos_weight for imbalance (no resampling needed for trees).
    """
    import optuna
    from xgboost import XGBClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info("─" * 50)
    log.info("Training XGBoost + Optuna …")

    # Preprocess once — trees don't need scaling
    pp = build_preprocessor(numeric_cols, cat_cols, scale_numeric=False)
    X_tr_proc = pp.fit_transform(X_train)
    X_vl_proc = pp.transform(X_val)

    # Class imbalance ratio for scale_pos_weight
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos
    log.info(f"  scale_pos_weight = {spw:.1f}")

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "scale_pos_weight":  spw,
            "tree_method":       "hist",
            "eval_metric":       "aucpr",
            "random_state":      RANDOM_STATE,
            "use_label_encoder": False,
        }
        model = XGBClassifier(**params)
        model.fit(
            X_tr_proc, y_train,
            eval_set=[(X_vl_proc, y_val)],
            verbose=False,
        )
        return average_precision_score_safe(y_val,
               model.predict_proba(X_vl_proc)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    log.info(f"  Running {cfg.models.optuna.n_trials} Optuna trials "
             f"(timeout={cfg.models.optuna.timeout}s) …")
    study.optimize(
        objective,
        n_trials=cfg.models.optuna.n_trials,
        timeout=cfg.models.optuna.timeout,
        show_progress_bar=False,
    )

    best = study.best_params
    log.info(f"  Best Optuna PR-AUC: {study.best_value:.4f}")
    log.info(f"  Best params: {best}")

    # Retrain with best params on full train set
    best_model = XGBClassifier(
        **best,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        use_label_encoder=False,
    )
    best_model.fit(X_tr_proc, y_train, verbose=False)

    y_prob = best_model.predict_proba(X_vl_proc)[:, 1]
    best_t, _ = find_best_f2_threshold(y_val, y_prob)
    metrics = compute_all_metrics(y_val, y_prob, best_t, "XGBoost")

    log.info(
        f"  PR-AUC={metrics['pr_auc']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F2={metrics['f2_score']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f}"
    )

    # Wrap in a simple namespace so downstream code can call .predict_proba
    return XGBWrapper(pp, best_model), y_prob, metrics, best


def average_precision_score_safe(y_true, y_prob):
    from sklearn.metrics import average_precision_score
    try:
        return average_precision_score(y_true, y_prob)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — LightGBM + Optuna
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    numeric_cols: list,
    cat_cols: list,
) -> tuple:
    """
    LightGBM with Optuna hyperparameter tuning.
    Uses is_unbalance=True for imbalance handling.
    """
    import optuna
    from lightgbm import LGBMClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info("─" * 50)
    log.info("Training LightGBM + Optuna …")

    pp = build_preprocessor(numeric_cols, cat_cols, scale_numeric=False)
    X_tr_proc = pp.fit_transform(X_train)
    X_vl_proc = pp.transform(X_val)

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 600),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 100),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "is_unbalance":    True,
            "metric":          "average_precision",
            "verbose":         -1,
            "random_state":    RANDOM_STATE,
        }
        model = LGBMClassifier(**params)
        model.fit(
            X_tr_proc, y_train,
            eval_set=[(X_vl_proc, y_val)],
            callbacks=[],
        )
        return average_precision_score_safe(y_val,
               model.predict_proba(X_vl_proc)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    log.info(f"  Running {cfg.models.optuna.n_trials} Optuna trials "
             f"(timeout={cfg.models.optuna.timeout}s) …")
    study.optimize(
        objective,
        n_trials=cfg.models.optuna.n_trials,
        timeout=cfg.models.optuna.timeout,
        show_progress_bar=False,
    )

    best = study.best_params
    log.info(f"  Best Optuna PR-AUC: {study.best_value:.4f}")
    log.info(f"  Best params: {best}")

    best_model = LGBMClassifier(
        **best,
        is_unbalance=True,
        metric="average_precision",
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    best_model.fit(X_tr_proc, y_train)

    y_prob = best_model.predict_proba(X_vl_proc)[:, 1]
    best_t, _ = find_best_f2_threshold(y_val, y_prob)
    metrics = compute_all_metrics(y_val, y_prob, best_t, "LightGBM")

    log.info(
        f"  PR-AUC={metrics['pr_auc']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F2={metrics['f2_score']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f}"
    )

    return LGBMWrapper(pp, best_model), y_prob, metrics, best


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_val: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    out_dir: Path,
) -> Path:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"],
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format=",d",
    )
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(
        f"Confusion Matrix — {model_name}\n"
        f"Threshold={threshold:.3f}  |  "
        f"Recall={tp/(tp+fn):.3f}  |  "
        f"Precision={tp/(tp+fp):.3f}",
        fontsize=11,
    )

    # Annotate FN cost explicitly
    ax.text(
        0.98, 0.02,
        f"False Negatives (missed fraud): {fn:,}\n"
        f"False Positives (wrong flags):  {fp:,}\n"
        f"True Positives  (caught fraud): {tp:,}",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=NEUTRAL, alpha=0.85),
    )

    out = out_dir / "12_confusion_matrix.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


def plot_roc_curves(curve_data: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random")

    for name, data in curve_data.items():
        ax.plot(
            data["fpr"], data["tpr"],
            label=f"{name}  (AUC={data['roc_auc']:.3f})",
            color=MODEL_COLORS[name], linewidth=2.2,
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    out = out_dir / "13_roc_curves.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


def plot_pr_curves(curve_data: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, data in curve_data.items():
        ax.plot(
            data["recall"], data["precision"],
            label=f"{name}  (PR-AUC={data['pr_auc']:.3f})",
            color=MODEL_COLORS[name], linewidth=2.2,
        )

    ax.set_xlabel("Recall (Fraud Caught / All Fraud)")
    ax.set_ylabel("Precision (Flagged Fraud / All Flagged)")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)

    out = out_dir / "14_pr_curves.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


def plot_threshold_analysis(
    y_val: np.ndarray,
    y_prob: np.ndarray,
    best_threshold: float,
    model_name: str,
    out_dir: Path,
) -> Path:
    from sklearn.metrics import fbeta_score, precision_score, recall_score

    thresholds = np.linspace(0.01, 0.99, 150)
    f2s, precs, recs = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f2s.append(fbeta_score(y_val, y_pred, beta=2, zero_division=0))
        precs.append(precision_score(y_val, y_pred, zero_division=0))
        recs.append(recall_score(y_val, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, recs,  color=LEGIT_COLOR,  linewidth=2,   label="Recall")
    ax.plot(thresholds, precs, color=FRAUD_COLOR,  linewidth=2,   label="Precision")
    ax.plot(thresholds, f2s,   color=ACCENT,       linewidth=2.5, label="F2 Score",
            linestyle="--")

    ax.axvline(best_threshold, color=NEUTRAL, linestyle=":",
               linewidth=1.8, label=f"Optimal F2 threshold ({best_threshold:.3f})")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Analysis — {model_name}")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)

    out = out_dir / "15_threshold_analysis.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


def plot_model_comparison(all_metrics: list[dict], out_dir: Path) -> Path:
    """Render a clean metric comparison table as a figure."""
    cols  = ["Model", "PR-AUC", "ROC-AUC", "F2", "Recall", "Precision", "Threshold"]
    rows  = []
    for m in all_metrics:
        rows.append([
            m["model"],
            f"{m['pr_auc']:.4f}",
            f"{m['roc_auc']:.4f}",
            f"{m['f2_score']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['threshold']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(13, 2.5 + 0.6 * len(rows)))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    # Header styling
    for c in range(len(cols)):
        tbl[(0, c)].set_facecolor("#2C3E50")
        tbl[(0, c)].set_text_props(color="white", fontweight="bold")

    # Highlight best row (highest PR-AUC)
    best_idx = max(range(len(all_metrics)),
                   key=lambda i: all_metrics[i]["pr_auc"])
    for c in range(len(cols)):
        tbl[(best_idx + 1, c)].set_facecolor("#D5F5E3")
        tbl[(best_idx + 1, c)].set_text_props(fontweight="bold")

    # Alternating row colours
    for r in range(1, len(rows) + 1):
        if r != best_idx + 1:
            bg = "#F8F9FA" if r % 2 == 0 else "white"
            for c in range(len(cols)):
                tbl[(r, c)].set_facecolor(bg)

    ax.set_title("Model Comparison — Validation Set Metrics",
                 fontsize=13, fontweight="bold", pad=20)

    out = out_dir / "16_model_comparison.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Results logging
# ─────────────────────────────────────────────────────────────────────────────

def log_final_table(all_metrics: list[dict]) -> None:
    hdr = (f"{'Model':<22} {'PR-AUC':>7} {'ROC-AUC':>8} "
           f"{'F2':>7} {'Recall':>7} {'Precision':>10} {'Threshold':>10}")
    log.info("=" * 75)
    log.info("FINAL MODEL COMPARISON")
    log.info("=" * 75)
    log.info(hdr)
    log.info("-" * 75)
    for m in all_metrics:
        log.info(
            f"{m['model']:<22} {m['pr_auc']:>7.4f} {m['roc_auc']:>8.4f} "
            f"{m['f2_score']:>7.4f} {m['recall']:>7.4f} "
            f"{m['precision']:>10.4f} {m['threshold']:>10.4f}"
        )
    log.info("=" * 75)
    best = max(all_metrics, key=lambda m: m["pr_auc"])
    log.info(f"Best model by PR-AUC: {best['model']}  "
             f"(PR-AUC={best['pr_auc']:.4f}, F2={best['f2_score']:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    apply_style()
    log.info("=" * 60)
    log.info("fraud-shield-ml | Phase 5: Model Training")
    log.info("=" * 60)

    out_dir    = PROJECT_ROOT / cfg.paths.outputs_figures
    models_dir = PROJECT_ROOT / cfg.paths.outputs_models
    out_dir.mkdir(parents=True,    exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val, num_cols, cat_cols = load_and_split()

    all_metrics   = []
    curve_data    = {}
    trained_models = {}

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_pipeline, lr_prob, lr_metrics = train_logistic_regression(
        X_train, y_train, X_val, y_val, num_cols, cat_cols
    )
    all_metrics.append(lr_metrics)
    curve_data["Logistic Regression"] = get_curve_data(y_val, lr_prob)
    trained_models["Logistic Regression"] = (lr_pipeline, lr_prob)
    joblib.dump(lr_pipeline, models_dir / "lr_baseline.joblib")
    log.info("  Saved: lr_baseline.joblib")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_wrapper, xgb_prob, xgb_metrics, xgb_best_params = train_xgboost(
        X_train, y_train, X_val, y_val, num_cols, cat_cols
    )
    all_metrics.append(xgb_metrics)
    curve_data["XGBoost"] = get_curve_data(y_val, xgb_prob)
    trained_models["XGBoost"] = (xgb_wrapper, xgb_prob)
    joblib.dump(xgb_wrapper, models_dir / "xgboost_model.joblib")
    log.info("  Saved: xgboost_model.joblib")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    lgbm_wrapper, lgbm_prob, lgbm_metrics, lgbm_best_params = train_lightgbm(
        X_train, y_train, X_val, y_val, num_cols, cat_cols
    )
    all_metrics.append(lgbm_metrics)
    curve_data["LightGBM"] = get_curve_data(y_val, lgbm_prob)
    trained_models["LightGBM"] = (lgbm_wrapper, lgbm_prob)
    joblib.dump(lgbm_wrapper, models_dir / "lightgbm_model.joblib")
    log.info("  Saved: lightgbm_model.joblib")

    # ── Select best model ─────────────────────────────────────────────────────
    log_final_table(all_metrics)
    best_metrics = max(all_metrics, key=lambda m: m["pr_auc"])
    best_name    = best_metrics["model"]
    best_wrapper, best_prob = trained_models[best_name]
    best_threshold = best_metrics["threshold"]

    log.info(f"\nBest model: {best_name}")
    log.info(f"  Operating threshold (F2-optimal): {best_threshold:.4f}")

    # High-precision threshold for hard-block
    hp_threshold, hp_recall = find_precision_threshold(y_val, best_prob, 0.80)
    log.info(f"  Hard-block threshold (≥80% precision): {hp_threshold:.4f} "
             f"(recall={hp_recall:.4f})")

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = models_dir / "model_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"  Saved: model_results.json")

    best_config = {
        "best_model":        best_name,
        "best_model_file":   {
            "Logistic Regression": "lr_baseline.joblib",
            "XGBoost":             "xgboost_model.joblib",
            "LightGBM":            "lightgbm_model.joblib",
        }[best_name],
        "f2_threshold":      round(best_threshold, 4),
        "hard_block_threshold": round(hp_threshold, 4),
        "metrics":           best_metrics,
        "feature_cols": {
            "numeric": num_cols,
            "categorical": cat_cols,
        },
        "xgboost_best_params":  xgb_best_params,
        "lgbm_best_params":     lgbm_best_params,
    }
    config_path = models_dir / "best_model_config.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    log.info(f"  Saved: best_model_config.json")

    # ── Figures ───────────────────────────────────────────────────────────────
    log.info("Generating figures …")
    plot_confusion_matrix(y_val, best_prob, best_threshold, best_name, out_dir)
    plot_roc_curves(curve_data, out_dir)
    plot_pr_curves(curve_data, out_dir)
    plot_threshold_analysis(y_val, best_prob, best_threshold, best_name, out_dir)
    plot_model_comparison(all_metrics, out_dir)

    log.info("=" * 60)
    log.info("Phase 5 complete.")
    log.info(f"  Best model:  {best_name}")
    log.info(f"  PR-AUC:      {best_metrics['pr_auc']:.4f}")
    log.info(f"  F2 Score:    {best_metrics['f2_score']:.4f}")
    log.info(f"  Recall:      {best_metrics['recall']:.4f}")
    log.info(f"  Precision:   {best_metrics['precision']:.4f}")
    log.info("  Next: python3 src/explainability/shap_analysis.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
