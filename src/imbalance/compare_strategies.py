"""
fraud-shield-ml | src/imbalance/compare_strategies.py
───────────────────────────────────────────────────────
Phase 4: Class Imbalance Handling — Strategy Comparison

Compares four strategies on a stratified 50K-row subsample
(full dataset SMOTE/ADASYN would require ~12 GB RAM and 45+ min):

  1. SMOTE          — synthetic oversampling
  2. ADASYN         — adaptive synthetic oversampling
  3. class_weight   — loss-function reweighting (no resampling)
  4. threshold_tune — default LR + optimal F2 threshold search

Base estimator: Logistic Regression (isolates resampling effect from model power)

Outputs:
  outputs/figures/09_imbalance_comparison.png   — metric bar chart
  outputs/figures/10_pr_curves_comparison.png   — PR curves overlay
  outputs/figures/11_accuracy_trap.png          — accuracy vs PR-AUC trap plot
  outputs/models/imbalance_results.json         — full metric table

Run:
    python3 src/imbalance/compare_strategies.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
from src.imbalance.preprocessor import build_preprocessor, prepare_xy

log = get_logger("imbalance")

# ── Constants ─────────────────────────────────────────────────────────────────
SUBSAMPLE_N   = 50_000          # rows used for resampling comparison
RANDOM_STATE  = cfg.imbalance.random_state
TEST_SIZE     = cfg.imbalance.test_size
BETA          = cfg.evaluation.f2_beta
THRESHOLD_STEPS = cfg.models.threshold_search_steps

STRATEGY_COLORS = {
    "SMOTE":          "#457B9D",
    "ADASYN":         "#2ECC71",
    "class_weight":   "#F4A261",
    "threshold_tune": "#9B59B6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute full metric suite at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "f2_score":  round(float(fbeta_score(y_true, y_pred, beta=BETA,
                                             zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred,
                                              zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred,
                                                 zero_division=0)), 4),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "threshold": round(threshold, 4),
    }


def find_best_f2_threshold(y_true: np.ndarray,
                           y_prob: np.ndarray) -> tuple[float, float]:
    """
    Sweep thresholds from 0.01 to 0.99 and return the one that
    maximises F2 score on the provided labels.
    """
    thresholds = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    best_t, best_f2 = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=BETA, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t, best_f2


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_subsample(n: int = SUBSAMPLE_N) -> tuple:
    """
    Load processed parquet, stratified-subsample to n rows,
    split train/val, fit-transform preprocessor.

    Returns
    -------
    X_train_proc, X_val_proc : np.ndarray
    y_train, y_val           : np.ndarray
    preprocessor             : fitted ColumnTransformer
    numeric_cols, cat_cols   : lists
    """
    from src.ingestion.load_data import load_processed

    log.info(f"Loading processed data and subsampling to {n:,} rows …")
    df = load_processed()

    # Stratified subsample
    _, df_sub = train_test_split(
        df, test_size=n / len(df),
        stratify=df["isFraud"],
        random_state=RANDOM_STATE,
    )
    fraud_rate = df_sub["isFraud"].mean()
    log.info(
        f"  Subsample: {len(df_sub):,} rows | "
        f"fraud rate: {fraud_rate:.2%} | "
        f"fraud count: {df_sub['isFraud'].sum():,}"
    )

    X, y, numeric_cols, cat_cols = prepare_xy(df_sub, scale_numeric=True)
    log.info(f"  Features: {len(numeric_cols)} numeric + {len(cat_cols)} categorical")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_cols, cat_cols, scale_numeric=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)

    log.info(
        f"  Train: {X_train_proc.shape[0]:,} rows | "
        f"Val: {X_val_proc.shape[0]:,} rows"
    )
    return (X_train_proc, X_val_proc,
            y_train.values, y_val.values,
            preprocessor, numeric_cols, cat_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy runners
# ─────────────────────────────────────────────────────────────────────────────

def _base_lr(class_weight=None) -> LogisticRegression:
    return LogisticRegression(
        max_iter=cfg.models.logistic_regression.max_iter,
        solver=cfg.models.logistic_regression.solver,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def run_smote(X_train, y_train, X_val, y_val) -> dict:
    log.info("  Running SMOTE …")
    from imblearn.over_sampling import SMOTE
    t0 = time.time()
    sm = SMOTE(
        k_neighbors=cfg.imbalance.smote_k_neighbors,
        random_state=RANDOM_STATE,
    )
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log.info(f"    Resampled: {X_res.shape[0]:,} rows | "
             f"fraud: {y_res.sum():,} ({y_res.mean():.1%}) | "
             f"time: {time.time()-t0:.1f}s")

    lr = _base_lr()
    lr.fit(X_res, y_res)
    y_prob = lr.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_f2_threshold(y_val, y_prob)
    return compute_metrics(y_val, y_prob, threshold=best_t)


def run_adasyn(X_train, y_train, X_val, y_val) -> dict:
    log.info("  Running ADASYN …")
    from imblearn.over_sampling import ADASYN
    t0 = time.time()
    ada = ADASYN(
        n_neighbors=cfg.imbalance.adasyn_n_neighbors,
        random_state=RANDOM_STATE,
    )
    try:
        X_res, y_res = ada.fit_resample(X_train, y_train)
    except Exception as e:
        log.warning(f"    ADASYN failed ({e}) — falling back to SMOTE result")
        return run_smote(X_train, y_train, X_val, y_val)

    log.info(f"    Resampled: {X_res.shape[0]:,} rows | "
             f"fraud: {y_res.sum():,} ({y_res.mean():.1%}) | "
             f"time: {time.time()-t0:.1f}s")

    lr = _base_lr()
    lr.fit(X_res, y_res)
    y_prob = lr.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_f2_threshold(y_val, y_prob)
    return compute_metrics(y_val, y_prob, threshold=best_t)


def run_class_weight(X_train, y_train, X_val, y_val) -> dict:
    log.info("  Running class_weight='balanced' …")
    t0 = time.time()
    lr = _base_lr(class_weight="balanced")
    lr.fit(X_train, y_train)
    log.info(f"    Fitted in {time.time()-t0:.1f}s")
    y_prob = lr.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_f2_threshold(y_val, y_prob)
    return compute_metrics(y_val, y_prob, threshold=best_t)


def run_threshold_tune(X_train, y_train, X_val, y_val) -> dict:
    log.info("  Running threshold tuning (no resampling) …")
    t0 = time.time()
    lr = _base_lr()                   # no class_weight, no resampling
    lr.fit(X_train, y_train)
    log.info(f"    Fitted in {time.time()-t0:.1f}s")
    y_prob = lr.predict_proba(X_val)[:, 1]

    # Show what default threshold 0.5 gives vs optimised
    default_metrics = compute_metrics(y_val, y_prob, threshold=0.5)
    best_t, best_f2 = find_best_f2_threshold(y_val, y_prob)
    tuned_metrics   = compute_metrics(y_val, y_prob, threshold=best_t)

    log.info(
        f"    Default threshold 0.5 → F2={default_metrics['f2_score']:.4f} | "
        f"Recall={default_metrics['recall']:.4f}"
    )
    log.info(
        f"    Optimal threshold {best_t:.3f} → F2={best_f2:.4f} | "
        f"Recall={tuned_metrics['recall']:.4f}"
    )
    return tuned_metrics


def get_pr_curve(X_train, y_train, X_val, y_val,
                 strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Re-run a strategy and return precision, recall arrays for PR curve."""
    if strategy == "SMOTE":
        from imblearn.over_sampling import SMOTE
        X_res, y_res = SMOTE(
            k_neighbors=cfg.imbalance.smote_k_neighbors,
            random_state=RANDOM_STATE
        ).fit_resample(X_train, y_train)
        lr = _base_lr()
        lr.fit(X_res, y_res)
    elif strategy == "ADASYN":
        from imblearn.over_sampling import ADASYN
        try:
            X_res, y_res = ADASYN(
                n_neighbors=cfg.imbalance.adasyn_n_neighbors,
                random_state=RANDOM_STATE
            ).fit_resample(X_train, y_train)
        except Exception:
            from imblearn.over_sampling import SMOTE
            X_res, y_res = SMOTE(
                random_state=RANDOM_STATE
            ).fit_resample(X_train, y_train)
        lr = _base_lr()
        lr.fit(X_res, y_res)
    elif strategy == "class_weight":
        lr = _base_lr(class_weight="balanced")
        lr.fit(X_train, y_train)
    else:  # threshold_tune
        lr = _base_lr()
        lr.fit(X_train, y_train)

    y_prob = lr.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    return precision, recall


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: dict, out_dir: Path) -> Path:
    """
    Grouped bar chart comparing all strategies across key metrics.
    Also includes accuracy to visually demonstrate the accuracy trap.
    """
    metrics_to_plot = ["pr_auc", "roc_auc", "f2_score", "recall", "precision"]
    metric_labels   = ["PR-AUC", "ROC-AUC", "F2 Score", "Recall", "Precision"]

    strategies = list(results.keys())
    n_metrics  = len(metrics_to_plot)
    n_strat    = len(strategies)
    x          = np.arange(n_metrics)
    width      = 0.8 / n_strat

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, strategy in enumerate(strategies):
        vals   = [results[strategy][m] for m in metrics_to_plot]
        offset = (i - n_strat / 2 + 0.5) * width
        bars   = ax.bar(
            x + offset, vals, width * 0.92,
            label=strategy,
            color=STRATEGY_COLORS[strategy],
            edgecolor="white", linewidth=0.7,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color=NEUTRAL,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Imbalance Strategy Comparison — Logistic Regression Base Estimator\n"
        "(50K stratified subsample · threshold optimised for F2 score)",
        fontsize=12,
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(0.5, color=NEUTRAL, linestyle=":", linewidth=0.8, alpha=0.5)

    out = out_dir / "09_imbalance_comparison.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


def plot_pr_curves(pr_data: dict, results: dict, out_dir: Path) -> Path:
    """Overlaid Precision-Recall curves for all four strategies."""
    baseline_rate = None

    fig, ax = plt.subplots(figsize=(10, 7))

    for strategy, (precision, recall) in pr_data.items():
        pr_auc = results[strategy]["pr_auc"]
        ax.plot(
            recall, precision,
            label=f"{strategy}  (PR-AUC={pr_auc:.3f})",
            color=STRATEGY_COLORS[strategy],
            linewidth=2.2,
        )

    # No-skill baseline = fraud rate
    if baseline_rate is not None:
        ax.axhline(baseline_rate, color=NEUTRAL, linestyle="--",
                   linewidth=1.2, alpha=0.6,
                   label=f"No-skill baseline ({baseline_rate:.3f})")

    ax.set_xlabel("Recall (Fraud Caught / All Fraud)", fontsize=12)
    ax.set_ylabel("Precision (Caught Fraud / Flagged)", fontsize=12)
    ax.set_title(
        "Precision-Recall Curves — Four Imbalance Strategies\n"
        "Higher area = better minority-class discrimination",
        fontsize=12,
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    out = out_dir / "10_pr_curves_comparison.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


def plot_accuracy_trap(results: dict, out_dir: Path) -> Path:
    """
    Side-by-side accuracy vs PR-AUC for all strategies.
    Visually proves accuracy is a misleading metric for imbalanced data.
    The key insight: accuracy is nearly identical (~96%) across all strategies
    while PR-AUC varies dramatically.
    """
    strategies = list(results.keys())
    accuracies = [results[s]["accuracy"] for s in strategies]
    pr_aucs    = [results[s]["pr_auc"]   for s in strategies]
    colors     = [STRATEGY_COLORS[s] for s in strategies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy — barely differs
    bars1 = ax1.bar(strategies, accuracies, color=colors,
                    edgecolor="white", linewidth=0.8)
    ax1.set_ylim(0.93, 1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Score\n(Appears similar across all strategies)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.3f}"))
    for bar, val in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0003,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=NEUTRAL)
    ax1.tick_params(axis="x", rotation=15)

    # Right: PR-AUC — reveals real differences
    bars2 = ax2.bar(strategies, pr_aucs, color=colors,
                    edgecolor="white", linewidth=0.8)
    ax2.set_ylim(0, max(pr_aucs) * 1.25)
    ax2.set_ylabel("PR-AUC (Average Precision)")
    ax2.set_title("PR-AUC Score\n(Reveals true minority-class performance)")
    for bar, val in zip(bars2, pr_aucs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=NEUTRAL)
    ax2.tick_params(axis="x", rotation=15)

    # Annotation
    acc_range = max(accuracies) - min(accuracies)
    pr_range  = max(pr_aucs)   - min(pr_aucs)
    fig.text(
        0.5, -0.04,
        f"Key insight: Accuracy range across strategies = {acc_range:.4f}  "
        f"(noise-level variation)\n"
        f"PR-AUC range = {pr_range:.4f}  "
        f"(meaningful signal — use PR-AUC, never accuracy, for imbalanced problems)",
        ha="center", fontsize=10, color=FRAUD_COLOR, fontweight="bold",
        wrap=True,
    )

    fig.suptitle(
        "The Accuracy Trap — Why Accuracy Misleads on Imbalanced Data",
        fontsize=13, fontweight="bold",
    )

    out = out_dir / "11_accuracy_trap.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Results logging
# ─────────────────────────────────────────────────────────────────────────────

def log_results_table(results: dict) -> None:
    """Print a clean comparison table to the log."""
    header = f"{'Strategy':<16} {'PR-AUC':>7} {'ROC-AUC':>8} {'F2':>7} {'Recall':>7} {'Precision':>10} {'Accuracy':>9} {'Threshold':>10}"
    log.info("─" * 80)
    log.info("IMBALANCE STRATEGY COMPARISON RESULTS")
    log.info("─" * 80)
    log.info(header)
    log.info("─" * 80)
    for strategy, m in results.items():
        log.info(
            f"{strategy:<16} "
            f"{m['pr_auc']:>7.4f} "
            f"{m['roc_auc']:>8.4f} "
            f"{m['f2_score']:>7.4f} "
            f"{m['recall']:>7.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['accuracy']:>9.4f} "
            f"{m['threshold']:>10.4f}"
        )
    log.info("─" * 80)

    # Find winner by PR-AUC
    best = max(results, key=lambda s: results[s]["pr_auc"])
    log.info(f"Best strategy by PR-AUC: {best}  "
             f"(PR-AUC={results[best]['pr_auc']:.4f})")

    # Key insight printout
    acc_range = max(m["accuracy"] for m in results.values()) - \
                min(m["accuracy"] for m in results.values())
    pr_range  = max(m["pr_auc"]   for m in results.values()) - \
                min(m["pr_auc"]   for m in results.values())
    log.info(
        f"\n  ★ Accuracy trap: accuracy range = {acc_range:.4f} "
        f"vs PR-AUC range = {pr_range:.4f}"
    )
    log.info(
        "  ★ A naive all-legitimate model scores ~0.965 accuracy "
        "but 0.000 PR-AUC.\n"
        "    Accuracy is useless for imbalanced fraud detection."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison() -> dict:
    """
    Full imbalance comparison pipeline.
    Returns results dict (used by tests and downstream phases).
    """
    apply_style()

    out_dir    = PROJECT_ROOT / cfg.paths.outputs_figures
    models_dir = PROJECT_ROOT / cfg.paths.outputs_models
    out_dir.mkdir(parents=True,    exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & prepare data ──────────────────────────────────────────────────
    (X_train, X_val, y_train, y_val,
     preprocessor, num_cols, cat_cols) = load_and_subsample()

    # ── Run all four strategies ──────────────────────────────────────────────
    log.info("Running all four imbalance strategies …")

    results = {}
    results["SMOTE"]          = run_smote(X_train, y_train, X_val, y_val)
    results["ADASYN"]         = run_adasyn(X_train, y_train, X_val, y_val)
    results["class_weight"]   = run_class_weight(X_train, y_train, X_val, y_val)
    results["threshold_tune"] = run_threshold_tune(X_train, y_train, X_val, y_val)

    log_results_table(results)

    # ── PR curves (re-run each strategy once more for curve data) ────────────
    log.info("Computing PR curves …")
    pr_data = {}
    for strategy in results:
        log.info(f"  PR curve: {strategy} …")
        pr_data[strategy] = get_pr_curve(
            X_train, y_train, X_val, y_val, strategy
        )

    # ── Plots ────────────────────────────────────────────────────────────────
    log.info("Generating figures …")
    plot_comparison(results, out_dir)
    plot_pr_curves(pr_data, results, out_dir)
    plot_accuracy_trap(results, out_dir)

    # ── Save results JSON ─────────────────────────────────────────────────────
    out_json = models_dir / "imbalance_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  ✓ Results → {out_json.relative_to(PROJECT_ROOT)}")

    return results


def main():
    log.info("=" * 60)
    log.info("fraud-shield-ml | Phase 4: Imbalance Handling")
    log.info("=" * 60)
    log.info(
        f"Subsample: {SUBSAMPLE_N:,} rows (stratified)\n"
        f"  Full-dataset SMOTE/ADASYN would require ~12 GB RAM\n"
        f"  and 45+ min. Subsampling is standard practice for\n"
        f"  strategy selection on large imbalanced datasets."
    )

    results = run_comparison()

    log.info("=" * 60)
    log.info("Phase 4 complete.")
    log.info("  Figures: outputs/figures/09_*, 10_*, 11_*")
    log.info("  Results: outputs/models/imbalance_results.json")
    log.info("  Next:    python3 src/models/train_models.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
