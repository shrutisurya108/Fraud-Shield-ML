"""
fraud-shield-ml | src/explainability/lime_analysis.py
───────────────────────────────────────────────────────
Phase 6: LIME Local Explainability

Produces:
  outputs/figures/19_lime_false_positive.png

Frames the explanation as: "Why did we incorrectly flag this
legitimate customer?" — the key business narrative for false positives.

Run:
    python3 src/explainability/lime_analysis.py
    (or called from shap_analysis.py as part of run_xai())
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import cfg
from src.eda.plot_style import apply_style, save_fig, FRAUD_COLOR, LEGIT_COLOR, NEUTRAL, ACCENT
from src.imbalance.preprocessor import prepare_xy
from src.models.train_models import XGBWrapper, LGBMWrapper  # noqa

log = get_logger("lime")

LIME_FEATURES = cfg.explainability.lime_num_features   # 15
LIME_SAMPLES  = cfg.explainability.lime_num_samples    # 5000
RANDOM_STATE  = cfg.models.random_state


# ─────────────────────────────────────────────────────────────────────────────
# Load data and model
# ─────────────────────────────────────────────────────────────────────────────

def load_for_lime() -> tuple:
    """
    Load the best model wrapper, validation data (raw + processed),
    feature names, and the operating threshold.
    """
    import __main__
    __main__.XGBWrapper  = XGBWrapper
    __main__.LGBMWrapper = LGBMWrapper

    models_dir = PROJECT_ROOT / cfg.paths.outputs_models
    with open(models_dir / "best_model_config.json") as f:
        config = json.load(f)

    wrapper   = joblib.load(models_dir / config["best_model_file"])
    threshold = config["f2_threshold"]
    num_cols  = config["feature_cols"]["numeric"]
    cat_cols  = config["feature_cols"]["categorical"]

    from src.ingestion.load_data import load_processed
    from sklearn.model_selection import train_test_split

    df = load_processed()
    X, y, _, _ = prepare_xy(df, scale_numeric=False)

    _, X_val_raw, _, y_val = train_test_split(
        X, y,
        test_size=cfg.imbalance.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    y_val = y_val.values

    # Pre-process
    X_val_proc = wrapper.pp.transform(X_val_raw)

    # Feature names
    try:
        feature_names = list(wrapper.pp.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(X_val_proc.shape[1])]
    feature_names = [n.replace("numeric__", "").replace("categorical__", "")
                     for n in feature_names]

    log.info(f"Loaded: {len(y_val):,} val rows | threshold={threshold:.4f}")
    return wrapper, X_val_proc, X_val_raw, y_val, feature_names, threshold


# ─────────────────────────────────────────────────────────────────────────────
# Find a false positive
# ─────────────────────────────────────────────────────────────────────────────

def find_false_positive(
    wrapper,
    X_val_proc: np.ndarray,
    y_val: np.ndarray,
    threshold: float,
) -> int:
    """
    Return the index of a false positive with fraud probability
    closest to (threshold + 0.15) — a clear but not extreme false positive,
    which makes for the most instructive LIME explanation.
    """
    y_prob = wrapper.model.predict_proba(X_val_proc)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # False positives: predicted fraud but actually legitimate
    fp_mask = (y_val == 0) & (y_pred == 1)
    log.info(f"  False positives in val set: {fp_mask.sum():,}")

    if fp_mask.sum() == 0:
        log.warning("  No false positives found — using most uncertain legitimate")
        legit_mask = y_val == 0
        fp_mask    = legit_mask

    fp_probs = y_prob.copy()
    fp_probs[~fp_mask] = -1

    # Pick the FP closest to threshold + 0.15 (instructive middle ground)
    target     = threshold + 0.15
    fp_indices = np.where(fp_mask)[0]
    closest    = fp_indices[np.argmin(np.abs(y_prob[fp_indices] - target))]

    log.info(
        f"  Selected FP idx={closest} | "
        f"fraud_prob={y_prob[closest]:.4f} | "
        f"threshold={threshold:.4f}"
    )
    return int(closest), float(y_prob[closest])


# ─────────────────────────────────────────────────────────────────────────────
# LIME explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_lime(
    wrapper,
    X_val_proc: np.ndarray,
    fp_idx: int,
    feature_names: list[str],
) -> object:
    """
    Fit a LIME tabular explainer around the false positive transaction.
    Returns the LIME explanation object.
    """
    from lime.lime_tabular import LimeTabularExplainer

    log.info(f"Fitting LIME explainer (n_samples={LIME_SAMPLES}) …")

    # Use the pre-processed validation set as training distribution for LIME
    explainer = LimeTabularExplainer(
        training_data=X_val_proc,
        feature_names=feature_names,
        class_names=["Legitimate", "Fraud"],
        mode="classification",
        random_state=RANDOM_STATE,
        discretize_continuous=True,
    )

    def predict_fn(X):
        return wrapper.model.predict_proba(X)

    explanation = explainer.explain_instance(
        data_row=X_val_proc[fp_idx],
        predict_fn=predict_fn,
        num_features=LIME_FEATURES,
        num_samples=LIME_SAMPLES,
        labels=(1,),  # explain fraud class
    )

    log.info("  LIME explanation computed ✓")
    return explanation


# ─────────────────────────────────────────────────────────────────────────────
# Plot LIME explanation
# ─────────────────────────────────────────────────────────────────────────────

def plot_lime_false_positive(
    explanation,
    fp_idx: int,
    fp_prob: float,
    threshold: float,
    out_dir: Path,
) -> Path:
    """
    Horizontal bar chart of LIME feature contributions for the false positive.
    Framed as: 'Why did we incorrectly flag this legitimate transaction?'
    """
    log.info("Plotting LIME false positive explanation …")
    apply_style()

    # Extract top features and weights for fraud class (label=1)
    exp_list = explanation.as_list(label=1)
    # exp_list: [(feature_description, weight), ...]
    # Positive weight = pushes toward fraud prediction

    # Sort by absolute weight
    exp_list_sorted = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
    features = [e[0] for e in exp_list_sorted]
    weights  = [e[1] for e in exp_list_sorted]

    # Reverse for bottom-to-top plot
    features = features[::-1]
    weights  = weights[::-1]

    colors = [FRAUD_COLOR if w > 0 else LEGIT_COLOR for w in weights]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(features))

    bars = ax.barh(y_pos, weights, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)

    # Value labels
    for bar, w in zip(bars, weights):
        sign = "+" if w > 0 else ""
        ax.text(
            w + (0.001 if w > 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{w:.4f}",
            va="center",
            ha="left" if w > 0 else "right",
            fontsize=8.5,
            color=NEUTRAL,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8.5)
    ax.axvline(0, color=NEUTRAL, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("LIME Weight  (positive = contributed to FRAUD flag)")

    ax.set_title(
        f"LIME Explanation — Why Did We Incorrectly Flag This Customer?\n"
        f"This transaction is LEGITIMATE but was flagged as fraud\n"
        f"Fraud probability: {fp_prob:.1%}  |  Decision threshold: {threshold:.3f}",
        fontsize=11,
        pad=12,
    )

    # Annotation box with business interpretation
    ax.text(
        0.98, 0.02,
        "Red bars = features that pushed toward a FRAUD flag\n"
        "Blue bars = features that pushed toward LEGITIMATE\n"
        "This false positive cost: one frustrated legitimate customer",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=NEUTRAL, alpha=0.9),
    )

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=FRAUD_COLOR, label="Pushed toward FRAUD flag"),
            Patch(facecolor=LEGIT_COLOR, label="Pushed toward LEGIT"),
        ],
        loc="upper left", fontsize=9,
    )

    out = out_dir / "19_lime_false_positive.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_lime() -> Path:
    """Full LIME pipeline. Returns path to saved figure."""
    apply_style()
    out_dir = PROJECT_ROOT / cfg.paths.outputs_figures
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("fraud-shield-ml | Phase 6b: LIME Analysis")
    log.info("=" * 60)

    wrapper, X_val_proc, X_val_raw, y_val, feature_names, threshold = \
        load_for_lime()

    fp_idx, fp_prob = find_false_positive(
        wrapper, X_val_proc, y_val, threshold
    )

    explanation = explain_with_lime(
        wrapper, X_val_proc, fp_idx, feature_names
    )

    out = plot_lime_false_positive(
        explanation, fp_idx, fp_prob, threshold, out_dir
    )

    log.info("=" * 60)
    log.info("LIME analysis complete.")
    log.info(f"  {out.name}")
    log.info("  Next: python3 src/reporting/generate_report.py")
    log.info("=" * 60)

    return out


if __name__ == "__main__":
    run_lime()
