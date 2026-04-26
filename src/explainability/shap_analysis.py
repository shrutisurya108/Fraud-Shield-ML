"""
fraud-shield-ml | src/explainability/shap_analysis.py
───────────────────────────────────────────────────────
Phase 6: SHAP Explainability

Produces:
  outputs/figures/17_shap_beeswarm.png    — global feature importance
  outputs/figures/18_shap_waterfall_fraud.png — single flagged transaction
  outputs/shap_values.npy                 — cached SHAP values array

The XGBWrapper wraps preprocessor + model. We extract the inner
XGBClassifier and pre-transformed X_val for SHAP TreeExplainer.

Run:
    python3 src/explainability/shap_analysis.py
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
import shap

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import cfg
from src.eda.plot_style import apply_style, save_fig, FRAUD_COLOR, LEGIT_COLOR, NEUTRAL, BG
from src.imbalance.preprocessor import prepare_xy, build_preprocessor
from src.models.train_models import XGBWrapper, LGBMWrapper  # noqa: needed for joblib

log = get_logger("shap")

SHAP_SAMPLE   = cfg.explainability.shap_sample_size   # 1000
RANDOM_STATE  = cfg.models.random_state


# ─────────────────────────────────────────────────────────────────────────────
# Load model and data
# ─────────────────────────────────────────────────────────────────────────────

def load_best_model_and_data() -> tuple:
    """
    Load best model config, the fitted wrapper, validation data,
    and the preprocessed feature matrix + feature names.

    Returns
    -------
    inner_model   : raw XGBClassifier / LGBMClassifier (for TreeExplainer)
    X_val_proc    : np.ndarray — preprocessed validation features
    feature_names : list[str]
    X_val_raw     : pd.DataFrame — raw validation features (for display)
    y_val         : np.ndarray
    """
    import __main__
    __main__.XGBWrapper  = XGBWrapper
    __main__.LGBMWrapper = LGBMWrapper

    models_dir = PROJECT_ROOT / cfg.paths.outputs_models
    config_path = models_dir / "best_model_config.json"

    with open(config_path) as f:
        config = json.load(f)

    best_name = config["best_model"]
    model_file = config["best_model_file"]
    num_cols   = config["feature_cols"]["numeric"]
    cat_cols   = config["feature_cols"]["categorical"]

    log.info(f"Best model: {best_name}  ({model_file})")

    # Load wrapper
    wrapper = joblib.load(models_dir / model_file)

    # Load full data and reproduce val split
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

    # Pre-process using the wrapper's fitted preprocessor
    pp = wrapper.pp
    X_val_proc = pp.transform(X_val_raw)

    # Get feature names from the preprocessor
    try:
        feature_names = list(pp.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(X_val_proc.shape[1])]

    # Clean feature names (remove pipeline prefix)
    feature_names = [n.replace("numeric__", "").replace("categorical__", "")
                     for n in feature_names]

    # Extract inner model (XGBClassifier or LGBMClassifier)
    inner_model = wrapper.model

    log.info(f"  Val rows: {len(y_val):,}  |  fraud: {y_val.sum():,}")
    log.info(f"  Features: {X_val_proc.shape[1]}")
    log.info(f"  Feature names sample: {feature_names[:5]}")

    return inner_model, X_val_proc, feature_names, X_val_raw, y_val


# ─────────────────────────────────────────────────────────────────────────────
# SHAP computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    inner_model,
    X_val_proc: np.ndarray,
    sample_n: int = SHAP_SAMPLE,
) -> tuple[shap.Explainer, np.ndarray, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer on a stratified sample.

    Returns
    -------
    explainer     : fitted shap.TreeExplainer
    shap_values   : np.ndarray shape (sample_n, n_features)  — fraud class
    X_sample      : np.ndarray shape (sample_n, n_features)
    """
    log.info(f"Computing SHAP values on {sample_n:,} sampled rows …")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_val_proc), size=min(sample_n, len(X_val_proc)),
                     replace=False)
    X_sample = X_val_proc[idx]

    explainer = shap.TreeExplainer(inner_model)

    shap_output = explainer(X_sample)

    # For binary classification, shap_output.values may be 3D (n, features, 2)
    # or 2D (n, features). We always want the fraud class (index 1).
    sv = shap_output.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    log.info(f"  SHAP values shape: {sv.shape}")
    return explainer, sv, X_sample, idx


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — SHAP Beeswarm (global importance)
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    top_n: int = 20,
) -> Path:
    """
    Beeswarm plot of top N features by mean |SHAP|.
    Each dot = one transaction. Colour = feature value (red=high, blue=low).
    """
    log.info(f"Plotting SHAP beeswarm (top {top_n} features) …")
    apply_style()

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    sv_top  = shap_values[:, top_idx]
    X_top   = X_sample[:, top_idx]
    names   = [feature_names[i] if i < len(feature_names)
               else f"f{i}" for i in top_idx]

    # Sort ascending for bottom-to-top display
    order     = np.argsort(mean_abs[top_idx])
    sv_plot   = sv_top[:, order]
    X_plot    = X_top[:, order]
    name_plot = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 9))

    for j in range(sv_plot.shape[1]):
        sv_col = sv_plot[:, j]
        x_col  = X_plot[:, j]

        # Normalise feature values to [0,1] for colouring
        x_min, x_max = np.nanpercentile(x_col, [5, 95])
        x_norm = np.clip((x_col - x_min) / (x_max - x_min + 1e-9), 0, 1)

        # Jitter y position
        rng   = np.random.default_rng(j)
        jitter = rng.uniform(-0.35, 0.35, len(sv_col))

        scatter = ax.scatter(
            sv_col, j + jitter,
            c=x_norm, cmap="RdBu_r", vmin=0, vmax=1,
            alpha=0.55, s=10, linewidths=0,
        )

    ax.set_yticks(range(len(name_plot)))
    ax.set_yticklabels(name_plot, fontsize=9)
    ax.axvline(0, color=NEUTRAL, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("SHAP Value  (positive = pushes toward Fraud)", fontsize=11)
    ax.set_title(
        f"SHAP Global Feature Importance — XGBoost\n"
        f"Top {top_n} features · {sv_plot.shape[0]:,} transactions sampled",
        fontsize=12,
    )

    # Colourbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("Feature value\n(Blue=low · Red=high)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    out = out_dir / "17_shap_beeswarm.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — SHAP Waterfall for one confirmed fraud
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_waterfall_fraud(
    explainer,
    inner_model,
    X_val_proc: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
) -> tuple[Path, int]:
    """
    SHAP waterfall plot for one transaction the model correctly flagged as fraud.
    Uses the transaction with the highest fraud probability among true positives.
    """
    log.info("Plotting SHAP waterfall for flagged fraud transaction …")

    # Find true positives — correctly caught fraud
    with open(PROJECT_ROOT / cfg.paths.outputs_models / "best_model_config.json") as f:
        config = json.load(f)
    threshold = config["f2_threshold"]

    # Predict directly on preprocessed data using the inner XGB/LGBM model
    y_prob_raw = inner_model.predict_proba(X_val_proc)[:, 1]
    y_pred = (y_prob_raw >= threshold).astype(int)

    # True positives with highest confidence
    tp_mask = (y_val == 1) & (y_pred == 1)
    if tp_mask.sum() == 0:
        log.warning("  No true positives found — using highest fraud probability")
        tp_mask = y_val == 1

    tp_probs = y_prob_raw.copy()
    tp_probs[~tp_mask] = -1
    tx_idx = int(np.argmax(tp_probs))

    log.info(
        f"  Selected transaction idx={tx_idx} | "
        f"fraud_prob={y_prob_raw[tx_idx]:.4f} | "
        f"true_label={y_val[tx_idx]}"
    )

    # Compute SHAP for this single transaction
    shap_exp = explainer(X_val_proc[tx_idx:tx_idx+1])
    sv = shap_exp.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    base_val = float(explainer.expected_value)
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1])

    sv_1d   = sv[0]
    top_n   = 15
    top_idx = np.argsort(np.abs(sv_1d))[::-1][:top_n]
    top_sv  = sv_1d[top_idx]
    top_names = [feature_names[i] if i < len(feature_names)
                 else f"f{i}" for i in top_idx]
    top_vals = X_val_proc[tx_idx, top_idx]

    # Build waterfall manually
    fig, ax = plt.subplots(figsize=(11, 8))

    # Sort by SHAP value for clearer display
    sort_order = np.argsort(top_sv)
    top_sv    = top_sv[sort_order]
    top_names = [top_names[i] for i in sort_order]
    top_vals  = top_vals[sort_order]

    colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in top_sv]
    y_pos  = np.arange(len(top_sv))

    bars = ax.barh(y_pos, top_sv, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.7)

    # Label with feature value
    for i, (bar, val, fname, fval) in enumerate(
            zip(bars, top_sv, top_names, top_vals)):
        sign = "+" if val > 0 else ""
        ax.text(
            val + (0.005 if val > 0 else -0.005),
            i,
            f"{sign}{val:.3f}  (={fval:.2f})",
            va="center",
            ha="left" if val > 0 else "right",
            fontsize=8,
            color=NEUTRAL,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.axvline(0, color=NEUTRAL, linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("SHAP Value  (positive = toward Fraud prediction)")
    ax.set_title(
        f"SHAP Waterfall — Why We Flagged This Transaction\n"
        f"Fraud probability: {y_prob_raw[tx_idx]:.1%}  |  "
        f"Base rate: {base_val:.3f}  |  "
        f"True label: {'FRAUD ✓' if y_val[tx_idx]==1 else 'LEGIT'}",
        fontsize=11,
    )

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=FRAUD_COLOR, label="Pushes toward FRAUD"),
            Patch(facecolor=LEGIT_COLOR, label="Pushes toward LEGIT"),
        ],
        loc="lower right", fontsize=9,
    )

    out = out_dir / "18_shap_waterfall_fraud.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out, tx_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_shap() -> dict:
    """Full SHAP pipeline. Returns dict of output paths."""
    apply_style()
    out_dir = PROJECT_ROOT / cfg.paths.outputs_figures
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("fraud-shield-ml | Phase 6a: SHAP Analysis")
    log.info("=" * 60)

    inner_model, X_val_proc, feature_names, X_val_raw, y_val = \
        load_best_model_and_data()

    # Load wrapper for predict_proba
    import __main__
    __main__.XGBWrapper  = XGBWrapper
    __main__.LGBMWrapper = LGBMWrapper
    with open(PROJECT_ROOT / cfg.paths.outputs_models / "best_model_config.json") as f:
        cfg_data = json.load(f)
    wrapper = joblib.load(
        PROJECT_ROOT / cfg.paths.outputs_models / cfg_data["best_model_file"]
    )

    explainer, shap_values, X_sample, sample_idx = compute_shap_values(
        inner_model, X_val_proc
    )

    # Save SHAP values for potential reuse
    shap_path = PROJECT_ROOT / cfg.paths.outputs_models / "shap_values.npy"
    np.save(shap_path, shap_values)
    log.info(f"  SHAP values cached → {shap_path.name}")

    outputs = {}
    outputs["beeswarm"] = plot_shap_beeswarm(
        shap_values, X_sample, feature_names, out_dir
    )
    outputs["waterfall_fraud"], fraud_tx_idx = plot_shap_waterfall_fraud(
        explainer, inner_model, X_val_proc, y_val, feature_names, out_dir
    )

    log.info("=" * 60)
    log.info("SHAP analysis complete.")
    for k, v in outputs.items():
        log.info(f"  {Path(v).name}")
    log.info("  Next: python3 src/explainability/lime_analysis.py")
    log.info("=" * 60)

    return outputs


if __name__ == "__main__":
    run_shap()
