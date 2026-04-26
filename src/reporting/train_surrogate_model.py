"""
fraud-shield-ml | src/reporting/train_surrogate_model.py
──────────────────────────────────────────────────────────
Trains a lightweight surrogate XGBoost model for Streamlit Cloud deployment.

The full XGBoost model (594 estimators, depth 8) produces a ~50MB joblib
that cannot be committed to GitHub or loaded on Streamlit Cloud's free tier.

This surrogate uses:
  - 80K stratified training rows (subset of full 472K)
  - 150 estimators, max_depth=5
  - Same feature engineering and preprocessor as the full model
  - Produces a ~3-5MB joblib suitable for git + Streamlit Cloud

Performance: PR-AUC ~0.75-0.78 (vs full model's 0.824)
This is clearly documented in the dashboard UI.

Output:
  outputs/models/surrogate_model.joblib   (~3-5 MB — committed to git)
  outputs/models/surrogate_config.json    (threshold, metrics, metadata)

Run:
    python3 src/reporting/train_surrogate_model.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import cfg
from src.imbalance.preprocessor import prepare_xy, build_preprocessor
from src.models.evaluate import compute_all_metrics, find_best_f2_threshold
from src.models.train_models import XGBWrapper

log = get_logger("surrogate")


def train_surrogate() -> Path:
    log.info("=" * 60)
    log.info("Training surrogate model for Streamlit Cloud deployment")
    log.info("=" * 60)

    from src.ingestion.load_data import load_processed
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    models_dir = PROJECT_ROOT / cfg.paths.outputs_models
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and subsample ────────────────────────────────────────────────────
    log.info("Loading processed data …")
    df = load_processed()

    X, y, num_cols, cat_cols = prepare_xy(df, scale_numeric=False)

    # Stratified subsample — 80K rows for training, 20% holdout
    SURROGATE_TRAIN_N = 80_000
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=cfg.models.random_state,
    )

    # Further subsample training set for speed + small model size
    if len(X_tr) > SURROGATE_TRAIN_N:
        _, X_tr, _, y_tr = train_test_split(
            X_tr, y_tr,
            test_size=SURROGATE_TRAIN_N / len(X_tr),
            stratify=y_tr,
            random_state=cfg.models.random_state,
        )

    log.info(f"  Surrogate train: {len(X_tr):,} rows | "
             f"fraud: {y_tr.sum():,} ({y_tr.mean():.2%})")
    log.info(f"  Validation:      {len(X_val):,} rows | "
             f"fraud: {y_val.sum():,} ({y_val.mean():.2%})")
    log.info(f"  Features: {len(num_cols)} numeric + {len(cat_cols)} categorical")

    # ── Preprocess ────────────────────────────────────────────────────────────
    pp = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
    X_tr_proc  = pp.fit_transform(X_tr)
    X_val_proc = pp.transform(X_val)

    # ── Train lightweight XGBoost ─────────────────────────────────────────────
    neg = int((y_tr.values == 0).sum())
    pos = int((y_tr.values == 1).sum())
    spw = neg / pos
    log.info(f"  scale_pos_weight = {spw:.1f}")

    t0 = time.time()
    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.18,
        subsample=0.75,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_alpha=0.0003,
        reg_lambda=0.04,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=cfg.models.random_state,
        use_label_encoder=False,
    )
    model.fit(X_tr_proc, y_tr.values, verbose=False)
    log.info(f"  Trained in {time.time()-t0:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_prob  = model.predict_proba(X_val_proc)[:, 1]
    best_t, best_f2 = find_best_f2_threshold(y_val.values, y_prob)
    metrics = compute_all_metrics(y_val.values, y_prob, best_t, "SurrogateXGBoost")

    log.info(
        f"  PR-AUC={metrics['pr_auc']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"F2={metrics['f2_score']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f}"
    )

    # ── Save wrapper ──────────────────────────────────────────────────────────
    wrapper = XGBWrapper(pp, model)
    out_joblib = models_dir / "surrogate_model.joblib"
    joblib.dump(wrapper, out_joblib)

    size_mb = out_joblib.stat().st_size / 1e6
    log.info(f"  Saved surrogate_model.joblib  ({size_mb:.1f} MB)")

    if size_mb > 95:
        log.warning(
            f"  Surrogate is {size_mb:.0f} MB — still too large for GitHub (100MB limit). "
            "Consider reducing n_estimators further."
        )

    # ── Save config ───────────────────────────────────────────────────────────
    surrogate_config = {
        "model_name":       "SurrogateXGBoost",
        "model_file":       "surrogate_model.joblib",
        "f2_threshold":     round(best_t, 4),
        "hard_block_threshold": 0.90,
        "metrics":          metrics,
        "feature_cols": {
            "numeric":      num_cols,
            "categorical":  cat_cols,
        },
        "training_rows":    len(X_tr),
        "note": (
            "Lightweight surrogate for Streamlit Cloud deployment. "
            "Full model (XGBoost, 594 estimators) achieves PR-AUC=0.824. "
            "Surrogate trades ~5pp PR-AUC for a 10x smaller file size."
        ),
    }

    config_path = models_dir / "surrogate_config.json"
    with open(config_path, "w") as f:
        json.dump(surrogate_config, f, indent=2)
    log.info(f"  Saved surrogate_config.json")

    log.info("=" * 60)
    log.info(f"Surrogate model ready for deployment.")
    log.info(f"  File size:  {size_mb:.1f} MB")
    log.info(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    log.info(f"  Threshold:  {best_t:.4f}")
    log.info("  Next: git add outputs/models/surrogate_model.joblib")
    log.info("               outputs/models/surrogate_config.json")
    log.info("        git commit -m 'feat: add surrogate model for Streamlit Cloud'")
    log.info("=" * 60)

    return out_joblib


if __name__ == "__main__":
    train_surrogate()
