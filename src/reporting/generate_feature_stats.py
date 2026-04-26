"""
fraud-shield-ml | src/reporting/generate_feature_stats.py
───────────────────────────────────────────────────────────
Precomputes feature medians (numeric) and modes (categorical)
from the validation set. These are used by the Streamlit live
prediction tab to fill unspecified V-features.

Output:
  outputs/models/feature_stats.json

Run:
    python3 src/reporting/generate_feature_stats.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import cfg
from src.imbalance.preprocessor import prepare_xy

log = get_logger("feature-stats")


def generate_feature_stats() -> Path:
    log.info("Generating feature statistics for Streamlit live prediction …")

    from src.ingestion.load_data import load_processed
    from sklearn.model_selection import train_test_split

    df = load_processed()
    X, y, num_cols, cat_cols = prepare_xy(df, scale_numeric=False)

    _, X_val, _, _ = train_test_split(
        X, y,
        test_size=cfg.imbalance.test_size,
        stratify=y,
        random_state=cfg.models.random_state,
    )

    log.info(f"  Val set: {len(X_val):,} rows")

    medians = {}
    for col in num_cols:
        val = X_val[col].median()
        medians[col] = float(val) if not np.isnan(val) else 0.0

    modes = {}
    for col in cat_cols:
        mode_val = X_val[col].mode()
        modes[col] = str(mode_val.iloc[0]) if len(mode_val) > 0 else ""

    stats = {
        "medians":      medians,
        "modes":        modes,
        "numeric_cols": num_cols,
        "cat_cols":     cat_cols,
        "n_val_rows":   len(X_val),
    }

    out_path = PROJECT_ROOT / cfg.paths.outputs_models / "feature_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}")
    log.info(f"  Numeric features: {len(num_cols)}")
    log.info(f"  Categorical features: {len(cat_cols)}")
    return out_path


if __name__ == "__main__":
    generate_feature_stats()