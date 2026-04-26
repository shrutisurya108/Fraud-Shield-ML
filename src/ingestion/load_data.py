"""
fraud-shield-ml | src/ingestion/load_data.py
─────────────────────────────────────────────
Phase 2: Data Ingestion Pipeline

Responsibilities:
  1. Load train_transaction.csv and train_identity.csv from data/raw/
  2. Validate raw file integrity (shape, columns, target presence)
  3. Merge on TransactionID via left join
  4. Convert TransactionDT (seconds offset) → real datetime
  5. Audit dtypes, memory usage, missing value rates
  6. Save merged dataset to data/processed/merged.parquet

Run:
    python3 src/ingestion/load_data.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("ingestion")

# ── Reference date for TransactionDT conversion ───────────────────────────────
# Vesta confirmed the dataset starts around 2017-11-30
REFERENCE_DATE = pd.Timestamp(cfg.dataset.reference_date)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_transactions(path: Path) -> pd.DataFrame:
    """Load train_transaction.csv with appropriate dtypes."""
    log.info(f"Loading transactions from: {path.name}")
    t0 = time.time()

    df = pd.read_csv(path)

    elapsed = time.time() - t0
    log.info(
        f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} cols "
        f"in {elapsed:.1f}s"
    )
    log.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def load_identity(path: Path) -> pd.DataFrame:
    """Load train_identity.csv."""
    log.info(f"Loading identity from: {path.name}")
    t0 = time.time()

    df = pd.read_csv(path)

    elapsed = time.time() - t0
    log.info(
        f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} cols "
        f"in {elapsed:.1f}s"
    )
    log.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_raw(tx: pd.DataFrame, identity: pd.DataFrame) -> None:
    """Assert minimum integrity requirements on raw files."""
    log.info("Validating raw data integrity …")

    # Transaction table
    assert cfg.dataset.target_column in tx.columns, \
        f"Target column '{cfg.dataset.target_column}' missing from transactions"
    assert cfg.dataset.transaction_id in tx.columns, \
        f"'{cfg.dataset.transaction_id}' missing from transactions"
    assert cfg.dataset.timestamp_col in tx.columns, \
        f"Timestamp column '{cfg.dataset.timestamp_col}' missing"
    assert tx[cfg.dataset.transaction_id].nunique() == len(tx), \
        "Duplicate TransactionIDs found in transaction table"
    assert tx.shape[0] > 500_000, \
        f"Transaction table too small: {tx.shape[0]} rows (expected ~590K)"
    assert tx.shape[1] > 300, \
        f"Transaction table too few columns: {tx.shape[1]} (expected ~394)"

    # Identity table
    assert cfg.dataset.transaction_id in identity.columns, \
        f"'{cfg.dataset.transaction_id}' missing from identity"
    assert identity.shape[0] > 100_000, \
        f"Identity table too small: {identity.shape[0]} rows (expected ~144K)"

    # Target distribution
    fraud_rate = tx[cfg.dataset.target_column].mean()
    assert 0.02 < fraud_rate < 0.06, \
        f"Unexpected fraud rate: {fraud_rate:.4f} (expected ~0.035)"

    log.info(f"  ✓ Transaction table: {tx.shape[0]:,} × {tx.shape[1]:,}")
    log.info(f"  ✓ Identity table:    {identity.shape[0]:,} × {identity.shape[1]:,}")
    log.info(f"  ✓ Fraud rate:        {fraud_rate:.4%}")
    log.info(f"  ✓ Unique TxIDs (tx): {tx[cfg.dataset.transaction_id].nunique():,}")
    log.info("  ✓ Raw validation passed")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_tables(tx: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join transactions with identity on TransactionID.

    Left join is deliberate:
    - 144,233 of 590,540 transactions have identity data (~24%)
    - Missingness in identity columns is itself a behavioural signal
    - We do not discard transactions without identity records
    """
    log.info("Merging transaction + identity tables (left join on TransactionID) …")

    n_before = len(tx)
    merged = tx.merge(identity, on=cfg.dataset.transaction_id, how="left")
    n_after = len(merged)

    assert n_after == n_before, \
        f"Row count changed after merge: {n_before} → {n_after}"

    identity_coverage = identity[cfg.dataset.transaction_id].nunique()
    log.info(
        f"  Merged shape:      {merged.shape[0]:,} × {merged.shape[1]:,}"
    )
    log.info(
        f"  Identity coverage: {identity_coverage:,} / {n_before:,} "
        f"transactions ({identity_coverage/n_before:.1%})"
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature engineering — timestamp conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TransactionDT (seconds since reference date) to datetime features.

    TransactionDT is an offset in seconds from a reference date.
    We derive:
      - TransactionDateTime : actual timestamp (for display/EDA)
      - tx_hour             : 0–23  (time-of-day fraud pattern)
      - tx_day_of_week      : 0–6   (weekday vs weekend pattern)
      - tx_day_of_month     : 1–31
      - tx_month            : 1–6   (dataset spans ~6 months)
    """
    log.info("Converting TransactionDT to datetime features …")

    df = df.copy()
    dt_series = REFERENCE_DATE + pd.to_timedelta(df[cfg.dataset.timestamp_col], unit="s")

    df["TransactionDateTime"] = dt_series
    df["tx_hour"]             = dt_series.dt.hour.astype(np.int8)
    df["tx_day_of_week"]      = dt_series.dt.dayofweek.astype(np.int8)
    df["tx_day_of_month"]     = dt_series.dt.day.astype(np.int8)
    df["tx_month"]            = dt_series.dt.month.astype(np.int8)

    date_min = dt_series.min().date()
    date_max = dt_series.max().date()
    log.info(f"  Date range: {date_min} → {date_max}")
    log.info("  ✓ Temporal features added: tx_hour, tx_day_of_week, tx_day_of_month, tx_month")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_merged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a missing value audit and dtype summary.
    Drops columns where missing rate exceeds config threshold.
    Returns the cleaned dataframe.
    """
    log.info("Running missing value audit …")

    total = len(df)
    missing = df.isnull().sum()
    missing_pct = (missing / total * 100).round(2)

    # Columns above threshold
    threshold = cfg.eda.missing_threshold  # default 0.50
    high_missing = missing_pct[missing_pct > threshold * 100]

    log.info(f"  Total columns:         {df.shape[1]:,}")
    log.info(f"  Columns with any NaN:  {(missing > 0).sum():,}")
    log.info(f"  Columns >{threshold*100:.0f}% missing: {len(high_missing):,}")

    if len(high_missing) > 0:
        log.info(f"  Dropping {len(high_missing)} high-missing columns …")
        df = df.drop(columns=high_missing.index.tolist())
        log.info(f"  Shape after drop: {df.shape[0]:,} × {df.shape[1]:,}")

    # Memory after cleanup
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    log.info(f"  Memory (merged+cleaned): {mem_mb:.1f} MB")

    # Dtype breakdown
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        log.info(f"  dtype {str(dtype):<10}: {count:,} columns")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, output_dir: Path) -> Path:
    """Save merged dataframe as parquet for fast downstream loading."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "merged.parquet"

    # Parquet cannot serialize datetime columns with mixed tz — drop display col
    save_df = df.drop(columns=["TransactionDateTime"], errors="ignore")
    save_df.to_parquet(out_path, index=False, engine="pyarrow")

    size_mb = out_path.stat().st_size / 1e6
    log.info(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")

    # Also save a small metadata file
    meta = {
        "rows": len(df),
        "cols": df.shape[1],
        "fraud_rate": float(df[cfg.dataset.target_column].mean()),
        "fraud_count": int(df[cfg.dataset.target_column].sum()),
        "legit_count": int((df[cfg.dataset.target_column] == 0).sum()),
        "date_min": str(df["TransactionDateTime"].min().date()),
        "date_max": str(df["TransactionDateTime"].max().date()),
        "columns": df.columns.tolist(),
    }

    import json
    meta_path = output_dir / "merged_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Metadata → {meta_path.relative_to(PROJECT_ROOT)}")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. Public API — used by EDA and all downstream modules
# ─────────────────────────────────────────────────────────────────────────────

def load_processed() -> pd.DataFrame:
    """
    Load the processed parquet file.
    Called by EDA, imbalance, model training, and XAI modules.
    """
    parquet_path = PROJECT_ROOT / cfg.paths.data_processed / "merged.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {parquet_path}\n"
            f"Run: python3 src/ingestion/load_data.py"
        )
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("fraud-shield-ml | Phase 2: Data Ingestion")
    log.info("=" * 60)

    raw_dir       = PROJECT_ROOT / cfg.paths.data_raw
    processed_dir = PROJECT_ROOT / cfg.paths.data_processed

    tx_path  = raw_dir / cfg.dataset.transaction_file
    id_path  = raw_dir / cfg.dataset.identity_file

    # ── Check raw files exist ─────────────────────────────────────────────
    for p in [tx_path, id_path]:
        if not p.exists():
            log.error(
                f"Missing file: {p}\n"
                f"Place raw CSVs in data/raw/ — see data/raw/README.md"
            )
            sys.exit(1)

    # ── Pipeline ─────────────────────────────────────────────────────────
    tx       = load_transactions(tx_path)
    identity = load_identity(id_path)

    validate_raw(tx, identity)

    merged   = merge_tables(tx, identity)
    merged   = convert_timestamps(merged)
    merged   = audit_merged(merged)

    out_path = save_processed(merged, processed_dir)

    log.info("=" * 60)
    log.info("Phase 2 ingestion complete.")
    log.info(f"  Output: {out_path.relative_to(PROJECT_ROOT)}")
    log.info(f"  Shape:  {merged.shape[0]:,} rows × {merged.shape[1]:,} cols")
    log.info(f"  Fraud:  {merged[cfg.dataset.target_column].sum():,} "
             f"({merged[cfg.dataset.target_column].mean():.2%})")
    log.info("  Next:   python3 src/eda/run_eda.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
