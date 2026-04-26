"""
fraud-shield-ml | tests/test_phase2_ingestion.py
─────────────────────────────────────────────────
Phase 2 test suite: validates the data ingestion pipeline.

Two test modes:
  - UNIT tests  : run always, use synthetic mini-dataframes (no CSVs needed)
  - INTEGRATION : run only when data/raw/ CSVs are present (skipped on CI)

Run:
    pytest tests/test_phase2_ingestion.py -v
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import cfg

# ── Helpers ───────────────────────────────────────────────────────────────────

RAW_DIR       = PROJECT_ROOT / cfg.paths.data_raw
PROCESSED_DIR = PROJECT_ROOT / cfg.paths.data_processed
TX_PATH       = RAW_DIR / cfg.dataset.transaction_file
ID_PATH       = RAW_DIR / cfg.dataset.identity_file
PARQUET_PATH  = PROCESSED_DIR / "merged.parquet"
META_PATH     = PROCESSED_DIR / "merged_meta.json"

DATA_AVAILABLE = TX_PATH.exists() and ID_PATH.exists()
PROCESSED_AVAILABLE = PARQUET_PATH.exists()

skip_no_raw       = pytest.mark.skipif(not DATA_AVAILABLE,      reason="Raw CSVs not present")
skip_no_processed = pytest.mark.skipif(not PROCESSED_AVAILABLE, reason="Processed parquet not present — run load_data.py first")


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS — synthetic data, always run
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeLogic:
    """Validate merge behaviour on synthetic mini-frames."""

    def _make_tx(self, n=1000, fraud_rate=0.035):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "TransactionID": np.arange(1, n + 1),
            "isFraud": rng.choice([0, 1], size=n, p=[1 - fraud_rate, fraud_rate]),
            "TransactionDT": rng.integers(0, 15_897_600, size=n),
            "TransactionAmt": rng.exponential(scale=100, size=n).round(2),
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], size=n),
        })

    def _make_identity(self, tx_ids, coverage=0.24):
        rng = np.random.default_rng(42)
        n_id = int(len(tx_ids) * coverage)
        chosen = rng.choice(tx_ids, size=n_id, replace=False)
        return pd.DataFrame({
            "TransactionID": chosen,
            "DeviceType": rng.choice(["desktop", "mobile", None], size=n_id),
            "DeviceInfo": rng.choice(["Windows", "iOS", "Android", None], size=n_id),
        })

    def test_left_join_preserves_all_tx_rows(self):
        from src.ingestion.load_data import merge_tables
        tx = self._make_tx(1000)
        identity = self._make_identity(tx["TransactionID"].values)
        merged = merge_tables(tx, identity)
        assert len(merged) == len(tx), "Left join must not drop transaction rows"

    def test_merge_adds_identity_columns(self):
        from src.ingestion.load_data import merge_tables
        tx = self._make_tx(500)
        identity = self._make_identity(tx["TransactionID"].values)
        merged = merge_tables(tx, identity)
        assert "DeviceType" in merged.columns
        assert "DeviceInfo" in merged.columns

    def test_no_duplicate_transaction_ids_after_merge(self):
        from src.ingestion.load_data import merge_tables
        tx = self._make_tx(500)
        identity = self._make_identity(tx["TransactionID"].values)
        merged = merge_tables(tx, identity)
        assert merged["TransactionID"].nunique() == len(merged)

    def test_unmatched_transactions_have_nan_identity(self):
        from src.ingestion.load_data import merge_tables
        tx = self._make_tx(500)
        identity = self._make_identity(tx["TransactionID"].values, coverage=0.20)
        merged = merge_tables(tx, identity)
        n_matched   = identity["TransactionID"].nunique()
        n_unmatched = len(tx) - n_matched
        nan_count   = merged["DeviceType"].isna().sum()
        assert nan_count >= n_unmatched * 0.9, \
            "Unmatched transactions should have NaN identity fields"


class TestTimestampConversion:
    """Validate TransactionDT → datetime feature engineering."""

    def _make_df_with_dt(self):
        return pd.DataFrame({
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionDT": [
                0,           # reference date itself
                86400,       # +1 day
                86400 * 30,  # +30 days
            ],
            "TransactionAmt": [100.0, 50.0, 200.0],
        })

    def test_transaction_datetime_column_created(self):
        from src.ingestion.load_data import convert_timestamps
        df = self._make_df_with_dt()
        result = convert_timestamps(df)
        assert "TransactionDateTime" in result.columns

    def test_datetime_offset_is_correct(self):
        from src.ingestion.load_data import convert_timestamps, REFERENCE_DATE
        df = self._make_df_with_dt()
        result = convert_timestamps(df)
        expected_day1 = REFERENCE_DATE + pd.Timedelta(days=1)
        assert result.loc[1, "TransactionDateTime"].date() == expected_day1.date()

    def test_hour_range(self):
        from src.ingestion.load_data import convert_timestamps
        df = self._make_df_with_dt()
        result = convert_timestamps(df)
        assert result["tx_hour"].between(0, 23).all()

    def test_day_of_week_range(self):
        from src.ingestion.load_data import convert_timestamps
        df = self._make_df_with_dt()
        result = convert_timestamps(df)
        assert result["tx_day_of_week"].between(0, 6).all()

    def test_temporal_features_all_present(self):
        from src.ingestion.load_data import convert_timestamps
        df = self._make_df_with_dt()
        result = convert_timestamps(df)
        for col in ["tx_hour", "tx_day_of_week", "tx_day_of_month", "tx_month"]:
            assert col in result.columns, f"Missing temporal feature: {col}"


class TestValidation:
    """Validate the raw-data integrity checker."""

    def _make_valid_tx(self, n=600_000):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "TransactionID": np.arange(1, n + 1),
            "isFraud": rng.choice([0, 1], size=n, p=[0.965, 0.035]),
            "TransactionDT": rng.integers(0, 15_000_000, size=n),
            **{f"V{i}": rng.random(n) for i in range(1, 350)},
        })

    def _make_valid_id(self, n=144_000):
        rng = np.random.default_rng(1)
        return pd.DataFrame({
            "TransactionID": np.arange(1, n + 1),
            "DeviceType": rng.choice(["desktop", "mobile"], size=n),
        })

    def test_valid_data_passes_validation(self):
        from src.ingestion.load_data import validate_raw
        tx = self._make_valid_tx()
        identity = self._make_valid_id()
        validate_raw(tx, identity)  # should not raise

    def test_missing_target_raises(self):
        from src.ingestion.load_data import validate_raw
        tx = self._make_valid_tx().drop(columns=["isFraud"])
        identity = self._make_valid_id()
        with pytest.raises(AssertionError, match="Target column"):
            validate_raw(tx, identity)

    def test_wrong_fraud_rate_raises(self):
        from src.ingestion.load_data import validate_raw
        tx = self._make_valid_tx()
        tx["isFraud"] = 0  # 0% fraud — invalid
        identity = self._make_valid_id()
        with pytest.raises(AssertionError, match="fraud rate"):
            validate_raw(tx, identity)


class TestMissingValueAudit:
    """Validate the high-missing-column drop logic."""

    def test_high_missing_columns_dropped(self):
        from src.ingestion.load_data import audit_merged
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame({
            "TransactionID": np.arange(n),
            "isFraud": rng.integers(0, 2, n),
            "good_col": rng.random(n),
            "bad_col": [np.nan] * n,           # 100% missing → should be dropped
            "borderline": [np.nan if i < 510 else 1.0 for i in range(n)],  # 51% → dropped
        })
        result = audit_merged(df)
        assert "bad_col" not in result.columns
        assert "borderline" not in result.columns
        assert "good_col" in result.columns

    def test_low_missing_columns_kept(self):
        from src.ingestion.load_data import audit_merged
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame({
            "TransactionID": np.arange(n),
            "isFraud": rng.integers(0, 2, n),
            "sparse_col": [np.nan if i < 400 else 1.0 for i in range(n)],  # 40% → kept
        })
        result = audit_merged(df)
        assert "sparse_col" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS — require actual data files
# ─────────────────────────────────────────────────────────────────────────────

class TestRawFilesPresent:
    @skip_no_raw
    def test_transaction_csv_exists(self):
        assert TX_PATH.exists()

    @skip_no_raw
    def test_identity_csv_exists(self):
        assert ID_PATH.exists()

    @skip_no_raw
    def test_transaction_csv_is_large(self):
        size_mb = TX_PATH.stat().st_size / 1e6
        assert size_mb > 400, f"train_transaction.csv too small: {size_mb:.0f} MB"

    @skip_no_raw
    def test_identity_csv_exists_and_nonzero(self):
        size_mb = ID_PATH.stat().st_size / 1e6
        assert size_mb > 10, f"train_identity.csv too small: {size_mb:.0f} MB"


class TestProcessedOutput:
    @skip_no_processed
    def test_parquet_file_exists(self):
        assert PARQUET_PATH.exists()

    @skip_no_processed
    def test_meta_json_exists(self):
        assert META_PATH.exists()

    @skip_no_processed
    def test_parquet_loads_correctly(self):
        df = pd.read_parquet(PARQUET_PATH)
        assert len(df) > 500_000
        assert "isFraud" in df.columns
        assert "TransactionID" in df.columns

    @skip_no_processed
    def test_fraud_rate_realistic(self):
        df = pd.read_parquet(PARQUET_PATH)
        rate = df["isFraud"].mean()
        assert 0.02 < rate < 0.06, f"Unexpected fraud rate: {rate:.4f}"

    @skip_no_processed
    def test_temporal_features_present(self):
        df = pd.read_parquet(PARQUET_PATH)
        for col in ["tx_hour", "tx_day_of_week", "tx_day_of_month", "tx_month"]:
            assert col in df.columns, f"Missing: {col}"

    @skip_no_processed
    def test_no_duplicate_transaction_ids(self):
        df = pd.read_parquet(PARQUET_PATH)
        assert df["TransactionID"].nunique() == len(df)

    @skip_no_processed
    def test_meta_json_has_required_keys(self):
        with open(META_PATH) as f:
            meta = json.load(f)
        for key in ["rows", "cols", "fraud_rate", "fraud_count", "legit_count",
                    "date_min", "date_max", "columns"]:
            assert key in meta, f"Missing key in metadata: {key}"

    @skip_no_processed
    def test_meta_row_count_matches_parquet(self):
        df = pd.read_parquet(PARQUET_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
        assert meta["rows"] == len(df)

    @skip_no_processed
    def test_load_processed_function_works(self):
        from src.ingestion.load_data import load_processed
        df = load_processed()
        assert df is not None
        assert len(df) > 0
