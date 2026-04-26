"""
fraud-shield-ml | tests/test_phase3_eda.py
───────────────────────────────────────────
Phase 3 test suite: EDA pipeline.

Unit tests use a synthetic 5K-row dataframe — no real data needed.
Integration tests check figure files exist after run_eda.py completes.

Run:
    pytest tests/test_phase3_eda.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import cfg

FIGURES_DIR = PROJECT_ROOT / cfg.paths.outputs_figures

EXPECTED_FIGURES = [
    "01_class_imbalance.png",
    "02_amount_distribution.png",
    "03_fraud_by_hour.png",
    "04_fraud_by_dow.png",
    "05_fraud_over_time.png",
    "06_missing_values.png",
    "07_correlation_heatmap.png",
    "08_categorical_fraud_rates.png",
]

FIGURES_AVAILABLE = all((FIGURES_DIR / f).exists() for f in EXPECTED_FIGURES)
skip_no_figures = pytest.mark.skipif(
    not FIGURES_AVAILABLE,
    reason="EDA figures not yet generated — run python3 src/eda/run_eda.py first"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_df():
    """5K-row synthetic dataframe that mimics the IEEE-CIS merged schema."""
    rng = np.random.default_rng(42)
    n = 5_000
    fraud_mask = rng.random(n) < 0.035

    df = pd.DataFrame({
        "TransactionID":   np.arange(1, n + 1),
        "isFraud":         fraud_mask.astype(int),
        "TransactionDT":   rng.integers(0, 15_897_600, n),
        "TransactionAmt":  np.abs(rng.normal(loc=120, scale=200, size=n)).clip(0.5),
        "ProductCD":       rng.choice(["W", "H", "C", "S", "R"], n),
        "card4":           rng.choice(["visa", "mastercard", "discover", "amex"], n),
        "card6":           rng.choice(["debit", "credit"], n),
        "P_emaildomain":   rng.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
             "aol.com", "icloud.com", "live.com", "protonmail.com",
             "comcast.net", "msn.com"],
            n
        ),
        "tx_hour":         rng.integers(0, 24, n).astype(np.int8),
        "tx_day_of_week":  rng.integers(0, 7, n).astype(np.int8),
        "tx_day_of_month": rng.integers(1, 32, n).astype(np.int8),
        "tx_month":        rng.integers(1, 7, n).astype(np.int8),
        **{f"V{i}": rng.random(n) * np.where(rng.random(n) < 0.3, np.nan, 1)
           for i in range(1, 21)},
        **{f"C{i}": rng.integers(0, 10, n).astype(float)
           for i in range(1, 6)},
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — plot_style
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotStyle:
    def test_apply_style_runs_without_error(self):
        from src.eda.plot_style import apply_style
        apply_style()  # should not raise

    def test_colour_constants_are_hex(self):
        from src.eda.plot_style import FRAUD_COLOR, LEGIT_COLOR, ACCENT
        for c in [FRAUD_COLOR, LEGIT_COLOR, ACCENT]:
            assert c.startswith("#"), f"Expected hex colour, got: {c}"
            assert len(c) == 7,      f"Expected 7-char hex, got: {c}"

    def test_save_fig_creates_file(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.eda.plot_style import save_fig, apply_style
        apply_style()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        out = tmp_path / "test_fig.png"
        save_fig(fig, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotClassImbalance:
    def test_creates_file(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_class_imbalance
        out = plot_class_imbalance(synthetic_df, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_correct_filename(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_class_imbalance
        out = plot_class_imbalance(synthetic_df, tmp_path)
        assert out.name == "01_class_imbalance.png"


class TestPlotAmountDistribution:
    def test_creates_file(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_amount_distribution
        out = plot_amount_distribution(synthetic_df, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_correct_filename(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_amount_distribution
        out = plot_amount_distribution(synthetic_df, tmp_path)
        assert out.name == "02_amount_distribution.png"


class TestPlotFraudByHour:
    def test_creates_file(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_fraud_by_hour
        out = plot_fraud_by_hour(synthetic_df, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_covers_all_24_hours(self, synthetic_df):
        assert synthetic_df["tx_hour"].nunique() == 24


class TestPlotFraudByDow:
    def test_creates_file(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_fraud_by_dow
        out = plot_fraud_by_dow(synthetic_df, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_covers_all_7_days(self, synthetic_df):
        assert synthetic_df["tx_day_of_week"].nunique() == 7


class TestPlotFraudOverTime:
    def test_creates_file(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_fraud_over_time
        out = plot_fraud_over_time(synthetic_df, tmp_path)
        assert out.exists() and out.stat().st_size > 0


class TestPlotMissingValues:
    def test_creates_file_with_missing_data(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_missing_values
        out = plot_missing_values(synthetic_df, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_creates_file_with_no_missing_data(self, tmp_path):
        from src.eda.run_eda import plot_missing_values
        df = pd.DataFrame({
            "isFraud": [0, 1, 0, 1],
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [5.0, 6.0, 7.0, 8.0],
        })
        out = plot_missing_values(df, tmp_path)
        assert out.exists() and out.stat().st_size > 0


class TestPlotCorrelationHeatmap:
    def test_creates_file(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_correlation_heatmap
        out = plot_correlation_heatmap(synthetic_df, tmp_path)
        assert out.exists() and out.stat().st_size > 0


class TestPlotCategoricalFraudRates:
    def test_creates_file(self, synthetic_df, tmp_path):
        from src.eda.run_eda import plot_categorical_fraud_rates
        out = plot_categorical_fraud_rates(synthetic_df, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_handles_missing_column_gracefully(self, tmp_path):
        """Should still run if some categorical columns are absent."""
        from src.eda.run_eda import plot_categorical_fraud_rates
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            "isFraud":   rng.integers(0, 2, n),
            "ProductCD": rng.choice(["W", "H", "C"], n),
            # card4, card6, P_emaildomain intentionally omitted
        })
        out = plot_categorical_fraud_rates(df, tmp_path)
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — EDA summary
# ─────────────────────────────────────────────────────────────────────────────

class TestEdaSummary:
    def test_summary_returns_dict(self, synthetic_df):
        from src.eda.run_eda import log_eda_summary
        result = log_eda_summary(synthetic_df)
        assert isinstance(result, dict)

    def test_summary_has_required_keys(self, synthetic_df):
        from src.eda.run_eda import log_eda_summary
        result = log_eda_summary(synthetic_df)
        for key in ["n_total", "n_fraud", "n_legit", "fraud_rate",
                    "imbalance_ratio", "n_features"]:
            assert key in result, f"Missing key: {key}"

    def test_fraud_rate_matches_data(self, synthetic_df):
        from src.eda.run_eda import log_eda_summary
        result = log_eda_summary(synthetic_df)
        expected = synthetic_df["isFraud"].mean()
        assert abs(result["fraud_rate"] - expected) < 1e-9

    def test_n_total_correct(self, synthetic_df):
        from src.eda.run_eda import log_eda_summary
        result = log_eda_summary(synthetic_df)
        assert result["n_total"] == len(synthetic_df)

    def test_fraud_plus_legit_equals_total(self, synthetic_df):
        from src.eda.run_eda import log_eda_summary
        result = log_eda_summary(synthetic_df)
        assert result["n_fraud"] + result["n_legit"] == result["n_total"]

    def test_imbalance_ratio_positive(self, synthetic_df):
        from src.eda.run_eda import log_eda_summary
        result = log_eda_summary(synthetic_df)
        assert result["imbalance_ratio"] > 1


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — figures on disk (skip if not yet generated)
# ─────────────────────────────────────────────────────────────────────────────

class TestEDAFiguresOnDisk:
    @skip_no_figures
    @pytest.mark.parametrize("fname", EXPECTED_FIGURES)
    def test_figure_file_exists(self, fname):
        path = FIGURES_DIR / fname
        assert path.exists(), f"Missing figure: {fname}"

    @skip_no_figures
    @pytest.mark.parametrize("fname", EXPECTED_FIGURES)
    def test_figure_is_valid_png(self, fname):
        from PIL import Image
        path = FIGURES_DIR / fname
        assert path.stat().st_size > 10_000, \
            f"Figure too small (likely empty): {fname} — {path.stat().st_size} bytes"
        img = Image.open(path)
        assert img.format == "PNG"
        w, h = img.size
        assert w >= 800 and h >= 400, \
            f"Figure too small: {fname} — {w}×{h}px"

    @skip_no_figures
    def test_all_eight_figures_present(self):
        missing = [f for f in EXPECTED_FIGURES
                   if not (FIGURES_DIR / f).exists()]
        assert missing == [], f"Missing figures: {missing}"
