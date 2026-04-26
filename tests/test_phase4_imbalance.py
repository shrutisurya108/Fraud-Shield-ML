"""
fraud-shield-ml | tests/test_phase4_imbalance.py
──────────────────────────────────────────────────
Phase 4 test suite: imbalance handling pipeline.

Unit tests use a synthetic 3K-row dataframe — no real data needed.
Integration tests check figures and JSON exist after the pipeline runs.

Run:
    pytest tests/test_phase4_imbalance.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import cfg

FIGURES_DIR = PROJECT_ROOT / cfg.paths.outputs_figures
MODELS_DIR  = PROJECT_ROOT / cfg.paths.outputs_models

EXPECTED_FIGURES = [
    "09_imbalance_comparison.png",
    "10_pr_curves_comparison.png",
    "11_accuracy_trap.png",
]
RESULTS_JSON = MODELS_DIR / "imbalance_results.json"

OUTPUTS_AVAILABLE = (
    all((FIGURES_DIR / f).exists() for f in EXPECTED_FIGURES)
    and RESULTS_JSON.exists()
)

skip_no_outputs = pytest.mark.skipif(
    not OUTPUTS_AVAILABLE,
    reason="Phase 4 outputs not present — run compare_strategies.py first"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """3K-row synthetic dataset with ~3.5% fraud rate."""
    rng = np.random.default_rng(42)
    n   = 3_000

    X = np.column_stack([
        rng.normal(0, 1, (n, 15)),
        rng.choice([0, 1, 2, 3], size=(n, 3)).astype(float),
    ])
    # Fraud slightly separable from legit
    y = (rng.random(n) < 0.035).astype(int)
    X[y == 1, 0] += 1.2   # make fraud slightly distinguishable on feature 0

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessor:
    def _make_df(self, n=500):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "isFraud":        rng.integers(0, 2, n),
            "TransactionID":  np.arange(n),
            "TransactionDT":  rng.integers(0, 1_000_000, n),
            "TransactionAmt": rng.exponential(100, n),
            "V1":             np.where(rng.random(n) < 0.3, np.nan, rng.random(n)),
            "V2":             rng.random(n),
            "C1":             rng.integers(0, 5, n).astype(float),
            "ProductCD":      rng.choice(["W", "H", "C"], n),
            "card4":          rng.choice(["visa", "mastercard"], n),
        })

    def test_get_feature_columns_excludes_target(self):
        from src.imbalance.preprocessor import get_feature_columns
        df = self._make_df()
        num_cols, cat_cols = get_feature_columns(df)
        assert "isFraud" not in num_cols
        assert "isFraud" not in cat_cols

    def test_get_feature_columns_excludes_id_cols(self):
        from src.imbalance.preprocessor import get_feature_columns
        df = self._make_df()
        num_cols, cat_cols = get_feature_columns(df)
        assert "TransactionID" not in num_cols + cat_cols
        assert "TransactionDT" not in num_cols + cat_cols

    def test_categorical_cols_correctly_identified(self):
        from src.imbalance.preprocessor import get_feature_columns
        df = self._make_df()
        num_cols, cat_cols = get_feature_columns(df)
        assert "ProductCD" in cat_cols
        assert "card4" in cat_cols

    def test_numeric_cols_correctly_identified(self):
        from src.imbalance.preprocessor import get_feature_columns
        df = self._make_df()
        num_cols, cat_cols = get_feature_columns(df)
        assert "TransactionAmt" in num_cols
        assert "V1" in num_cols

    def test_build_preprocessor_returns_column_transformer(self):
        from src.imbalance.preprocessor import build_preprocessor
        from sklearn.compose import ColumnTransformer
        pp = build_preprocessor(["V1", "V2"], ["ProductCD"])
        assert isinstance(pp, ColumnTransformer)

    def test_preprocessor_fits_and_transforms(self):
        from src.imbalance.preprocessor import build_preprocessor
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "V1": np.where(rng.random(n) < 0.2, np.nan, rng.random(n)),
            "V2": rng.random(n),
            "ProductCD": rng.choice(["W", "H", "C"], n),
        })
        pp = build_preprocessor(["V1", "V2"], ["ProductCD"], scale_numeric=True)
        result = pp.fit_transform(df)
        assert result.shape == (n, 3)
        assert not np.isnan(result).any(), "Preprocessor should fill all NaNs"

    def test_preprocessor_no_scaling_option(self):
        from src.imbalance.preprocessor import build_preprocessor
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "V1": rng.random(n) * 1000,   # large values
            "V2": rng.random(n),
        })
        pp_scaled   = build_preprocessor(["V1", "V2"], [], scale_numeric=True)
        pp_unscaled = build_preprocessor(["V1", "V2"], [], scale_numeric=False)
        X_scaled   = pp_scaled.fit_transform(df)
        X_unscaled = pp_unscaled.fit_transform(df)
        # Scaled V1 should have std ~1; unscaled should have large values
        assert X_scaled[:, 0].std() < 5
        assert X_unscaled[:, 0].std() > 50

    def test_prepare_xy_returns_correct_shapes(self):
        from src.imbalance.preprocessor import prepare_xy
        df = self._make_df(n=200)
        X, y, num_cols, cat_cols = prepare_xy(df)
        assert len(X) == len(y) == 200
        assert "isFraud" not in X.columns
        assert set(y.unique()).issubset({0, 1})

    def test_unknown_categorical_handled(self):
        """Encoder should not crash on unseen categories at transform time."""
        from src.imbalance.preprocessor import build_preprocessor
        rng = np.random.default_rng(0)
        train_df = pd.DataFrame({"cat": ["A", "B", "A", "B", "A"]})
        test_df  = pd.DataFrame({"cat": ["A", "C", "D"]})  # C, D unseen
        pp = build_preprocessor([], ["cat"], scale_numeric=False)
        pp.fit(train_df)
        result = pp.transform(test_df)
        assert not np.isnan(result).any()


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — metric helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricHelpers:
    def test_compute_metrics_returns_required_keys(self, synthetic_data):
        from src.imbalance.compare_strategies import compute_metrics
        X, y = synthetic_data
        rng = np.random.default_rng(0)
        y_prob = np.clip(rng.random(len(y)), 0, 1)
        result = compute_metrics(y, y_prob, threshold=0.5)
        for key in ["pr_auc", "roc_auc", "f2_score", "recall",
                    "precision", "accuracy", "threshold"]:
            assert key in result

    def test_compute_metrics_all_values_in_range(self, synthetic_data):
        from src.imbalance.compare_strategies import compute_metrics
        X, y = synthetic_data
        rng = np.random.default_rng(0)
        y_prob = np.clip(rng.random(len(y)), 0, 1)
        result = compute_metrics(y, y_prob, threshold=0.5)
        for key in ["pr_auc", "roc_auc", "f2_score", "recall",
                    "precision", "accuracy"]:
            assert 0.0 <= result[key] <= 1.0, \
                f"{key} out of range: {result[key]}"

    def test_perfect_predictor_scores_high(self):
        from src.imbalance.compare_strategies import compute_metrics
        y      = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.01, 0.01, 0.01, 0.99, 0.99, 0.99])
        result = compute_metrics(y, y_prob, threshold=0.5)
        assert result["recall"]    == 1.0
        assert result["precision"] == 1.0
        assert result["f2_score"]  == 1.0

    def test_find_best_f2_threshold_is_in_range(self, synthetic_data):
        from src.imbalance.compare_strategies import find_best_f2_threshold
        X, y = synthetic_data
        rng    = np.random.default_rng(42)
        y_prob = np.clip(rng.random(len(y)) * 0.5 + (y * 0.3), 0, 1)
        best_t, best_f2 = find_best_f2_threshold(y, y_prob)
        assert 0.0 < best_t < 1.0
        assert 0.0 <= best_f2 <= 1.0

    def test_threshold_tune_beats_default(self, synthetic_data):
        """
        Optimal threshold should achieve equal or better F2 than 0.5 default.
        """
        from src.imbalance.compare_strategies import (
            compute_metrics, find_best_f2_threshold
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X, y = synthetic_data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        lr = LogisticRegression(
            class_weight="balanced", max_iter=200,
            solver="lbfgs", random_state=42
        )
        lr.fit(X_tr, y_tr)
        y_prob = lr.predict_proba(X_val)[:, 1]

        default  = compute_metrics(y_val, y_prob, threshold=0.5)
        best_t, _ = find_best_f2_threshold(y_val, y_prob)
        tuned    = compute_metrics(y_val, y_prob, threshold=best_t)

        assert tuned["f2_score"] >= default["f2_score"] - 1e-9

    def test_accuracy_misleads_on_imbalanced(self):
        """
        Key insight test: a model predicting all-legitimate gets >96% accuracy
        but 0% recall (and thus 0 PR-AUC) on a ~3.5% fraud dataset.
        """
        from src.imbalance.compare_strategies import compute_metrics

        rng = np.random.default_rng(0)
        n   = 10_000
        y   = (rng.random(n) < 0.035).astype(int)

        # All-zero predictor
        y_prob_all_legit = np.zeros(n)
        result = compute_metrics(y, y_prob_all_legit, threshold=0.5)

        assert result["accuracy"] > 0.96, \
            "All-legit predictor should have >96% accuracy"
        assert result["recall"] == 0.0, \
            "All-legit predictor catches zero fraud"
        assert result["pr_auc"] < 0.10, \
            "All-legit predictor should have near-zero PR-AUC"


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — plot functions (smoke tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestPlots:
    @pytest.fixture
    def mock_results(self):
        return {
            "SMOTE":          {"pr_auc": 0.62, "roc_auc": 0.88, "f2_score": 0.41,
                               "recall": 0.72, "precision": 0.28, "accuracy": 0.963,
                               "threshold": 0.15},
            "ADASYN":         {"pr_auc": 0.60, "roc_auc": 0.87, "f2_score": 0.40,
                               "recall": 0.70, "precision": 0.27, "accuracy": 0.962,
                               "threshold": 0.14},
            "class_weight":   {"pr_auc": 0.64, "roc_auc": 0.89, "f2_score": 0.43,
                               "recall": 0.74, "precision": 0.30, "accuracy": 0.961,
                               "threshold": 0.18},
            "threshold_tune": {"pr_auc": 0.61, "roc_auc": 0.87, "f2_score": 0.38,
                               "recall": 0.68, "precision": 0.25, "accuracy": 0.960,
                               "threshold": 0.12},
        }

    @pytest.fixture
    def mock_pr_data(self, mock_results):
        rng = np.random.default_rng(0)
        data = {}
        for s in mock_results:
            recall    = np.linspace(1, 0, 50)
            precision = np.clip(rng.random(50) * 0.4 + 0.2, 0, 1)
            data[s] = (precision, recall)
        return data

    def test_plot_comparison_creates_file(self, mock_results, tmp_path):
        from src.imbalance.compare_strategies import plot_comparison
        out = plot_comparison(mock_results, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_pr_curves_creates_file(self, mock_pr_data, mock_results, tmp_path):
        from src.imbalance.compare_strategies import plot_pr_curves
        out = plot_pr_curves(mock_pr_data, mock_results, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_accuracy_trap_creates_file(self, mock_results, tmp_path):
        from src.imbalance.compare_strategies import plot_accuracy_trap
        out = plot_accuracy_trap(mock_results, tmp_path)
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — outputs on disk
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase4Outputs:
    @skip_no_outputs
    @pytest.mark.parametrize("fname", EXPECTED_FIGURES)
    def test_figure_exists(self, fname):
        assert (FIGURES_DIR / fname).exists(), f"Missing: {fname}"

    @skip_no_outputs
    @pytest.mark.parametrize("fname", EXPECTED_FIGURES)
    def test_figure_is_valid_png(self, fname):
        from PIL import Image
        path = FIGURES_DIR / fname
        assert path.stat().st_size > 5_000
        img = Image.open(path)
        assert img.format == "PNG"

    @skip_no_outputs
    def test_results_json_exists(self):
        assert RESULTS_JSON.exists()

    @skip_no_outputs
    def test_results_json_has_all_strategies(self):
        with open(RESULTS_JSON) as f:
            data = json.load(f)
        for s in ["SMOTE", "ADASYN", "class_weight", "threshold_tune"]:
            assert s in data, f"Missing strategy: {s}"

    @skip_no_outputs
    def test_results_json_metrics_in_range(self):
        with open(RESULTS_JSON) as f:
            data = json.load(f)
        for strategy, metrics in data.items():
            for key in ["pr_auc", "roc_auc", "f2_score",
                        "recall", "precision", "accuracy"]:
                val = metrics[key]
                assert 0.0 <= val <= 1.0, \
                    f"{strategy}.{key} = {val} out of range"

    @skip_no_outputs
    def test_accuracy_trap_demonstrated(self):
        """
        Integration version: verify the accuracy trap is real in actual results.
        Accuracy range should be tiny, PR-AUC range should be non-trivial.
        """
        with open(RESULTS_JSON) as f:
            data = json.load(f)
        accs    = [data[s]["accuracy"] for s in data]
        pr_aucs = [data[s]["pr_auc"]   for s in data]
        acc_range = max(accs)    - min(accs)
        pr_range  = max(pr_aucs) - min(pr_aucs)
        assert acc_range < 0.15, \
            f"Accuracy range too large ({acc_range:.4f}) — expected <0.15"
        assert pr_range > 0.001, \
            f"PR-AUC range too small ({pr_range:.4f}) — strategies indistinguishable"
