"""
fraud-shield-ml | tests/test_phase5_models.py
───────────────────────────────────────────────
Phase 5 test suite: model training pipeline.

Unit tests use synthetic data — no real data needed.
Integration tests check model artefacts and figures after training.

Run:
    pytest tests/test_phase5_models.py -v
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

MODELS_DIR  = PROJECT_ROOT / cfg.paths.outputs_models
FIGURES_DIR = PROJECT_ROOT / cfg.paths.outputs_figures

EXPECTED_FIGURES = [
    "12_confusion_matrix.png",
    "13_roc_curves.png",
    "14_pr_curves.png",
    "15_threshold_analysis.png",
    "16_model_comparison.png",
]
EXPECTED_MODEL_FILES = [
    "lr_baseline.joblib",
    "xgboost_model.joblib",
    "lightgbm_model.joblib",
    "model_results.json",
    "best_model_config.json",
]

OUTPUTS_AVAILABLE = (
    all((FIGURES_DIR / f).exists() for f in EXPECTED_FIGURES) and
    all((MODELS_DIR  / f).exists() for f in EXPECTED_MODEL_FILES)
)

skip_no_outputs = pytest.mark.skipif(
    not OUTPUTS_AVAILABLE,
    reason="Phase 5 outputs not present — run train_models.py first"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_binary():
    """2K-row binary classification dataset with ~3.5% positive rate."""
    rng = np.random.default_rng(42)
    n   = 2_000
    y   = (rng.random(n) < 0.035).astype(int)
    X   = rng.normal(0, 1, (n, 20))
    X[y == 1, :3] += 1.5   # make fraud slightly separable
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateMetrics:
    def test_compute_all_metrics_keys(self, synthetic_binary):
        from src.models.evaluate import compute_all_metrics
        X, y = synthetic_binary
        rng   = np.random.default_rng(0)
        y_prob = np.clip(rng.random(len(y)), 0, 1)
        result = compute_all_metrics(y, y_prob, threshold=0.5, model_name="test")
        required = ["model", "threshold", "pr_auc", "roc_auc", "f2_score",
                    "recall", "precision", "accuracy", "tp", "fp", "tn", "fn"]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_compute_all_metrics_values_in_range(self, synthetic_binary):
        from src.models.evaluate import compute_all_metrics
        X, y = synthetic_binary
        rng   = np.random.default_rng(0)
        y_prob = np.clip(rng.random(len(y)), 0, 1)
        result = compute_all_metrics(y, y_prob, 0.5, "test")
        for k in ["pr_auc", "roc_auc", "f2_score", "recall", "precision", "accuracy"]:
            assert 0.0 <= result[k] <= 1.0, f"{k}={result[k]} out of [0,1]"

    def test_perfect_model_scores_ones(self):
        from src.models.evaluate import compute_all_metrics
        y      = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.01, 0.01, 0.01, 0.99, 0.99, 0.99])
        result = compute_all_metrics(y, y_prob, 0.5)
        assert result["recall"]    == 1.0
        assert result["precision"] == 1.0
        assert result["f2_score"]  == 1.0

    def test_confusion_matrix_elements_sum_to_n(self, synthetic_binary):
        from src.models.evaluate import compute_all_metrics
        X, y = synthetic_binary
        rng   = np.random.default_rng(0)
        y_prob = np.clip(rng.random(len(y)), 0, 1)
        m = compute_all_metrics(y, y_prob, 0.5)
        assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(y)

    def test_find_best_f2_threshold_range(self, synthetic_binary):
        from src.models.evaluate import find_best_f2_threshold
        X, y = synthetic_binary
        lr = LogisticRegression(class_weight="balanced", max_iter=200,
                                solver="lbfgs", random_state=42)
        lr.fit(X, y)
        y_prob = lr.predict_proba(X)[:, 1]
        t, f2 = find_best_f2_threshold(y, y_prob)
        assert 0.0 < t < 1.0
        assert 0.0 <= f2 <= 1.0

    def test_optimal_threshold_beats_default(self, synthetic_binary):
        from src.models.evaluate import compute_all_metrics, find_best_f2_threshold
        X, y = synthetic_binary
        lr = LogisticRegression(class_weight="balanced", max_iter=300,
                                solver="lbfgs", random_state=42)
        lr.fit(X, y)
        y_prob = lr.predict_proba(X)[:, 1]
        t_opt, _ = find_best_f2_threshold(y, y_prob)
        m_default = compute_all_metrics(y, y_prob, 0.5)
        m_optimal = compute_all_metrics(y, y_prob, t_opt)
        assert m_optimal["f2_score"] >= m_default["f2_score"] - 1e-9

    def test_find_precision_threshold(self, synthetic_binary):
        from src.models.evaluate import find_precision_threshold
        X, y = synthetic_binary
        lr = LogisticRegression(class_weight="balanced", max_iter=200,
                                solver="lbfgs", random_state=42)
        lr.fit(X, y)
        y_prob = lr.predict_proba(X)[:, 1]
        t, recall = find_precision_threshold(y, y_prob, target_precision=0.50)
        assert 0.0 < t <= 1.0
        assert 0.0 <= recall <= 1.0

    def test_get_curve_data_keys(self, synthetic_binary):
        from src.models.evaluate import get_curve_data
        X, y = synthetic_binary
        rng   = np.random.default_rng(0)
        y_prob = np.clip(rng.random(len(y)), 0, 1)
        result = get_curve_data(y, y_prob)
        for k in ["precision", "recall", "fpr", "tpr", "pr_auc", "roc_auc"]:
            assert k in result

    def test_get_curve_data_arrays_nonempty(self, synthetic_binary):
        from src.models.evaluate import get_curve_data
        X, y = synthetic_binary
        rng   = np.random.default_rng(0)
        y_prob = np.clip(rng.random(len(y)), 0, 1)
        result = get_curve_data(y, y_prob)
        assert len(result["precision"]) > 1
        assert len(result["fpr"])       > 1


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — plot functions (smoke tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotFunctions:
    @pytest.fixture
    def mock_curve_data(self, synthetic_binary):
        from src.models.evaluate import get_curve_data
        X, y = synthetic_binary
        lr = LogisticRegression(class_weight="balanced", max_iter=200,
                                solver="lbfgs", random_state=42)
        lr.fit(X, y)
        y_prob = lr.predict_proba(X)[:, 1]
        return {"Logistic Regression": get_curve_data(y, y_prob)}, y, y_prob

    def test_plot_roc_curves(self, mock_curve_data, tmp_path):
        from src.models.train_models import plot_roc_curves
        curve_data, _, _ = mock_curve_data
        out = plot_roc_curves(curve_data, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_pr_curves(self, mock_curve_data, tmp_path):
        from src.models.train_models import plot_pr_curves
        curve_data, _, _ = mock_curve_data
        out = plot_pr_curves(curve_data, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_confusion_matrix(self, mock_curve_data, tmp_path):
        from src.models.train_models import plot_confusion_matrix
        _, y, y_prob = mock_curve_data
        out = plot_confusion_matrix(y, y_prob, 0.5, "TestModel", tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_threshold_analysis(self, mock_curve_data, tmp_path):
        from src.models.train_models import plot_threshold_analysis
        _, y, y_prob = mock_curve_data
        out = plot_threshold_analysis(y, y_prob, 0.3, "TestModel", tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_model_comparison(self, tmp_path):
        from src.models.train_models import plot_model_comparison
        metrics = [
            {"model": "LR",      "pr_auc": 0.40, "roc_auc": 0.85,
             "f2_score": 0.35, "recall": 0.60, "precision": 0.22,
             "accuracy": 0.93, "threshold": 0.15,
             "tp": 100, "fp": 300, "tn": 9400, "fn": 200},
            {"model": "XGBoost", "pr_auc": 0.78, "roc_auc": 0.95,
             "f2_score": 0.62, "recall": 0.82, "precision": 0.45,
             "accuracy": 0.97, "threshold": 0.35,
             "tp": 140, "fp": 170, "tn": 9530, "fn": 160},
        ]
        out = plot_model_comparison(metrics, tmp_path)
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — LR training smoke test on synthetic data
# ─────────────────────────────────────────────────────────────────────────────

class TestLogisticRegressionTraining:
    def test_lr_trains_and_predicts(self, synthetic_binary, tmp_path):
        from src.models.train_models import train_logistic_regression
        X, y = synthetic_binary
        df = pd.DataFrame(
            X, columns=[f"V{i}" for i in range(X.shape[1])]
        )
        df["isFraud"] = y
        from src.imbalance.preprocessor import prepare_xy
        X_df, y_s, num_cols, cat_cols = prepare_xy(df)

        from sklearn.model_selection import train_test_split
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_df, y_s, test_size=0.2, stratify=y_s, random_state=42
        )
        pipeline, y_prob, metrics = train_logistic_regression(
            X_tr, y_tr.values, X_vl, y_vl.values, num_cols, cat_cols
        )
        assert len(y_prob) == len(y_vl)
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert pipeline is not None


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — outputs on disk
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase5Outputs:
    @skip_no_outputs
    @pytest.mark.parametrize("fname", EXPECTED_FIGURES)
    def test_figure_exists(self, fname):
        assert (FIGURES_DIR / fname).exists()

    @skip_no_outputs
    @pytest.mark.parametrize("fname", EXPECTED_FIGURES)
    def test_figure_is_valid_png(self, fname):
        from PIL import Image
        path = FIGURES_DIR / fname
        assert path.stat().st_size > 5_000
        assert Image.open(path).format == "PNG"

    @skip_no_outputs
    @pytest.mark.parametrize("fname", EXPECTED_MODEL_FILES)
    def test_model_file_exists(self, fname):
        assert (MODELS_DIR / fname).exists()

    @skip_no_outputs
    def test_model_results_json_structure(self):
        with open(MODELS_DIR / "model_results.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3
        for m in data:
            assert "model"     in m
            assert "pr_auc"    in m
            assert "f2_score"  in m
            assert "threshold" in m

    @skip_no_outputs
    def test_best_model_config_structure(self):
        with open(MODELS_DIR / "best_model_config.json") as f:
            cfg_data = json.load(f)
        for k in ["best_model", "best_model_file", "f2_threshold",
                  "hard_block_threshold", "metrics", "feature_cols"]:
            assert k in cfg_data, f"Missing key: {k}"

    @skip_no_outputs
    def test_best_model_is_valid_name(self):
        with open(MODELS_DIR / "best_model_config.json") as f:
            cfg_data = json.load(f)
        assert cfg_data["best_model"] in [
            "Logistic Regression", "XGBoost", "LightGBM"
        ]

    @skip_no_outputs
    def test_best_model_pr_auc_beats_lr_baseline(self):
        """The winning model must beat LR — otherwise tuning failed."""
        with open(MODELS_DIR / "model_results.json") as f:
            results = json.load(f)
        lr_pr   = next(m["pr_auc"] for m in results
                       if m["model"] == "Logistic Regression")
        best_pr = max(m["pr_auc"] for m in results)
        assert best_pr > lr_pr, \
            f"Best PR-AUC ({best_pr:.4f}) did not beat LR ({lr_pr:.4f})"

    @skip_no_outputs
    def test_thresholds_in_range(self):
        with open(MODELS_DIR / "best_model_config.json") as f:
            cfg_data = json.load(f)
        assert 0.0 < cfg_data["f2_threshold"]         < 1.0
        assert 0.0 < cfg_data["hard_block_threshold"]  < 1.0

    @skip_no_outputs
    def test_feature_cols_saved(self):
        with open(MODELS_DIR / "best_model_config.json") as f:
            cfg_data = json.load(f)
        assert len(cfg_data["feature_cols"]["numeric"])     > 0
        assert len(cfg_data["feature_cols"]["categorical"]) > 0

    @skip_no_outputs
    def test_joblib_models_loadable(self):
        import joblib
        import __main__
        # joblib pickle stored wrappers as __main__.XGBWrapper / LGBMWrapper
        # (the namespace when train_models.py ran as a script). Inject them
        # into __main__ so pickle's find_class() resolves correctly in pytest.
        from src.models.train_models import XGBWrapper, LGBMWrapper
        __main__.XGBWrapper  = XGBWrapper
        __main__.LGBMWrapper = LGBMWrapper

        for fname in ["lr_baseline.joblib", "xgboost_model.joblib",
                      "lightgbm_model.joblib"]:
            model = joblib.load(MODELS_DIR / fname)
            assert model is not None
            assert hasattr(model, "predict_proba")
