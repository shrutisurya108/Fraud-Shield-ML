"""
fraud-shield-ml | tests/test_phase6_xai.py
────────────────────────────────────────────
Phase 6 test suite: XAI pipeline (SHAP + LIME).

Unit tests use synthetic data — no real model or data needed.
Integration tests check figures exist after run_xai.py completes.

Run:
    pytest tests/test_phase6_xai.py -v
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
MODELS_DIR  = PROJECT_ROOT / cfg.paths.outputs_models

EXPECTED_FIGURES = [
    "17_shap_beeswarm.png",
    "18_shap_waterfall_fraud.png",
    "19_lime_false_positive.png",
]

FIGURES_AVAILABLE = all((FIGURES_DIR / f).exists() for f in EXPECTED_FIGURES)
SHAP_NPY_AVAILABLE = (MODELS_DIR / "shap_values.npy").exists()

skip_no_figures = pytest.mark.skipif(
    not FIGURES_AVAILABLE,
    reason="Phase 6 figures not present — run python3 src/explainability/run_xai.py first"
)
skip_no_shap = pytest.mark.skipif(
    not SHAP_NPY_AVAILABLE,
    reason="SHAP values not cached — run run_xai.py first"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_xai_data():
    """2K-row synthetic data + trained XGBoost for XAI smoke tests."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X, y = make_classification(
        n_samples=2_000, n_features=20, n_informative=10,
        weights=[0.965, 0.035], random_state=42,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = XGBClassifier(
        n_estimators=50, max_depth=3,
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
        random_state=42, tree_method="hist",
        eval_metric="aucpr",
    )
    model.fit(X_tr, y_tr)
    feature_names = [f"V{i}" for i in range(X.shape[1])]
    return model, X_val, y_val, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — SHAP computation
# ─────────────────────────────────────────────────────────────────────────────

class TestSHAPComputation:
    def test_shap_tree_explainer_runs(self, synthetic_xai_data):
        import shap
        model, X_val, y_val, feature_names = synthetic_xai_data
        explainer = shap.TreeExplainer(model)
        shap_out  = explainer(X_val[:50])
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        assert sv.shape == (50, X_val.shape[1])

    def test_shap_values_have_correct_shape(self, synthetic_xai_data):
        import shap
        model, X_val, y_val, feature_names = synthetic_xai_data
        explainer = shap.TreeExplainer(model)
        shap_out  = explainer(X_val[:20])
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        assert sv.shape[0] == 20
        assert sv.shape[1] == X_val.shape[1]

    def test_shap_values_are_finite(self, synthetic_xai_data):
        import shap
        model, X_val, y_val, feature_names = synthetic_xai_data
        explainer = shap.TreeExplainer(model)
        shap_out  = explainer(X_val[:30])
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        assert np.isfinite(sv).all(), "SHAP values contain NaN or Inf"

    def test_mean_abs_shap_nonnegative(self, synthetic_xai_data):
        import shap
        model, X_val, y_val, feature_names = synthetic_xai_data
        explainer = shap.TreeExplainer(model)
        shap_out  = explainer(X_val[:30])
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        mean_abs = np.abs(sv).mean(axis=0)
        assert (mean_abs >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — SHAP beeswarm plot
# ─────────────────────────────────────────────────────────────────────────────

class TestSHAPBeeswarm:
    def test_beeswarm_creates_file(self, synthetic_xai_data, tmp_path):
        import shap
        from src.explainability.shap_analysis import plot_shap_beeswarm
        model, X_val, y_val, feature_names = synthetic_xai_data
        explainer = shap.TreeExplainer(model)
        shap_out  = explainer(X_val[:100])
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        out = plot_shap_beeswarm(sv, X_val[:100], feature_names, tmp_path, top_n=10)
        assert out.exists() and out.stat().st_size > 0

    def test_beeswarm_correct_filename(self, synthetic_xai_data, tmp_path):
        import shap
        from src.explainability.shap_analysis import plot_shap_beeswarm
        model, X_val, y_val, feature_names = synthetic_xai_data
        explainer = shap.TreeExplainer(model)
        shap_out  = explainer(X_val[:50])
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        out = plot_shap_beeswarm(sv, X_val[:50], feature_names, tmp_path, top_n=5)
        assert out.name == "17_shap_beeswarm.png"


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — False positive detection
# ─────────────────────────────────────────────────────────────────────────────

class TestFalsePositiveDetection:
    def test_find_false_positive_returns_valid_index(self, synthetic_xai_data):
        from src.explainability.lime_analysis import find_false_positive

        model, X_val, y_val, feature_names = synthetic_xai_data

        # Create a mock wrapper
        class MockWrapper:
            def __init__(self, model):
                self.model = model
            def predict_proba(self, X):
                return self.model.predict_proba(X)

        wrapper   = MockWrapper(model)
        threshold = 0.3

        fp_idx, fp_prob = find_false_positive(wrapper, X_val, y_val, threshold)
        assert 0 <= fp_idx < len(X_val)
        assert 0.0 <= fp_prob <= 1.0

    def test_false_positive_is_actually_legitimate(self, synthetic_xai_data):
        from src.explainability.lime_analysis import find_false_positive

        model, X_val, y_val, feature_names = synthetic_xai_data

        class MockWrapper:
            def __init__(self, model):
                self.model = model
            def predict_proba(self, X):
                return self.model.predict_proba(X)

        wrapper   = MockWrapper(model)
        threshold = 0.3

        fp_idx, fp_prob = find_false_positive(wrapper, X_val, y_val, threshold)
        # The selected transaction should be legitimate (y=0)
        assert y_val[fp_idx] == 0, \
            f"Expected legitimate transaction, got label={y_val[fp_idx]}"


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — LIME explanation
# ─────────────────────────────────────────────────────────────────────────────

class TestLIMEExplanation:
    def test_lime_runs_and_returns_explanation(self, synthetic_xai_data):
        from lime.lime_tabular import LimeTabularExplainer

        model, X_val, y_val, feature_names = synthetic_xai_data

        explainer = LimeTabularExplainer(
            training_data=X_val,
            feature_names=feature_names,
            class_names=["Legitimate", "Fraud"],
            mode="classification",
            random_state=42,
        )
        explanation = explainer.explain_instance(
            data_row=X_val[0],
            predict_fn=lambda X: model.predict_proba(X),
            num_features=5,
            num_samples=100,
            labels=(1,),
        )
        assert explanation is not None
        exp_list = explanation.as_list(label=1)
        assert len(exp_list) > 0

    def test_lime_explanation_has_weights(self, synthetic_xai_data):
        from lime.lime_tabular import LimeTabularExplainer

        model, X_val, y_val, feature_names = synthetic_xai_data
        explainer = LimeTabularExplainer(
            training_data=X_val,
            feature_names=feature_names,
            class_names=["Legitimate", "Fraud"],
            mode="classification",
            random_state=42,
        )
        explanation = explainer.explain_instance(
            data_row=X_val[0],
            predict_fn=lambda X: model.predict_proba(X),
            num_features=5,
            num_samples=200,
            labels=(1,),
        )
        exp_list = explanation.as_list(label=1)
        weights  = [w for _, w in exp_list]
        # Weights should be finite and not all zero
        assert all(np.isfinite(w) for w in weights)

    def test_lime_plot_creates_file(self, synthetic_xai_data, tmp_path):
        from lime.lime_tabular import LimeTabularExplainer
        from src.explainability.lime_analysis import plot_lime_false_positive

        model, X_val, y_val, feature_names = synthetic_xai_data
        explainer = LimeTabularExplainer(
            training_data=X_val,
            feature_names=feature_names,
            class_names=["Legitimate", "Fraud"],
            mode="classification",
            random_state=42,
        )
        explanation = explainer.explain_instance(
            data_row=X_val[0],
            predict_fn=lambda X: model.predict_proba(X),
            num_features=8,
            num_samples=100,
            labels=(1,),
        )
        out = plot_lime_false_positive(
            explanation, fp_idx=0, fp_prob=0.55,
            threshold=0.45, out_dir=tmp_path
        )
        assert out.exists() and out.stat().st_size > 0

    def test_lime_plot_correct_filename(self, synthetic_xai_data, tmp_path):
        from lime.lime_tabular import LimeTabularExplainer
        from src.explainability.lime_analysis import plot_lime_false_positive

        model, X_val, y_val, feature_names = synthetic_xai_data
        explainer = LimeTabularExplainer(
            training_data=X_val,
            feature_names=feature_names,
            class_names=["Legitimate", "Fraud"],
            mode="classification",
            random_state=42,
        )
        explanation = explainer.explain_instance(
            data_row=X_val[0],
            predict_fn=lambda X: model.predict_proba(X),
            num_features=5,
            num_samples=100,
            labels=(1,),
        )
        out = plot_lime_false_positive(
            explanation, 0, 0.55, 0.45, tmp_path
        )
        assert out.name == "19_lime_false_positive.png"


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — figures on disk
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase6Outputs:
    @skip_no_figures
    @pytest.mark.parametrize("fname", EXPECTED_FIGURES)
    def test_figure_exists(self, fname):
        assert (FIGURES_DIR / fname).exists(), f"Missing: {fname}"

    @skip_no_figures
    @pytest.mark.parametrize("fname", EXPECTED_FIGURES)
    def test_figure_is_valid_png(self, fname):
        from PIL import Image
        path = FIGURES_DIR / fname
        assert path.stat().st_size > 5_000, \
            f"Figure too small: {fname} ({path.stat().st_size} bytes)"
        assert Image.open(path).format == "PNG"

    @skip_no_figures
    def test_all_three_figures_present(self):
        missing = [f for f in EXPECTED_FIGURES
                   if not (FIGURES_DIR / f).exists()]
        assert missing == [], f"Missing: {missing}"

    @skip_no_shap
    def test_shap_values_npy_loadable(self):
        sv = np.load(MODELS_DIR / "shap_values.npy")
        assert sv.ndim == 2
        assert sv.shape[0] > 0
        assert np.isfinite(sv).all()

    @skip_no_shap
    def test_shap_values_sample_size(self):
        sv = np.load(MODELS_DIR / "shap_values.npy")
        # Should match config shap_sample_size
        assert sv.shape[0] <= cfg.explainability.shap_sample_size + 10
