"""
fraud-shield-ml | tests/test_phase7_report.py
───────────────────────────────────────────────
Phase 7 test suite: PDF report generation.

Unit tests validate individual builder functions.
Integration test validates the final PDF exists and is readable.

Run:
    pytest tests/test_phase7_report.py -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import cfg

REPORTS_DIR = PROJECT_ROOT / cfg.paths.outputs_reports
REPORT_PDF  = REPORTS_DIR / "fraud_detection_report.pdf"
MODELS_DIR  = PROJECT_ROOT / cfg.paths.outputs_models

REPORT_AVAILABLE = REPORT_PDF.exists()
skip_no_report   = pytest.mark.skipif(
    not REPORT_AVAILABLE,
    reason="Report PDF not present — run generate_report.py first"
)

RESULTS_AVAILABLE = (MODELS_DIR / "model_results.json").exists()
skip_no_results   = pytest.mark.skipif(
    not RESULTS_AVAILABLE,
    reason="model_results.json not present — run train_models.py first"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture — mock results for unit tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_results():
    return [
        {"model": "Logistic Regression", "pr_auc": 0.31, "roc_auc": 0.84,
         "f2_score": 0.40, "recall": 0.54, "precision": 0.19,
         "accuracy": 0.96, "threshold": 0.66,
         "tp": 2200, "fp": 9400, "tn": 104575, "fn": 1933},
        {"model": "XGBoost", "pr_auc": 0.82, "roc_auc": 0.97,
         "f2_score": 0.77, "recall": 0.79, "precision": 0.70,
         "accuracy": 0.98, "threshold": 0.45,
         "tp": 3265, "fp": 1400, "tn": 112575, "fn": 868},
        {"model": "LightGBM", "pr_auc": 0.79, "roc_auc": 0.96,
         "f2_score": 0.74, "recall": 0.77, "precision": 0.63,
         "accuracy": 0.97, "threshold": 0.64,
         "tp": 3183, "fp": 1870, "tn": 112105, "fn": 950},
    ]


@pytest.fixture(scope="module")
def mock_config(mock_results):
    return {
        "best_model":             "XGBoost",
        "best_model_file":        "xgboost_model.joblib",
        "f2_threshold":           0.4456,
        "hard_block_threshold":   0.9900,
        "metrics":                mock_results[1],
        "feature_cols": {
            "numeric":    [f"V{i}" for i in range(1, 213)],
            "categorical": ["ProductCD", "card4", "card6"],
        },
        "xgboost_best_params":  {"n_estimators": 594, "max_depth": 8},
        "lgbm_best_params":     {"n_estimators": 556, "num_leaves": 50},
    }


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — style builder
# ─────────────────────────────────────────────────────────────────────────────

class TestStyleBuilder:
    def test_build_styles_returns_dict(self):
        from src.reporting.generate_report import build_styles
        styles = build_styles()
        assert isinstance(styles, dict)

    def test_required_style_keys_present(self):
        from src.reporting.generate_report import build_styles
        styles = build_styles()
        required = ["title", "subtitle", "h1", "h2", "body",
                    "bullet", "caption", "model_card_key", "model_card_val"]
        for k in required:
            assert k in styles, f"Missing style: {k}"

    def test_styles_are_paragraph_styles(self):
        from src.reporting.generate_report import build_styles
        from reportlab.lib.styles import ParagraphStyle
        styles = build_styles()
        for k, v in styles.items():
            assert isinstance(v, ParagraphStyle), \
                f"Style '{k}' is not a ParagraphStyle"


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — section builders
# ─────────────────────────────────────────────────────────────────────────────

class TestSectionBuilders:
    def test_build_cover_returns_list(self, mock_results, mock_config):
        from src.reporting.generate_report import build_cover, build_styles
        styles = build_styles()
        story  = build_cover(styles, mock_results, mock_config)
        assert isinstance(story, list)
        assert len(story) > 0

    def test_build_exec_summary_returns_list(self, mock_results, mock_config):
        from src.reporting.generate_report import build_exec_summary, build_styles
        styles = build_styles()
        story  = build_exec_summary(styles, mock_results, mock_config)
        assert isinstance(story, list)
        assert len(story) > 0

    def test_build_methodology_returns_list(self):
        from src.reporting.generate_report import build_methodology, build_styles
        styles = build_styles()
        story  = build_methodology(styles)
        assert isinstance(story, list)
        assert len(story) > 0

    def test_build_model_card_returns_list(self, mock_config, mock_results):
        from src.reporting.generate_report import build_model_card, build_styles
        styles = build_styles()
        story  = build_model_card(styles, mock_config, mock_results)
        assert isinstance(story, list)
        assert len(story) > 0

    def test_build_limitations_returns_list(self):
        from src.reporting.generate_report import build_limitations, build_styles
        styles = build_styles()
        story  = build_limitations(styles)
        assert isinstance(story, list)
        assert len(story) > 0

    def test_embed_figure_missing_returns_placeholder(self, tmp_path):
        from src.reporting.generate_report import embed_figure, build_styles
        styles  = build_styles()
        missing = tmp_path / "nonexistent.png"
        result  = embed_figure(missing, caption="test", styles=styles)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_build_results_returns_list(self, mock_results, mock_config, tmp_path):
        from src.reporting.generate_report import build_results, build_styles
        styles = build_styles()
        # tmp_path has no figures — embed_figure will return placeholders
        story  = build_results(styles, mock_results, mock_config, tmp_path)
        assert isinstance(story, list)
        assert len(story) > 0

    def test_build_explainability_returns_list(self, tmp_path):
        from src.reporting.generate_report import build_explainability, build_styles
        styles = build_styles()
        story  = build_explainability(styles, tmp_path)
        assert isinstance(story, list)
        assert len(story) > 0


# ─────────────────────────────────────────────────────────────────────────────
# UNIT — full PDF generation on mock data
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFGeneration:
    def test_full_pdf_generates_without_error(self,
                                               mock_results, mock_config,
                                               tmp_path):
        """Generate a complete PDF using mock data into a temp directory."""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate
        from src.reporting.generate_report import (
            build_styles, build_cover, build_exec_summary,
            build_methodology, build_model_card, build_limitations,
            build_results, build_explainability, make_header_footer,
            MARGIN,
        )
        import reportlab.lib.units as units

        out = tmp_path / "test_report.pdf"
        doc = SimpleDocTemplate(
            str(out),
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=1.6 * units.cm,
            bottomMargin=1.3 * units.cm,
        )
        styles = build_styles()
        story  = []
        story += build_cover(styles, mock_results, mock_config)
        story += build_exec_summary(styles, mock_results, mock_config)
        story += build_methodology(styles)
        story += build_results(styles, mock_results, mock_config, tmp_path)
        story += build_model_card(styles, mock_config, mock_results)
        story += build_explainability(styles, tmp_path)
        story += build_limitations(styles)

        doc.build(story, onFirstPage=make_header_footer,
                  onLaterPages=make_header_footer)

        assert out.exists()
        assert out.stat().st_size > 10_000, \
            f"PDF too small: {out.stat().st_size} bytes"

    def test_pdf_is_valid_pdf_format(self, mock_results, mock_config, tmp_path):
        """Verify generated file has PDF magic bytes."""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from src.reporting.generate_report import (
            build_styles, build_limitations, make_header_footer, MARGIN
        )
        import reportlab.lib.units as units

        out = tmp_path / "mini_report.pdf"
        doc = SimpleDocTemplate(str(out), pagesize=A4,
                                leftMargin=MARGIN, rightMargin=MARGIN,
                                topMargin=1.6*units.cm, bottomMargin=1.3*units.cm)
        styles = build_styles()
        doc.build(build_limitations(styles),
                  onFirstPage=make_header_footer,
                  onLaterPages=make_header_footer)

        with open(out, "rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-", f"Not a valid PDF: magic={magic}"


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — real PDF on disk
# ─────────────────────────────────────────────────────────────────────────────

class TestReportOnDisk:
    @skip_no_report
    def test_report_pdf_exists(self):
        assert REPORT_PDF.exists()

    @skip_no_report
    def test_report_pdf_is_nonzero(self):
        size = REPORT_PDF.stat().st_size
        assert size > 100_000, f"Report too small: {size} bytes (expected >100KB)"

    @skip_no_report
    def test_report_is_valid_pdf(self):
        with open(REPORT_PDF, "rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-", f"Not a valid PDF: magic={magic}"

    @skip_no_report
    def test_report_has_multiple_pages(self):
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(REPORT_PDF))
            assert len(reader.pages) >= 6, \
                f"Expected at least 6 pages, got {len(reader.pages)}"
        except ImportError:
            pytest.skip("pypdf not installed — skipping page count check")

    @skip_no_results
    def test_model_results_consistent(self):
        """Results JSON used for report must have all 3 models."""
        with open(MODELS_DIR / "model_results.json") as f:
            results = json.load(f)
        model_names = [r["model"] for r in results]
        assert "Logistic Regression" in model_names
        assert "XGBoost"             in model_names
        assert "LightGBM"            in model_names
