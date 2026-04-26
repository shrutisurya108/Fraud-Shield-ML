"""
fraud-shield-ml | tests/test_phase1_scaffold.py
────────────────────────────────────────────────
Phase 1 test suite: validates project scaffold,
config loading, and logger setup.

Run with:  pytest tests/test_phase1_scaffold.py -v
"""

import sys
from pathlib import Path
import pytest

# ── Ensure src is on the path ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Directory Structure Tests
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src/ingestion",
    "src/eda",
    "src/imbalance",
    "src/models",
    "src/explainability",
    "src/reporting",
    "src/utils",
    "outputs/figures",
    "outputs/models",
    "outputs/reports",
    "tests",
    "logs",
    "config",
    "docs",
    "streamlit_app",
]

REQUIRED_FILES = [
    "requirements.txt",
    "setup.py",
    "README.md",
    ".gitignore",
    "config/config.yaml",
    "src/__init__.py",
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/config_loader.py",
    "src/ingestion/__init__.py",
]


@pytest.mark.parametrize("directory", REQUIRED_DIRS)
def test_directory_exists(directory):
    """All required directories must exist."""
    path = PROJECT_ROOT / directory
    assert path.is_dir(), f"Missing directory: {directory}"


@pytest.mark.parametrize("filepath", REQUIRED_FILES)
def test_file_exists(filepath):
    """All required files must exist."""
    path = PROJECT_ROOT / filepath
    assert path.is_file(), f"Missing file: {filepath}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Config Loader Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigLoader:
    def test_config_loads_without_error(self):
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert cfg is not None

    def test_config_has_required_sections(self):
        from src.utils.config_loader import load_config
        cfg = load_config()
        required = ["project", "paths", "dataset", "eda", "imbalance", "models", "evaluation", "explainability"]
        for section in required:
            assert hasattr(cfg, section), f"Config missing section: {section}"

    def test_config_paths_are_strings(self):
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert isinstance(cfg.paths.data_raw, str)
        assert isinstance(cfg.paths.data_processed, str)

    def test_config_dataset_fields(self):
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert cfg.dataset.target_column == "isFraud"
        assert cfg.dataset.transaction_id == "TransactionID"

    def test_config_f2_beta(self):
        from src.utils.config_loader import load_config
        cfg = load_config()
        assert cfg.evaluation.f2_beta == 2

    def test_config_dot_access(self):
        """Nested dot access must work (SimpleNamespace)."""
        from src.utils.config_loader import load_config
        cfg = load_config()
        # Three levels deep
        _ = cfg.models.xgboost.n_estimators
        assert cfg.models.xgboost.n_estimators == 500

    def test_config_missing_file_raises(self):
        from src.utils.config_loader import load_config
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Logger Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLogger:
    def test_logger_returns_bound_logger(self):
        from src.utils.logger import get_logger
        log = get_logger("test-module")
        assert log is not None

    def test_logger_can_log_info(self):
        from src.utils.logger import get_logger
        log = get_logger("test-module")
        # Should not raise
        log.info("Test info message from pytest")

    def test_logger_creates_log_dir(self):
        from src.utils.logger import get_logger
        get_logger("test-module")
        log_dir = PROJECT_ROOT / "logs"
        assert log_dir.is_dir(), "logs/ directory was not created by logger"

    def test_multiple_loggers_dont_crash(self):
        from src.utils.logger import get_logger
        log1 = get_logger("module-a")
        log2 = get_logger("module-b")
        log1.info("Logger A working")
        log2.info("Logger B working")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Requirements & Setup Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRequirements:
    def test_requirements_file_not_empty(self):
        req_path = PROJECT_ROOT / "requirements.txt"
        content = req_path.read_text()
        assert len(content.strip()) > 0

    def test_requirements_has_key_packages(self):
        req_path = PROJECT_ROOT / "requirements.txt"
        content = req_path.read_text().lower()
        key_packages = [
            "pandas", "numpy", "scikit-learn", "xgboost",
            "lightgbm", "imbalanced-learn", "shap", "lime",
            "optuna", "streamlit", "loguru",
        ]
        for pkg in key_packages:
            assert pkg in content, f"Missing package in requirements.txt: {pkg}"

    def test_gitignore_excludes_data(self):
        gitignore = (PROJECT_ROOT / ".gitignore").read_text()
        assert "data/raw" in gitignore
        assert "*.csv" in gitignore or "data/raw/*.csv" in gitignore

    def test_gitignore_excludes_logs(self):
        gitignore = (PROJECT_ROOT / ".gitignore").read_text()
        assert "logs/" in gitignore

    def test_gitignore_excludes_kaggle_json(self):
        gitignore = (PROJECT_ROOT / ".gitignore").read_text()
        assert "kaggle.json" in gitignore
