"""
fraud-shield-ml | src/utils/config_loader.py
─────────────────────────────────────────────
Loads config/config.yaml and exposes it as a
dot-accessible dictionary (SimpleNamespace tree).
"""

from pathlib import Path
import yaml
from types import SimpleNamespace


def _dict_to_ns(d: dict) -> SimpleNamespace:
    """Recursively convert nested dicts to SimpleNamespace for dot access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def load_config(config_path: str | Path | None = None) -> SimpleNamespace:
    """
    Load the project YAML config file.

    Parameters
    ----------
    config_path : str | Path | None
        Explicit path to config.yaml.  If None, auto-resolves to
        <project_root>/config/config.yaml.

    Returns
    -------
    SimpleNamespace  (dot-accessible, e.g. cfg.paths.data_raw)
    """
    if config_path is None:
        # Walk up from this file to find project root
        config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return _dict_to_ns(raw)


# Convenience: module-level singleton (import once, use everywhere)
cfg = load_config()
