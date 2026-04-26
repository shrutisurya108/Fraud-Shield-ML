"""
fraud-shield-ml | src/utils/logger.py
─────────────────────────────────────
Centralised logging setup using loguru.
Every module imports get_logger() from here.
Logs go to:  logs/fraud_shield_{date}.log  (rotating daily, retained 7 days)
             stdout (coloured, human-readable)
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger


def get_logger(module_name: str = "fraud-shield"):
    """
    Return a configured loguru logger bound to the calling module name.

    Parameters
    ----------
    module_name : str
        Name shown in the log prefix (e.g., 'eda', 'model_training').

    Returns
    -------
    loguru.Logger (bound with module context)
    """
    # Determine project root (two levels up from this file: src/utils → project root)
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"fraud_shield_{datetime.now().strftime('%Y-%m-%d')}.log"

    # Remove any default handlers added by previous calls to avoid duplication
    logger.remove()

    # ── Stdout handler (coloured, for local dev) ─────────────────────────
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[module]}</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    # ── File handler (plain text, rotating daily) ─────────────────────────
    logger.add(
        str(log_file),
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{extra[module]} | "
            "{message}"
        ),
        level="DEBUG",
        rotation="00:00",       # rotate at midnight
        retention="7 days",
        compression="zip",
        enqueue=True,           # thread-safe
    )

    return logger.bind(module=module_name)


# ── Module-level default logger (import-safe) ────────────────────────────────
log = get_logger("fraud-shield")
