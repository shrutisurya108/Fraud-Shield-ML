"""
fraud-shield-ml | src/explainability/run_xai.py
─────────────────────────────────────────────────
Phase 6 orchestrator — runs SHAP then LIME in sequence.

Outputs:
  outputs/figures/17_shap_beeswarm.png
  outputs/figures/18_shap_waterfall_fraud.png
  outputs/figures/19_lime_false_positive.png
  outputs/models/shap_values.npy

Run:
    python3 src/explainability/run_xai.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.explainability.shap_analysis import run_shap
from src.explainability.lime_analysis import run_lime

log = get_logger("xai")


def main():
    log.info("=" * 60)
    log.info("fraud-shield-ml | Phase 6: XAI (SHAP + LIME)")
    log.info("=" * 60)

    log.info("\n── Phase 6a: SHAP ──────────────────────────────────────")
    shap_outputs = run_shap()

    log.info("\n── Phase 6b: LIME ──────────────────────────────────────")
    lime_output = run_lime()

    log.info("\n" + "=" * 60)
    log.info("Phase 6 XAI complete. All figures saved:")
    for k, v in shap_outputs.items():
        log.info(f"  {Path(v).name}")
    log.info(f"  {Path(lime_output).name}")
    log.info("  Next: python3 src/reporting/generate_report.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
