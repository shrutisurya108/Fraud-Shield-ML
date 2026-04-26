"""
fraud-shield-ml | src/models/evaluate.py
─────────────────────────────────────────
Shared evaluation utilities for Phase 5.

Functions:
  compute_all_metrics()   — full metric suite at a given threshold
  find_best_threshold()   — sweep thresholds, return best F2 threshold
  find_precision_threshold() — return threshold achieving target precision
  get_pr_curve_data()     — precision/recall arrays for plotting
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config_loader import cfg

BETA = cfg.evaluation.f2_beta          # 2
STEPS = cfg.models.threshold_search_steps  # 100


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "",
) -> dict:
    """Full metric suite at a given classification threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model":     model_name,
        "threshold": round(float(threshold), 4),
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "f2_score":  round(float(fbeta_score(y_true, y_pred, beta=BETA,
                                             zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred,
                                              zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred,
                                                 zero_division=0)), 4),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def find_best_f2_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Return (threshold, f2) that maximises F2 on y_true."""
    thresholds = np.linspace(0.01, 0.99, STEPS)
    best_t, best_f2 = 0.5, 0.0
    for t in thresholds:
        f2 = fbeta_score(y_true, (y_prob >= t).astype(int),
                         beta=BETA, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, float(t)
    return best_t, best_f2


def find_precision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float = 0.80,
) -> tuple[float, float]:
    """
    Return the lowest threshold that achieves >= target_precision.
    Used to identify the hard-block operating point.
    """
    thresholds = np.linspace(0.99, 0.01, STEPS)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        if prec >= target_precision:
            rec = recall_score(y_true, y_pred, zero_division=0)
            return float(t), float(rec)
    return 0.99, 0.0


def get_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Return PR curve and ROC curve arrays for plotting."""
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob)
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    return {
        "precision": precision,
        "recall":    recall,
        "fpr":       fpr,
        "tpr":       tpr,
        "pr_auc":    average_precision_score(y_true, y_prob),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }
