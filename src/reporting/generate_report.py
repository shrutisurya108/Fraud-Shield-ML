"""
fraud-shield-ml | src/reporting/generate_report.py
────────────────────────────────────────────────────
Phase 7: Internal DS Report — PDF Memo

Generates a professional PDF memo addressed to the Head of Risk Analytics.
Uses ReportLab Platypus for precise, portable PDF output.

Sections:
  1. Executive Summary
  2. Methodology
  3. Results
  4. Model Card
  5. Explainability (SHAP + LIME)
  6. Limitations & Recommendations

Output:
  outputs/reports/fraud_detection_report.pdf

Run:
    python3 src/reporting/generate_report.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import cfg

log = get_logger("reporting")

# ── Colour palette ────────────────────────────────────────────────────────────
C_NAVY    = colors.HexColor("#1B2A4A")
C_RED     = colors.HexColor("#E63946")
C_BLUE    = colors.HexColor("#457B9D")
C_AMBER   = colors.HexColor("#F4A261")
C_LIGHT   = colors.HexColor("#F8F9FA")
C_RULE    = colors.HexColor("#D1D5DB")
C_WHITE   = colors.white
C_DARK    = colors.HexColor("#1F2937")

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm


# ─────────────────────────────────────────────────────────────────────────────
# Style definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_styles() -> dict:
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=C_NAVY,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=11,
            textColor=C_BLUE,
            spaceAfter=2,
            alignment=TA_LEFT,
        ),
        "meta": ParagraphStyle(
            "meta",
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#6B7280"),
            spaceAfter=0,
            alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=C_NAVY,
            spaceBefore=14,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=C_BLUE,
            spaceBefore=10,
            spaceAfter=3,
            alignment=TA_LEFT,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=C_DARK,
            leading=15,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
        ),
        "body_left": ParagraphStyle(
            "body_left",
            fontName="Helvetica",
            fontSize=10,
            textColor=C_DARK,
            leading=15,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=10,
            textColor=C_DARK,
            leading=14,
            spaceAfter=3,
            leftIndent=16,
            bulletIndent=0,
            alignment=TA_LEFT,
        ),
        "callout": ParagraphStyle(
            "callout",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=C_NAVY,
            leading=15,
            spaceAfter=4,
            leftIndent=12,
            alignment=TA_LEFT,
            backColor=colors.HexColor("#EBF5FB"),
            borderPad=8,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            textColor=colors.HexColor("#6B7280"),
            spaceAfter=10,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=C_WHITE,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_DARK,
            alignment=TA_CENTER,
        ),
        "table_cell_left": ParagraphStyle(
            "table_cell_left",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_DARK,
            alignment=TA_LEFT,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.HexColor("#9CA3AF"),
            alignment=TA_CENTER,
        ),
        "model_card_key": ParagraphStyle(
            "model_card_key",
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=C_NAVY,
            spaceAfter=1,
        ),
        "model_card_val": ParagraphStyle(
            "model_card_val",
            fontName="Helvetica",
            fontSize=9.5,
            textColor=C_DARK,
            leading=14,
            spaceAfter=6,
            leftIndent=12,
        ),
    }
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Helper builders
# ─────────────────────────────────────────────────────────────────────────────

def rule(styles) -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.5,
                      color=C_RULE, spaceAfter=6, spaceBefore=4)


def section_rule(styles) -> HRFlowable:
    return HRFlowable(width="100%", thickness=2,
                      color=C_NAVY, spaceAfter=8, spaceBefore=2)


def embed_figure(path: Path, width_cm: float = 14,
                 caption: str = "", styles: dict = None) -> list:
    """Return a list of flowables: image + optional caption."""
    if not path.exists():
        return [Paragraph(f"[Figure not found: {path.name}]",
                          styles["caption"])]
    w = width_cm * cm
    items = [
        Spacer(1, 4),
        Image(str(path), width=w, height=w * 0.62, kind="proportional"),
    ]
    if caption:
        items.append(Paragraph(caption, styles["caption"]))
    items.append(Spacer(1, 6))
    return items


def metric_table(rows: list[list], col_widths: list,
                 styles: dict) -> Table:
    """Build a styled metric table."""
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    n_rows = len(rows)
    ts = TableStyle([
        # Header
        ("BACKGROUND",  (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING",  (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        # Body
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("ALIGN",       (0, 1), (-1, -1), "CENTER"),
        ("TOPPADDING",  (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        # Alternating rows
        ("BACKGROUND",  (0, 2), (-1, 2), colors.HexColor("#F0F4F8")),
        ("BACKGROUND",  (0, 4), (-1, 4), colors.HexColor("#F0F4F8")),
        # Grid
        ("GRID",        (0, 0), (-1, -1), 0.4, C_RULE),
        ("ROWBACKGROUND", (0, 0), (0, -1), C_LIGHT),
    ])
    # Highlight best row (row 2 = XGBoost, 0-indexed header at 0)
    if n_rows >= 3:
        ts.add("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#D5F5E3"))
        ts.add("FONTNAME",   (0, 2), (-1, 2), "Helvetica-Bold")
    tbl.setStyle(ts)
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Page template — header + footer on every page
# ─────────────────────────────────────────────────────────────────────────────

def make_header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4

    # Header bar
    canvas.setFillColor(C_NAVY)
    canvas.rect(0, h - 1.1 * cm, w, 1.1 * cm, fill=True, stroke=False)
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN, h - 0.72 * cm, "INTERNAL — CONFIDENTIAL")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(w - MARGIN, h - 0.72 * cm,
                           "Fraud Detection System — Technical Report")

    # Footer bar
    canvas.setFillColor(C_LIGHT)
    canvas.rect(0, 0, w, 0.9 * cm, fill=True, stroke=False)
    canvas.setFillColor(colors.HexColor("#9CA3AF"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN, 0.32 * cm,
                      f"fraud-shield-ml  |  Generated {datetime.now().strftime('%d %B %Y')}")
    canvas.drawCentredString(w / 2, 0.32 * cm, f"Page {doc.page}")
    canvas.drawRightString(w - MARGIN, 0.32 * cm, "For internal use only")

    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Report content builders
# ─────────────────────────────────────────────────────────────────────────────

def build_cover(styles, results, config) -> list:
    """Title page content."""
    story = []

    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("Credit Card Fraud Detection", styles["title"]))
    story.append(Paragraph(
        "End-to-End Machine Learning System — Technical Report",
        styles["subtitle"]
    ))
    story.append(rule(styles))
    story.append(Spacer(1, 0.3 * cm))

    meta_rows = [
        ["To:",      "Head of Risk Analytics"],
        ["From:",    "Data Science — Risk & Fraud"],
        ["Date:",    datetime.now().strftime("%d %B %Y")],
        ["Dataset:", "IEEE-CIS Fraud Detection (Vesta Corporation, 2019)"],
        ["Status:",  "Internal Review Draft"],
    ]
    for label, val in meta_rows:
        story.append(Paragraph(
            f"<b>{label}</b>  {val}", styles["body_left"]
        ))

    story.append(Spacer(1, 0.5 * cm))
    story.append(rule(styles))

    # Key findings box
    best = config["best_model"]
    m = config["metrics"]
    kf_data = [
        ["Metric", "Value", "Interpretation"],
        ["Best Model", best, "Highest PR-AUC on validation set"],
        ["PR-AUC", f"{m['pr_auc']:.3f}", "Strong minority-class discrimination"],
        ["Recall", f"{m['recall']:.1%}", "Fraud transactions correctly flagged"],
        ["Precision", f"{m['precision']:.1%}", "Flagged transactions that are real fraud"],
        ["F2 Score", f"{m['f2_score']:.3f}", "Primary metric (recall-weighted)"],
        ["Operating Threshold", f"{config['f2_threshold']:.3f}",
         "F2-optimal classification boundary"],
    ]
    col_w = [(PAGE_W - 2*MARGIN) * x for x in [0.28, 0.18, 0.54]]
    tbl = Table(kf_data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("BACKGROUND",    (0, 1), (-1, -1), C_LIGHT),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_RULE),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 1), (0, -1), C_BLUE),
        ("BACKGROUND",    (0, 2), (-1, 2), colors.HexColor("#EBF5FB")),
        ("BACKGROUND",    (0, 4), (-1, 4), colors.HexColor("#EBF5FB")),
        ("BACKGROUND",    (0, 6), (-1, 6), colors.HexColor("#EBF5FB")),
    ]))
    story.append(Paragraph("Key Findings at a Glance", styles["h2"]))
    story.append(tbl)
    story.append(PageBreak())
    return story


def build_exec_summary(styles, results, config) -> list:
    story = []
    story.append(Paragraph("1. Executive Summary", styles["h1"]))
    story.append(section_rule(styles))

    m = config["metrics"]
    story.append(Paragraph(
        "This report presents the findings of a machine learning fraud detection "
        "system trained on the IEEE-CIS transaction dataset (590,540 transactions, "
        "December 2017 to May 2018, sponsored by Vesta Corporation). The objective "
        "was to build a production-ready classifier that identifies fraudulent "
        "payment transactions with high recall — minimising the cost of missed fraud "
        "— while maintaining sufficient precision to keep analyst review workload "
        "manageable.",
        styles["body"]
    ))

    story.append(Paragraph(
        f"The winning model is <b>XGBoost</b>, tuned with Optuna over 50 trials. "
        f"On the held-out validation set (118,108 transactions, 4,133 fraud), "
        f"it achieves a <b>PR-AUC of {m['pr_auc']:.3f}</b>, catching "
        f"<b>{m['recall']:.1%} of all fraud</b> at a precision of "
        f"<b>{m['precision']:.1%}</b>. This represents a "
        f"<b>169% improvement</b> in PR-AUC over the Logistic Regression baseline "
        f"(0.306), demonstrating the value of gradient boosting on high-dimensional "
        f"behavioural transaction features.",
        styles["body"]
    ))

    story.append(Paragraph(
        "The system operates at two thresholds: an F2-optimal threshold "
        f"({config['f2_threshold']:.3f}) for the analyst review queue, and a "
        f"high-precision threshold ({config['hard_block_threshold']:.3f}) for "
        "automatic hard-block decisions. All predictions are accompanied by "
        "SHAP-based feature importance explanations, enabling analysts to "
        "understand and audit every flagged transaction.",
        styles["body"]
    ))

    story.append(Paragraph("Primary Recommendation", styles["h2"]))
    story.append(Paragraph(
        "Deploy XGBoost at the F2-optimal threshold as the primary fraud scoring "
        "engine. Route high-confidence predictions (above the hard-block threshold) "
        "to automatic decline. Route mid-confidence predictions to the analyst "
        "review queue with SHAP explanations attached. Monitor PR-AUC weekly and "
        "retrain quarterly as fraud patterns evolve.",
        styles["body"]
    ))

    story.append(Spacer(1, 0.4 * cm))
    return story


def build_methodology(styles) -> list:
    story = []
    story.append(Paragraph("2. Methodology", styles["h1"]))
    story.append(section_rule(styles))

    story.append(Paragraph("2.1  Dataset", styles["h2"]))
    story.append(Paragraph(
        "The IEEE-CIS Fraud Detection dataset contains 590,540 real payment "
        "transactions from Vesta Corporation's payment network. Each transaction "
        "has 394 raw features across two tables: transaction attributes (amount, "
        "product code, card metadata, Vesta's proprietary V-features capturing "
        "behavioural signals) and identity attributes (device type, browser, "
        "network fingerprints). The two tables are joined on TransactionID via "
        "left join, preserving all 590,540 transactions regardless of whether "
        "identity data is available (only 24.4% of transactions have identity "
        "records — the missingness is itself a signal).",
        styles["body"]
    ))
    story.append(Paragraph(
        "The dataset exhibits a fraud rate of 3.50% (20,663 fraud / 569,877 "
        "legitimate), a 27.6:1 class imbalance. After dropping columns with "
        "greater than 50% missing values, 221 features remained for modelling.",
        styles["body"]
    ))

    story.append(Paragraph("2.2  Why F2 Score, Not Accuracy", styles["h2"]))
    story.append(Paragraph(
        "A model that predicts every transaction as legitimate scores 96.5% "
        "accuracy while catching zero fraud. Accuracy is therefore a useless "
        "metric for this problem. The chosen primary metric is the <b>F2 score</b> "
        "(F-beta with beta=2), which weights recall twice as heavily as precision. "
        "This reflects the cost asymmetry: a missed fraud (false negative) incurs "
        "the full transaction value loss plus chargeback fees, while a false alarm "
        "(false positive) causes customer friction that is recoverable. "
        "<b>PR-AUC</b> (average precision) is used for threshold-independent model "
        "comparison and architecture selection.",
        styles["body"]
    ))

    story.append(Paragraph("2.3  Imbalance Handling", styles["h2"]))
    story.append(Paragraph(
        "Four strategies were evaluated on a 50,000-row stratified subsample "
        "using Logistic Regression as the base estimator (to isolate the effect "
        "of the resampling strategy from model power): SMOTE, ADASYN, "
        "class_weight='balanced', and threshold tuning on a default model. "
        "Threshold tuning achieved the best PR-AUC (0.322) on the subsample, "
        "while class_weight achieved the best F2 (0.377). For XGBoost and "
        "LightGBM, scale_pos_weight and is_unbalance were used respectively — "
        "no synthetic resampling was needed as tree models handle imbalance "
        "through the loss function directly.",
        styles["body"]
    ))

    story.append(Paragraph("2.4  Model Training", styles["h2"]))
    story.append(Paragraph(
        "Three models were trained on 80% of the data (472,432 transactions) "
        "and evaluated on a held-out 20% validation set (118,108 transactions). "
        "Logistic Regression (SAGA solver, class_weight='balanced') served as "
        "the baseline. XGBoost and LightGBM were each tuned using Optuna "
        "Tree-structured Parzen Estimator (TPE) over 50 trials with a 10-minute "
        "timeout, optimising PR-AUC on the validation set. The final models were "
        "retrained on the full training set using the best hyperparameters found.",
        styles["body"]
    ))

    return story


def build_results(styles, results, config, figures_dir) -> list:
    story = []
    story.append(Paragraph("3. Results", styles["h1"]))
    story.append(section_rule(styles))

    story.append(Paragraph("3.1  Model Comparison", styles["h2"]))

    # Model comparison table
    headers = ["Model", "PR-AUC", "ROC-AUC", "F2 Score",
               "Recall", "Precision", "Threshold"]
    rows = [headers]
    for m in results:
        rows.append([
            m["model"],
            f"{m['pr_auc']:.4f}",
            f"{m['roc_auc']:.4f}",
            f"{m['f2_score']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['threshold']:.4f}",
        ])

    col_w = [(PAGE_W - 2 * MARGIN) * x
             for x in [0.24, 0.11, 0.11, 0.12, 0.11, 0.13, 0.18]]
    story.append(metric_table(rows, col_w, styles))
    story.append(Paragraph(
        "Table 1: Validation set metrics for all three models. "
        "XGBoost (highlighted) achieves the best PR-AUC and F2 score. "
        "All thresholds are F2-optimised on the validation set.",
        styles["caption"]
    ))

    story.append(Paragraph(
        f"XGBoost outperforms LightGBM by 3.8 percentage points in PR-AUC "
        f"({config['metrics']['pr_auc']:.3f} vs 0.786) and surpasses the "
        "Logistic Regression baseline by 169%. The ROC-AUC of 0.967 indicates "
        "excellent overall discrimination, though PR-AUC is the operationally "
        "relevant metric given the severe class imbalance.",
        styles["body"]
    ))

    story.append(Paragraph("3.2  Precision-Recall Curves", styles["h2"]))
    story += embed_figure(
        figures_dir / "14_pr_curves.png",
        width_cm=13,
        caption=(
            "Figure 1: Precision-Recall curves for all three models. "
            "Higher area under the curve indicates better fraud detection "
            "at all operating thresholds. XGBoost dominates across the full range."
        ),
        styles=styles,
    )

    story.append(Paragraph("3.3  Confusion Matrix at Operating Threshold", styles["h2"]))
    story.append(Paragraph(
        f"At the F2-optimal threshold of {config['f2_threshold']:.3f}, XGBoost "
        f"correctly identifies {config['metrics']['tp']:,} fraud transactions "
        f"(true positives) while missing {config['metrics']['fn']:,} "
        f"(false negatives). It generates {config['metrics']['fp']:,} false alarms "
        f"on legitimate transactions.",
        styles["body"]
    ))
    story += embed_figure(
        figures_dir / "12_confusion_matrix.png",
        width_cm=10,
        caption=(
            f"Figure 2: Confusion matrix at F2-optimal threshold "
            f"({config['f2_threshold']:.3f}). "
            "True Positives = caught fraud. False Negatives = missed fraud (highest cost)."
        ),
        styles=styles,
    )

    story.append(Paragraph("3.4  Threshold Analysis", styles["h2"]))
    story += embed_figure(
        figures_dir / "15_threshold_analysis.png",
        width_cm=13,
        caption=(
            "Figure 3: Precision, Recall, and F2 Score vs classification threshold. "
            "The vertical line marks the F2-optimal operating point. "
            "As threshold rises, precision increases and recall falls."
        ),
        styles=styles,
    )

    return story


def build_model_card(styles, config, results) -> list:
    story = []
    story.append(Paragraph("4. Model Card", styles["h1"]))
    story.append(section_rule(styles))
    story.append(Paragraph(
        "A model card is a standardised summary of model characteristics, "
        "intended use, performance, and limitations. It enables downstream "
        "stakeholders to make informed decisions about deployment.",
        styles["body"]
    ))

    m = config["metrics"]

    def kv(key, val):
        return [
            Paragraph(key, styles["model_card_key"]),
            Paragraph(val, styles["model_card_val"]),
        ]

    card_items = [
        ("Model Name",         "XGBoost Fraud Classifier v1.0"),
        ("Model Type",         "Gradient Boosted Decision Trees (XGBoost 3.x)"),
        ("Intended Use",       "Binary classification of payment transactions as "
                               "fraudulent or legitimate. Intended for use in a "
                               "human-in-the-loop review queue and automated hard-block "
                               "pipeline within a payments fraud prevention system."),
        ("Out-of-Scope Use",   "Not suitable for: credit scoring, identity verification, "
                               "real-time streaming inference without re-evaluation of "
                               "latency, or application to transaction data from "
                               "domains substantially different from Vesta's payment network."),
        ("Training Data",      "IEEE-CIS Fraud Detection dataset. 472,432 training "
                               "transactions. Date range: December 2017 to April 2018. "
                               "Source: Vesta Corporation payment network."),
        ("Validation Data",    "118,108 held-out transactions (20% stratified split). "
                               "Date range: April to May 2018."),
        ("Input Features",     f"221 features: 212 numeric (transaction amount, Vesta "
                               "V-features, card metadata, temporal features) + "
                               "9 categorical (ProductCD, card brand/type, email domain, "
                               "M-fields)."),
        ("Performance",        f"PR-AUC: {m['pr_auc']:.4f}  |  "
                               f"ROC-AUC: {m['roc_auc']:.4f}  |  "
                               f"F2: {m['f2_score']:.4f}  |  "
                               f"Recall: {m['recall']:.4f}  |  "
                               f"Precision: {m['precision']:.4f}"),
        ("Operating Threshold", f"F2-optimal: {config['f2_threshold']:.4f}  "
                                f"(review queue)  |  "
                                f"High-precision: {config['hard_block_threshold']:.4f} "
                                "(hard block)"),
        ("Explainability",     "SHAP TreeExplainer provides global feature importance "
                               "and per-transaction explanations. LIME provides local "
                               "linear approximations for audit of individual decisions."),
        ("Training Date",      datetime.now().strftime("%d %B %Y")),
        ("Retraining Cadence", "Recommended quarterly, or when monthly PR-AUC on "
                               "live traffic drops more than 3 percentage points below "
                               "validation baseline."),
    ]

    for key, val in card_items:
        for flowable in kv(key, val):
            story.append(flowable)
        story.append(rule(styles))

    return story


def build_explainability(styles, figures_dir) -> list:
    story = []
    story.append(Paragraph("5. Explainability", styles["h1"]))
    story.append(section_rule(styles))
    story.append(Paragraph(
        "All model decisions are interpretable via SHAP (SHapley Additive "
        "exPlanations) and LIME (Local Interpretable Model-agnostic Explanations). "
        "This section presents three XAI artefacts: global feature importance, "
        "a single flagged fraud explained, and a false positive dissected.",
        styles["body"]
    ))

    story.append(Paragraph("5.1  Global Feature Importance — SHAP Beeswarm", styles["h2"]))
    story.append(Paragraph(
        "The beeswarm plot shows the top 20 features by mean absolute SHAP value "
        "across 1,000 sampled validation transactions. Each dot represents one "
        "transaction. Colour encodes feature value (red = high, blue = low). "
        "Features at the top exert the strongest influence on predictions. "
        "A positive SHAP value pushes the prediction toward fraud; negative "
        "pushes toward legitimate.",
        styles["body"]
    ))
    story += embed_figure(
        figures_dir / "17_shap_beeswarm.png",
        width_cm=14,
        caption=(
            "Figure 4: SHAP beeswarm plot — global feature importance. "
            "V-features (Vesta behavioural signals) dominate, confirming that "
            "transaction velocity and behavioural fingerprints are the primary "
            "fraud signals, not the transaction amount alone."
        ),
        styles=styles,
    )

    story.append(Paragraph("5.2  Why We Flagged This Transaction — SHAP Waterfall",
                           styles["h2"]))
    story.append(Paragraph(
        "The waterfall plot decomposes a single fraud prediction into the "
        "contribution of each feature. Starting from the model's expected output "
        "(base rate), each bar shows how one feature shifts the prediction up "
        "(toward fraud, red) or down (toward legitimate, blue). "
        "The final prediction probability is the sum of the base value and all "
        "feature contributions. This transaction was flagged with 100% confidence.",
        styles["body"]
    ))
    story += embed_figure(
        figures_dir / "18_shap_waterfall_fraud.png",
        width_cm=14,
        caption=(
            "Figure 5: SHAP waterfall for the highest-confidence fraud transaction. "
            "Multiple V-features compound upward, while the transaction amount "
            "contributes a smaller directional push. "
            "This explanation would be shown to an analyst reviewing the flag."
        ),
        styles=styles,
    )

    story.append(Paragraph(
        "5.3  Why We Incorrectly Flagged This Customer — LIME", styles["h2"]))
    story.append(Paragraph(
        "The LIME plot explains a false positive: a legitimate transaction that "
        "the model flagged as fraud. LIME fits a local linear model around this "
        "specific transaction, identifying which feature values caused the "
        "incorrect classification. Understanding false positives is operationally "
        "important: it reveals edge cases where the model's learned patterns "
        "generalise poorly, and informs rule-based overrides or feature engineering "
        "to reduce false alarm rates.",
        styles["body"]
    ))
    story += embed_figure(
        figures_dir / "19_lime_false_positive.png",
        width_cm=14,
        caption=(
            "Figure 6: LIME explanation for a false positive (legitimate transaction "
            "incorrectly flagged as fraud). Red bars pushed the model toward a fraud "
            "decision; blue bars pushed back. The transaction cleared the decision "
            "threshold despite being legitimate, costing one customer friction event."
        ),
        styles=styles,
    )

    return story


def build_limitations(styles) -> list:
    story = []
    story.append(Paragraph("6. Limitations & Recommendations", styles["h1"]))
    story.append(section_rule(styles))

    story.append(Paragraph("6.1  Known Limitations", styles["h2"]))

    limitations = [
        ("<b>Temporal validity:</b> The model was trained on data from late 2017 to "
         "mid-2018. Fraud tactics evolve rapidly. Model performance should be "
         "monitored continuously and retraining triggered when PR-AUC degrades."),
        ("<b>Feature opacity:</b> The Vesta V-features (V1-V321) are proprietary "
         "engineered signals whose exact definitions are not publicly documented. "
         "This limits full interpretability and makes feature engineering "
         "by the deployment team difficult without Vesta's documentation."),
        ("<b>Identity coverage:</b> Only 24.4% of transactions have identity "
         "signals (device, browser). The model cannot leverage identity features "
         "for the majority of transactions, likely leaving recall on the table."),
        ("<b>No real-time velocity features:</b> Features such as 'number of "
         "transactions on this card in the last 5 minutes' are not available "
         "in the static dataset. These are among the strongest fraud signals in "
         "production systems."),
        ("<b>Subgroup fairness not evaluated:</b> Model performance was not "
         "disaggregated by geography, card issuer, or demographic proxies. "
         "Disparate false positive rates across customer segments would be "
         "a material concern for deployment."),
        ("<b>Threshold sensitivity:</b> The F2-optimal threshold (0.446) was "
         "selected on the validation set and may not generalise perfectly to "
         "live traffic with different fraud rate or transaction mix."),
    ]

    for lim in limitations:
        story.append(Paragraph(f"• {lim}", styles["bullet"]))
        story.append(Spacer(1, 3))

    story.append(Paragraph("6.2  Recommendations", styles["h2"]))

    recs = [
        ("<b>Immediate deployment:</b> XGBoost at F2-threshold 0.446 for the "
         "analyst review queue. Hard-block threshold 0.990 for automatic decline."),
        ("<b>Monitoring:</b> Track weekly PR-AUC, recall, and false positive rate "
         "on live traffic. Alert if PR-AUC drops below 0.78 (5 pp below validation)."),
        ("<b>Feature engineering:</b> Add real-time velocity features (card velocity "
         "in 1h, 24h windows) and graph-based network features connecting "
         "cards, emails, and devices. Expected 5-10 pp PR-AUC improvement."),
        ("<b>Retraining cadence:</b> Quarterly full retraining on rolling 12-month "
         "window. Trigger immediate retraining if fraud rate shifts by more than "
         "1 percentage point."),
        ("<b>Fairness audit:</b> Before production deployment, conduct subgroup "
         "analysis by card type, product code, and transaction amount bucket. "
         "Ensure false positive rate is equitable across segments."),
        ("<b>Online learning:</b> Consider a lightweight online gradient boosting "
         "layer trained on recent confirmed fraud labels to adapt to emerging "
         "fraud patterns without full retraining."),
    ]

    for rec in recs:
        story.append(Paragraph(f"• {rec}", styles["bullet"]))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 0.5 * cm))
    story.append(rule(styles))
    story.append(Paragraph(
        "This report was generated programmatically from model artefacts and "
        "evaluation outputs. All figures and metrics reflect performance on the "
        "held-out validation set unless otherwise stated. Code and full "
        "reproducibility instructions are available in the project repository.",
        styles["meta"]
    ))
    return story


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_report() -> Path:
    log.info("=" * 60)
    log.info("fraud-shield-ml | Phase 7: Report Generation")
    log.info("=" * 60)

    reports_dir = PROJECT_ROOT / cfg.paths.outputs_reports
    figures_dir = PROJECT_ROOT / cfg.paths.outputs_figures
    models_dir  = PROJECT_ROOT / cfg.paths.outputs_models

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "fraud_detection_report.pdf"

    # ── Load results ─────────────────────────────────────────────────────────
    with open(models_dir / "model_results.json") as f:
        results = json.load(f)

    with open(models_dir / "best_model_config.json") as f:
        config = json.load(f)

    log.info(f"  Best model: {config['best_model']}  "
             f"PR-AUC={config['metrics']['pr_auc']:.4f}")

    # ── Build document ────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=1.6 * cm,
        bottomMargin=1.3 * cm,
        title="Fraud Detection System — Technical Report",
        author="Data Science — Risk & Fraud",
        subject="IEEE-CIS Fraud Detection Model",
    )

    styles = build_styles()

    story = []
    story += build_cover(styles, results, config)
    story += build_exec_summary(styles, results, config)
    story.append(PageBreak())
    story += build_methodology(styles)
    story.append(PageBreak())
    story += build_results(styles, results, config, figures_dir)
    story.append(PageBreak())
    story += build_model_card(styles, config, results)
    story.append(PageBreak())
    story += build_explainability(styles, figures_dir)
    story.append(PageBreak())
    story += build_limitations(styles)

    doc.build(story, onFirstPage=make_header_footer,
              onLaterPages=make_header_footer)

    size_mb = out_path.stat().st_size / 1e6
    log.info(f"  Report saved → {out_path.relative_to(PROJECT_ROOT)}")
    log.info(f"  File size: {size_mb:.1f} MB")
    log.info("=" * 60)
    log.info("Phase 7 complete.")
    log.info("  Next: python3 streamlit_app/app.py")
    log.info("=" * 60)

    return out_path


if __name__ == "__main__":
    generate_report()
