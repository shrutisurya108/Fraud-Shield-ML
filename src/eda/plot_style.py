"""
fraud-shield-ml | src/eda/plot_style.py
────────────────────────────────────────
Shared matplotlib / seaborn style applied to every EDA figure.
Import apply_style() at the top of any plotting module.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on all OS, no display needed

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Palette ───────────────────────────────────────────────────────────────────
FRAUD_COLOR  = "#E63946"   # vivid red   — fraud / positive class
LEGIT_COLOR  = "#457B9D"   # steel blue  — legitimate / negative class
ACCENT       = "#F4A261"   # amber       — highlights, secondary bars
NEUTRAL      = "#6B7280"   # slate grey  — annotations, gridlines
BG           = "#F8F9FA"   # off-white   — figure background
PALETTE      = [LEGIT_COLOR, FRAUD_COLOR]

# ── Figure defaults ───────────────────────────────────────────────────────────
DPI      = 150          # 150 DPI — crisp on screen and in PDF, not overkill
FIGSIZE  = (12, 6)      # default wide layout
FONTSIZE = 11


def apply_style() -> None:
    """
    Apply project-wide matplotlib / seaborn style.
    Call once at the top of each plotting script.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)

    plt.rcParams.update({
        # Figure
        "figure.facecolor":     BG,
        "figure.dpi":           DPI,
        "figure.autolayout":    True,

        # Axes
        "axes.facecolor":       BG,
        "axes.edgecolor":       NEUTRAL,
        "axes.linewidth":       0.8,
        "axes.labelsize":       FONTSIZE,
        "axes.titlesize":       FONTSIZE + 2,
        "axes.titleweight":     "bold",
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # Grid
        "grid.color":           "#D1D5DB",
        "grid.linewidth":       0.6,
        "grid.linestyle":       "--",

        # Ticks
        "xtick.labelsize":      FONTSIZE - 1,
        "ytick.labelsize":      FONTSIZE - 1,
        "xtick.color":          NEUTRAL,
        "ytick.color":          NEUTRAL,

        # Legend
        "legend.fontsize":      FONTSIZE - 1,
        "legend.framealpha":    0.85,
        "legend.edgecolor":     "#D1D5DB",

        # Font
        "font.family":          "DejaVu Sans",

        # Save
        "savefig.dpi":          150,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    BG,
    })


def save_fig(fig: plt.Figure, path, tight: bool = True) -> None:
    """Save figure and close it to free memory."""
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
