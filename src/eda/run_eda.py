"""
fraud-shield-ml | src/eda/run_eda.py
─────────────────────────────────────
Phase 3: Exploratory Data Analysis Pipeline

Produces 8 publication-quality figures saved to outputs/figures/:
  01_class_imbalance.png
  02_amount_distribution.png
  03_fraud_by_hour.png
  04_fraud_by_dow.png
  05_fraud_over_time.png
  06_missing_values.png
  07_correlation_heatmap.png
  08_categorical_fraud_rates.png

Run:
    python3 src/eda/run_eda.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config_loader import cfg
from src.eda.plot_style import (
    apply_style, save_fig,
    FRAUD_COLOR, LEGIT_COLOR, ACCENT, NEUTRAL, BG, PALETTE, FIGSIZE, DPI
)

log = get_logger("eda")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _label_bars(ax, fmt="{:.0f}", fontsize=10, color="white", offset_frac=0.02):
    """Place value labels inside/on top of bars."""
    y_max = max(p.get_height() for p in ax.patches if p.get_height() > 0)
    for patch in ax.patches:
        h = patch.get_height()
        if h == 0:
            continue
        x = patch.get_x() + patch.get_width() / 2
        # Put label inside if bar is tall enough, otherwise above
        if h > y_max * 0.15:
            y   = h * (1 - offset_frac * 4)
            va  = "top"
            col = color
        else:
            y   = h + y_max * offset_frac
            va  = "bottom"
            col = NEUTRAL
        ax.text(x, y, fmt.format(h), ha="center", va=va,
                fontsize=fontsize, fontweight="bold", color=col)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Class Imbalance
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_imbalance(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Two-panel figure:
      Left  — raw transaction counts (legitimate vs fraud)
      Right — percentage breakdown with annotation
    """
    log.info("Plotting class imbalance …")

    counts = df["isFraud"].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]
    pcts   = (counts / counts.sum() * 100).values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: counts
    bars = axes[0].bar(labels, counts.values, color=PALETTE, width=0.5,
                       edgecolor="white", linewidth=1.2)
    axes[0].set_title("Transaction Count by Class")
    axes[0].set_ylabel("Number of Transactions")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}K"))
    _label_bars(axes[0], fmt="{:,.0f}", fontsize=9)

    # Right: percentages
    bars2 = axes[1].bar(labels, pcts, color=PALETTE, width=0.5,
                        edgecolor="white", linewidth=1.2)
    axes[1].set_title("Class Distribution (%)")
    axes[1].set_ylabel("Percentage of Transactions")
    axes[1].set_ylim(0, 110)
    for bar, pct in zip(bars2, pcts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{pct:.2f}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=NEUTRAL
        )

    # Annotation box
    fraud_n = int(counts[1])
    legit_n = int(counts[0])
    ratio   = legit_n / fraud_n
    axes[1].text(
        0.97, 0.95,
        f"Imbalance ratio ≈ {ratio:.0f}:1\n"
        f"Fraud: {fraud_n:,} ({pcts[1]:.2f}%)\n"
        f"Legit: {legit_n:,} ({pcts[0]:.2f}%)",
        transform=axes[1].transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=NEUTRAL, alpha=0.9)
    )

    fig.suptitle(
        "Class Imbalance — IEEE-CIS Fraud Detection Dataset",
        fontsize=14, fontweight="bold", y=1.02
    )

    out = out_dir / "01_class_imbalance.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Transaction Amount Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_amount_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Log-scale overlapping histograms of TransactionAmt for fraud vs legit.
    Inset table shows median / mean / 95th pct for each class.
    """
    log.info("Plotting transaction amount distribution …")

    fraud = df[df["isFraud"] == 1]["TransactionAmt"].clip(upper=10_000)
    legit = df[df["isFraud"] == 0]["TransactionAmt"].clip(upper=10_000)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    bins = np.logspace(np.log10(0.5), np.log10(10_001), 80)

    ax.hist(legit, bins=bins, alpha=0.65, color=LEGIT_COLOR,
            label=f"Legitimate (n={len(legit):,})", density=True)
    ax.hist(fraud, bins=bins, alpha=0.70, color=FRAUD_COLOR,
            label=f"Fraud (n={len(fraud):,})", density=True)

    ax.set_xscale("log")
    ax.set_xlabel("Transaction Amount (USD, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction Amount Distribution — Fraud vs Legitimate")
    ax.legend(framealpha=0.9)

    # Stats table inset
    stats = pd.DataFrame({
        "Class":   ["Legitimate", "Fraud"],
        "Median":  [f"${legit.median():.0f}", f"${fraud.median():.0f}"],
        "Mean":    [f"${legit.mean():.0f}",   f"${fraud.mean():.0f}"],
        "95th %":  [f"${legit.quantile(.95):.0f}", f"${fraud.quantile(.95):.0f}"],
    })
    tbl = ax.table(
        cellText=stats.values,
        colLabels=stats.columns,
        cellLoc="center", loc="upper right",
        bbox=[0.60, 0.60, 0.38, 0.30]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if r == 0:
            cell.set_facecolor(NEUTRAL)
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#EBF5FB")
        else:
            cell.set_facecolor("#FDEDEC")

    out = out_dir / "02_amount_distribution.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Fraud Rate by Hour of Day
# ─────────────────────────────────────────────────────────────────────────────

def plot_fraud_by_hour(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Dual-axis plot:
      Bar  — total transaction volume per hour (context)
      Line — fraud rate per hour (the signal)
    """
    log.info("Plotting fraud rate by hour of day …")

    hourly = df.groupby("tx_hour").agg(
        total=("isFraud", "count"),
        fraud=("isFraud", "sum")
    )
    hourly["fraud_rate"] = hourly["fraud"] / hourly["total"] * 100
    overall_rate = df["isFraud"].mean() * 100

    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    # Bar — volume
    ax1.bar(hourly.index, hourly["total"], color=LEGIT_COLOR,
            alpha=0.4, label="Transaction Volume", zorder=2)
    ax1.set_xlabel("Hour of Day (0 = midnight)")
    ax1.set_ylabel("Transaction Volume", color=LEGIT_COLOR)
    ax1.tick_params(axis="y", labelcolor=LEGIT_COLOR)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}K"))
    ax1.set_xticks(range(0, 24, 2))

    # Line — fraud rate
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly["fraud_rate"], color=FRAUD_COLOR,
             linewidth=2.5, marker="o", markersize=5,
             label="Fraud Rate (%)", zorder=3)
    ax2.axhline(overall_rate, color=FRAUD_COLOR, linestyle="--",
                linewidth=1.2, alpha=0.6,
                label=f"Overall rate ({overall_rate:.2f}%)")
    ax2.set_ylabel("Fraud Rate (%)", color=FRAUD_COLOR)
    ax2.tick_params(axis="y", labelcolor=FRAUD_COLOR)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", framealpha=0.9)

    ax1.set_title("Fraud Rate & Transaction Volume by Hour of Day")

    # Shade night hours
    for span in [(0, 6), (22, 24)]:
        ax1.axvspan(span[0] - 0.5, span[1] - 0.5,
                    alpha=0.07, color=FRAUD_COLOR, zorder=0)

    out = out_dir / "03_fraud_by_hour.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Fraud Rate by Day of Week
# ─────────────────────────────────────────────────────────────────────────────

def plot_fraud_by_dow(df: pd.DataFrame, out_dir: Path) -> Path:
    """Fraud rate and volume by day of week."""
    log.info("Plotting fraud rate by day of week …")

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df.groupby("tx_day_of_week").agg(
        total=("isFraud", "count"),
        fraud=("isFraud", "sum")
    )
    dow["fraud_rate"] = dow["fraud"] / dow["total"] * 100
    overall_rate = df["isFraud"].mean() * 100

    fig, ax1 = plt.subplots(figsize=(10, 5))

    colors = [FRAUD_COLOR if r > overall_rate else LEGIT_COLOR
              for r in dow["fraud_rate"].values]

    ax1.bar(range(len(dow)), dow["total"], color=LEGIT_COLOR,
            alpha=0.35, label="Transaction Volume")
    ax1.set_xticks(range(len(dow)))
    ax1.set_xticklabels(day_names)
    ax1.set_ylabel("Transaction Volume", color=LEGIT_COLOR)
    ax1.tick_params(axis="y", labelcolor=LEGIT_COLOR)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}K"))

    ax2 = ax1.twinx()
    ax2.plot(range(len(dow)), dow["fraud_rate"], color=FRAUD_COLOR,
             linewidth=2.5, marker="D", markersize=7, zorder=3)
    ax2.axhline(overall_rate, color=FRAUD_COLOR, linestyle="--",
                linewidth=1.2, alpha=0.6,
                label=f"Overall rate ({overall_rate:.2f}%)")

    for i, (idx, row) in enumerate(dow.iterrows()):
        ax2.annotate(
            f"{row['fraud_rate']:.2f}%",
            xy=(i, row["fraud_rate"]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8.5, color=FRAUD_COLOR, fontweight="bold"
        )

    ax2.set_ylabel("Fraud Rate (%)", color=FRAUD_COLOR)
    ax2.tick_params(axis="y", labelcolor=FRAUD_COLOR)
    ax1.set_title("Fraud Rate & Transaction Volume by Day of Week")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    out = out_dir / "04_fraud_by_dow.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Fraud Volume Over Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_fraud_over_time(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Weekly fraud count and fraud rate over the dataset's 6-month window.
    Uses TransactionDT converted to week number.
    """
    log.info("Plotting fraud volume over time …")

    from src.ingestion.load_data import REFERENCE_DATE

    df = df.copy()
    df["_dt"] = REFERENCE_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["_week"] = df["_dt"].dt.to_period("W").dt.start_time

    weekly = df.groupby("_week").agg(
        total=("isFraud", "count"),
        fraud=("isFraud", "sum")
    ).reset_index()
    weekly["fraud_rate"] = weekly["fraud"] / weekly["total"] * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Top: weekly fraud count
    ax1.fill_between(weekly["_week"], weekly["fraud"],
                     color=FRAUD_COLOR, alpha=0.55, linewidth=0)
    ax1.plot(weekly["_week"], weekly["fraud"],
             color=FRAUD_COLOR, linewidth=1.8)
    ax1.set_ylabel("Weekly Fraud Count")
    ax1.set_title("Fraud Trends Over Time (Weekly Aggregation)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}"))

    # Bottom: fraud rate
    overall = df["isFraud"].mean() * 100
    ax2.fill_between(weekly["_week"], weekly["fraud_rate"],
                     color=ACCENT, alpha=0.55, linewidth=0)
    ax2.plot(weekly["_week"], weekly["fraud_rate"],
             color=ACCENT, linewidth=1.8)
    ax2.axhline(overall, color=NEUTRAL, linestyle="--",
                linewidth=1.2, alpha=0.7,
                label=f"Overall rate ({overall:.2f}%)")
    ax2.set_ylabel("Weekly Fraud Rate (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")

    fig.autofmt_xdate(rotation=30)

    out = out_dir / "05_fraud_over_time.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Missing Value Audit
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Horizontal bar chart of missing rate per column, grouped by column family.
    Shows top 40 columns with highest missingness for readability.
    """
    log.info("Plotting missing value audit …")

    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    # Top 40 only for readability — drop columns with 0% missing
    missing_pct = missing_pct[missing_pct > 0].head(40)

    if missing_pct.empty:
        log.info("  No missing values found — skipping missing value plot")
        # Create a placeholder
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No missing values in processed dataset",
                ha="center", va="center", fontsize=14, color=NEUTRAL)
        out = out_dir / "06_missing_values.png"
        save_fig(fig, out)
        return out

    # Colour by column family
    def _col_color(col):
        if col.startswith("V"):       return LEGIT_COLOR
        if col.startswith("id_"):     return ACCENT
        if col.startswith("card"):    return "#2ECC71"
        if col.startswith("addr"):    return "#9B59B6"
        if col.startswith("D"):       return "#E67E22"
        if col.startswith("C"):       return "#1ABC9C"
        if col.startswith("M"):       return "#F39C12"
        return NEUTRAL

    colors = [_col_color(c) for c in missing_pct.index]

    fig, ax = plt.subplots(figsize=(12, max(6, len(missing_pct) * 0.28)))

    bars = ax.barh(missing_pct.index[::-1], missing_pct.values[::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    ax.axvline(50, color=FRAUD_COLOR, linestyle="--", linewidth=1.5,
               alpha=0.8, label="50% threshold (drop boundary)")
    ax.set_xlabel("Missing Value Rate (%)")
    ax.set_title(
        "Missing Value Audit — Top 40 Columns\n"
        "(Colour = column family: V=blue, id=amber, card=green, D=orange, C=teal, M=yellow)",
        fontsize=11
    )
    ax.legend(loc="lower right", fontsize=9)

    # Value labels on bars
    for bar, val in zip(bars, missing_pct.values[::-1]):
        if val > 3:
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=7.5, color=NEUTRAL)

    out = out_dir / "06_missing_values.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Correlation heatmap of the top N numeric features most correlated
    with isFraud. Sampled to 50K rows for speed on 8GB RAM.
    """
    log.info("Plotting correlation heatmap …")

    top_n = cfg.eda.top_n_features  # 30

    # Sample for speed — correlation is stable at 50K rows
    sample = df.sample(n=min(50_000, len(df)), random_state=42)

    # Select numeric columns only
    numeric = sample.select_dtypes(include=[np.number])

    # Get top_n features most correlated with target
    target_corr = numeric.corr()["isFraud"].abs().drop("isFraud")
    top_features = target_corr.nlargest(top_n).index.tolist()
    top_features = ["isFraud"] + top_features

    corr_matrix = numeric[top_features].corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True   # upper triangle masked

    sns.heatmap(
        corr_matrix,
        mask=mask,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=top_n <= 20,          # only annotate if small enough to read
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        linecolor="#E5E7EB",
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
        square=True,
    )

    ax.set_title(
        f"Correlation Heatmap — Top {top_n} Features by |corr| with isFraud\n"
        f"(Sampled 50K rows · Lower triangle only)",
        fontsize=12, pad=15
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    out = out_dir / "07_correlation_heatmap.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 8 — Categorical Fraud Rates
# ─────────────────────────────────────────────────────────────────────────────

def plot_categorical_fraud_rates(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    2×2 grid of bar charts showing fraud rate by key categorical features:
      ProductCD, card4 (card brand), card6 (card type), P_emaildomain
    """
    log.info("Plotting categorical fraud rates …")

    overall_rate = df["isFraud"].mean() * 100

    # Define which categoricals to plot and their display names
    cats = [
        ("ProductCD",      "Product Code"),
        ("card4",          "Card Brand"),
        ("card6",          "Card Type"),
        ("P_emaildomain",  "Purchaser Email Domain (top 10)"),
    ]

    # Filter to cols that actually exist post-processing
    cats = [(c, lbl) for c, lbl in cats if c in df.columns]

    n_plots = len(cats)
    ncols = 2
    nrows = (n_plots + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, (col, label) in enumerate(cats):
        ax = axes[i]

        grp = (
            df.groupby(col)["isFraud"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "fraud_rate", "count": "n"})
            .sort_values("fraud_rate", ascending=False)
        )

        # Limit email domains to top 10 by volume
        if col == "P_emaildomain":
            top10 = df[col].value_counts().head(10).index
            grp = grp.loc[grp.index.isin(top10)].sort_values(
                "fraud_rate", ascending=False)

        # Drop NaN index
        grp = grp[grp.index.notna()]

        bar_colors = [FRAUD_COLOR if r * 100 > overall_rate else LEGIT_COLOR
                      for r in grp["fraud_rate"].values]

        bars = ax.bar(
            range(len(grp)),
            grp["fraud_rate"] * 100,
            color=bar_colors, edgecolor="white", linewidth=0.8
        )
        ax.axhline(overall_rate, color=NEUTRAL, linestyle="--",
                   linewidth=1.2, alpha=0.7,
                   label=f"Overall ({overall_rate:.2f}%)")

        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp.index.astype(str), rotation=30,
                           ha="right", fontsize=9)
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_title(f"Fraud Rate by {label}")
        ax.legend(fontsize=8)

        # Count labels below x-axis (volume context)
        for j, (bar, (idx, row)) in enumerate(zip(bars, grp.iterrows())):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                -overall_rate * 0.4,
                f"n={row['n']:,}",
                ha="center", va="top", fontsize=7,
                color=NEUTRAL, rotation=0
            )

    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Fraud Rate by Key Categorical Features",
        fontsize=14, fontweight="bold", y=1.01
    )

    out = out_dir / "08_categorical_fraud_rates.png"
    save_fig(fig, out)
    log.info(f"  ✓ Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EDA Summary — print to log
# ─────────────────────────────────────────────────────────────────────────────

def log_eda_summary(df: pd.DataFrame) -> dict:
    """Compute and log key EDA statistics. Returns dict for downstream use."""
    log.info("Computing EDA summary statistics …")

    fraud   = df[df["isFraud"] == 1]
    legit   = df[df["isFraud"] == 0]
    missing = df.isnull().mean()

    summary = {
        "n_total":            len(df),
        "n_fraud":            len(fraud),
        "n_legit":            len(legit),
        "fraud_rate":         df["isFraud"].mean(),
        "imbalance_ratio":    len(legit) / len(fraud),
        "n_features":         df.shape[1] - 1,
        "n_features_missing": (missing > 0).sum(),
        "avg_fraud_amount":   fraud["TransactionAmt"].mean() if "TransactionAmt" in df.columns else None,
        "avg_legit_amount":   legit["TransactionAmt"].mean() if "TransactionAmt" in df.columns else None,
        "peak_fraud_hour":    df.groupby("tx_hour")["isFraud"].mean().idxmax() if "tx_hour" in df.columns else None,
    }

    log.info(f"  Total transactions:     {summary['n_total']:,}")
    log.info(f"  Fraud transactions:     {summary['n_fraud']:,} ({summary['fraud_rate']:.2%})")
    log.info(f"  Legitimate:             {summary['n_legit']:,}")
    log.info(f"  Imbalance ratio:        {summary['imbalance_ratio']:.1f}:1")
    log.info(f"  Features (post-clean):  {summary['n_features']}")
    log.info(f"  Features with any NaN:  {summary['n_features_missing']}")
    if summary["avg_fraud_amount"]:
        log.info(f"  Avg fraud amount:       ${summary['avg_fraud_amount']:.2f}")
        log.info(f"  Avg legit amount:       ${summary['avg_legit_amount']:.2f}")
    if summary["peak_fraud_hour"] is not None:
        log.info(f"  Peak fraud hour:        {int(summary['peak_fraud_hour']):02d}:00")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame | None = None) -> dict:
    """
    Full EDA pipeline. Accepts an optional pre-loaded dataframe
    (used by tests). If None, loads from processed parquet.

    Returns dict of output file paths.
    """
    apply_style()

    out_dir = PROJECT_ROOT / cfg.paths.outputs_figures
    out_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        from src.ingestion.load_data import load_processed
        log.info("Loading processed data …")
        df = load_processed()
        log.info(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]:,} cols")

    summary = log_eda_summary(df)

    outputs = {}
    outputs["class_imbalance"]       = plot_class_imbalance(df, out_dir)
    outputs["amount_distribution"]   = plot_amount_distribution(df, out_dir)
    outputs["fraud_by_hour"]         = plot_fraud_by_hour(df, out_dir)
    outputs["fraud_by_dow"]          = plot_fraud_by_dow(df, out_dir)
    outputs["fraud_over_time"]       = plot_fraud_over_time(df, out_dir)
    outputs["missing_values"]        = plot_missing_values(df, out_dir)
    outputs["correlation_heatmap"]   = plot_correlation_heatmap(df, out_dir)
    outputs["categorical_rates"]     = plot_categorical_fraud_rates(df, out_dir)

    return {"outputs": outputs, "summary": summary}


def main():
    log.info("=" * 60)
    log.info("fraud-shield-ml | Phase 3: EDA")
    log.info("=" * 60)

    result = run_eda()

    log.info("=" * 60)
    log.info("Phase 3 EDA complete. Figures saved to outputs/figures/:")
    for name, path in result["outputs"].items():
        log.info(f"  {path.name}")
    log.info("  Next: python3 src/imbalance/compare_strategies.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
