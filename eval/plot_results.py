"""
eval/plot_results.py — Visualization Utilities for SynCVE Evaluation

Publication-quality plots with a dark theme and colorblind-friendly palette.
Uses matplotlib + seaborn. Standalone: no imports from src.backend.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Global style constants
# ---------------------------------------------------------------------------
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#ffffff"
GRID_COLOR = "#2e2e4a"

# Colorblind-friendly palette (Wong 2011, extended)
CB_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#999999",  # grey (extra)
]


def _apply_dark_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply dark-background styling to a figure and axes."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)


def _save_fig(fig: plt.Figure, save_path: str, dpi: int = 200) -> None:
    """Save figure and close."""
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Plot saved to {out.resolve()}")


# ---------------------------------------------------------------------------
# 1. Confusion matrix heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    save_path: str = "confusion_matrix.png",
) -> None:
    """Heatmap with raw counts and row-percentages.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes x n_classes).
    labels : list of str
        Ordered class labels.
    title : str
        Figure title.
    save_path : str
        Output image path.
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), max(7, n * 1.0)))
    _apply_dark_style(ax, fig)

    # Row-normalised percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_pct = cm / row_sums * 100

    # Annotation strings: count + percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor=GRID_COLOR,
        cbar_kws={"label": "Row %"},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Colorbar text colour
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 2. ROC curves
# ---------------------------------------------------------------------------
def plot_roc_curves(
    roc_data: dict,
    title: str = "ROC Curves",
    save_path: str = "roc_curves.png",
) -> None:
    """Plot per-class ROC curves on one figure with micro/macro averages.

    Parameters
    ----------
    roc_data : dict
        Output of ``metrics.compute_roc_auc``; keys are emotion labels plus
        ``"micro_avg"`` and ``"macro_avg"``.
    title : str
        Figure title.
    save_path : str
        Output image path.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    _apply_dark_style(ax, fig)

    # Per-class curves — skip classes where AUC is NaN/None
    class_labels = [k for k in roc_data if k not in ("micro_avg", "macro_avg")]
    for idx, label in enumerate(class_labels):
        d = roc_data[label]
        color = CB_PALETTE[idx % len(CB_PALETTE)]
        auc_val = d.get("auc")

        # Skip classes with undefined AUC (e.g., 0 positive samples)
        if auc_val is None or (isinstance(auc_val, float) and np.isnan(auc_val)):
            ax.plot([], [], " ", label=f"{label} (AUC=N/A)")
            continue

        ax.plot(
            d["fpr"],
            d["tpr"],
            color=color,
            linewidth=1.5,
            label=f'{label} (AUC={auc_val:.3f})',
        )

    # Micro-average
    if "micro_avg" in roc_data and "fpr" in roc_data["micro_avg"]:
        d = roc_data["micro_avg"]
        micro_auc = d.get("auc")
        if micro_auc is not None and not (isinstance(micro_auc, float) and np.isnan(micro_auc)):
            ax.plot(
                d["fpr"],
                d["tpr"],
                color="#ffffff",
                linewidth=2.5,
                linestyle="--",
                label=f'micro-avg (AUC={micro_auc:.3f})',
            )
        else:
            ax.plot([], [], " ", label="micro-avg (AUC=N/A)")

    # Macro-average (just AUC number, no curve)
    if "macro_avg" in roc_data:
        macro_auc = roc_data["macro_avg"]["auc"]
        if macro_auc is not None and not (isinstance(macro_auc, float) and np.isnan(macro_auc)):
            ax.plot([], [], " ", label=f"macro-avg AUC={macro_auc:.3f}")
        else:
            ax.plot([], [], " ", label="macro-avg AUC=N/A")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="#555555", linewidth=1, linestyle=":")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(
        loc="lower right",
        fontsize=9,
        facecolor="#2e2e4a",
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Per-class grouped bar chart
# ---------------------------------------------------------------------------
def plot_per_class_metrics(
    report: dict,
    title: str = "Per-Class Metrics",
    save_path: str = "per_class_metrics.png",
) -> None:
    """Grouped bar chart showing precision, recall, F1 per emotion.

    Parameters
    ----------
    report : dict
        Output of ``metrics.compute_classification_report`` (or
        ``sklearn.metrics.classification_report(output_dict=True)``).
    title : str
        Figure title.
    save_path : str
        Output image path.
    """
    # Filter to per-class entries only
    skip = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    class_labels = [k for k in report if k not in skip]

    precision = [report[c]["precision"] for c in class_labels]
    recall = [report[c]["recall"] for c in class_labels]
    f1 = [report[c]["f1-score"] for c in class_labels]

    x = np.arange(len(class_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(class_labels) * 1.3), 6))
    _apply_dark_style(ax, fig)

    bars_p = ax.bar(
        x - width, precision, width, label="Precision", color=CB_PALETTE[0]
    )
    bars_r = ax.bar(
        x, recall, width, label="Recall", color=CB_PALETTE[1]
    )
    bars_f = ax.bar(
        x + width, f1, width, label="F1-Score", color=CB_PALETTE[2]
    )

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(
        facecolor="#2e2e4a",
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )

    # Value labels on bars
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=TEXT_COLOR,
            )

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Latency histogram
# ---------------------------------------------------------------------------
def plot_latency_histogram(
    latencies: List[float],
    title: str = "Inference Latency Distribution",
    save_path: str = "latency_histogram.png",
) -> None:
    """Latency distribution histogram with p50/p95/p99 marker lines.

    Parameters
    ----------
    latencies : list of float
        Latency values in milliseconds.
    title : str
        Figure title.
    save_path : str
        Output image path.
    """
    arr = np.array(latencies)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(ax, fig)

    ax.hist(
        arr,
        bins=min(80, max(20, len(arr) // 50)),
        color=CB_PALETTE[4],
        edgecolor=BG_COLOR,
        alpha=0.85,
    )

    # Marker lines
    for val, label, color, ls in [
        (p50, f"p50={p50:.1f}ms", CB_PALETTE[2], "--"),
        (p95, f"p95={p95:.1f}ms", CB_PALETTE[0], "--"),
        (p99, f"p99={p99:.1f}ms", CB_PALETTE[5], "--"),
    ]:
        ax.axvline(val, color=color, linewidth=2, linestyle=ls, label=label)

    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(
        fontsize=10,
        facecolor="#2e2e4a",
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )

    _save_fig(fig, save_path)
