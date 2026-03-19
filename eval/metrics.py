"""
eval/metrics.py — Reusable Metrics Library for SynCVE Evaluation

Provides functions for computing classification metrics, confusion matrices,
ROC/AUC curves, latency statistics, and saving results to JSON.

Standalone module: no imports from src.backend.
"""

import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)


def compute_classification_report(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> dict:
    """Compute per-class precision, recall, F1, support + macro/weighted avgs.

    Parameters
    ----------
    y_true : list of str
        Ground-truth emotion labels.
    y_pred : list of str
        Predicted emotion labels.
    labels : list of str
        Ordered list of all class labels.

    Returns
    -------
    dict
        sklearn classification_report output with per-class and aggregate
        metrics.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return report


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> np.ndarray:
    """Compute a standard confusion matrix.

    Parameters
    ----------
    y_true : list of str
        Ground-truth labels.
    y_pred : list of str
        Predicted labels.
    labels : list of str
        Ordered class labels (row/column order).

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm


def compute_roc_auc(
    y_true_onehot: np.ndarray,
    y_scores: np.ndarray,
    labels: List[str],
) -> dict:
    """Compute per-class ROC curve data and AUC scores.

    Parameters
    ----------
    y_true_onehot : np.ndarray, shape (n_samples, n_classes)
        One-hot encoded ground-truth labels.
    y_scores : np.ndarray, shape (n_samples, n_classes)
        Predicted probability scores for each class, values in [0, 1].
    labels : list of str
        Ordered class labels matching column order.

    Returns
    -------
    dict
        Mapping of emotion -> {"fpr": list, "tpr": list, "auc": float}.
    """
    roc_data = {}
    n_classes = len(labels)

    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc_val = auc(fpr, tpr)
        roc_data[label] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc_val),
        }

    # Micro-average ROC: flatten all classes
    fpr_micro, tpr_micro, _ = roc_curve(
        y_true_onehot.ravel(), y_scores.ravel()
    )
    roc_data["micro_avg"] = {
        "fpr": fpr_micro.tolist(),
        "tpr": tpr_micro.tolist(),
        "auc": float(auc(fpr_micro, tpr_micro)),
    }

    # Macro-average AUC (average of per-class AUCs)
    macro_auc = float(
        np.mean([roc_data[label]["auc"] for label in labels])
    )
    roc_data["macro_avg"] = {"auc": macro_auc}

    return roc_data


def compute_latency_stats(latencies: List[float]) -> dict:
    """Compute descriptive statistics for inference latencies.

    Parameters
    ----------
    latencies : list of float
        Latency measurements in milliseconds.

    Returns
    -------
    dict
        Keys: mean_ms, median_ms, p95_ms, p99_ms, min_ms, max_ms,
        std_ms, total_samples.
    """
    arr = np.array(latencies, dtype=np.float64)
    stats = {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
        "total_samples": len(latencies),
    }
    return stats


def save_results_json(results: dict, path: str) -> None:
    """Save results dict as pretty-printed JSON with metadata.

    Adds a ``metadata`` block containing timestamp, Python version,
    platform, and machine info if not already present.

    Parameters
    ----------
    results : dict
        Results dictionary to persist.
    path : str
        Output file path (.json).
    """
    # Inject metadata if not already present
    if "metadata" not in results:
        results["metadata"] = {}
    meta = results["metadata"]
    meta.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    meta.setdefault("python_version", platform.python_version())
    meta.setdefault("platform", platform.platform())
    meta.setdefault("machine", platform.machine())
    meta.setdefault("processor", platform.processor())

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom encoder for numpy types
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    print(f"Results saved to {out_path.resolve()}")
