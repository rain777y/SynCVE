"""
eval/compare_results.py — CLI Result Comparator

Side-by-side comparison of two evaluation JSON files.

Usage
-----
    python eval/compare_results.py eval/results/baseline/fer2013_retinaface.json eval/results/pipeline/pipeline_vs_b0.json
"""

import json
import math
import sys
from pathlib import Path


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {p}")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for NaN/None/missing."""
    if val is None:
        return None
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _fmt(val: float | None, pct: bool = False, digits: int = 4) -> str:
    if val is None:
        return "N/A".center(10)
    if pct:
        return f"{val * 100:.{digits - 2}f}%".rjust(10)
    return f"{val:.{digits}f}".rjust(10)


def _delta(a: float | None, b: float | None, pct: bool = False) -> str:
    if a is None or b is None:
        return "N/A".center(10)
    d = b - a
    sign = "+" if d >= 0 else ""
    if pct:
        return f"{sign}{d * 100:.2f}%".rjust(10)
    return f"{sign}{d:.4f}".rjust(10)


def _extract_metrics(data: dict) -> dict:
    """
    Extract a flat dict of common metrics from various JSON structures.
    Handles baseline files, pipeline files, and ablation files.
    """
    metrics = {}

    # Direct baseline structure
    metrics["accuracy"] = _safe_float(data.get("overall_accuracy") or data.get("accuracy"))

    cr = data.get("classification_report", {})
    if "weighted avg" in cr:
        metrics["weighted_f1"] = _safe_float(cr["weighted avg"].get("f1-score"))
        metrics["macro_f1"] = _safe_float(cr.get("macro avg", {}).get("f1-score"))
    else:
        metrics["weighted_f1"] = _safe_float(data.get("weighted_f1"))
        metrics["macro_f1"] = None

    metrics["detection_rate"] = _safe_float(data.get("detection_rate"))

    lat = data.get("latency", {})
    metrics["mean_latency_ms"] = _safe_float(lat.get("mean_ms"))
    metrics["median_latency_ms"] = _safe_float(lat.get("median_ms"))
    metrics["p95_latency_ms"] = _safe_float(lat.get("p95_ms"))
    metrics["p99_latency_ms"] = _safe_float(lat.get("p99_ms"))

    roc = data.get("roc_auc", {})
    metrics["macro_auc"] = _safe_float(roc.get("macro_avg") if not isinstance(roc.get("macro_avg"), dict) else roc.get("macro_avg", {}).get("auc"))
    metrics["micro_auc"] = _safe_float(roc.get("micro_avg") if not isinstance(roc.get("micro_avg"), dict) else roc.get("micro_avg", {}).get("auc"))

    # Pipeline-vs-baseline: try to extract from nested structure
    comparisons = data.get("comparisons", {})
    if comparisons and metrics["accuracy"] is None:
        # Use the first comparison entry
        first_key = next(iter(comparisons))
        comp = comparisons[first_key]
        pipe = comp.get("pipeline", {})
        metrics["accuracy"] = _safe_float(pipe.get("accuracy"))
        metrics["weighted_f1"] = _safe_float(pipe.get("weighted_f1"))
        metrics["detection_rate"] = _safe_float(pipe.get("detection_rate"))
        plat = pipe.get("latency", {})
        metrics["mean_latency_ms"] = _safe_float(plat.get("mean_ms"))
        metrics["median_latency_ms"] = _safe_float(plat.get("median_ms"))
        metrics["p95_latency_ms"] = _safe_float(plat.get("p95_ms"))
        metrics["p99_latency_ms"] = _safe_float(plat.get("p99_ms"))

    return metrics


def compare(path_a: str, path_b: str) -> None:
    data_a = _load(path_a)
    data_b = _load(path_b)

    m_a = _extract_metrics(data_a)
    m_b = _extract_metrics(data_b)

    name_a = Path(path_a).stem
    name_b = Path(path_b).stem

    # Header
    header = f"{'Metric':<25} | {name_a[:10]:>10} | {name_b[:10]:>10} | {'Delta':>10}"
    sep = "-" * 25 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12

    print()
    print(header)
    print(sep)

    # Rows
    row_defs = [
        ("Overall Accuracy", "accuracy", False),
        ("Weighted F1", "weighted_f1", False),
        ("Macro F1", "macro_f1", False),
        ("Detection Rate", "detection_rate", True),
        ("Macro AUC", "macro_auc", False),
        ("Micro AUC", "micro_auc", False),
        ("Mean Latency (ms)", "mean_latency_ms", False),
        ("Median Latency (ms)", "median_latency_ms", False),
        ("P95 Latency (ms)", "p95_latency_ms", False),
        ("P99 Latency (ms)", "p99_latency_ms", False),
    ]

    for label, key, is_pct in row_defs:
        va = m_a.get(key)
        vb = m_b.get(key)

        # Skip rows where both sides are N/A
        if va is None and vb is None:
            continue

        if key.endswith("_ms"):
            # Latency: show as raw numbers
            fa = f"{va:.1f}".rjust(10) if va is not None else "N/A".center(10)
            fb = f"{vb:.1f}".rjust(10) if vb is not None else "N/A".center(10)
            if va is not None and vb is not None:
                d = vb - va
                fd = f"{d:+.1f}".rjust(10)
            else:
                fd = "N/A".center(10)
        else:
            fa = _fmt(va, pct=is_pct)
            fb = _fmt(vb, pct=is_pct)
            fd = _delta(va, vb, pct=is_pct)

        print(f"{label:<25} | {fa} | {fb} | {fd}")

    print()


def main():
    if len(sys.argv) < 3:
        print("Usage: python eval/compare_results.py <file_a.json> <file_b.json>")
        print("Example: python eval/compare_results.py eval/results/baseline/fer2013_retinaface.json eval/results/pipeline/pipeline_vs_b0.json")
        sys.exit(1)

    compare(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
