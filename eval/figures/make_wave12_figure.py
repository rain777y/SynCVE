"""
Render a figure summarising Wave-1+2 latency / parallelism / debounce gains.

Inputs : the four most recent flood_e2e_*.json under eval/reports/
Output : eval/figures/fig3_wave12_results.png  (and .svg)

Usage  : E:/conda/envs/SynCVE/python.exe eval/figures/make_wave12_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "eval" / "reports"
OUT_DIR = ROOT / "eval" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_runs(n: int = 4) -> List[Dict]:
    files = sorted(REPORTS.glob("flood_e2e_*.json"), reverse=True)[:n]
    runs = []
    for fp in reversed(files):  # chronological
        with fp.open() as f:
            runs.append(json.load(f))
    return runs


def fig() -> None:
    runs = load_runs(4)
    if not runs:
        raise SystemExit("no flood_e2e results in eval/reports/")

    labels = [f"R{i+1}" for i in range(len(runs))]

    health_ms = [r["health"]["latency_ms"] for r in runs]
    events_speedup = [r["events_cache"]["speedup_x"] for r in runs]
    metrics_speedup = [
        (r.get("clinical_metrics_cache", {}).get("speedup_x") or 0) for r in runs
    ]
    pdf_parallel = [r["concurrent_pdf"]["parallel_score"] for r in runs]
    async_submit = [r["async_stop"]["submit_ms"] for r in runs]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=120)
    fig.suptitle(
        "SynCVE Wave-1 + Wave-2 — Engineering Validation (4 runs)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # 1. Health latency
    ax = axes[0, 0]
    ax.bar(labels, health_ms, color="#4A90D9", edgecolor="#2C3E50")
    ax.set_title("/health latency (ms, lower is better)")
    ax.set_ylabel("ms")
    for i, v in enumerate(health_ms):
        ax.text(i, v + 0.3, f"{v}", ha="center", fontsize=9)
    ax.set_ylim(0, max(health_ms) * 1.4 + 1)

    # 2. Cache speed-up
    ax = axes[0, 1]
    x = range(len(labels))
    w = 0.35
    bars1 = ax.bar([i - w / 2 for i in x], events_speedup, w,
                   label="/events", color="#FFB020", edgecolor="#2C3E50")
    bars2 = ax.bar([i + w / 2 for i in x], metrics_speedup, w,
                   label="/clinical_metrics", color="#27AE60", edgecolor="#2C3E50")
    ax.axhline(1.0, color="#E74C3C", linestyle="--", linewidth=1, label="no speed-up")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("LRU cache speed-up (cold/warm median, higher is better)")
    ax.set_ylabel("× speed-up")
    ax.legend(fontsize=8, loc="upper left")
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        if h:
            ax.text(b.get_x() + b.get_width() / 2, h + 0.05,
                    f"{h:.2f}×", ha="center", fontsize=8)

    # 3. PDF parallel score
    ax = axes[1, 0]
    ax.bar(labels, pdf_parallel, color="#9B59B6", edgecolor="#2C3E50")
    ax.axhline(1.0, color="#7F8C8D", linestyle=":", linewidth=1, label="serial")
    ax.axhline(1.5, color="#E74C3C", linestyle="--", linewidth=1, label="threshold")
    ax.set_title("4× concurrent PDF — parallel score (higher is better)")
    ax.set_ylabel("× parallelism (n × med / wall)")
    ax.set_ylim(0, max(pdf_parallel) * 1.25 + 0.5)
    ax.legend(fontsize=8, loc="lower left")
    for i, v in enumerate(pdf_parallel):
        ax.text(i, v + 0.05, f"{v:.2f}×", ha="center", fontsize=9)

    # 4. Async stop submit latency
    ax = axes[1, 1]
    ax.bar(labels, async_submit, color="#16A085", edgecolor="#2C3E50")
    ax.axhline(16000, color="#E74C3C", linestyle="--", linewidth=1,
               label="prior sync ≈16 s")
    ax.set_title("/session/stop_async — submit latency (ms, log scale)")
    ax.set_yscale("log")
    ax.set_ylabel("ms (log)")
    ax.legend(fontsize=8, loc="upper right")
    for i, v in enumerate(async_submit):
        ax.text(i, v * 1.2, f"{v} ms", ha="center", fontsize=9)
    ax.set_ylim(1, 30000)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = OUT_DIR / "fig3_wave12_results.png"
    out_svg = OUT_DIR / "fig3_wave12_results.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    fig()
