"""
Render Figure 2 for the paper — the worked example.

Plots the smoothed emotion-probability stream from a synthetic session,
overlays the ground-truth change-points, and marks the consensus events
detected by the system.

Output: ``eval/figures/fig2_worked_example.png`` and ``.svg``.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.event_eval import synth_session  # noqa: E402
from src.backend.event_detector import EventDetector  # noqa: E402

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
COLORS = {
    "angry": "#E74C3C", "disgust": "#27AE60", "fear": "#9B59B6",
    "happy": "#FFB020", "sad": "#4A90D9", "surprise": "#F39C12",
    "neutral": "#95A5A6",
}


def main() -> None:
    import matplotlib.pyplot as plt
    history, gt = synth_session(n_segments=8, seg_len=30, seed=42)
    det = EventDetector(method="ensemble", consensus_min_methods=2)
    events = det.detect_batch(history)

    n = len(history)
    xs = list(range(n))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Stacked area for each emotion's probability
    bottom = [0.0] * n
    for emo in EMOTIONS:
        ys = [h.get(emo, 0.0) for h in history]
        ax.fill_between(xs, bottom, [b + y for b, y in zip(bottom, ys)],
                        color=COLORS[emo], alpha=0.65, label=emo, linewidth=0)
        bottom = [b + y for b, y in zip(bottom, ys)]

    # Ground-truth change-points (red dotted vertical lines)
    for g in gt:
        ax.axvline(g.frame_idx, color="#000", linestyle=":", alpha=0.45,
                   linewidth=1.0)

    # System-confirmed events (large filled markers, top of plot)
    for ev in events:
        ax.scatter([ev.frame_idx], [1.04], marker="v", s=70,
                   facecolor="#ff4081", edgecolor="white", linewidth=1.0,
                   zorder=5)
        ax.annotate(
            f"{ev.from_emotion}→{ev.to_emotion}\nconf={ev.confidence:.2f}",
            xy=(ev.frame_idx, 1.04),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.18",
                      fc="#fff7fa", ec="#ff4081", linewidth=0.8, alpha=0.92),
        )

    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Probability mass")
    ax.set_title("SynCVE — Worked Example: Synthetic Stream with Detected Events",
                 fontsize=12, weight="bold")

    # Custom-line legend extras for GT and events
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], color="#000", linestyle=":", lw=1.2,
               label="Ground-truth change-point"),
        Line2D([0], [0], marker="v", color="w",
               markerfacecolor="#ff4081", markeredgecolor="white",
               markersize=10, label="Consensus event (system)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra, labels + [e.get_label() for e in extra],
              loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=5, fontsize=8, frameon=False)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ("png", "svg"):
        out = os.path.join(out_dir, f"fig2_worked_example.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=160)
        print(f"saved -> {out}")
    plt.close(fig)

    # Also dump a small JSON next to the figure documenting the numbers
    import json
    summary_path = os.path.join(out_dir, "fig2_worked_example.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_frames": n,
            "ground_truth_changepoints": [g.frame_idx for g in gt],
            "system_events": [
                {"frame_idx": ev.frame_idx,
                 "from_emotion": ev.from_emotion,
                 "to_emotion": ev.to_emotion,
                 "magnitude": ev.magnitude,
                 "confidence": ev.confidence,
                 "methods": ev.methods}
                for ev in events
            ],
        }, f, indent=2)
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
