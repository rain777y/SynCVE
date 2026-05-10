"""
Render Figure 1 for the paper — the SynCVE pipeline diagram.

Output: ``eval/figures/fig1_pipeline.png`` and ``.svg``.
Pure matplotlib, no external dataset.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        # (x, y, w, h, label, color, sub_label)
        (0.2, 2.5, 1.8, 1.0, "Webcam\nframe", "#cfd8dc", ""),
        (2.4, 2.5, 2.2, 1.0, "Detector\nensemble",
         "#bbdefb", "RetinaFace + MTCNN"),
        (5.0, 2.5, 2.6, 1.0, "Uncertainty-aware\nfusion (Axis 1C)",
         "#90caf9", "entropy-weighted softmax"),
        (8.0, 2.5, 2.2, 1.0, "EMA\nsmoothing",
         "#a5d6a7", "alpha = 0.2"),
        (10.6, 2.5, 3.0, 1.0, "Event detector\n(Axis 1A)",
         "#fff59d", "sliding · cusum · pelt"),
        (5.0, 0.4, 2.6, 1.0, "Clinical metrics\n(Axis 1A applied)",
         "#ffcc80", "valence · drift · blunting"),
        (10.6, 0.4, 3.0, 1.0, "Clinical report\n+ timeline UI (Axis 4)",
         "#ef9a9a", "PDF / Markdown / 3-track"),
    ]

    for x, y, w, h, label, color, sub in boxes:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.65, label,
                ha="center", va="center",
                fontsize=11, weight="bold")
        if sub:
            ax.text(x + w / 2, y + h * 0.22, sub,
                    ha="center", va="center",
                    fontsize=8.5, color="#555", style="italic")

    arrows = [
        ((2.0, 3.0), (2.4, 3.0)),
        ((4.6, 3.0), (5.0, 3.0)),
        ((7.6, 3.0), (8.0, 3.0)),
        ((10.2, 3.0), (10.6, 3.0)),
        ((12.1, 2.5), (12.1, 1.4)),       # event detector → clinical report
        ((9.6, 2.5), (9.0, 1.4)),          # event detector → clinical metrics (left bend)
        ((6.3, 2.5), (6.3, 1.4)),          # smoothing/fusion → clinical metrics
    ]
    for (x0, y0), (x1, y1) in arrows:
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                             mutation_scale=14, color="#444", linewidth=1.4)
        ax.add_patch(a)

    ax.text(7.0, 5.4, "SynCVE event-level emotion analysis pipeline",
            ha="center", va="center", fontsize=14, weight="bold")
    ax.text(7.0, 4.9,
            "Frozen pretrained detectors → entropy-weighted fusion → "
            "consensus event detection → clinical-readable report.",
            ha="center", va="center", fontsize=10, color="#555", style="italic")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ("png", "svg"):
        out = os.path.join(out_dir, f"fig1_pipeline.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=160)
        print(f"saved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
