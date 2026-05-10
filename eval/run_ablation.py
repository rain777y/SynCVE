"""
Ablation runner — convenience wrapper around ``event_eval.py`` that runs
every method × every fusion method on the synthetic stream and dumps a
single JSON to ``eval/reports/full_ablation.json``.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.event_eval import (  # noqa: E402
    run_ablation,
    run_fusion_ablation,
    synth_session,
    synthesise_per_detector,
    print_table,
)


def main() -> None:
    history, gt = synth_session(n_segments=8, seed=42)
    method_rows = run_ablation(history, gt, tolerance_frames=4)
    per_det = synthesise_per_detector(history, n_detectors=3, noise=0.12, seed=0)
    fusion_rows = list(run_fusion_ablation(per_det, gt, tolerance_frames=4).values())

    print("=== Method ablation ===")
    print_table(method_rows)
    print("\n=== Fusion ablation ===")
    print_table(fusion_rows)

    out = {
        "method_rows": [r.__dict__ for r in method_rows],
        "fusion_rows": [r.__dict__ for r in fusion_rows],
        "n_history": len(history),
        "n_ground_truth": len(gt),
    }
    out_path = os.path.join(ROOT, "eval", "reports", "full_ablation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
