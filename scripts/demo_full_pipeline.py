"""
End-to-end synthetic demo of the SynCVE Axis 1A/1C pipeline.

Runs without a webcam, without DeepFace, without Supabase, without the
Gemini API — purely on a synthesised emotion-probability stream. Exercises:

    1. ``uncertainty_fusion.fuse_probabilities``      (Axis 1C)
    2. ``temporal_analysis.TemporalAnalyzer``         (EMA + transitions)
    3. ``event_detector.EventDetector``               (Axis 1A consensus)
    4. ``clinical_metrics.compute_session_metrics``   (Axis 1A metrics)
    5. ``clinical_report.build_markdown / build_pdf`` (Axis 4 export)

Useful for:
    - 答辩 / demo without hardware in the room.
    - Regression checking that the integrated pipeline still works.
    - Generating a sample PDF for the paper.

Usage::

    python -m scripts.demo_full_pipeline
    python -m scripts.demo_full_pipeline --pdf  # also render PDF
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile

# Project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.backend.uncertainty_fusion import fuse_probabilities  # noqa: E402
from src.backend.event_detector import (  # noqa: E402
    EventDetector,
    event_to_dict,
)
from src.backend.temporal_analysis import TemporalAnalyzer  # noqa: E402
from src.backend.clinical_metrics import (  # noqa: E402
    compute_session_metrics,
    metrics_to_dict,
)
from src.backend.clinical_report import (  # noqa: E402
    build_markdown,
    build_pdf,
)

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def _peaked(target: str, intensity: float = 0.65) -> dict:
    base = (1.0 - intensity) / (len(EMOTIONS) - 1)
    return {e: (intensity if e == target else base) for e in EMOTIONS}


def synth_two_detector_stream(
    seed: int = 7,
    n_segments: int = 8,
    seg_len: int = 30,
    detector_noise: tuple = (0.04, 0.14),
):
    """
    Build a synthetic two-detector stream where detector A is reliable and
    detector B is noisier. Returns a per-frame list of pairs of probability
    dicts plus the ground-truth segment boundaries.
    """
    rng = random.Random(seed)
    non_neutral = [e for e in EMOTIONS if e != "neutral"]
    rng.shuffle(non_neutral)
    history_pairs = []
    transitions = []
    cur = "neutral"
    nn_idx = 0

    for s in range(n_segments):
        target = "neutral" if s % 2 == 0 else non_neutral[nn_idx % len(non_neutral)]
        if s % 2 == 1:
            nn_idx += 1
        if target != cur and s > 0:
            transitions.append({"frame_idx": len(history_pairs), "kind": f"{cur}->{target}"})
        cur = target
        for _ in range(seg_len):
            base = _peaked(target, intensity=0.65 + rng.uniform(-0.02, 0.02))
            jitter_a = {k: max(1e-6, v + rng.uniform(-detector_noise[0], detector_noise[0]))
                        for k, v in base.items()}
            jitter_b = {k: max(1e-6, v + rng.uniform(-detector_noise[1], detector_noise[1]))
                        for k, v in base.items()}
            sa = sum(jitter_a.values()); jitter_a = {k: v / sa for k, v in jitter_a.items()}
            sb = sum(jitter_b.values()); jitter_b = {k: v / sb for k, v in jitter_b.items()}
            history_pairs.append([jitter_a, jitter_b])
    return history_pairs, transitions


def main() -> int:
    parser = argparse.ArgumentParser(description="SynCVE end-to-end synthetic demo")
    parser.add_argument("--pdf", action="store_true", help="Also render the PDF report.")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "eval", "reports"),
                        help="Directory for the demo artefacts.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 60)
    print("SynCVE Synthetic Pipeline Demo")
    print("=" * 60)

    # 1. Build a 2-detector synthetic stream
    pairs, gt_transitions = synth_two_detector_stream(seed=args.seed)
    print(f"\n[1/5] Synthesised {len(pairs)} frames across "
          f"{len(gt_transitions) + 1} ground-truth segments "
          f"({len(gt_transitions)} transitions).")

    # 2. Per-frame fusion (Axis 1C) and EMA smoothing via TemporalAnalyzer
    detectors = ["A_clean", "B_noisy"]
    analyzer = TemporalAnalyzer(
        alpha=0.3,
        fps_estimate=0.5,
        event_detector=EventDetector(method="ensemble", consensus_min_methods=2),
    )
    per_frame_ensemble_meta = []
    for i, per_det in enumerate(pairs):
        fusion_out = fuse_probabilities(
            per_det, method="uncertainty", temperature=1.0, entropy_floor=0.05,
        )
        fused = fusion_out["fused"]
        ensemble_meta = {
            "weights": {d: round(w, 4) for d, w in zip(detectors, fusion_out["weights"])},
            "per_entropy": {d: e for d, e in zip(detectors, fusion_out["per_entropy"])},
            "fused_entropy": fusion_out["fused_entropy"],
            "max_entropy": fusion_out["max_entropy"],
        }
        per_frame_ensemble_meta.append({"ensemble": ensemble_meta})
        analyzer.add_frame(fused, ensemble_meta=ensemble_meta)
    print(f"[2/5] Uncertainty fusion + EMA done "
          f"({len(per_frame_ensemble_meta)} frames).")

    # 3. Events
    events = analyzer.get_events()
    print(f"[3/5] Event detector confirmed {len(events)} consensus events:")
    for ev in events[:5]:
        print(f"        frame {ev['frame_idx']:>4}  "
              f"{ev['from_emotion']:>9} -> {ev['to_emotion']:<9}  "
              f"mag={ev['magnitude']:.2f}  conf={ev['confidence']:.2f}  "
              f"methods={ev['methods']}")

    # 4. Clinical metrics
    triggers = [
        {"word": "father", "frame_idx": 28},
        {"word": "house", "frame_idx": 88},
    ]
    metrics = compute_session_metrics(
        analyzer.get_smoothed_history_dicts(),
        events,
        fps=0.5,
        triggers=triggers,
        per_frame_ensemble=per_frame_ensemble_meta,
        detectors=detectors,
    )
    metrics_d = metrics_to_dict(metrics)
    print(f"[4/5] Clinical metrics:")
    print(f"        valence_mean      = {metrics.valence_mean}")
    print(f"        valence_drift/min = {metrics.valence_drift_per_min}")
    print(f"        affect_blunting   = {metrics.affect_blunting_score}")
    print(f"        reactivity        = {metrics.reactivity_events_per_min}")
    print(f"        suppression_index = {metrics.suppression_index}")
    print(f"        detector_reliab   = {metrics.detector_reliability}")

    # 5. Clinical report
    report_data = {
        "session_id": "demo-synthetic",
        "samples": metrics.samples,
        "duration_sec": metrics.duration_sec,
        "fps_estimate": 0.5,
        "clinical_metrics": {
            k: v for k, v in metrics_d.items()
            if k not in {"events", "session_id", "valence_trace", "reaction_latencies"}
        },
        "events": events,
        "reaction_latencies": None,
    }
    md = build_markdown(report_data)
    md_path = os.path.join(args.out_dir, "demo_clinical_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[5/5] Markdown report written -> {md_path}")

    if args.pdf:
        pdf_path = build_pdf(md, output_dir=args.out_dir)
        print(f"        PDF report written -> {pdf_path}")

    # Persist a JSON summary too
    json_path = os.path.join(args.out_dir, "demo_session_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "events": events,
            "metrics": metrics_d,
            "ground_truth_transitions": gt_transitions,
        }, f, indent=2)
    print(f"        JSON summary written -> {json_path}")
    print()
    print("Demo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
