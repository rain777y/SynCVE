"""
Event-level evaluation harness for Axis 1A.

Computes Precision / Recall / F1 with a temporal tolerance window
(``tolerance_frames``), runs three method ablations
(sliding-only / cusum-only / ensemble), and synthesises a probability
stream from CASME-II-style ground-truth annotations when no real
emotion-prob trace is available.

This script is intentionally framework-light: it depends only on the
backend ``event_detector`` module and the standard library. Plotting is
done with matplotlib when available.

Usage::

    # 1. Evaluate against synthetic ground-truth (sanity check)
    python -m eval.event_eval --synth

    # 2. Evaluate against an exported session JSON
    python -m eval.event_eval --session-json eval/data/session.json

    # 3. Compare fusion ablations on a session
    python -m eval.event_eval --session-json ... --ablate-fusion
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure project root on path when run as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.backend.event_detector import EventDetector  # noqa: E402
from src.backend.uncertainty_fusion import fuse_probabilities  # noqa: E402

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


# ---------------------------------------------------------------------------
# Ground-truth & matching
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthEvent:
    frame_idx: int
    label: str = ""           # optional (e.g. "trigger:father")


@dataclass
class EvalResult:
    method: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_pred: int
    n_gt: int
    tolerance_frames: int


def event_match(
    pred: Sequence[Dict[str, Any]],
    gt: Sequence[GroundTruthEvent],
    tolerance_frames: int = 4,
) -> Tuple[int, int, int]:
    """
    Greedy event matching: each ground-truth event consumes the closest
    prediction within ``tolerance_frames``. Returns (tp, fp, fn).
    """
    used_pred = [False] * len(pred)
    pred_frames = [int(p.get("frame_idx", -1)) for p in pred]

    tp = 0
    for g in gt:
        # Find the closest unused prediction within tolerance
        best_idx = -1
        best_dist = tolerance_frames + 1
        for i, pf in enumerate(pred_frames):
            if used_pred[i]:
                continue
            d = abs(pf - g.frame_idx)
            if d <= tolerance_frames and d < best_dist:
                best_idx = i
                best_dist = d
        if best_idx >= 0:
            used_pred[best_idx] = True
            tp += 1
    fn = len(gt) - tp
    fp = sum(1 for u in used_pred if not u)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------

def _peaked_dist(target: str, intensity: float = 0.7) -> Dict[str, float]:
    base = (1.0 - intensity) / (len(EMOTIONS) - 1)
    return {e: (intensity if e == target else base) for e in EMOTIONS}


def synth_session(
    n_segments: int = 6,
    seg_len: int = 30,
    noise: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Dict[str, float]], List[GroundTruthEvent]]:
    """
    Build a synthetic emotion-probability stream with planted change points.

    Segments alternate between ``neutral`` and a non-neutral emotion drawn
    from a fixed (seeded) cycle so every odd segment is guaranteed to be a
    transition. Returns (probability_history, ground_truth_events).
    """
    rng = random.Random(seed)
    non_neutral = [e for e in EMOTIONS if e != "neutral"]
    rng.shuffle(non_neutral)

    history: List[Dict[str, float]] = []
    gt: List[GroundTruthEvent] = []
    cur = "neutral"
    nn_idx = 0
    for s in range(n_segments):
        if s % 2 == 0:
            target = "neutral"
        else:
            target = non_neutral[nn_idx % len(non_neutral)]
            nn_idx += 1
        if target != cur and s > 0:
            gt.append(GroundTruthEvent(frame_idx=len(history), label=f"{cur}->{target}"))
        cur = target
        for _ in range(seg_len):
            d = _peaked_dist(target, intensity=0.7 + rng.uniform(-0.05, 0.05))
            # Add small Dirichlet-like noise
            for k in d:
                d[k] = max(1e-6, d[k] + rng.uniform(-noise, noise))
            s_ = sum(d.values())
            d = {k: v / s_ for k, v in d.items()}
            history.append(d)
    return history, gt


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_method(
    name: str,
    history: List[Dict[str, float]],
    gt: List[GroundTruthEvent],
    *,
    tolerance_frames: int = 4,
    **det_kwargs,
) -> EvalResult:
    det = EventDetector(method=name, **det_kwargs)
    events = [vars(e) if hasattr(e, "frame_idx") else e for e in det.detect_batch(history)]
    tp, fp, fn = event_match(events, gt, tolerance_frames=tolerance_frames)
    p, r, f1 = prf(tp, fp, fn)
    return EvalResult(
        method=name, precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
        tp=tp, fp=fp, fn=fn, n_pred=len(events), n_gt=len(gt),
        tolerance_frames=tolerance_frames,
    )


def run_ablation(
    history: List[Dict[str, float]],
    gt: List[GroundTruthEvent],
    *,
    tolerance_frames: int = 4,
) -> List[EvalResult]:
    return [
        run_method("sliding", history, gt, tolerance_frames=tolerance_frames),
        run_method("cusum", history, gt, tolerance_frames=tolerance_frames),
        run_method(
            "ensemble", history, gt, tolerance_frames=tolerance_frames,
            consensus_min_methods=2,
        ),
    ]


def run_fusion_ablation(
    per_frame_per_detector: List[List[Dict[str, float]]],
    gt: List[GroundTruthEvent],
    *,
    tolerance_frames: int = 4,
) -> Dict[str, EvalResult]:
    """
    Compare fusion methods on a multi-detector probability sequence.
    ``per_frame_per_detector`` is a list of per-frame lists, where each
    per-frame list contains one probability dict per ensemble backend.

    For each fusion method, fuse per-frame, then run the ensemble event
    detector and compute event-level F1 against the same GT.
    """
    results: Dict[str, EvalResult] = {}
    for method in ("fixed", "uncertainty", "max_confidence"):
        fused_history: List[Dict[str, float]] = []
        for per_frame in per_frame_per_detector:
            f = fuse_probabilities(per_frame, method=method)
            fused_history.append(f["fused"])
        det = EventDetector(method="ensemble", consensus_min_methods=2)
        events = [vars(e) for e in det.detect_batch(fused_history)]
        tp, fp, fn = event_match(events, gt, tolerance_frames=tolerance_frames)
        p, r, f1 = prf(tp, fp, fn)
        results[method] = EvalResult(
            method=f"fusion::{method}", precision=round(p, 4), recall=round(r, 4),
            f1=round(f1, 4), tp=tp, fp=fp, fn=fn, n_pred=len(events), n_gt=len(gt),
            tolerance_frames=tolerance_frames,
        )
    return results


def synthesise_per_detector(
    history: List[Dict[str, float]], n_detectors: int = 3, noise: float = 0.10, seed: int = 0,
) -> List[List[Dict[str, float]]]:
    """For fusion ablation: produce noisy per-detector replicas of the GT history."""
    rng = random.Random(seed)
    out: List[List[Dict[str, float]]] = []
    for p in history:
        per_frame: List[Dict[str, float]] = []
        for _ in range(n_detectors):
            jittered = {
                k: max(1e-6, v + rng.uniform(-noise, noise)) for k, v in p.items()
            }
            s = sum(jittered.values())
            per_frame.append({k: v / s for k, v in jittered.items()})
        out.append(per_frame)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(rows: List[EvalResult]) -> None:
    print(f"{'method':<22s} {'P':>6s} {'R':>6s} {'F1':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'N_pred':>8s} {'N_gt':>5s}")
    print("-" * 70)
    for r in rows:
        print(f"{r.method:<22s} {r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f} "
              f"{r.tp:>4d} {r.fp:>4d} {r.fn:>4d} {r.n_pred:>8d} {r.n_gt:>5d}")


def save_json(path: str, rows: List[EvalResult]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_session_json(path: str) -> Tuple[List[Dict[str, float]], List[GroundTruthEvent]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = data["smoothed_history"]
    gt = [GroundTruthEvent(**g) for g in data.get("ground_truth", [])]
    return history, gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-level evaluation for Axis 1A")
    parser.add_argument("--synth", action="store_true",
                        help="Run on a synthetic stream (sanity / smoke).")
    parser.add_argument("--session-json", type=str, default=None,
                        help="Path to a session export with smoothed_history + ground_truth.")
    parser.add_argument("--tolerance-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="eval/reports/event_eval.json")
    parser.add_argument("--ablate-fusion", action="store_true",
                        help="Run fusion-method ablation on the supplied session.")
    args = parser.parse_args()

    if not args.synth and not args.session_json:
        args.synth = True
        print("[info] no input given — defaulting to --synth")

    if args.synth:
        history, gt = synth_session(seed=args.seed)
    else:
        history, gt = _load_session_json(args.session_json)

    print(f"history frames: {len(history)} | ground-truth events: {len(gt)}")
    rows = run_ablation(history, gt, tolerance_frames=args.tolerance_frames)
    print()
    print("=== Method ablation (Axis 1A) ===")
    print_table(rows)

    if args.ablate_fusion:
        per_frame_per_det = synthesise_per_detector(history, seed=args.seed)
        fusion_rows = list(run_fusion_ablation(per_frame_per_det, gt,
                                                tolerance_frames=args.tolerance_frames).values())
        print()
        print("=== Fusion ablation (Axis 1C) ===")
        print_table(fusion_rows)
        rows.extend(fusion_rows)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_json(out_path, rows)


if __name__ == "__main__":
    main()
