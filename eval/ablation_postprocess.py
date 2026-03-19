"""
eval/ablation_postprocess.py — Post-processing Ablation Study (Temporal)

Tests EMA smoothing and noise-floor suppression on sequential data by
grouping FER2013 test images BY emotion into same-emotion batches of 20,
simulating a temporal sequence.

Configurations:
    raw              — no post-processing
    ema_0.1          — EMA alpha=0.1
    ema_0.2          — EMA alpha=0.2
    ema_0.3          — EMA alpha=0.3
    nf_0.05          — noise floor 5%
    nf_0.10          — noise floor 10%
    nf_0.15          — noise floor 15%
    ema_0.3_nf_0.10  — EMA alpha=0.3 + noise floor 10%

Metrics:
    - Consistency Score: 1 - mean pairwise dominant-emotion-change-rate
    - Flicker Rate: dominant emotion changes per 20-frame window
    - Accuracy: per-frame accuracy

Usage
-----
    python -m eval.ablation_postprocess --limit 100       # quick test
    python -m eval.ablation_postprocess                   # full run
"""

import argparse
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BATCH_SIZE = 20

# Post-processing configurations
POSTPROCESS_CONFIGS = {
    "raw":              {"ema_alpha": None,  "noise_floor": None},
    "ema_0.1":          {"ema_alpha": 0.1,   "noise_floor": None},
    "ema_0.2":          {"ema_alpha": 0.2,   "noise_floor": None},
    "ema_0.3":          {"ema_alpha": 0.3,   "noise_floor": None},
    "nf_0.05":          {"ema_alpha": None,  "noise_floor": 0.05},
    "nf_0.10":          {"ema_alpha": None,  "noise_floor": 0.10},
    "nf_0.15":          {"ema_alpha": None,  "noise_floor": 0.15},
    "ema_0.3_nf_0.10":  {"ema_alpha": 0.3,   "noise_floor": 0.10},
}


# ---------------------------------------------------------------------------
# Full preprocessing (matches the pipeline)
# ---------------------------------------------------------------------------
def _apply_full_preprocess(frame: np.ndarray) -> np.ndarray:
    """Apply super-resolution + unsharp mask + CLAHE."""
    if frame is None or not hasattr(frame, "shape"):
        return frame

    height, width = frame.shape[:2]
    min_size = min(height, width)
    target_min = 256
    if min_size < target_min:
        scale = target_min / float(min_size)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    frame = cv2.addWeighted(frame, 1.25, blurred, -0.25, 0)

    if len(frame.shape) >= 3 and frame.shape[2] == 3:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        merged = cv2.merge((l_eq, a_channel, b_channel))
        frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return frame


# ---------------------------------------------------------------------------
# Post-processing primitives
# ---------------------------------------------------------------------------
def apply_ema(scores_sequence: list, alpha: float) -> list:
    """Exponential moving average over a sequence of emotion score dicts.

    Parameters
    ----------
    scores_sequence : list of dict
        Each dict maps emotion -> score (0-100 from DeepFace).
    alpha : float
        Smoothing factor (0 < alpha <= 1). Higher = more responsive.

    Returns
    -------
    list of dict
        Smoothed score dicts.
    """
    if not scores_sequence:
        return []
    smoothed = scores_sequence[0].copy()
    results = [smoothed.copy()]
    for scores in scores_sequence[1:]:
        for emotion in smoothed:
            smoothed[emotion] = alpha * scores.get(emotion, 0) + (1 - alpha) * smoothed[emotion]
        results.append(smoothed.copy())
    return results


def apply_noise_floor(scores: dict, threshold: float) -> dict:
    """Suppress low-confidence emotions below threshold (as fraction of 100)."""
    floor_val = threshold * 100.0  # Convert fraction to percentage scale
    filtered = {}
    for emotion, score in scores.items():
        filtered[emotion] = score if score >= floor_val else 0.0
    # Re-normalize so scores sum to 100
    total = sum(filtered.values())
    if total > 0:
        for emotion in filtered:
            filtered[emotion] = filtered[emotion] / total * 100.0
    return filtered


def get_dominant(scores: dict) -> str:
    """Return the dominant emotion from a score dict."""
    if not scores:
        return "neutral"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Temporal metrics
# ---------------------------------------------------------------------------
def compute_flicker_rate(dominant_sequence: list) -> float:
    """Number of dominant emotion changes in a sequence."""
    if len(dominant_sequence) < 2:
        return 0.0
    changes = sum(
        1 for i in range(1, len(dominant_sequence))
        if dominant_sequence[i] != dominant_sequence[i - 1]
    )
    return float(changes)


def compute_consistency_score(dominant_sequence: list) -> float:
    """1 - mean pairwise dominant-emotion-change-rate."""
    if len(dominant_sequence) < 2:
        return 1.0
    changes = sum(
        1 for i in range(1, len(dominant_sequence))
        if dominant_sequence[i] != dominant_sequence[i - 1]
    )
    change_rate = changes / (len(dominant_sequence) - 1)
    return 1.0 - change_rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-processing ablation study (temporal simulation)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "datasets" / "FER2013" / "test"),
        help="Path to FER2013 test directory (subfolders per emotion).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max images total (0 = all). Images are drawn per-emotion.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for output files.",
    )
    return parser.parse_args()


def collect_image_paths_by_emotion(dataset_dir: str) -> dict:
    """Return {emotion: [img_path, ...]} for building same-emotion batches."""
    root = Path(dataset_dir)
    if not root.is_dir():
        print(f"ERROR: dataset directory not found: {root}")
        sys.exit(1)

    by_emotion = defaultdict(list)
    for emotion in EMOTION_LABELS:
        emotion_dir = root / emotion
        if not emotion_dir.is_dir():
            print(f"  WARNING: missing sub-folder {emotion_dir}")
            continue
        for img_path in sorted(emotion_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                by_emotion[emotion].append(str(img_path))

    total = sum(len(v) for v in by_emotion.values())
    print(f"Found {total} images across {root}")
    return by_emotion


def build_batches(by_emotion: dict, batch_size: int, limit: int) -> list:
    """Build two types of batches for complementary temporal evaluation.

    Type 1 — Stability batches (same-emotion):
        All frames share ground truth emotion.  Tests: does EMA reduce
        within-emotion jitter without changing the correct dominant label?

    Type 2 — Transition batches (multi-emotion):
        Realistic scenario: e.g. neutral(5) → happy(8) → neutral(4) → sad(3).
        Tests: does EMA preserve real transitions while still reducing noise?

    Returns list of (batch_type, ground_truth_seq, [img_path, ...]) tuples.
        ground_truth_seq is a list of per-frame ground-truth labels.
    """
    batches = []

    # --- Type 1: same-emotion stability batches ---
    for emotion, paths in by_emotion.items():
        shuffled = paths.copy()
        random.shuffle(shuffled)
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i : i + batch_size]
            if len(batch) == batch_size:
                gt_seq = [emotion] * batch_size
                batches.append(("stability", gt_seq, batch))

    # --- Type 2: transition batches (realistic multi-emotion sequences) ---
    # Simulate a user whose emotion shifts: pick 2-4 emotions, allocate
    # frames proportionally, stitch images together in order.
    emotions_available = [e for e in by_emotion if len(by_emotion[e]) >= batch_size]
    if len(emotions_available) >= 2:
        # Generate transition scenarios
        transition_patterns = [
            # (emotion, frame_count) sequences
            [("neutral", 6), ("happy", 8), ("neutral", 6)],
            [("happy", 5), ("sad", 5), ("happy", 5), ("sad", 5)],
            [("neutral", 4), ("surprise", 3), ("happy", 6), ("neutral", 4), ("sad", 3)],
            [("angry", 4), ("neutral", 8), ("happy", 5), ("neutral", 3)],
            [("sad", 5), ("neutral", 5), ("happy", 5), ("surprise", 5)],
        ]
        for pattern in transition_patterns:
            # Adjust pattern to match batch_size
            total = sum(n for _, n in pattern)
            if total != batch_size:
                # Scale proportionally
                scaled = []
                remaining = batch_size
                for i, (emo, n) in enumerate(pattern):
                    if i == len(pattern) - 1:
                        scaled.append((emo, remaining))
                    else:
                        adj = max(1, round(n * batch_size / total))
                        adj = min(adj, remaining - (len(pattern) - i - 1))
                        scaled.append((emo, adj))
                        remaining -= adj
                pattern = scaled

            # Only use if we have all required emotions
            needed = set(e for e, _ in pattern)
            if not needed.issubset(set(emotions_available)):
                continue

            batch_paths = []
            gt_seq = []
            valid = True
            for emo, count in pattern:
                available = by_emotion[emo]
                if len(available) < count:
                    valid = False
                    break
                chosen = random.sample(available, count)
                batch_paths.extend(chosen)
                gt_seq.extend([emo] * count)

            if valid and len(batch_paths) == batch_size:
                batches.append(("transition", gt_seq, batch_paths))

    random.shuffle(batches)

    if limit > 0:
        max_batches = max(1, limit // batch_size)
        batches = batches[:max_batches]

    return batches


def run_ablation(args: argparse.Namespace) -> None:
    from deepface import DeepFace
    from tqdm import tqdm

    from eval.metrics import save_results_json
    from eval.plot_results import (
        _apply_dark_style,
        _save_fig,
        CB_PALETTE,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # 1. Build stability + transition batches
    # ------------------------------------------------------------------
    by_emotion = collect_image_paths_by_emotion(args.dataset_dir)
    batches = build_batches(by_emotion, BATCH_SIZE, args.limit)
    total_frames = sum(len(paths) for _, _, paths in batches)
    n_stability = sum(1 for t, _, _ in batches if t == "stability")
    n_transition = sum(1 for t, _, _ in batches if t == "transition")
    print(f"\nBuilt {len(batches)} batches of {BATCH_SIZE} frames ({total_frames} total)")
    print(f"  Stability batches (same emotion): {n_stability}")
    print(f"  Transition batches (multi-emotion): {n_transition}")

    if len(batches) == 0:
        print("No complete batches found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Run DeepFace on all frames (once, shared across configs)
    # ------------------------------------------------------------------
    print("\nRunning DeepFace on all batch frames...")
    # raw_results[img_path] = {"emotion": {...}, "dominant_emotion": str} or None
    raw_results = {}
    all_paths = set()
    for _, _, batch_paths in batches:
        all_paths.update(batch_paths)

    for img_path in tqdm(sorted(all_paths), desc="Inference", unit="img"):
        frame = cv2.imread(img_path)
        if frame is None:
            raw_results[img_path] = None
            continue

        processed = _apply_full_preprocess(frame)
        try:
            result = DeepFace.analyze(
                img_path=processed,
                actions=["emotion"],
                detector_backend="retinaface",
                enforce_detection=False,
                silent=True,
            )
            face = result[0] if isinstance(result, list) else result
            raw_results[img_path] = {
                "emotion": face["emotion"],
                "dominant_emotion": face["dominant_emotion"],
            }
        except Exception:
            raw_results[img_path] = None

    successful = sum(1 for v in raw_results.values() if v is not None)
    print(f"Successful inference: {successful}/{len(all_paths)}")

    # ------------------------------------------------------------------
    # 3. Apply each post-processing config and compute metrics
    # ------------------------------------------------------------------
    all_config_results = {}

    for config_id, config in POSTPROCESS_CONFIGS.items():
        ema_alpha = config["ema_alpha"]
        noise_floor = config["noise_floor"]

        # Separate metrics for stability vs transition batches
        stability_consistencies = []
        stability_flicker_rates = []
        stability_accuracies = []
        transition_accuracies = []
        transition_flicker_rates = []
        transition_correct_transitions = []  # did EMA preserve real emotion shifts?

        for batch_type, gt_seq, batch_paths in batches:
            batch_scores = []
            batch_true_labels = []
            for img_path, true_label in zip(batch_paths, gt_seq):
                r = raw_results.get(img_path)
                if r is None:
                    continue
                batch_scores.append(r["emotion"])
                batch_true_labels.append(true_label)

            if len(batch_scores) < 2:
                continue

            # Apply EMA
            if ema_alpha is not None:
                processed_scores = apply_ema(batch_scores, ema_alpha)
            else:
                processed_scores = [s.copy() for s in batch_scores]

            # Apply noise floor
            if noise_floor is not None:
                processed_scores = [
                    apply_noise_floor(s, noise_floor) for s in processed_scores
                ]

            dominant_seq = [get_dominant(s) for s in processed_scores]
            consistency = compute_consistency_score(dominant_seq)
            flicker = compute_flicker_rate(dominant_seq)
            frame_accuracy = sum(
                1 for d, t in zip(dominant_seq, batch_true_labels) if d == t
            ) / len(dominant_seq)

            if batch_type == "stability":
                stability_consistencies.append(consistency)
                stability_flicker_rates.append(flicker)
                stability_accuracies.append(frame_accuracy)
            else:  # transition
                transition_accuracies.append(frame_accuracy)
                transition_flicker_rates.append(flicker)
                # Check if real transitions were preserved:
                # Count ground-truth transitions and see how many the model detected
                gt_transitions = sum(
                    1 for i in range(1, len(batch_true_labels))
                    if batch_true_labels[i] != batch_true_labels[i - 1]
                )
                pred_transitions = sum(
                    1 for i in range(1, len(dominant_seq))
                    if dominant_seq[i] != dominant_seq[i - 1]
                )
                # Ratio: how many transitions survived post-processing
                # (too few = over-smoothing, too many = under-smoothing)
                if gt_transitions > 0:
                    transition_preservation = min(pred_transitions / gt_transitions, 2.0)
                else:
                    transition_preservation = 1.0 if pred_transitions == 0 else 0.0
                transition_correct_transitions.append(transition_preservation)

        # Aggregate per batch type
        def _safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        result = {
            "config_id": config_id,
            "settings": config,
            # Stability batches: same-emotion jitter reduction
            "stability_consistency": round(_safe_mean(stability_consistencies), 4),
            "stability_flicker": round(_safe_mean(stability_flicker_rates), 4),
            "stability_accuracy": round(_safe_mean(stability_accuracies), 4),
            "stability_batches": len(stability_consistencies),
            # Transition batches: real emotion shift preservation
            "transition_accuracy": round(_safe_mean(transition_accuracies), 4),
            "transition_flicker": round(_safe_mean(transition_flicker_rates), 4),
            "transition_preservation": round(_safe_mean(transition_correct_transitions), 4),
            "transition_batches": len(transition_accuracies),
            # Combined (backward compat)
            "consistency_score": round(_safe_mean(stability_consistencies), 4),
            "flicker_rate": round(_safe_mean(stability_flicker_rates + transition_flicker_rates), 4),
            "accuracy": round(_safe_mean(stability_accuracies + transition_accuracies), 4),
            "num_batches": len(stability_consistencies) + len(transition_accuracies),
        }
        all_config_results[config_id] = result

        stab = f"stab(cons={result['stability_consistency']:.2f} flk={result['stability_flicker']:.1f} acc={result['stability_accuracy']:.2f})"
        trans = f"trans(acc={result['transition_accuracy']:.2f} pres={result['transition_preservation']:.2f})"
        print(f"  {config_id:20s}  {stab}  {trans}")

    # ------------------------------------------------------------------
    # 4. Save JSON results
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    ablation_dir = out_dir / "ablation"
    plots_dir = ablation_dir / "plots"
    results_payload = {
        "study": "postprocessing_ablation",
        "dataset": "FER2013",
        "batch_size": BATCH_SIZE,
        "total_batches": len(batches),
        "total_frames": total_frames,
        "configs": all_config_results,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed": SEED,
            "limit": args.limit,
        },
    }
    save_results_json(results_payload, str(ablation_dir / "postprocess.json"))

    # ------------------------------------------------------------------
    # 5. Comparison plots
    # ------------------------------------------------------------------
    config_ids = list(all_config_results.keys())
    x = np.arange(len(config_ids))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Stability — Consistency Score
    ax = axes[0][0]
    _apply_dark_style(ax, fig)
    vals = [all_config_results[c]["stability_consistency"] for c in config_ids]
    ax.bar(x, vals, color=CB_PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Consistency Score")
    ax.set_title("Stability: Within-Emotion Consistency", fontweight="bold")
    ax.set_ylim(0, 1.05)

    # Top-right: Stability — Flicker Rate
    ax = axes[0][1]
    _apply_dark_style(ax, fig)
    vals = [all_config_results[c]["stability_flicker"] for c in config_ids]
    ax.bar(x, vals, color=CB_PALETTE[5])
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Flicker Rate (changes / batch)")
    ax.set_title("Stability: Emotion Flicker Rate", fontweight="bold")

    # Bottom-left: Transition — Accuracy
    ax = axes[1][0]
    _apply_dark_style(ax, fig)
    stab_acc = [all_config_results[c]["stability_accuracy"] for c in config_ids]
    trans_acc = [all_config_results[c]["transition_accuracy"] for c in config_ids]
    width = 0.35
    ax.bar(x - width / 2, stab_acc, width, label="Stability batches", color=CB_PALETTE[0])
    ax.bar(x + width / 2, trans_acc, width, label="Transition batches", color=CB_PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Per-Frame Accuracy")
    ax.set_title("Accuracy by Batch Type", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#2e2e4a", edgecolor="#2e2e4a", labelcolor="white", fontsize=8)

    # Bottom-right: Transition Preservation
    ax = axes[1][1]
    _apply_dark_style(ax, fig)
    vals = [all_config_results[c]["transition_preservation"] for c in config_ids]
    colors = [CB_PALETTE[2] if 0.5 <= v <= 1.5 else CB_PALETTE[5] for v in vals]
    ax.bar(x, vals, color=colors)
    ax.axhline(y=1.0, color="white", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Transition Preservation Ratio")
    ax.set_title("Transition Preservation (1.0 = ideal)", fontweight="bold")
    ax.set_ylim(0, 2.1)

    fig.suptitle(
        "Post-Processing Ablation — Stability + Transition Analysis",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=1.02,
    )
    _save_fig(fig, str(plots_dir / "postprocess_comparison.png"))

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("POST-PROCESSING ABLATION SUMMARY")
    print("=" * 80)
    print(f"  {'Config':20s}  {'Stab.Cons':>9s}  {'Stab.Flk':>8s}  {'Stab.Acc':>8s}  {'Trans.Acc':>9s}  {'Trans.Pres':>10s}")
    print("  " + "-" * 76)
    for cid in config_ids:
        r = all_config_results[cid]
        print(f"  {cid:20s}  {r['stability_consistency']:9.4f}  {r['stability_flicker']:8.2f}  "
              f"{r['stability_accuracy']:8.4f}  {r['transition_accuracy']:9.4f}  {r['transition_preservation']:10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args)
