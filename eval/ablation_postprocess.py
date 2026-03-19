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
    """Build same-emotion batches of `batch_size` images.

    Returns list of (emotion_label, [img_path, ...]) tuples.
    """
    batches = []
    for emotion, paths in by_emotion.items():
        # Shuffle within each emotion for variety
        shuffled = paths.copy()
        random.shuffle(shuffled)
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i : i + batch_size]
            if len(batch) == batch_size:
                batches.append((emotion, batch))

    random.shuffle(batches)

    # Limit total images (in batches)
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
    # 1. Build same-emotion batches
    # ------------------------------------------------------------------
    by_emotion = collect_image_paths_by_emotion(args.dataset_dir)
    batches = build_batches(by_emotion, BATCH_SIZE, args.limit)
    total_frames = sum(len(b) for _, b in batches)
    print(f"\nBuilt {len(batches)} batches of {BATCH_SIZE} frames ({total_frames} total)")

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
    for _, batch_paths in batches:
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

        batch_consistencies = []
        batch_flicker_rates = []
        batch_accuracies = []

        for true_emotion, batch_paths in batches:
            # Collect raw scores for this batch
            batch_scores = []
            batch_true_labels = []
            for img_path in batch_paths:
                r = raw_results.get(img_path)
                if r is None:
                    continue
                batch_scores.append(r["emotion"])
                batch_true_labels.append(true_emotion)

            if len(batch_scores) < 2:
                continue

            # Apply EMA if configured
            if ema_alpha is not None:
                processed_scores = apply_ema(batch_scores, ema_alpha)
            else:
                processed_scores = [s.copy() for s in batch_scores]

            # Apply noise floor if configured
            if noise_floor is not None:
                processed_scores = [
                    apply_noise_floor(s, noise_floor) for s in processed_scores
                ]

            # Compute per-frame dominant emotions
            dominant_seq = [get_dominant(s) for s in processed_scores]

            # Metrics
            consistency = compute_consistency_score(dominant_seq)
            flicker = compute_flicker_rate(dominant_seq)
            frame_accuracy = sum(
                1 for d, t in zip(dominant_seq, batch_true_labels) if d == t
            ) / len(dominant_seq)

            batch_consistencies.append(consistency)
            batch_flicker_rates.append(flicker)
            batch_accuracies.append(frame_accuracy)

        if batch_consistencies:
            mean_consistency = float(np.mean(batch_consistencies))
            mean_flicker = float(np.mean(batch_flicker_rates))
            mean_accuracy = float(np.mean(batch_accuracies))
        else:
            mean_consistency = 0.0
            mean_flicker = 0.0
            mean_accuracy = 0.0

        all_config_results[config_id] = {
            "config_id": config_id,
            "settings": config,
            "consistency_score": round(mean_consistency, 4),
            "flicker_rate": round(mean_flicker, 4),
            "accuracy": round(mean_accuracy, 4),
            "num_batches": len(batch_consistencies),
        }

        print(f"  {config_id:20s}  consistency={mean_consistency:.4f}  "
              f"flicker={mean_flicker:.2f}  acc={mean_accuracy:.4f}")

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
    consistencies = [all_config_results[c]["consistency_score"] for c in config_ids]
    flicker_rates = [all_config_results[c]["flicker_rate"] for c in config_ids]
    accuracies = [all_config_results[c]["accuracy"] for c in config_ids]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Consistency
    ax = axes[0]
    _apply_dark_style(ax, fig)
    x = np.arange(len(config_ids))
    ax.bar(x, consistencies, color=CB_PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Consistency Score")
    ax.set_title("Temporal Consistency", fontweight="bold")
    ax.set_ylim(0, 1.05)

    # Flicker rate
    ax = axes[1]
    _apply_dark_style(ax, fig)
    ax.bar(x, flicker_rates, color=CB_PALETTE[5])
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Flicker Rate (changes / batch)")
    ax.set_title("Emotion Flicker Rate", fontweight="bold")

    # Accuracy
    ax = axes[2]
    _apply_dark_style(ax, fig)
    ax.bar(x, accuracies, color=CB_PALETTE[0])
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Frame Accuracy", fontweight="bold")
    ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Post-Processing Ablation — FER2013 (Temporal Simulation)",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=1.02,
    )
    _save_fig(fig, str(plots_dir / "postprocess_comparison.png"))

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("POST-PROCESSING ABLATION SUMMARY")
    print("=" * 60)
    for cid in config_ids:
        r = all_config_results[cid]
        print(f"  {cid:20s}  consistency={r['consistency_score']:.4f}  "
              f"flicker={r['flicker_rate']:.2f}  acc={r['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args)
