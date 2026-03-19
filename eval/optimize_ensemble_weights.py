"""
eval/optimize_ensemble_weights.py — Ensemble Weight Optimization

Grid-searches over detector weights for (retinaface, mtcnn, centerface)
to find the optimal weighted-average combination for emotion recognition.

Workflow:
    1. Run each detector on a training subset (~2000 images from FER2013 train)
    2. Cache raw DeepFace results per (image, detector) to eval/cache/
    3. Grid search: w1 in [0.1..0.8], w2 in [0.1..remaining], w3 = 1 - w1 - w2
    4. Evaluate each weight combo on cached results
    5. Test optimal weights on full FER2013 test set
    6. Compare vs hand-tuned weights (0.50, 0.30, 0.20)

Usage
-----
    python -m eval.optimize_ensemble_weights --limit 200      # quick test
    python -m eval.optimize_ensemble_weights                  # full run
"""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

ENSEMBLE_DETECTORS = ["retinaface", "mtcnn", "centerface"]
HAND_TUNED_WEIGHTS = {"retinaface": 0.50, "mtcnn": 0.30, "centerface": 0.20}


# ---------------------------------------------------------------------------
# Full preprocessing (same as ablation_preprocess full_preprocess)
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
# Caching utilities
# ---------------------------------------------------------------------------
def _cache_key(img_path: str, detector: str) -> str:
    """Stable hash for (image_path, detector) pair."""
    raw = f"{Path(img_path).resolve()}|{detector}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cached_result(cache_dir: Path, img_path: str, detector: str) -> dict | None:
    """Load cached DeepFace result, or None if not cached."""
    key = _cache_key(img_path, detector)
    cache_file = cache_dir / f"{key}.json"
    if cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_cached_result(cache_dir: Path, img_path: str, detector: str, result: dict) -> None:
    """Persist DeepFace result to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(img_path, detector)
    cache_file = cache_dir / f"{key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble weight optimization via grid search",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "datasets" / "FER2013" / "train"),
        help="Path to FER2013 train directory for weight optimization.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "datasets" / "FER2013" / "test"),
        help="Path to FER2013 test directory for final evaluation.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=2000,
        help="Max training images for grid search (default: 2000).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max test images for final evaluation (0 = all).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "cache"),
        help="Directory for caching detector results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.1,
        help="Weight grid step size (default: 0.1).",
    )
    return parser.parse_args()


def collect_image_paths(dataset_dir: str) -> list:
    """Walk dataset_dir/<emotion>/*.{png,jpg,jpeg} and return (path, label) pairs."""
    root = Path(dataset_dir)
    if not root.is_dir():
        print(f"ERROR: dataset directory not found: {root}")
        sys.exit(1)

    samples = []
    for emotion in EMOTION_LABELS:
        emotion_dir = root / emotion
        if not emotion_dir.is_dir():
            print(f"  WARNING: missing sub-folder {emotion_dir}")
            continue
        for img_path in sorted(emotion_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                samples.append((str(img_path), emotion))

    print(f"Found {len(samples)} images across {root}")
    return samples


def run_detector_on_samples(
    samples: list,
    detector: str,
    cache_dir: Path,
) -> dict:
    """Run a single detector on all samples, caching results.

    Returns: {img_path: {"emotion": {...}, "dominant_emotion": str}} or None for failures.
    """
    from deepface import DeepFace
    from tqdm import tqdm

    results = {}
    cache_hits = 0
    cache_misses = 0

    for img_path, _true_label in tqdm(samples, desc=f"  {detector}", unit="img"):
        # Check cache first
        cached = load_cached_result(cache_dir, img_path, detector)
        if cached is not None:
            results[img_path] = cached
            cache_hits += 1
            continue

        cache_misses += 1
        frame = cv2.imread(img_path)
        if frame is None:
            results[img_path] = None
            continue

        processed = _apply_full_preprocess(frame)

        try:
            result = DeepFace.analyze(
                img_path=processed,
                actions=["emotion"],
                detector_backend=detector,
                enforce_detection=False,
                silent=True,
            )
            face = result[0] if isinstance(result, list) else result
            entry = {
                "emotion": face["emotion"],
                "dominant_emotion": face["dominant_emotion"],
            }
            results[img_path] = entry
            save_cached_result(cache_dir, img_path, detector, entry)

        except Exception:
            results[img_path] = None

    print(f"    Cache hits: {cache_hits}, misses: {cache_misses}")
    return results


def weighted_ensemble_predict(
    detector_results: dict,
    weights: dict,
    img_path: str,
) -> str | None:
    """Compute weighted ensemble prediction for a single image.

    Returns the dominant emotion label, or None if no valid results.
    """
    emotion_accumulator = {}
    total_weight = 0.0

    for detector, w in weights.items():
        if detector not in detector_results:
            continue
        result = detector_results[detector].get(img_path)
        if result is None:
            continue
        emotions = result.get("emotion", {})
        for emotion_label, score in emotions.items():
            emotion_accumulator[emotion_label] = emotion_accumulator.get(emotion_label, 0.0) + score * w
        total_weight += w

    if not emotion_accumulator or total_weight == 0:
        return None

    return max(emotion_accumulator, key=emotion_accumulator.get)


def run_optimization(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    from eval.metrics import (
        compute_classification_report,
        save_results_json,
    )

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # 1. Collect training images for grid search
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Collecting training images ---")
    train_samples = collect_image_paths(args.train_dir)
    if args.train_limit > 0 and len(train_samples) > args.train_limit:
        random.shuffle(train_samples)
        train_samples = train_samples[: args.train_limit]
        print(f"Using {len(train_samples)} training images (--train-limit {args.train_limit})")

    if len(train_samples) == 0:
        print("No training images found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Run each detector on training set (with caching)
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Running detectors on training set ---")
    train_detector_results = {}
    for detector in ENSEMBLE_DETECTORS:
        print(f"\nDetector: {detector}")
        train_detector_results[detector] = run_detector_on_samples(
            train_samples, detector, cache_dir
        )

    # ------------------------------------------------------------------
    # 3. Grid search over weights
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Grid search over ensemble weights ---")
    step = args.grid_step
    best_accuracy = 0.0
    best_weights = {}
    grid_results = []

    # Generate weight combinations: w1 from 0.1 to 0.8, w2 from 0.1 to (1-w1-0.1), w3 = 1-w1-w2
    w1_values = np.arange(step, 0.8 + step / 2, step)

    for w1 in w1_values:
        w2_max = 1.0 - w1 - step + step / 2  # ensure w3 >= step
        w2_values = np.arange(step, w2_max + step / 2, step)
        for w2 in w2_values:
            w3 = 1.0 - w1 - w2
            if w3 < step / 2:
                continue

            weights = {
                ENSEMBLE_DETECTORS[0]: round(float(w1), 2),
                ENSEMBLE_DETECTORS[1]: round(float(w2), 2),
                ENSEMBLE_DETECTORS[2]: round(float(w3), 2),
            }

            # Evaluate on training set
            y_true = []
            y_pred = []
            for img_path, true_label in train_samples:
                pred = weighted_ensemble_predict(train_detector_results, weights, img_path)
                if pred is not None:
                    y_true.append(true_label)
                    y_pred.append(pred)

            if len(y_true) == 0:
                continue

            accuracy = float(np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true))

            grid_results.append({
                "weights": weights,
                "accuracy": round(accuracy, 4),
                "num_evaluated": len(y_true),
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.copy()

    print(f"\nGrid search complete: {len(grid_results)} weight combinations tested")
    print(f"Best training accuracy: {best_accuracy:.4f}")
    print(f"Best weights: {best_weights}")

    # Sort grid results by accuracy (descending)
    grid_results.sort(key=lambda x: x["accuracy"], reverse=True)

    # ------------------------------------------------------------------
    # 4. Evaluate on test set: optimal vs hand-tuned
    # ------------------------------------------------------------------
    print("\n--- Phase 4: Evaluating on FER2013 test set ---")
    test_samples = collect_image_paths(args.test_dir)
    if args.limit > 0 and len(test_samples) > args.limit:
        random.shuffle(test_samples)
        test_samples = test_samples[: args.limit]
        print(f"Using {len(test_samples)} test images (--limit {args.limit})")

    if len(test_samples) == 0:
        print("No test images found. Exiting.")
        sys.exit(1)

    # Run detectors on test set
    test_detector_results = {}
    for detector in ENSEMBLE_DETECTORS:
        print(f"\nDetector: {detector}")
        test_detector_results[detector] = run_detector_on_samples(
            test_samples, detector, cache_dir
        )

    # Evaluate both weight sets
    weight_sets = {
        "optimized": best_weights,
        "hand_tuned": HAND_TUNED_WEIGHTS,
    }

    test_evaluations = {}
    for label, weights in weight_sets.items():
        y_true = []
        y_pred = []
        for img_path, true_label in test_samples:
            pred = weighted_ensemble_predict(test_detector_results, weights, img_path)
            if pred is not None:
                y_true.append(true_label)
                y_pred.append(pred)

        if len(y_true) > 0:
            report = compute_classification_report(y_true, y_pred, EMOTION_LABELS)
            accuracy = float(np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true))
            weighted_f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
        else:
            accuracy = 0.0
            weighted_f1 = 0.0
            report = {}

        test_evaluations[label] = {
            "weights": weights,
            "accuracy": round(accuracy, 4),
            "weighted_f1": round(weighted_f1, 4),
            "num_evaluated": len(y_true),
            "classification_report": report,
        }

        print(f"\n  {label}: accuracy={accuracy:.4f}, F1={weighted_f1:.4f}, "
              f"weights={weights}")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results_payload = {
        "study": "ensemble_weight_optimization",
        "dataset": "FER2013",
        "detectors": ENSEMBLE_DETECTORS,
        "grid_search": {
            "step": step,
            "total_combinations": len(grid_results),
            "best_training_accuracy": best_accuracy,
            "best_weights": best_weights,
            "top_10_combos": grid_results[:10],
        },
        "test_evaluation": test_evaluations,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed": SEED,
            "train_limit": args.train_limit,
            "test_limit": args.limit,
            "train_images": len(train_samples),
            "test_images": len(test_samples),
        },
    }
    save_results_json(results_payload, str(out_dir / "ablation_ensemble.json"))

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ENSEMBLE WEIGHT OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"  Grid step:        {step}")
    print(f"  Combos tested:    {len(grid_results)}")
    print(f"  Training images:  {len(train_samples)}")
    print(f"  Test images:      {len(test_samples)}")
    print(f"\n  Best training accuracy: {best_accuracy:.4f}")
    print(f"  Optimized weights:      {best_weights}")
    print(f"  Hand-tuned weights:     {HAND_TUNED_WEIGHTS}")
    for label, ev in test_evaluations.items():
        print(f"\n  [{label}]")
        print(f"    Test accuracy:  {ev['accuracy']:.4f}")
        print(f"    Test wt-F1:     {ev['weighted_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_optimization(args)
