"""
eval/ablation_preprocess.py — Preprocessing Ablation Study

Evaluates the impact of individual preprocessing steps (super-resolution,
CLAHE, unsharp mask) on emotion recognition accuracy using the FER2013 test
set with a fixed RetinaFace detector.

Configurations tested:
    none            — raw image, no preprocessing
    sr_only         — super-resolution upscale only
    clahe_only      — CLAHE lighting normalization only
    sr_clahe        — super-resolution + CLAHE
    full_preprocess — super-resolution + CLAHE + unsharp mask

Usage
-----
    python -m eval.ablation_preprocess --limit 100        # quick test
    python -m eval.ablation_preprocess                    # full run
"""

import argparse
import platform
import random
import sys
import time
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

# DeepFace canonical emotion labels (lowercase, alphabetical)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Fixed detector for this ablation — isolate preprocessing effects only
DETECTOR = "retinaface"


# ---------------------------------------------------------------------------
# Standalone preprocessing primitives (avoid config dependency)
# ---------------------------------------------------------------------------
def _apply_super_resolve(frame: np.ndarray) -> np.ndarray:
    """Upscale small images using cubic interpolation (no sharpening)."""
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
    return frame


def _apply_clahe(frame: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel in LAB space."""
    if frame is None or len(frame.shape) < 3 or frame.shape[2] != 3:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _apply_unsharp_mask(frame: np.ndarray) -> np.ndarray:
    """Unsharp mask for edge crispness."""
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(frame, 1.25, blurred, -0.25, 0)


# ---------------------------------------------------------------------------
# Preprocessing configurations
# ---------------------------------------------------------------------------
PREPROCESS_CONFIGS = {
    "none": {
        "super_resolve": False,
        "clahe": False,
        "unsharp_mask": False,
    },
    "sr_only": {
        "super_resolve": True,
        "clahe": False,
        "unsharp_mask": False,
    },
    "clahe_only": {
        "super_resolve": False,
        "clahe": True,
        "unsharp_mask": False,
    },
    "sr_clahe": {
        "super_resolve": True,
        "clahe": True,
        "unsharp_mask": False,
    },
    "full_preprocess": {
        "super_resolve": True,
        "clahe": True,
        "unsharp_mask": True,
    },
}


def apply_preprocessing(frame: np.ndarray, config: dict) -> np.ndarray:
    """Apply a specific preprocessing configuration to a frame."""
    result = frame.copy()
    if config["super_resolve"]:
        result = _apply_super_resolve(result)
    if config["unsharp_mask"]:
        result = _apply_unsharp_mask(result)
    if config["clahe"]:
        result = _apply_clahe(result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing ablation study on FER2013",
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
        help="Max images per configuration (0 = all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for output files.",
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


def run_ablation(args: argparse.Namespace) -> None:
    # Lazy-import heavy libs so --help is fast
    from deepface import DeepFace
    from tqdm import tqdm

    from eval.metrics import (
        compute_classification_report,
        compute_latency_stats,
        save_results_json,
    )
    from eval.plot_results import (
        _apply_dark_style,
        _save_fig,
        CB_PALETTE,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # 1. Collect images
    # ------------------------------------------------------------------
    samples = collect_image_paths(args.dataset_dir)
    if args.limit > 0:
        random.shuffle(samples)
        samples = samples[: args.limit]
        print(f"Limited to {len(samples)} images (--limit {args.limit})")

    total = len(samples)
    if total == 0:
        print("No images found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Run each preprocessing configuration
    # ------------------------------------------------------------------
    all_results = {}

    for config_id, config in PREPROCESS_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_id}  |  SR={config['super_resolve']}  "
              f"CLAHE={config['clahe']}  Unsharp={config['unsharp_mask']}")
        print(f"{'='*60}")

        y_true = []
        y_pred = []
        latencies = []
        failures = []

        for img_path, true_label in tqdm(samples, desc=config_id, unit="img"):
            # Load image
            frame = cv2.imread(img_path)
            if frame is None:
                failures.append({"image": img_path, "error": "cv2.imread returned None"})
                continue

            # Apply preprocessing
            processed = apply_preprocessing(frame, config)

            # Inference
            t0 = time.perf_counter()
            try:
                result = DeepFace.analyze(
                    img_path=processed,
                    actions=["emotion"],
                    detector_backend=DETECTOR,
                    enforce_detection=False,
                    silent=True,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                face = result[0] if isinstance(result, list) else result
                pred_label = face["dominant_emotion"]

                y_true.append(true_label)
                y_pred.append(pred_label)
                latencies.append(elapsed_ms)

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                failures.append({
                    "image": img_path,
                    "error": str(exc),
                    "latency_ms": elapsed_ms,
                })

        detected = len(y_true)
        detection_rate = detected / total if total > 0 else 0.0

        if detected > 0:
            report = compute_classification_report(y_true, y_pred, EMOTION_LABELS)
            accuracy = float(np.sum(np.array(y_true) == np.array(y_pred)) / detected)
            latency_stats = compute_latency_stats(latencies)

            # Per-class F1
            per_class_f1 = {
                e: report[e]["f1-score"]
                for e in EMOTION_LABELS
                if e in report
            }
        else:
            report = {}
            accuracy = 0.0
            latency_stats = {}
            per_class_f1 = {}

        config_result = {
            "config_id": config_id,
            "settings": config,
            "total_images": total,
            "detected_images": detected,
            "detection_rate": round(detection_rate, 4),
            "accuracy": round(accuracy, 4),
            "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0.0), 4),
            "per_class_f1": per_class_f1,
            "latency": latency_stats,
            "num_failures": len(failures),
        }
        all_results[config_id] = config_result

        print(f"  Accuracy:       {accuracy:.4f}")
        print(f"  Weighted F1:    {config_result['weighted_f1']:.4f}")
        print(f"  Detection rate: {detection_rate:.2%}")
        if latency_stats:
            print(f"  Mean latency:   {latency_stats.get('mean_ms', 0):.1f} ms")

    # ------------------------------------------------------------------
    # 3. Save JSON results
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    ablation_dir = out_dir / "ablation"
    plots_dir = ablation_dir / "plots"
    results_payload = {
        "study": "preprocessing_ablation",
        "dataset": "FER2013",
        "detector": DETECTOR,
        "configs": all_results,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed": SEED,
            "limit": args.limit,
            "total_images": total,
        },
    }
    save_results_json(results_payload, str(ablation_dir / "preprocess.json"))

    # ------------------------------------------------------------------
    # 4. Comparison bar chart
    # ------------------------------------------------------------------
    config_ids = list(all_results.keys())
    accuracies = [all_results[c]["accuracy"] for c in config_ids]
    f1_scores = [all_results[c]["weighted_f1"] for c in config_ids]
    det_rates = [all_results[c]["detection_rate"] for c in config_ids]
    mean_lats = [
        all_results[c]["latency"].get("mean_ms", 0) if all_results[c]["latency"] else 0
        for c in config_ids
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Accuracy + F1
    ax = axes[0]
    _apply_dark_style(ax, fig)
    x = np.arange(len(config_ids))
    width = 0.35
    ax.bar(x - width / 2, accuracies, width, label="Accuracy", color=CB_PALETTE[0])
    ax.bar(x + width / 2, f1_scores, width, label="Weighted F1", color=CB_PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy & F1 by Preprocessing", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#2e2e4a", edgecolor="#2e2e4a", labelcolor="white")

    # Detection rate
    ax = axes[1]
    _apply_dark_style(ax, fig)
    ax.bar(x, det_rates, color=CB_PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Detection Rate")
    ax.set_title("Detection Rate by Preprocessing", fontweight="bold")
    ax.set_ylim(0, 1.05)

    # Mean latency
    ax = axes[2]
    _apply_dark_style(ax, fig)
    ax.bar(x, mean_lats, color=CB_PALETTE[5])
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Mean Latency by Preprocessing", fontweight="bold")

    fig.suptitle(
        "Preprocessing Ablation — FER2013",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=1.02,
    )
    _save_fig(fig, str(plots_dir / "preprocess_comparison.png"))

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING ABLATION SUMMARY")
    print("=" * 60)
    for cid in config_ids:
        r = all_results[cid]
        lat = r["latency"].get("mean_ms", 0) if r["latency"] else 0
        print(f"  {cid:20s}  acc={r['accuracy']:.4f}  F1={r['weighted_f1']:.4f}  "
              f"det={r['detection_rate']:.2%}  lat={lat:.1f}ms")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args)
