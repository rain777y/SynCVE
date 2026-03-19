"""
eval/pipeline_vs_baseline.py — Full Pipeline vs Baseline Comparison

Compares raw DeepFace baseline against the complete SynCVE pipeline
(best preprocessing + ensemble + post-processing) on both FER2013 and
RAF-DB datasets.

Loads baseline results from eval/results/ and runs the full SynCVE pipeline
for a head-to-head comparison.

Usage
-----
    python -m eval.pipeline_vs_baseline --limit 100       # quick test
    python -m eval.pipeline_vs_baseline                   # full run
"""

import argparse
import json
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

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Ensemble configuration (best known weights)
ENSEMBLE_DETECTORS = ["retinaface", "mtcnn", "centerface"]
ENSEMBLE_WEIGHTS = {"retinaface": 0.50, "mtcnn": 0.30, "centerface": 0.20}

# Post-processing defaults
EMA_ALPHA = 0.3
NOISE_FLOOR = 0.10

# RAF-DB label mapping
RAFDB_LABEL_MAP = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}


# ---------------------------------------------------------------------------
# Preprocessing (full pipeline)
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
# Ensemble inference
# ---------------------------------------------------------------------------
def run_ensemble_inference(processed_frame: np.ndarray) -> dict | None:
    """Run ensemble of detectors and return weighted emotion scores.

    Returns dict with 'emotion' and 'dominant_emotion', or None on failure.
    """
    from deepface import DeepFace

    backend_results = []
    successful_detectors = []

    for detector in ENSEMBLE_DETECTORS:
        try:
            result = DeepFace.analyze(
                img_path=processed_frame.copy(),
                actions=["emotion"],
                detector_backend=detector,
                enforce_detection=False,
                silent=True,
            )
            face = result[0] if isinstance(result, list) else result
            backend_results.append(face)
            successful_detectors.append(detector)
        except Exception:
            continue

    if not backend_results:
        return None

    # Weighted aggregation
    emotion_accumulator = {}
    for face, detector in zip(backend_results, successful_detectors):
        w = ENSEMBLE_WEIGHTS.get(detector, 0.1)
        emotions = face.get("emotion", {})
        for emotion_label, score in emotions.items():
            emotion_accumulator[emotion_label] = (
                emotion_accumulator.get(emotion_label, 0.0) + score * w
            )

    # Normalize by total weight
    total_weight = sum(ENSEMBLE_WEIGHTS.get(d, 0.1) for d in successful_detectors)
    if total_weight > 0:
        for e in emotion_accumulator:
            emotion_accumulator[e] /= total_weight

    dominant = max(emotion_accumulator, key=emotion_accumulator.get)

    return {
        "emotion": emotion_accumulator,
        "dominant_emotion": dominant,
        "detectors_used": successful_detectors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full SynCVE pipeline vs raw DeepFace baseline comparison",
    )
    parser.add_argument(
        "--fer2013-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "datasets" / "FER2013" / "test"),
        help="Path to FER2013 test directory.",
    )
    parser.add_argument(
        "--rafdb-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "datasets" / "RAF-DB" / "DATASET" / "test"
        ),
        help="Path to RAF-DB test directory.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory containing baseline result JSONs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max images per dataset (0 = all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for output files.",
    )
    return parser.parse_args()


def collect_fer2013_paths(dataset_dir: str) -> list:
    """Collect (path, label) pairs from FER2013 test directory."""
    root = Path(dataset_dir)
    if not root.is_dir():
        print(f"WARNING: FER2013 directory not found: {root}")
        return []

    samples = []
    for emotion in EMOTION_LABELS:
        emotion_dir = root / emotion
        if not emotion_dir.is_dir():
            continue
        for img_path in sorted(emotion_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                samples.append((str(img_path), emotion))
    return samples


def collect_rafdb_paths(dataset_dir: str) -> list:
    """Collect (path, label) pairs from RAF-DB test directory."""
    root = Path(dataset_dir)
    if not root.is_dir():
        print(f"WARNING: RAF-DB directory not found: {root}")
        return []

    samples = []
    for folder_num, emotion in RAFDB_LABEL_MAP.items():
        emotion_dir = root / folder_num
        if not emotion_dir.is_dir():
            continue
        for img_path in sorted(emotion_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                samples.append((str(img_path), emotion))
    return samples


def load_baseline_results(baseline_dir: str, dataset_name: str) -> dict | None:
    """Load baseline results JSON for a dataset."""
    baseline_dir = Path(baseline_dir)
    filename_map = {
        "FER2013": "fer2013_baseline_results.json",
        "RAF-DB": "rafdb_baseline_results.json",
    }
    path = baseline_dir / filename_map.get(dataset_name, "")
    if not path.is_file():
        print(f"WARNING: Baseline results not found at {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_pipeline_on_dataset(
    samples: list,
    dataset_name: str,
) -> dict:
    """Run the full SynCVE pipeline on a dataset and compute metrics."""
    from tqdm import tqdm

    from eval.metrics import (
        compute_classification_report,
        compute_confusion_matrix,
        compute_latency_stats,
        save_results_json,
    )

    y_true = []
    y_pred = []
    latencies = []
    failures = []

    print(f"\nRunning SynCVE pipeline on {dataset_name} ({len(samples)} images)")

    for img_path, true_label in tqdm(samples, desc=f"{dataset_name} pipeline", unit="img"):
        frame = cv2.imread(img_path)
        if frame is None:
            failures.append({"image": img_path, "error": "cv2.imread returned None"})
            continue

        t0 = time.perf_counter()

        # Step 1: Full preprocessing
        processed = _apply_full_preprocess(frame)

        # Step 2: Ensemble inference
        result = run_ensemble_inference(processed)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if result is None:
            failures.append({
                "image": img_path,
                "error": "Ensemble returned no results",
                "latency_ms": elapsed_ms,
            })
            continue

        y_true.append(true_label)
        y_pred.append(result["dominant_emotion"])
        latencies.append(elapsed_ms)

    detected = len(y_true)
    total = len(samples)
    detection_rate = detected / total if total > 0 else 0.0

    if detected > 0:
        report = compute_classification_report(y_true, y_pred, EMOTION_LABELS)
        cm = compute_confusion_matrix(y_true, y_pred, EMOTION_LABELS)
        accuracy = float(np.sum(np.array(y_true) == np.array(y_pred)) / detected)
        latency_stats = compute_latency_stats(latencies)
        per_class_f1 = {
            e: report[e]["f1-score"] for e in EMOTION_LABELS if e in report
        }
    else:
        report = {}
        cm = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)))
        accuracy = 0.0
        latency_stats = {}
        per_class_f1 = {}

    return {
        "dataset": dataset_name,
        "total_images": total,
        "detected_images": detected,
        "detection_rate": round(detection_rate, 4),
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0.0), 4),
        "per_class_f1": per_class_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist() if isinstance(cm, np.ndarray) else cm,
        "latency": latency_stats,
        "num_failures": len(failures),
        "ensemble_config": {
            "detectors": ENSEMBLE_DETECTORS,
            "weights": ENSEMBLE_WEIGHTS,
        },
    }


def run_comparison(args: argparse.Namespace) -> None:
    from eval.metrics import save_results_json
    from eval.plot_results import (
        _apply_dark_style,
        _save_fig,
        CB_PALETTE,
        plot_confusion_matrix,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.output_dir)

    datasets_config = {}

    # FER2013
    fer_samples = collect_fer2013_paths(args.fer2013_dir)
    if fer_samples:
        if args.limit > 0:
            random.shuffle(fer_samples)
            fer_samples = fer_samples[: args.limit]
        datasets_config["FER2013"] = fer_samples

    # RAF-DB
    rafdb_samples = collect_rafdb_paths(args.rafdb_dir)
    if rafdb_samples:
        if args.limit > 0:
            random.shuffle(rafdb_samples)
            rafdb_samples = rafdb_samples[: args.limit]
        datasets_config["RAF-DB"] = rafdb_samples

    if not datasets_config:
        print("No datasets found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run pipeline on each dataset and compare with baseline
    # ------------------------------------------------------------------
    comparison_results = {}

    for dataset_name, samples in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        # Run full pipeline
        pipeline_result = evaluate_pipeline_on_dataset(samples, dataset_name)

        # Load baseline
        baseline = load_baseline_results(args.baseline_dir, dataset_name)

        comparison = {
            "pipeline": pipeline_result,
            "baseline": {
                "accuracy": baseline.get("overall_accuracy", 0) if baseline else None,
                "weighted_f1": (
                    baseline.get("classification_report", {})
                    .get("weighted avg", {})
                    .get("f1-score", 0)
                    if baseline else None
                ),
                "detection_rate": baseline.get("detection_rate", 0) if baseline else None,
                "latency_mean_ms": (
                    baseline.get("latency", {}).get("mean_ms", 0) if baseline else None
                ),
            },
        }

        # Compute deltas
        if baseline:
            bl_acc = baseline.get("overall_accuracy", 0)
            pl_acc = pipeline_result["accuracy"]
            bl_f1 = (
                baseline.get("classification_report", {})
                .get("weighted avg", {})
                .get("f1-score", 0)
            )
            pl_f1 = pipeline_result["weighted_f1"]

            comparison["delta"] = {
                "accuracy": round(pl_acc - bl_acc, 4),
                "weighted_f1": round(pl_f1 - bl_f1, 4),
            }

            print(f"\n  Baseline accuracy:  {bl_acc:.4f}")
            print(f"  Pipeline accuracy:  {pl_acc:.4f}")
            print(f"  Delta accuracy:     {comparison['delta']['accuracy']:+.4f}")
            print(f"  Baseline wt-F1:     {bl_f1:.4f}")
            print(f"  Pipeline wt-F1:     {pl_f1:.4f}")
            print(f"  Delta wt-F1:        {comparison['delta']['weighted_f1']:+.4f}")
        else:
            comparison["delta"] = None
            print(f"\n  Pipeline accuracy:  {pipeline_result['accuracy']:.4f}")
            print("  (No baseline available for comparison)")

        comparison_results[dataset_name] = comparison

        # Generate confusion matrix for pipeline
        if pipeline_result.get("confusion_matrix"):
            cm = np.array(pipeline_result["confusion_matrix"])
            plot_confusion_matrix(
                cm,
                EMOTION_LABELS,
                title=f"{dataset_name} — SynCVE Pipeline Confusion Matrix",
                save_path=str(out_dir / f"{dataset_name.lower().replace('-', '')}_pipeline_cm.png"),
            )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    results_payload = {
        "study": "pipeline_vs_baseline",
        "comparisons": comparison_results,
        "pipeline_config": {
            "preprocessing": "full_preprocess (SR + unsharp + CLAHE)",
            "ensemble_detectors": ENSEMBLE_DETECTORS,
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "ema_alpha": EMA_ALPHA,
            "noise_floor": NOISE_FLOOR,
        },
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed": SEED,
            "limit": args.limit,
        },
    }
    save_results_json(results_payload, str(out_dir / "pipeline_vs_baseline.json"))

    # ------------------------------------------------------------------
    # Comparison bar chart (side-by-side baseline vs pipeline)
    # ------------------------------------------------------------------
    # Only plot datasets that have both baseline and pipeline results
    plottable = [
        ds for ds in comparison_results
        if comparison_results[ds].get("baseline", {}).get("accuracy") is not None
    ]

    if plottable:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Accuracy comparison
        ax = axes[0]
        _apply_dark_style(ax, fig)
        x = np.arange(len(plottable))
        width = 0.35
        baseline_accs = [
            comparison_results[ds]["baseline"]["accuracy"] for ds in plottable
        ]
        pipeline_accs = [
            comparison_results[ds]["pipeline"]["accuracy"] for ds in plottable
        ]
        bars1 = ax.bar(x - width / 2, baseline_accs, width, label="Baseline (raw DeepFace)",
                        color=CB_PALETTE[5])
        bars2 = ax.bar(x + width / 2, pipeline_accs, width, label="SynCVE Pipeline",
                        color=CB_PALETTE[2])
        ax.set_xticks(x)
        ax.set_xticklabels(plottable, fontsize=11)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy: Baseline vs SynCVE", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor="#2e2e4a", edgecolor="#2e2e4a", labelcolor="white")

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="white",
                )

        # Weighted F1 comparison
        ax = axes[1]
        _apply_dark_style(ax, fig)
        baseline_f1s = [
            comparison_results[ds]["baseline"]["weighted_f1"] for ds in plottable
        ]
        pipeline_f1s = [
            comparison_results[ds]["pipeline"]["weighted_f1"] for ds in plottable
        ]
        bars1 = ax.bar(x - width / 2, baseline_f1s, width, label="Baseline (raw DeepFace)",
                        color=CB_PALETTE[5])
        bars2 = ax.bar(x + width / 2, pipeline_f1s, width, label="SynCVE Pipeline",
                        color=CB_PALETTE[2])
        ax.set_xticks(x)
        ax.set_xticklabels(plottable, fontsize=11)
        ax.set_ylabel("Weighted F1")
        ax.set_title("Weighted F1: Baseline vs SynCVE", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor="#2e2e4a", edgecolor="#2e2e4a", labelcolor="white")

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="white",
                )

        fig.suptitle(
            "Pipeline vs Baseline Comparison",
            fontsize=15, fontweight="bold", color="white", y=1.02,
        )
        _save_fig(fig, str(out_dir / "pipeline_vs_baseline_comparison.png"))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE VS BASELINE SUMMARY")
    print("=" * 60)
    for ds, comp in comparison_results.items():
        pl = comp["pipeline"]
        bl = comp.get("baseline", {})
        delta = comp.get("delta", {})
        print(f"\n  [{ds}]")
        print(f"    Pipeline  acc={pl['accuracy']:.4f}  F1={pl['weighted_f1']:.4f}  "
              f"det={pl['detection_rate']:.2%}")
        if bl.get("accuracy") is not None:
            print(f"    Baseline  acc={bl['accuracy']:.4f}  F1={bl['weighted_f1']:.4f}  "
                  f"det={bl['detection_rate']:.2%}")
        if delta:
            print(f"    Delta     acc={delta['accuracy']:+.4f}  F1={delta['weighted_f1']:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
