"""
eval/benchmark_rafdb.py — RAF-DB Baseline Evaluation Runner

Evaluates DeepFace emotion recognition on the RAF-DB test split.
Produces classification report, confusion matrix, ROC/AUC, latency stats,
and publication-quality plots.

Usage
-----
    python -m eval.benchmark_rafdb --limit 100          # quick test run
    python -m eval.benchmark_rafdb                      # full evaluation
    python -m eval.benchmark_rafdb --detector opencv    # alternate detector

Standalone: no imports from src.backend.
"""

import argparse
import csv
import platform
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._gpu_init import init_gpu; init_gpu()  # must run before TF/DeepFace

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# DeepFace canonical emotion labels (lowercase, alphabetical)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# RAF-DB numeric folder -> DeepFace label mapping
RAFDB_LABEL_MAP = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DeepFace on RAF-DB test set",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "datasets"
            / "RAF-DB"
            / "DATASET"
            / "test"
        ),
        help="Path to RAF-DB test directory (subfolders 1-7).",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="",
        help="Optional path to test_labels.csv for cross-validation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max images to evaluate (0 = all). Useful for quick tests.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface",
        help="DeepFace detector backend (default: retinaface).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for output files.",
    )
    return parser.parse_args()


def load_labels_csv(csv_path: str) -> dict:
    """Load test_labels.csv and return {filename: emotion_label} dict.

    Expected CSV format: filename,label (header optional).
    Label is numeric 1-7, mapped via RAFDB_LABEL_MAP.
    """
    mapping = {}
    p = Path(csv_path)
    if not p.is_file():
        print(f"  WARNING: labels CSV not found at {p}, skipping cross-validation")
        return mapping

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            fname, label_num = row[0].strip(), row[1].strip()
            if label_num in RAFDB_LABEL_MAP:
                mapping[fname] = RAFDB_LABEL_MAP[label_num]
    print(f"  Loaded {len(mapping)} entries from labels CSV")
    return mapping


def collect_image_paths(dataset_dir: str) -> list:
    """Walk dataset_dir/{1..7}/*.{png,jpg,jpeg} and return (path, label) pairs."""
    root = Path(dataset_dir)
    if not root.is_dir():
        print(f"ERROR: dataset directory not found: {root}")
        sys.exit(1)

    samples = []
    for folder_num, emotion in RAFDB_LABEL_MAP.items():
        emotion_dir = root / folder_num
        if not emotion_dir.is_dir():
            print(f"  WARNING: missing sub-folder {emotion_dir}")
            continue
        for img_path in sorted(emotion_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                samples.append((str(img_path), emotion))

    print(f"Found {len(samples)} images across {root}")
    return samples


def run_benchmark(args: argparse.Namespace) -> None:
    # Lazy-import heavy libs so --help is fast
    from deepface import DeepFace
    from tqdm import tqdm

    from eval.metrics import (
        compute_classification_report,
        compute_confusion_matrix,
        compute_latency_stats,
        compute_roc_auc,
        save_results_json,
    )
    from eval.plot_results import (
        plot_confusion_matrix,
        plot_latency_histogram,
        plot_per_class_metrics,
        plot_roc_curves,
    )

    # ------------------------------------------------------------------
    # 1. Collect images
    # ------------------------------------------------------------------
    samples = collect_image_paths(args.dataset)
    if args.limit > 0:
        random.shuffle(samples)
        samples = samples[: args.limit]
        print(f"Limited to {len(samples)} images (--limit {args.limit})")

    total = len(samples)
    if total == 0:
        print("No images found. Exiting.")
        sys.exit(1)

    # Optional: load CSV labels for cross-validation reporting
    csv_labels = {}
    if args.labels_csv:
        csv_labels = load_labels_csv(args.labels_csv)

    # ------------------------------------------------------------------
    # 2. Inference loop
    # ------------------------------------------------------------------
    y_true: list[str] = []
    y_pred: list[str] = []
    y_scores: list[list[float]] = []
    latencies: list[float] = []
    failures: list[dict] = []
    csv_mismatches: int = 0

    print(f"\nRunning DeepFace.analyze  |  detector={args.detector}")
    for img_path, true_label in tqdm(samples, desc="RAF-DB eval", unit="img"):
        # Cross-validate with CSV if available
        if csv_labels:
            fname = Path(img_path).name
            csv_label = csv_labels.get(fname)
            if csv_label and csv_label != true_label:
                csv_mismatches += 1

        t0 = time.perf_counter()
        try:
            result = DeepFace.analyze(
                img_path=img_path,
                actions=["emotion"],
                detector_backend=args.detector,
                enforce_detection=False,
                silent=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            face = result[0] if isinstance(result, list) else result
            emotion_dist = face["emotion"]
            pred_label = face["dominant_emotion"]

            # Normalise scores to [0, 1]
            scores = [emotion_dist.get(e, 0.0) / 100.0 for e in EMOTION_LABELS]

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_scores.append(scores)
            latencies.append(elapsed_ms)

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            failures.append({
                "image": img_path,
                "error": str(exc),
                "true_label": true_label,
                "latency_ms": elapsed_ms,
            })

    detected = len(y_true)
    detection_rate = detected / total if total > 0 else 0.0
    print(f"\nDetected: {detected}/{total}  ({detection_rate:.2%})")
    print(f"Failures: {len(failures)}")
    if csv_labels:
        print(f"CSV cross-validation mismatches: {csv_mismatches}")

    if detected == 0:
        print("No successful detections. Cannot compute metrics.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Metrics
    # ------------------------------------------------------------------
    report = compute_classification_report(y_true, y_pred, EMOTION_LABELS)
    cm = compute_confusion_matrix(y_true, y_pred, EMOTION_LABELS)
    latency_stats = compute_latency_stats(latencies)

    # One-hot encode ground truth for ROC/AUC
    label_to_idx = {e: i for i, e in enumerate(EMOTION_LABELS)}
    y_true_onehot = np.zeros((detected, len(EMOTION_LABELS)), dtype=np.float64)
    for i, lbl in enumerate(y_true):
        y_true_onehot[i, label_to_idx[lbl]] = 1.0
    y_scores_arr = np.array(y_scores, dtype=np.float64)

    roc_data = compute_roc_auc(y_true_onehot, y_scores_arr, EMOTION_LABELS)

    # Overall accuracy
    accuracy = float(np.sum(np.array(y_true) == np.array(y_pred)) / detected)

    roc_auc_summary = {e: roc_data[e]["auc"] for e in EMOTION_LABELS}
    roc_auc_summary["micro_avg"] = roc_data["micro_avg"]["auc"]
    roc_auc_summary["macro_avg"] = roc_data["macro_avg"]["auc"]

    # ------------------------------------------------------------------
    # 4. Assemble and save JSON results
    # ------------------------------------------------------------------
    deepface_version = "unknown"
    try:
        import importlib.metadata
        deepface_version = importlib.metadata.version("deepface")
    except Exception:
        pass

    gpu_info = "N/A"
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_info = str([g.name for g in gpus])
        else:
            gpu_info = "CPU only"
    except Exception:
        pass

    results = {
        "dataset": "RAF-DB",
        "split": "test",
        "total_images": total,
        "detected_images": detected,
        "detection_rate": round(detection_rate, 4),
        "overall_accuracy": round(accuracy, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc_summary,
        "latency": latency_stats,
        "metadata": {
            "detector": args.detector,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "device": gpu_info,
            "deepface_version": deepface_version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed": SEED,
            "limit": args.limit,
            "label_map": RAFDB_LABEL_MAP,
            "csv_mismatches": csv_mismatches if csv_labels else "N/A",
        },
    }

    out_dir = Path(args.output_dir)
    baseline_dir = out_dir / "baseline"
    plots_dir = baseline_dir / "plots"

    det = args.detector
    save_results_json(results, str(baseline_dir / f"rafdb_{det}.json"))

    # ------------------------------------------------------------------
    # 5. Generate plots
    # ------------------------------------------------------------------
    det_label = det.capitalize()
    plot_confusion_matrix(
        cm,
        EMOTION_LABELS,
        title=f"RAF-DB — Confusion Matrix (DeepFace + {det_label})",
        save_path=str(plots_dir / f"rafdb_{det}_confusion_matrix.png"),
    )
    plot_roc_curves(
        roc_data,
        title=f"RAF-DB — ROC Curves (DeepFace + {det_label})",
        save_path=str(plots_dir / f"rafdb_{det}_roc_curves.png"),
    )
    plot_per_class_metrics(
        report,
        title=f"RAF-DB — Per-Class P/R/F1 ({det_label})",
        save_path=str(plots_dir / f"rafdb_{det}_per_class_metrics.png"),
    )
    plot_latency_histogram(
        latencies,
        title=f"RAF-DB — Latency Distribution ({det_label})",
        save_path=str(plots_dir / f"rafdb_{det}_latency_histogram.png"),
    )

    # ------------------------------------------------------------------
    # 6. Summary to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RAF-DB BASELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total images:    {total}")
    print(f"  Detected:        {detected}  ({detection_rate:.2%})")
    print(f"  Overall accuracy:{accuracy:.4f}")
    print(f"  Macro-avg AUC:   {roc_auc_summary['macro_avg']:.4f}")
    print(f"  Latency (mean):  {latency_stats['mean_ms']:.1f} ms")
    print(f"  Latency (p95):   {latency_stats['p95_ms']:.1f} ms")
    print(f"  Detector:        {args.detector}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
