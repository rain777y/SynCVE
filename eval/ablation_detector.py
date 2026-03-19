"""
eval/ablation_detector.py — Detector Backend Ablation Study

Evaluates different face detector backends with full preprocessing applied
to isolate detector impact on emotion recognition accuracy and speed.

Detectors tested: retinaface, mtcnn, centerface, opencv, ssd

Usage
-----
    python -m eval.ablation_detector --limit 100        # quick test
    python -m eval.ablation_detector                    # full run
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

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

DETECTORS = ["retinaface", "mtcnn", "centerface", "opencv", "ssd"]


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (matches full_preprocess config)
# ---------------------------------------------------------------------------
def _apply_full_preprocess(frame: np.ndarray) -> np.ndarray:
    """Apply super-resolution + unsharp mask + CLAHE."""
    if frame is None or not hasattr(frame, "shape"):
        return frame

    # Super-resolution upscale
    height, width = frame.shape[:2]
    min_size = min(height, width)
    target_min = 256
    if min_size < target_min:
        scale = target_min / float(min_size)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Unsharp mask
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    frame = cv2.addWeighted(frame, 1.25, blurred, -0.25, 0)

    # CLAHE
    if len(frame.shape) >= 3 and frame.shape[2] == 3:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        merged = cv2.merge((l_eq, a_channel, b_channel))
        frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detector backend ablation study on FER2013",
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
        help="Max images per detector (0 = all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--detectors",
        type=str,
        nargs="+",
        default=DETECTORS,
        help="Detector backends to test.",
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
    # 1. Collect and limit images
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

    # Pre-process all images once (same preprocessing for all detectors)
    print("\nPre-processing all images with full pipeline...")
    preprocessed = []
    for img_path, true_label in tqdm(samples, desc="Preprocessing", unit="img"):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        processed = _apply_full_preprocess(frame)
        preprocessed.append((processed, true_label, img_path))
    print(f"Successfully preprocessed {len(preprocessed)}/{total} images")

    # ------------------------------------------------------------------
    # 2. Run each detector
    # ------------------------------------------------------------------
    all_results = {}

    for detector in args.detectors:
        print(f"\n{'='*60}")
        print(f"Detector: {detector}")
        print(f"{'='*60}")

        y_true = []
        y_pred = []
        latencies = []
        failures = []

        for processed, true_label, img_path in tqdm(
            preprocessed, desc=detector, unit="img"
        ):
            t0 = time.perf_counter()
            try:
                result = DeepFace.analyze(
                    img_path=processed.copy(),
                    actions=["emotion"],
                    detector_backend=detector,
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
        detection_rate = detected / len(preprocessed) if len(preprocessed) > 0 else 0.0

        if detected > 0:
            report = compute_classification_report(y_true, y_pred, EMOTION_LABELS)
            accuracy = float(np.sum(np.array(y_true) == np.array(y_pred)) / detected)
            latency_stats = compute_latency_stats(latencies)
        else:
            report = {}
            accuracy = 0.0
            latency_stats = {
                "mean_ms": 0, "median_ms": 0, "p95_ms": 0, "p99_ms": 0,
                "min_ms": 0, "max_ms": 0, "std_ms": 0, "total_samples": 0,
            }

        det_result = {
            "detector": detector,
            "total_images": len(preprocessed),
            "detected_images": detected,
            "detection_rate": round(detection_rate, 4),
            "accuracy": round(accuracy, 4),
            "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0.0), 4),
            "latency": latency_stats,
            "num_failures": len(failures),
        }
        all_results[detector] = det_result

        print(f"  Accuracy:       {accuracy:.4f}")
        print(f"  Weighted F1:    {det_result['weighted_f1']:.4f}")
        print(f"  Detection rate: {detection_rate:.2%}")
        if latency_stats:
            print(f"  Mean latency:   {latency_stats.get('mean_ms', 0):.1f} ms")
            print(f"  P95 latency:    {latency_stats.get('p95_ms', 0):.1f} ms")

    # ------------------------------------------------------------------
    # 3. Save JSON results
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    ablation_dir = out_dir / "ablation"
    plots_dir = ablation_dir / "plots"
    results_payload = {
        "study": "detector_ablation",
        "dataset": "FER2013",
        "preprocessing": "full_preprocess",
        "detectors": all_results,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed": SEED,
            "limit": args.limit,
            "total_images": total,
        },
    }
    save_results_json(results_payload, str(ablation_dir / "detector.json"))

    # ------------------------------------------------------------------
    # 4. Scatter plot: accuracy vs latency
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_dark_style(ax, fig)

    for idx, detector in enumerate(args.detectors):
        if detector not in all_results:
            continue
        r = all_results[detector]
        lat = r["latency"].get("mean_ms", 0) if r["latency"] else 0
        p95 = r["latency"].get("p95_ms", 0) if r["latency"] else 0
        color = CB_PALETTE[idx % len(CB_PALETTE)]

        # Main point: accuracy vs mean latency
        ax.scatter(
            lat, r["accuracy"],
            s=200, color=color, edgecolors="white", linewidths=1.5,
            zorder=5, label=f"{detector}",
        )
        # Annotate with detector name
        ax.annotate(
            f"  {detector}\n  acc={r['accuracy']:.3f}\n  F1={r['weighted_f1']:.3f}",
            xy=(lat, r["accuracy"]),
            fontsize=8,
            color="white",
        )
        # P95 latency marker (smaller, same color)
        ax.scatter(
            p95, r["accuracy"],
            s=60, color=color, marker="x", zorder=4,
        )
        # Connect mean to p95
        ax.plot(
            [lat, p95], [r["accuracy"], r["accuracy"]],
            color=color, linewidth=1, linestyle=":", alpha=0.7,
        )

    ax.set_xlabel("Latency (ms)  [circle=mean, x=p95]", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        "Detector Ablation — Accuracy vs Latency (FER2013)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.legend(
        facecolor="#2e2e4a", edgecolor="#2e2e4a", labelcolor="white",
        fontsize=10, loc="lower right",
    )

    _save_fig(fig, str(plots_dir / "detector_comparison.png"))

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DETECTOR ABLATION SUMMARY")
    print("=" * 60)
    for det in args.detectors:
        if det not in all_results:
            continue
        r = all_results[det]
        lat = r["latency"].get("mean_ms", 0) if r["latency"] else 0
        p95 = r["latency"].get("p95_ms", 0) if r["latency"] else 0
        print(f"  {det:15s}  acc={r['accuracy']:.4f}  F1={r['weighted_f1']:.4f}  "
              f"det={r['detection_rate']:.2%}  mean={lat:.1f}ms  p95={p95:.1f}ms")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args)
