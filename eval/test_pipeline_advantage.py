"""
eval/test_pipeline_advantage.py — Pipeline-Friendly Evaluation

Two tests that showcase pipeline advantages over B0 baseline:

  Test A: Corruption Robustness
    Apply real-world degradations (blur, noise, JPEG compression, low light)
    to RAF-DB images. Pipeline's SR + CLAHE should recover degraded images
    better than raw B0.

  Test B: Full-Frame Detection
    Embed RAF-DB faces into 640×480 canvases at various sizes/positions.
    Pipeline's retinaface+mtcnn should detect faces better than opencv cascade.

Usage:
    python -m eval.test_pipeline_advantage --test corruption --limit 200
    python -m eval.test_pipeline_advantage --test fullframe --limit 200
    python -m eval.test_pipeline_advantage --test all --limit 200
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._gpu_init import init_gpu; init_gpu()

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Same config as pipeline_vs_baseline.py
ENSEMBLE_DETECTORS = ["retinaface", "mtcnn"]
ENSEMBLE_WEIGHTS = {"retinaface": 0.50, "mtcnn": 0.50}
ADAPTIVE_THRESHOLD = 128
B0_DETECTOR = "opencv"

RAFDB_LABEL_MAP = {
    "1": "surprise", "2": "fear", "3": "disgust", "4": "happy",
    "5": "sad", "6": "angry", "7": "neutral",
}


# ---------------------------------------------------------------------------
# Image corruption functions
# ---------------------------------------------------------------------------
def apply_gaussian_blur(img, ksize=9):
    """Simulate out-of-focus camera."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_gaussian_noise(img, sigma=30):
    """Simulate low-light sensor noise."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_jpeg_compression(img, quality=15):
    """Simulate heavy JPEG compression (webcam streaming)."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_low_brightness(img, factor=0.35):
    """Simulate dark room / backlit conditions."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_high_brightness(img, factor=2.0):
    """Simulate overexposure / bright window behind."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_motion_blur(img, ksize=15):
    """Simulate subject head movement."""
    kernel = np.zeros((ksize, ksize))
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(img, -1, kernel)


def apply_downscale(img, scale=0.25):
    """Simulate low-res webcam (downscale then upscale back)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


CORRUPTIONS = {
    "clean":           lambda img: img,
    "gaussian_blur":   apply_gaussian_blur,
    "gaussian_noise":  apply_gaussian_noise,
    "jpeg_q15":        apply_jpeg_compression,
    "low_brightness":  apply_low_brightness,
    "high_brightness": apply_high_brightness,
    "motion_blur":     apply_motion_blur,
    "downscale_4x":    apply_downscale,
}


# ---------------------------------------------------------------------------
# Preprocessing (same as pipeline_vs_baseline.py)
# ---------------------------------------------------------------------------
def _apply_full_preprocess(frame: np.ndarray) -> np.ndarray:
    """Resolution-adaptive preprocessing: SR always, CLAHE+unsharp only on large inputs."""
    if frame is None or not hasattr(frame, "shape"):
        return frame

    original_min = min(frame.shape[:2])

    # Super-resolution upscale (always applied)
    height, width = frame.shape[:2]
    min_size = min(height, width)
    target_min = 256
    if min_size < target_min:
        scale = target_min / float(min_size)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # CLAHE + unsharp only if original was large enough
    if original_min >= ADAPTIVE_THRESHOLD:
        # Unsharp mask
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
        frame = cv2.addWeighted(frame, 1.25, blurred, -0.25, 0)

        # CLAHE on luminance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        merged = cv2.merge((l_eq, a_channel, b_channel))
        frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return frame


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------
def run_b0(frame: np.ndarray) -> dict | None:
    """B0 baseline: raw DeepFace + opencv, no preprocessing."""
    from deepface import DeepFace
    try:
        result = DeepFace.analyze(
            img_path=frame.copy(), actions=["emotion"],
            detector_backend=B0_DETECTOR, enforce_detection=False, silent=True,
        )
        face = result[0] if isinstance(result, list) else result
        return {"emotion": face.get("emotion", {}),
                "dominant_emotion": face.get("dominant_emotion", "neutral")}
    except Exception:
        return None


def run_pipeline(frame: np.ndarray) -> dict | None:
    """Full pipeline: preprocess + ensemble detection."""
    from deepface import DeepFace

    processed = _apply_full_preprocess(frame)
    backend_results = []
    successful_detectors = []

    input_min = min(processed.shape[:2]) if hasattr(processed, "shape") else 0

    for detector in ENSEMBLE_DETECTORS:
        try:
            result = DeepFace.analyze(
                img_path=processed.copy(), actions=["emotion"],
                detector_backend=detector, enforce_detection=False, silent=True,
            )
            face = result[0] if isinstance(result, list) else result
            backend_results.append(face)
            successful_detectors.append(detector)
        except Exception:
            continue

    if not backend_results:
        return None

    emotion_acc = {}
    for face, det in zip(backend_results, successful_detectors):
        w = ENSEMBLE_WEIGHTS.get(det, 0.1)
        for label, score in face.get("emotion", {}).items():
            emotion_acc[label] = emotion_acc.get(label, 0.0) + score * w
    total_w = sum(ENSEMBLE_WEIGHTS.get(d, 0.1) for d in successful_detectors)
    if total_w > 0:
        for e in emotion_acc:
            emotion_acc[e] /= total_w

    dominant = max(emotion_acc, key=emotion_acc.get)
    return {"emotion": emotion_acc, "dominant_emotion": dominant}


# ---------------------------------------------------------------------------
# Dataset collection (stratified)
# ---------------------------------------------------------------------------
def collect_rafdb_stratified(dataset_dir: str, limit: int) -> list:
    """Collect RAF-DB test samples with stratified sampling."""
    root = Path(dataset_dir)
    if not root.is_dir():
        print(f"RAF-DB directory not found: {root}")
        return []

    by_class = defaultdict(list)
    for label_id in sorted(RAFDB_LABEL_MAP.keys()):
        label_name = RAFDB_LABEL_MAP[label_id]
        class_dir = root / label_id
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                by_class[label_name].append((str(f), label_name))

    if limit > 0 and by_class:
        per_class = max(1, limit // len(by_class))
        result = []
        for label, items in by_class.items():
            random.shuffle(items)
            result.extend(items[:per_class])
        random.shuffle(result)
        return result[:limit]

    all_samples = []
    for items in by_class.values():
        all_samples.extend(items)
    return all_samples


# ---------------------------------------------------------------------------
# Test A: Corruption Robustness
# ---------------------------------------------------------------------------
def test_corruption_robustness(samples: list, output_dir: Path) -> dict:
    """Compare B0 vs Pipeline under various image corruptions."""
    print("\n" + "=" * 60)
    print("TEST A: Corruption Robustness")
    print("=" * 60)

    results = {}

    for corruption_name, corruption_fn in CORRUPTIONS.items():
        print(f"\n--- Corruption: {corruption_name} ---")

        b0_correct = 0
        pipeline_correct = 0
        b0_detected = 0
        pipeline_detected = 0
        total = 0

        for img_path, true_label in tqdm(samples, desc=corruption_name, unit="img"):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            total += 1

            # Apply corruption
            corrupted = corruption_fn(frame)

            # B0
            b0_result = run_b0(corrupted)
            if b0_result:
                b0_detected += 1
                if b0_result["dominant_emotion"] == true_label:
                    b0_correct += 1

            # Pipeline
            pipe_result = run_pipeline(corrupted)
            if pipe_result:
                pipeline_detected += 1
                if pipe_result["dominant_emotion"] == true_label:
                    pipeline_correct += 1

        b0_acc = b0_correct / total if total > 0 else 0
        pipe_acc = pipeline_correct / total if total > 0 else 0
        b0_det_rate = b0_detected / total if total > 0 else 0
        pipe_det_rate = pipeline_detected / total if total > 0 else 0

        results[corruption_name] = {
            "total": total,
            "b0": {"accuracy": round(b0_acc, 4), "detected": b0_detected,
                   "detection_rate": round(b0_det_rate, 4)},
            "pipeline": {"accuracy": round(pipe_acc, 4), "detected": pipeline_detected,
                         "detection_rate": round(pipe_det_rate, 4)},
            "delta_accuracy": round(pipe_acc - b0_acc, 4),
            "delta_detection": round(pipe_det_rate - b0_det_rate, 4),
        }

        print(f"  B0:       acc={b0_acc:.1%}  det={b0_det_rate:.1%}")
        print(f"  Pipeline: acc={pipe_acc:.1%}  det={pipe_det_rate:.1%}")
        print(f"  Delta:    acc={results[corruption_name]['delta_accuracy']:+.1%}"
              f"  det={results[corruption_name]['delta_detection']:+.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("CORRUPTION ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"{'Corruption':<20} {'B0 Acc':>8} {'Pipe Acc':>10} {'Delta':>8} {'B0 Det':>8} {'Pipe Det':>10}")
    print("-" * 66)
    for name, r in results.items():
        print(f"{name:<20} {r['b0']['accuracy']:>7.1%} {r['pipeline']['accuracy']:>9.1%} "
              f"{r['delta_accuracy']:>+7.1%} {r['b0']['detection_rate']:>7.1%} "
              f"{r['pipeline']['detection_rate']:>9.1%}")

    # Save
    out_file = output_dir / "corruption_robustness.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"study": "corruption_robustness", "results": results,
                   "metadata": {"seed": SEED, "n_samples": len(samples),
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}},
                  f, indent=2)
    print(f"\nResults saved to {out_file}")
    return results


# ---------------------------------------------------------------------------
# Test B: Full-Frame Detection
# ---------------------------------------------------------------------------
def embed_face_in_canvas(face_img: np.ndarray, canvas_size=(480, 640),
                         face_size=None, position=None) -> np.ndarray:
    """Embed a face image into a larger canvas with random background."""
    ch, cw = canvas_size
    # Random dark-ish background (simulates room environment)
    canvas = np.random.randint(20, 80, (ch, cw, 3), dtype=np.uint8)

    if face_size is None:
        face_size = random.choice([40, 60, 80, 100, 120, 150])

    face_resized = cv2.resize(face_img, (face_size, face_size),
                              interpolation=cv2.INTER_AREA if face_size < face_img.shape[0]
                              else cv2.INTER_CUBIC)

    max_x = cw - face_size
    max_y = ch - face_size
    if max_x <= 0 or max_y <= 0:
        return canvas

    if position is None:
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
    else:
        x, y = position

    canvas[y:y + face_size, x:x + face_size] = face_resized
    return canvas


def test_fullframe_detection(samples: list, output_dir: Path) -> dict:
    """Compare B0 vs Pipeline on full-frame face detection + recognition."""
    print("\n" + "=" * 60)
    print("TEST B: Full-Frame Detection (faces embedded in 640×480 canvas)")
    print("=" * 60)

    face_sizes = [40, 60, 80, 100, 120, 150]
    results = {}

    for face_size in face_sizes:
        print(f"\n--- Face size: {face_size}×{face_size}px in 640×480 canvas ---")

        b0_correct = 0
        pipeline_correct = 0
        b0_detected = 0
        pipeline_detected = 0
        total = 0

        for img_path, true_label in tqdm(samples, desc=f"face={face_size}px", unit="img"):
            face_img = cv2.imread(img_path)
            if face_img is None:
                continue
            total += 1

            # Create full-frame image with embedded face
            canvas = embed_face_in_canvas(face_img, canvas_size=(480, 640),
                                          face_size=face_size)

            # B0: opencv cascade on full frame
            b0_result = run_b0(canvas)
            if b0_result:
                b0_detected += 1
                if b0_result["dominant_emotion"] == true_label:
                    b0_correct += 1

            # Pipeline: retinaface+mtcnn ensemble on full frame
            pipe_result = run_pipeline(canvas)
            if pipe_result:
                pipeline_detected += 1
                if pipe_result["dominant_emotion"] == true_label:
                    pipeline_correct += 1

        b0_acc = b0_correct / total if total > 0 else 0
        pipe_acc = pipeline_correct / total if total > 0 else 0
        b0_det_rate = b0_detected / total if total > 0 else 0
        pipe_det_rate = pipeline_detected / total if total > 0 else 0

        size_key = f"{face_size}px"
        results[size_key] = {
            "face_size": face_size,
            "canvas_size": "640x480",
            "total": total,
            "b0": {"accuracy": round(b0_acc, 4), "detected": b0_detected,
                   "detection_rate": round(b0_det_rate, 4)},
            "pipeline": {"accuracy": round(pipe_acc, 4), "detected": pipeline_detected,
                         "detection_rate": round(pipe_det_rate, 4)},
            "delta_accuracy": round(pipe_acc - b0_acc, 4),
            "delta_detection": round(pipe_det_rate - b0_det_rate, 4),
        }

        print(f"  B0:       acc={b0_acc:.1%}  det={b0_det_rate:.1%}")
        print(f"  Pipeline: acc={pipe_acc:.1%}  det={pipe_det_rate:.1%}")
        print(f"  Delta:    acc={results[size_key]['delta_accuracy']:+.1%}"
              f"  det={results[size_key]['delta_detection']:+.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("FULL-FRAME DETECTION SUMMARY")
    print("=" * 60)
    print(f"{'Face Size':<12} {'B0 Acc':>8} {'Pipe Acc':>10} {'Delta':>8} {'B0 Det':>8} {'Pipe Det':>10}")
    print("-" * 58)
    for name, r in results.items():
        print(f"{name:<12} {r['b0']['accuracy']:>7.1%} {r['pipeline']['accuracy']:>9.1%} "
              f"{r['delta_accuracy']:>+7.1%} {r['b0']['detection_rate']:>7.1%} "
              f"{r['pipeline']['detection_rate']:>9.1%}")

    # Save
    out_file = output_dir / "fullframe_detection.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"study": "fullframe_detection", "results": results,
                   "metadata": {"seed": SEED, "n_samples": len(samples),
                                "canvas_size": "640x480",
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}},
                  f, indent=2)
    print(f"\nResults saved to {out_file}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pipeline-friendly evaluation tests")
    parser.add_argument("--test", choices=["corruption", "fullframe", "all"],
                        default="all", help="Which test to run")
    parser.add_argument("--rafdb-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "datasets" / "RAF-DB" / "DATASET" / "test"),
                        help="Path to RAF-DB test directory")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max images per test (stratified)")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "results" / "pipeline_advantage"),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Collect samples (stratified)
    samples = collect_rafdb_stratified(args.rafdb_dir, args.limit)
    if not samples:
        print("No RAF-DB samples found. Check --rafdb-dir path.")
        sys.exit(1)

    print(f"Loaded {len(samples)} RAF-DB samples (stratified, limit={args.limit})")

    if args.test in ("corruption", "all"):
        test_corruption_robustness(samples, output_dir)

    if args.test in ("fullframe", "all"):
        test_fullframe_detection(samples, output_dir)


if __name__ == "__main__":
    main()
