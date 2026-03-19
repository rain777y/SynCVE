# built-in dependencies
import uuid
from typing import Dict, List, Optional, Union

# 3rd party dependencies
import cv2
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.commons import image_utils
from deepface.commons.logger import Logger
from src.backend.config import get_config

logger = Logger()


def _get_deepface_config():
    return get_config().deepface


# Detector quality weights for ensemble voting.
# Higher weight = more influence on the final result.
# Available backends (DeepFace 0.0.99+):
#   opencv, ssd, dlib, mtcnn, fastmtcnn, retinaface, mediapipe,
#   yolov8n, yolov8m, yolov8l,
#   yolov11n, yolov11s, yolov11m, yolov11l,
#   yolov12n, yolov12s, yolov12m, yolov12l,
#   yunet, centerface
DETECTOR_WEIGHTS = {
    "retinaface": 0.50,
    "mtcnn": 0.30,
    "centerface": 0.20,
    "yolov8n": 0.15,
    "yolov11n": 0.15,
    "yolov12n": 0.15,
    "fastmtcnn": 0.20,
    "yunet": 0.15,
    "opencv": 0.10,
    "ssd": 0.10,
}


# pylint: disable=broad-except


def _clean_error_message(message: str) -> str:
    """Normalize known typos in upstream DeepFace error messages."""
    return message.replace("arraay", "array")


def _to_bool(value: Union[str, bool, None], *, default: bool = True) -> bool:
    """
    Normalize user-provided truthy/falsey values coming from JSON or form data.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    return lowered not in {"0", "false", "no", "off"}


def _preprocess_image(img: Union[str, np.ndarray]) -> np.ndarray:
    """
    Apply lightweight preprocessing to improve downstream emotion accuracy:
    - Load any supported DeepFace input into a numpy array
    - Upscale low-resolution inputs with mild sharpening (pseudo super-resolution)
    - Normalize lighting using CLAHE on the luminance channel
    """
    # DeepFace handles the heavy lifting of decoding paths/base64/URLs into arrays
    if isinstance(img, np.ndarray):
        frame = img.copy()
    else:
        frame, _ = image_utils.load_image(img)

    enhanced = _maybe_super_resolve(frame)
    normalized = _normalize_lighting(enhanced)

    return normalized


def _maybe_super_resolve(frame: np.ndarray) -> np.ndarray:
    """
    Lightweight super-resolution: upscale small faces and apply unsharp masking.
    This avoids dependency on heavyweight SR models while still restoring detail.
    """
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

    # Unsharp mask for extra crispness
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(frame, 1.25, blurred, -0.25, 0)

    return sharpened


def _normalize_lighting(frame: np.ndarray) -> np.ndarray:
    """
    Normalize illumination with CLAHE on the luminance channel in LAB space.
    """
    if frame is None or len(frame.shape) < 3 or frame.shape[2] != 3:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    merged = cv2.merge((l_eq, a_channel, b_channel))
    normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return normalized


def _parse_detector_backends(
    detector_backend: Union[str, List[str], None], enable_ensemble: bool
) -> List[str]:
    """
    Build a detector list for ensemble analysis. Accepts strings, comma-separated
    values, or explicit lists/tuples. If ensemble is enabled and we only have a
    single backend, automatically merge with configured ensemble defaults to
    avoid single-detector bias.
    """
    configured_detectors = _get_deepface_config().ensemble_detectors

    detectors: List[str] = []
    if isinstance(detector_backend, (list, tuple)):
        detectors = [str(item).strip() for item in detector_backend if str(item).strip()]
    elif isinstance(detector_backend, str):
        detectors = [
            backend.strip()
            for backend in detector_backend.replace("[", "").replace("]", "").split(",")
            if backend.strip()
        ]

    # Merge in configured ensemble defaults to guarantee diversity
    if enable_ensemble:
        merged: List[str] = []
        for backend in detectors or []:
            if backend and backend not in merged:
                merged.append(backend)
        for backend in configured_detectors:
            if backend and backend not in merged:
                merged.append(backend)
        detectors = merged or configured_detectors

    if not detectors:
        detectors = (configured_detectors if enable_ensemble else configured_detectors[:1]) or ["retinaface"]

    # If ensemble is disabled but the caller provided multiple detectors, honor
    # only the first to keep behavior predictable.
    if not enable_ensemble and len(detectors) > 1:
        detectors = detectors[:1]

    return detectors


def _normalize_weights(detectors: List[str]) -> List[float]:
    """
    Generate normalized weights for detector ensemble voting.
    """
    raw_weights = [DETECTOR_WEIGHTS.get(det, 1.0) for det in detectors]
    total = float(sum(raw_weights))
    if total == 0:
        return [1.0 / len(detectors)] * len(detectors)
    return [weight / total for weight in raw_weights]


def _aggregate_emotions(
    backend_results: List[Dict],
    detectors: List[str],
    weights: List[float],
    confidence_threshold: float,
) -> List[Dict]:
    """
    Combine emotion probabilities from multiple detector runs using weighted
    averaging. If confidence is too low, raise to let callers reject the result.
    """
    if not backend_results:
        raise ValueError("No emotion results available for ensemble aggregation")

    # Use the first successful result as the template for non-emotion fields
    template = backend_results[0].copy()
    emotion_accumulator: Dict[str, float] = {}

    for result, weight in zip(backend_results, weights):
        emotions = result.get("emotion") or {}
        for emotion_label, score in emotions.items():
            emotion_accumulator[emotion_label] = emotion_accumulator.get(emotion_label, 0.0) + score * weight

    if not emotion_accumulator:
        raise ValueError("Missing emotion scores from ensemble backends")

    dominant_emotion = max(emotion_accumulator, key=emotion_accumulator.get)
    dominant_score = emotion_accumulator[dominant_emotion]

    # DeepFace returns emotion scores as percentages (0-100)
    normalized_confidence = dominant_score / 100.0
    if normalized_confidence < confidence_threshold:
        logger.warn(
            f"Low confidence ensemble prediction ({dominant_score:.1f}% < {confidence_threshold * 100:.0f}% threshold)"
        )
        template["low_confidence"] = True
    else:
        template["low_confidence"] = False

    template["emotion"] = emotion_accumulator
    template["dominant_emotion"] = dominant_emotion
    template["detector_backends"] = detectors
    template["ensemble"] = {
        "weights": {det: weight for det, weight in zip(detectors, weights)},
        "max_emotion_confidence": dominant_score,
        "confidence_threshold": confidence_threshold,
    }

    return [template]


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs

        return result
    except Exception as err:
        error_id = str(uuid.uuid4())[:8]
        logger.error(f"[{error_id}] represent: {err}", exc_info=True)
        return {"error": "Representation failed. Please try again.", "error_id": error_id}, 500


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )

        return obj
    except Exception as err:
        error_id = str(uuid.uuid4())[:8]
        logger.error(f"[{error_id}] verify: {err}", exc_info=True)
        return {"error": "Verification failed. Please try again.", "error_id": error_id}, 500


def analyze(
    img_path: Union[str, np.ndarray],
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    enable_ensemble: Optional[bool] = None,
    confidence_threshold: Optional[float] = None,
):
    try:
        cfg = _get_deepface_config()
        result = {}
        processed_img = _preprocess_image(img_path)

        threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else cfg.confidence_threshold
        )

        default_ensemble = cfg.enable_ensemble
        ensemble_enabled = _to_bool(enable_ensemble, default=default_ensemble)
        detectors = _parse_detector_backends(detector_backend, ensemble_enabled)
        # Default to lenient detection when running an ensemble so a single miss
        # does not cascade into a total failure; still allow callers to override.
        enforce_detection_flag = _to_bool(enforce_detection, default=not ensemble_enabled)
        align_flag = _to_bool(align, default=True)
        anti_spoofing_flag = _to_bool(anti_spoofing, default=True)
        if not align_flag:
            logger.warn("Face alignment disabled; results may degrade.")

        def _analyze_single_backend(
            selected_backend: str,
            *,
            enforce_override: Optional[bool] = None,
            allow_spoof_fallback: bool = False,
        ):
            """
            Run DeepFace.analyze for a single detector backend with optional anti-spoof fallback.
            If anti-spoof triggers a false positive, retry once with anti_spoofing disabled and
            mark the result so the caller can trace the bypass.
            """
            enforce_flag = enforce_detection_flag if enforce_override is None else enforce_override
            try:
                analysis = DeepFace.analyze(
                    img_path=processed_img.copy(),
                    actions=actions,
                    detector_backend=selected_backend,
                    enforce_detection=enforce_flag,
                    align=align_flag,
                    silent=True,
                    anti_spoofing=anti_spoofing_flag,
                )
                result_dict = analysis[0] if isinstance(analysis, list) else analysis
                result_dict.setdefault("detector_backend", selected_backend)
                result_dict.setdefault("spoof_check", {"triggered": False, "bypassed": False})
                result_dict.setdefault("enforce_detection", enforce_flag)
                return result_dict
            except ValueError as backend_err:
                # Preserve strict anti-spoofing; surface a friendly error upstream
                if "Spoof detected" in str(backend_err):
                    raise backend_err
                raise ValueError(_clean_error_message(str(backend_err)))

        if ensemble_enabled and len(detectors) > 1:
            backend_analyses = []
            successful_detectors: List[str] = []
            spoof_detected = False
            backend_errors: Dict[str, str] = {}

            for backend in detectors:
                try:
                    analysis = _analyze_single_backend(backend)
                    successful_detectors.append(backend)
                    backend_analyses.append(analysis)
                    logger.info(f"Detector {backend} succeeded during ensemble analyze")
                except Exception as backend_err:
                    err_msg = _clean_error_message(str(backend_err))
                    if "Spoof detected" in str(backend_err):
                        spoof_detected = True
                        logger.warn(
                            f"Detector {backend} rejected input via anti-spoofing: {backend_err}"
                        )
                        backend_errors[backend] = err_msg
                    elif "Face could not be detected" in err_msg or "Face could not be detected in numpy array" in err_msg:
                        # Graceful degradation: retry once with enforce_detection disabled
                        try:
                            logger.warn(
                                f"Detector {backend} could not find a face; retrying with enforce_detection=False"
                            )
                            analysis = _analyze_single_backend(backend, enforce_override=False)
                            analysis.setdefault("detection_fallback", True)
                            successful_detectors.append(backend)
                            backend_analyses.append(analysis)
                            logger.info(f"Detector {backend} succeeded after enforce_detection=False fallback")
                            continue
                        except Exception as retry_err:
                            clean_retry = _clean_error_message(str(retry_err))
                            backend_errors[backend] = clean_retry
                            logger.warn(
                                f"Detector {backend} failed after fallback disable: {clean_retry}"
                            )
                    else:
                        backend_errors[backend] = err_msg
                        logger.warn(f"Detector {backend} failed during ensemble analyze: {err_msg}")
                    continue

            if not backend_analyses:
                if spoof_detected:
                    return (
                        {
                            "error": "Anti-spoofing triggered: please use a live face (no screens/photos).",
                            "details": {
                                "detectors_checked": detectors,
                                "anti_spoofing": True,
                            },
                        },
                        400,
                    )
                return (
                    {
                        "error": "All ensemble detector backends failed during analysis",
                        "details": {
                            "detectors_checked": detectors,
                            "detector_errors": backend_errors,
                        },
                    },
                    400,
                )

            ensemble_weights = _normalize_weights(successful_detectors)
            demographies = _aggregate_emotions(
                backend_results=backend_analyses,
                detectors=successful_detectors,
                weights=ensemble_weights,
                confidence_threshold=threshold,
            )
        else:
            try:
                analysis = _analyze_single_backend(detector_backend)
                logger.info(f"Detector {detector_backend} succeeded during analyze")
            except Exception as backend_err:
                cleaned_err = _clean_error_message(str(backend_err))
                is_detection_fail = "Face could not be detected" in cleaned_err or "Face could not be detected in numpy array" in cleaned_err
                if is_detection_fail and enforce_detection_flag:
                    try:
                        logger.warn(
                            f"{detector_backend} could not find a face with enforce_detection=True; retrying with enforce_detection=False"
                        )
                        analysis = _analyze_single_backend(detector_backend, enforce_override=False)
                        analysis.setdefault("detection_fallback", True)
                        logger.info(f"{detector_backend} succeeded after enforce_detection=False fallback")
                    except Exception as retry_err:
                        cleaned_retry = _clean_error_message(str(retry_err))
                        logger.error(f"{detector_backend} failed after fallback disable: {cleaned_retry}")
                        raise
                elif "Spoof detected" in str(backend_err):
                    return (
                        {
                            "error": "Anti-spoofing triggered: please use a live face (no screens/photos).",
                            "details": {
                                "detectors_checked": [detector_backend],
                                "anti_spoofing": True,
                            },
                        },
                        400,
                    )
                raise

            demographies = [analysis] if not isinstance(analysis, list) else analysis

            if "emotion" in actions:
                emotions = demographies[0].get("emotion") or {}
                max_score = max(emotions.values()) if emotions else 0.0
                if max_score / 100.0 < threshold:
                    logger.warn(
                        f"Low confidence prediction ({max_score:.1f}% < {threshold * 100:.0f}% threshold)"
                    )
                    demographies[0]["low_confidence"] = True
                else:
                    demographies[0]["low_confidence"] = False

        result["results"] = demographies

        return result
    except Exception as err:
        error_id = str(uuid.uuid4())[:8]
        logger.error(f"[{error_id}] analyze: {err}", exc_info=True)
        return {"error": "Analysis failed. Please try again.", "error_id": error_id}, 500
