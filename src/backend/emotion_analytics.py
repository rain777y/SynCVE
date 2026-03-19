"""
Emotion analytics: aggregation, statistics, and noise-floor filtering.
Pure computation -- no I/O, no API calls.
"""
from typing import Any, Dict, List, Optional

from src.backend.config import get_config


def aggregate_emotion_metrics(
    raw_vision_data: List[Dict[str, Any]],
    noise_floor: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Aggregate raw vision outputs into compact metrics.

    - Average probability per emotion
    - Peak probability per emotion (micro-expressions)
    - Noise filtering: drop emotions whose average is below *noise_floor*

    Returned scores are normalised to 0-1.

    Args:
        raw_vision_data: List of per-frame analysis dicts.
        noise_floor: Minimum average to keep an emotion (default from config).

    Returns:
        Dict with keys: samples, noise_floor, averages, peaks, filtered_out,
        dominant, dominant_score, peak_emotion, peak_score.

    Raises:
        ValueError: If no usable emotion data is found.
    """
    if not raw_vision_data:
        raise ValueError("No vision data supplied for aggregation")

    if noise_floor is None:
        noise_floor = get_config().gemini.noise_floor

    accumulator: Dict[str, Dict[str, float]] = {}
    samples = 0

    for entry in raw_vision_data:
        scores = _coerce_scores(entry)
        if not scores:
            continue
        samples += 1
        for emotion_label, raw_score in scores.items():
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            # DeepFace returns 0-100; accept 0-1 as well
            if score > 1.5:
                score = score / 100.0
            stats = accumulator.setdefault(emotion_label, {"sum": 0.0, "count": 0, "peak": 0.0})
            stats["sum"] += score
            stats["count"] += 1
            if score > stats["peak"]:
                stats["peak"] = score

    if not accumulator or samples == 0:
        raise ValueError("Unable to parse emotion scores from supplied data")

    averages_all = {
        label: values["sum"] / max(values["count"], 1)
        for label, values in accumulator.items()
    }
    peaks_all = {label: values["peak"] for label, values in accumulator.items()}

    filtered_out = [label for label, avg in averages_all.items() if avg < noise_floor]
    averages = {label: avg for label, avg in averages_all.items() if avg >= noise_floor}
    peaks = {label: peaks_all[label] for label in averages}

    # Guarantee at least one entry even if everything was filtered
    if not averages:
        dominant_label = max(averages_all, key=averages_all.get)  # type: ignore[arg-type]
        averages[dominant_label] = averages_all[dominant_label]
        peaks[dominant_label] = peaks_all[dominant_label]
        filtered_out = [lbl for lbl in filtered_out if lbl != dominant_label]

    dominant = max(averages, key=averages.get)  # type: ignore[arg-type]
    peak_emotion = max(peaks_all, key=peaks_all.get)  # type: ignore[arg-type]

    return {
        "samples": samples,
        "noise_floor": noise_floor,
        "averages": averages,
        "peaks": peaks,
        "filtered_out": filtered_out,
        "dominant": dominant,
        "dominant_score": averages.get(dominant, 0.0),
        "peak_emotion": peak_emotion,
        "peak_score": peaks_all.get(peak_emotion, 0.0),
    }


def calculate_emotion_stats(emotions_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate basic statistics for a single-frame emotion dict.

    Returns:
        Dict with dominant_emotion, confidence, and low_confidence flag.
    """
    if not emotions_data:
        return {"dominant_emotion": None, "confidence": 0.0, "low_confidence": True}

    dominant = max(emotions_data, key=emotions_data.get)  # type: ignore[arg-type]
    confidence = float(emotions_data.get(dominant, 0.0))
    threshold = get_config().deepface.confidence_threshold

    return {
        "dominant_emotion": dominant,
        "confidence": confidence,
        "low_confidence": (confidence / 100.0) < threshold,
    }


def summarize_for_art_direction(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact dominant/secondary view for image-focused prompting.

    Returns:
        Dict with dominant, score, secondary, secondary_score, peak_emotion, peak_score.
    """
    averages = metrics.get("averages", {})
    top_pairs = sorted(averages.items(), key=lambda kv: kv[1], reverse=True)

    dominant, dom_score = (None, 0.0)
    secondary, sec_score = (None, 0.0)

    if top_pairs:
        dominant, dom_score = top_pairs[0]
    if len(top_pairs) > 1:
        secondary, sec_score = top_pairs[1]

    return {
        "dominant": dominant,
        "score": float(dom_score),
        "secondary": secondary,
        "secondary_score": float(sec_score),
        "peak_emotion": metrics.get("peak_emotion"),
        "peak_score": metrics.get("peak_score"),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_scores(candidate: Any) -> Optional[Dict[str, float]]:
    """Extract emotion score dict from various payload shapes."""
    if not isinstance(candidate, dict):
        return None
    for key in ("emotions", "emotion"):
        maybe = candidate.get(key)
        if isinstance(maybe, dict):
            return maybe
    # DeepFace-style nested result
    if "results" in candidate and isinstance(candidate["results"], list) and candidate["results"]:
        nested = candidate["results"][0]
        if isinstance(nested, dict):
            return _coerce_scores(nested)
    return None
