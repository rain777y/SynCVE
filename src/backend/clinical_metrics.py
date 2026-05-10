"""
Clinical / research metrics derived from the smoothed emotion probability
stream and the event list emitted by ``event_detector``.

Implements the formulas in ``docs/clinical_metrics.md`` (1-9):
    1. valence(t)
    2. valence_drift
    3. affect_blunting_score (ABS) + per-emotion variant
    4. reactivity (events / minute)
    5. reaction_latency (ms) + per-trigger aggregation
    6. suppression_index
    7. affect_incongruence (stub when ASR semantic valence is missing)
    8. event_confidence (already produced by EventDetector — surfaced here)
    9. per_detector_reliability (entropy-based diagnostic from ensemble logs)

All metrics are pure-Python; no NumPy / SciPy dependency. Each function
defends against degenerate inputs (empty / single-frame sessions) by
returning ``None`` for fields that cannot be computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_VALENCE_MAP = {
    "happy": 1.0,
    "surprise": 0.5,
    "neutral": 0.0,
    "fear": -0.7,
    "sad": -0.8,
    "disgust": -0.6,
    "angry": -0.9,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _ols_slope(ys: Sequence[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom


def _entropy(p: Sequence[float], eps: float = 1e-9) -> float:
    return -sum(x * math.log(x + eps) for x in p if x > 0)


# ---------------------------------------------------------------------------
# Public metric API
# ---------------------------------------------------------------------------

@dataclass
class ClinicalMetrics:
    """Aggregated session-level metrics; all fields are JSON-safe."""

    samples: int = 0
    duration_sec: float = 0.0
    valence_mean: Optional[float] = None
    valence_std: Optional[float] = None
    valence_drift_per_min: Optional[float] = None
    valence_drift_ci95: Optional[Tuple[float, float]] = None
    affect_blunting_score: Optional[float] = None
    affect_blunting_per_emotion: Dict[str, float] = field(default_factory=dict)
    reactivity_events_per_min: Optional[float] = None
    expressive_range_per_emotion: Dict[str, float] = field(default_factory=dict)
    suppression_index: Optional[float] = None
    incongruence_index: Optional[float] = None
    incongruence_window_count: int = 0
    detector_reliability: Dict[str, float] = field(default_factory=dict)
    event_count: int = 0
    high_confidence_event_count: int = 0
    valence_trace: List[float] = field(default_factory=list)


def compute_valence_trace(
    smoothed_history: Sequence[Dict[str, float]],
    valence_map: Optional[Dict[str, float]] = None,
) -> List[float]:
    """
    Compute the per-frame valence signal from the smoothed probability history.
    """
    vmap = valence_map or DEFAULT_VALENCE_MAP
    return [
        sum(vmap.get(e, 0.0) * p.get(e, 0.0) for e in p)
        for p in smoothed_history
    ]


def compute_valence_drift(
    valence_trace: Sequence[float],
    fps: float,
    bootstrap_samples: int = 200,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """
    OLS slope of valence(t) reported in valence/minute, with a bootstrap 95% CI.
    """
    n = len(valence_trace)
    if n < 5 or fps <= 0:
        return {"slope_per_min": None, "ci_low": None, "ci_high": None}

    slope = _ols_slope(valence_trace) * fps * 60.0  # per-minute

    # Bootstrap with a deterministic LCG (no numpy dependency)
    seeds = []
    state = seed & 0xFFFFFFFF
    for _ in range(bootstrap_samples):
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        seeds.append(state)

    n_per = max(8, n // 2)
    slopes_boot: List[float] = []
    for s in seeds:
        rng_state = s
        sample = []
        for _ in range(n_per):
            rng_state = (1103515245 * rng_state + 12345) & 0xFFFFFFFF
            idx = rng_state % n
            sample.append(valence_trace[idx])
        slopes_boot.append(_ols_slope(sample) * fps * 60.0)
    slopes_boot.sort()
    lo = slopes_boot[int(0.025 * bootstrap_samples)]
    hi = slopes_boot[int(0.975 * bootstrap_samples) - 1]
    return {
        "slope_per_min": round(slope, 5),
        "ci_low": round(lo, 5),
        "ci_high": round(hi, 5),
    }


def compute_affect_blunting(
    valence_trace: Sequence[float],
    smoothed_history: Sequence[Dict[str, float]],
    sigma_baseline: float = 0.30,
    range_baseline: float = 1.20,
    min_samples: int = 8,
) -> Dict[str, Any]:
    """
    Affect Blunting Score (composite). See ``docs/clinical_metrics.md``.

    Per-emotion blunting is reported as the contraction ratio of the
    per-emotion probability range relative to a simple `1.0` cohort baseline
    (true cohort baselines should be supplied via the calibration JSON when
    available).
    """
    if len(valence_trace) < min_samples:
        return {"abs": None, "per_emotion": {}}
    sigma_v = _std(valence_trace)
    range_v = max(valence_trace) - min(valence_trace)
    sigma_ratio = sigma_v / max(sigma_baseline, 1e-6)
    range_ratio = range_v / max(range_baseline, 1e-6)
    abs_score = 1.0 - max(0.0, min(1.0, sigma_ratio * range_ratio))

    per_emotion: Dict[str, float] = {}
    if smoothed_history:
        keys = set()
        for h in smoothed_history:
            keys.update(h.keys())
        for k in sorted(keys):
            vals = [h.get(k, 0.0) for h in smoothed_history]
            spread = max(vals) - min(vals)
            per_emotion[k] = round(1.0 - max(0.0, min(1.0, spread / 1.0)), 4)

    return {
        "abs": round(abs_score, 4),
        "valence_std": round(sigma_v, 4),
        "valence_range": round(range_v, 4),
        "per_emotion": per_emotion,
    }


def compute_reactivity(event_count: int, duration_sec: float) -> Optional[float]:
    """Events per minute. Returns None for ill-defined windows."""
    if duration_sec <= 0:
        return None
    return round(event_count / (duration_sec / 60.0), 4)


def compute_reaction_latencies(
    triggers: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    fps: float,
    max_latency_sec: float = 3.0,
) -> Dict[str, Any]:
    """
    Match each trigger ``{word, frame_idx | t_sec}`` to the next event within
    ``max_latency_sec``. Returns per-trigger and aggregate stats in ms.

    `events` may be either ``DetectedEvent``-as-dict or full DetectedEvent
    instances; both expose a ``frame_idx`` field.
    """
    if not triggers or not events or fps <= 0:
        return {"per_trigger": [], "mean_ms": None, "std_ms": None, "n": 0}

    sorted_events = sorted(events, key=lambda e: _ev_get(e, "frame_idx"))
    out: List[Dict[str, Any]] = []
    matched_ms: List[float] = []
    for tr in triggers:
        t_frame = tr.get("frame_idx")
        if t_frame is None and "t_sec" in tr:
            t_frame = int(round(tr["t_sec"] * fps))
        if t_frame is None:
            continue
        # Find first event with frame_idx >= t_frame and within window
        next_ev = None
        for ev in sorted_events:
            ef = _ev_get(ev, "frame_idx")
            if ef >= t_frame:
                latency_sec = (ef - t_frame) / fps
                if 0 <= latency_sec <= max_latency_sec:
                    next_ev = ev
                    break
        if next_ev is None:
            out.append({**tr, "matched_event": None, "latency_ms": None})
            continue
        latency_ms = (_ev_get(next_ev, "frame_idx") - t_frame) / fps * 1000.0
        matched_ms.append(latency_ms)
        out.append({
            **tr,
            "matched_event": _ev_to_dict(next_ev),
            "latency_ms": round(latency_ms, 1),
        })

    return {
        "per_trigger": out,
        "mean_ms": round(_mean(matched_ms), 1) if matched_ms else None,
        "std_ms": round(_std(matched_ms), 1) if matched_ms else None,
        "n": len(matched_ms),
    }


def compute_suppression_index(
    events: List[Dict[str, Any]],
    smoothed_history: Sequence[Dict[str, float]],
    refractory_frames: int = 6,
    neutral_threshold: float = 0.5,
) -> Optional[float]:
    """
    Fraction of events whose magnitude collapses back into a neutral-dominated
    state within ``refractory_frames``.
    """
    if not events or not smoothed_history:
        return None

    suppressed = 0
    n_total = 0
    for ev in events:
        f = _ev_get(ev, "frame_idx")
        if f < 0 or f >= len(smoothed_history):
            continue
        n_total += 1
        end = min(len(smoothed_history), f + refractory_frames + 1)
        for j in range(f + 1, end):
            scores = smoothed_history[j]
            if not scores:
                continue
            dominant = max(scores, key=scores.get)
            if dominant == "neutral" and scores.get("neutral", 0.0) >= neutral_threshold:
                suppressed += 1
                break
    if n_total == 0:
        return None
    return round(suppressed / n_total, 4)


def compute_incongruence(
    valence_trace: Sequence[float],
    asr_segments: Optional[List[Dict[str, Any]]],
    fps: float,
    window_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Compare windowed facial valence against ASR-provided semantic valence.
    Falls back to ``None`` when no ASR data is supplied (Phase 2 dependency).

    ``asr_segments`` items are ``{t_start_sec, t_end_sec, valence}``.
    """
    if not asr_segments or not valence_trace or fps <= 0:
        return {"index": None, "window_count": 0, "windows": []}

    out_windows: List[Dict[str, Any]] = []
    diffs: List[float] = []
    for seg in asr_segments:
        t_s = float(seg.get("t_start_sec", 0.0))
        t_e = float(seg.get("t_end_sec", t_s + window_sec))
        sem_v = float(seg.get("valence", 0.0))
        i_s = max(0, int(round(t_s * fps)))
        i_e = min(len(valence_trace) - 1, int(round(t_e * fps)))
        if i_e <= i_s:
            continue
        face_v = _mean(valence_trace[i_s:i_e + 1])
        diff = abs(face_v - sem_v) / 2.0
        diffs.append(diff)
        out_windows.append({
            "t_start_sec": round(t_s, 3),
            "t_end_sec": round(t_e, 3),
            "facial_valence": round(face_v, 4),
            "semantic_valence": round(sem_v, 4),
            "incongruence": round(diff, 4),
        })
    if not diffs:
        return {"index": None, "window_count": 0, "windows": []}
    return {
        "index": round(_mean(diffs), 4),
        "window_count": len(diffs),
        "windows": out_windows,
    }


def compute_detector_reliability(
    per_frame_ensemble: Sequence[Dict[str, Any]],
    detectors: Sequence[str],
) -> Dict[str, float]:
    """
    Average (1 - H(p_b)/H_max) across the session for each backend ``b``.

    ``per_frame_ensemble`` items are expected to carry a
    ``ensemble.per_entropy`` map produced by ``uncertainty_fusion``.
    """
    sums: Dict[str, float] = {d: 0.0 for d in detectors}
    counts: Dict[str, int] = {d: 0 for d in detectors}
    for entry in per_frame_ensemble:
        ens = entry.get("ensemble") or {}
        per_e = ens.get("per_entropy") or {}
        H_max = ens.get("max_entropy") or math.log(7)
        for d in detectors:
            if d in per_e:
                sums[d] += 1.0 - per_e[d] / max(H_max, 1e-6)
                counts[d] += 1
    return {
        d: round(sums[d] / counts[d], 4) if counts[d] else 0.0
        for d in detectors
    }


def compute_expressive_range_per_emotion(
    smoothed_history: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """Per-emotion range of probability mass over the session."""
    if not smoothed_history:
        return {}
    keys = set()
    for h in smoothed_history:
        keys.update(h.keys())
    out = {}
    for k in sorted(keys):
        vals = [h.get(k, 0.0) for h in smoothed_history]
        out[k] = round(max(vals) - min(vals), 4)
    return out


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def compute_session_metrics(
    smoothed_history: Sequence[Dict[str, float]],
    events: List[Dict[str, Any]],
    *,
    fps: float = 0.5,
    valence_map: Optional[Dict[str, float]] = None,
    sigma_baseline: float = 0.30,
    range_baseline: float = 1.20,
    triggers: Optional[List[Dict[str, Any]]] = None,
    asr_segments: Optional[List[Dict[str, Any]]] = None,
    refractory_frames: int = 6,
    incongruence_window_sec: float = 2.0,
    reaction_latency_max_sec: float = 3.0,
    per_frame_ensemble: Optional[Sequence[Dict[str, Any]]] = None,
    detectors: Optional[Sequence[str]] = None,
    high_confidence_threshold: float = 0.7,
) -> ClinicalMetrics:
    """
    Single-call convenience that produces a fully-populated ClinicalMetrics.
    """
    metrics = ClinicalMetrics()
    metrics.samples = len(smoothed_history)
    metrics.duration_sec = round(metrics.samples / max(fps, 1e-6), 2)

    valence = compute_valence_trace(smoothed_history, valence_map)
    metrics.valence_trace = [round(v, 4) for v in valence]
    metrics.valence_mean = round(_mean(valence), 4) if valence else None
    metrics.valence_std = round(_std(valence), 4) if valence else None

    drift = compute_valence_drift(valence, fps)
    metrics.valence_drift_per_min = drift["slope_per_min"]
    if drift["ci_low"] is not None and drift["ci_high"] is not None:
        metrics.valence_drift_ci95 = (drift["ci_low"], drift["ci_high"])

    bl = compute_affect_blunting(
        valence, smoothed_history, sigma_baseline, range_baseline
    )
    metrics.affect_blunting_score = bl["abs"]
    metrics.affect_blunting_per_emotion = bl["per_emotion"]

    metrics.event_count = len(events)
    metrics.reactivity_events_per_min = compute_reactivity(
        len(events), metrics.duration_sec
    )
    metrics.high_confidence_event_count = sum(
        1 for e in events if _ev_get(e, "confidence", 0.0) >= high_confidence_threshold
    )
    metrics.expressive_range_per_emotion = compute_expressive_range_per_emotion(
        smoothed_history
    )

    metrics.suppression_index = compute_suppression_index(
        events, smoothed_history, refractory_frames=refractory_frames
    )

    incong = compute_incongruence(
        valence, asr_segments, fps, window_sec=incongruence_window_sec
    )
    metrics.incongruence_index = incong["index"]
    metrics.incongruence_window_count = incong["window_count"]

    if per_frame_ensemble is not None and detectors:
        metrics.detector_reliability = compute_detector_reliability(
            per_frame_ensemble, detectors
        )

    return metrics


def metrics_to_dict(m: ClinicalMetrics) -> Dict[str, Any]:
    """Asdict but with stable ordering and JSON-safe tuple coercion."""
    d = asdict(m)
    if d.get("valence_drift_ci95") is not None:
        d["valence_drift_ci95"] = list(d["valence_drift_ci95"])
    return d


# ---------------------------------------------------------------------------
# Internal getters that tolerate dict-or-dataclass event records
# ---------------------------------------------------------------------------

def _ev_get(ev: Any, key: str, default: Any = 0) -> Any:
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)


def _ev_to_dict(ev: Any) -> Dict[str, Any]:
    if isinstance(ev, dict):
        return dict(ev)
    if hasattr(ev, "__dataclass_fields__"):
        return asdict(ev)
    return {"frame_idx": _ev_get(ev, "frame_idx")}
