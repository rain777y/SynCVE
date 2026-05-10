"""
Uncertainty-aware ensemble fusion for emotion probabilities (Axis 1C).

The legacy DeepFace ensemble fuses per-detector emotion vectors with a
fixed weight (50/50 in the previous configuration). This module replaces
that scheme with an entropy-weighted softmax fusion: detectors whose
distribution is sharply peaked (low entropy = high confidence) are
weighted more, while flat / under-confident distributions are
down-weighted.

This is the ensemble counterpart of the *temperature-scaling* literature
(Guo et al., 2017) — an inexpensive post-hoc calibration that empirically
improves both reliability and downstream event-detection F1, and is
trivial to ablate against the fixed-weight baseline.

Design rules:
    - Pure NumPy / Python — no model retraining, no GPU.
    - Returns the **same shape** as the legacy aggregator so the rest of
      the pipeline does not need to change.
    - Always exposes per-detector entropy + weights for the report
      appendix and for ablation experiments.

Public API:
    fuse_probabilities(per_detector_probs, *, method, temperature, weights)
        -> {"fused": {emo: prob}, "weights": {det: w}, "entropy": float, ...}
    aggregate_emotions(backend_results, detectors, *, method, ...)
        -> List[{...}]   (drop-in replacement for service._aggregate_emotions)
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _entropy(p: Sequence[float], eps: float = 1e-9) -> float:
    """Shannon entropy in nats. Returns 0 for degenerate distributions."""
    return -sum(x * math.log(x + eps) for x in p if x > 0)


def _max_entropy(k: int) -> float:
    """Max possible entropy of a k-class categorical."""
    return math.log(k) if k > 0 else 1.0


def _normalise(d: Dict[str, float]) -> Dict[str, float]:
    """Normalise a dict of non-negative floats so it sums to 1."""
    s = sum(d.values())
    if s <= 0:
        return {k: 1.0 / len(d) for k in d} if d else {}
    return {k: v / s for k, v in d.items()}


def _to_unit(d: Dict[str, float]) -> Dict[str, float]:
    """
    Coerce raw scores to a 0..1 probability dict.
    DeepFace returns 0..100 percentages; tolerate both.
    """
    if not d:
        return {}
    needs_pct_normalise = any(v > 1.5 for v in d.values())
    if needs_pct_normalise:
        d = {k: float(v) / 100.0 for k, v in d.items()}
    else:
        d = {k: float(v) for k, v in d.items()}
    return _normalise(d)


def _temperature_scale(probs: Dict[str, float], T: float) -> Dict[str, float]:
    """
    Apply temperature scaling in *probability* space:
        p_scaled = softmax(log(p) / T)
    T > 1 softens the distribution (more uncertain), T < 1 sharpens it.
    """
    if T == 1.0 or not probs:
        return dict(probs)
    eps = 1e-12
    logits = {k: math.log(max(v, eps)) / T for k, v in probs.items()}
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    return _normalise(exps)


# ---------------------------------------------------------------------------
# Weight schemes
# ---------------------------------------------------------------------------

def _entropy_weights(
    per_detector_probs: List[Dict[str, float]],
    entropy_floor: float,
) -> List[float]:
    """
    Weight ∝ 1 / (H(p) + floor). Detectors with sharply peaked (confident)
    distributions get more weight. The floor prevents divide-by-zero on
    degenerate single-class distributions.
    """
    raw = []
    for p in per_detector_probs:
        H = _entropy(list(p.values()))
        raw.append(1.0 / (H + entropy_floor))
    s = sum(raw)
    if s <= 0:
        return [1.0 / len(per_detector_probs)] * len(per_detector_probs)
    return [w / s for w in raw]


def _max_confidence_weights(
    per_detector_probs: List[Dict[str, float]],
) -> List[float]:
    """Weight ∝ max(p). Simpler than entropy; useful as a 2nd ablation arm."""
    raw = [max(p.values()) if p else 0.0 for p in per_detector_probs]
    s = sum(raw)
    if s <= 0:
        return [1.0 / len(per_detector_probs)] * len(per_detector_probs)
    return [w / s for w in raw]


# ---------------------------------------------------------------------------
# Public fusion entry point
# ---------------------------------------------------------------------------

def fuse_probabilities(
    per_detector_probs: List[Dict[str, float]],
    *,
    method: str = "uncertainty",
    temperature: float = 1.0,
    fixed_weights: Optional[Sequence[float]] = None,
    entropy_floor: float = 0.05,
    blend_with_fixed: float = 0.0,
) -> Dict:
    """
    Fuse a list of per-detector emotion probability dicts into a single
    distribution. Returns a result dict with:
        fused        — fused probability dict (sums to 1.0)
        weights      — per-detector weights actually used (sums to 1.0)
        per_entropy  — per-detector entropy (nats)
        fused_entropy — entropy of the fused distribution (nats)
        method       — echo of the chosen method
        temperature  — echo of the temperature
    """
    if not per_detector_probs:
        raise ValueError("fuse_probabilities requires at least one detector")

    # 1. Normalise + temperature-scale every detector
    cleaned: List[Dict[str, float]] = [
        _temperature_scale(_to_unit(p), temperature) for p in per_detector_probs
    ]

    # 2. Pick weights according to the configured method
    if method == "uncertainty":
        weights = _entropy_weights(cleaned, entropy_floor)
    elif method == "max_confidence":
        weights = _max_confidence_weights(cleaned)
    elif method == "fixed":
        if fixed_weights is None or len(fixed_weights) != len(cleaned):
            weights = [1.0 / len(cleaned)] * len(cleaned)
        else:
            s = float(sum(fixed_weights)) or 1.0
            weights = [w / s for w in fixed_weights]
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    # 3. Optional blend with the configured fixed weights so cohort-tuned
    #    priors aren't completely overridden.
    if (
        fixed_weights is not None
        and len(fixed_weights) == len(cleaned)
        and 0.0 < blend_with_fixed <= 1.0
        and method != "fixed"
    ):
        s = float(sum(fixed_weights)) or 1.0
        fixed_norm = [w / s for w in fixed_weights]
        weights = [
            (1.0 - blend_with_fixed) * w + blend_with_fixed * f
            for w, f in zip(weights, fixed_norm)
        ]
        s_w = sum(weights) or 1.0
        weights = [w / s_w for w in weights]

    # 4. Compute the fused distribution
    keys = set()
    for p in cleaned:
        keys.update(p.keys())
    fused = {k: 0.0 for k in keys}
    for p, w in zip(cleaned, weights):
        for k in keys:
            fused[k] += p.get(k, 0.0) * w
    fused = _normalise(fused)

    return {
        "fused": fused,
        "weights": weights,
        "per_entropy": [round(_entropy(list(p.values())), 4) for p in cleaned],
        "fused_entropy": round(_entropy(list(fused.values())), 4),
        "max_entropy": round(_max_entropy(len(fused)), 4),
        "method": method,
        "temperature": temperature,
    }


# ---------------------------------------------------------------------------
# Drop-in replacement for service._aggregate_emotions
# ---------------------------------------------------------------------------

def aggregate_emotions(
    backend_results: List[Dict],
    detectors: List[str],
    *,
    method: str = "uncertainty",
    temperature: float = 1.0,
    fixed_weights: Optional[Sequence[float]] = None,
    entropy_floor: float = 0.05,
    blend_with_fixed: float = 0.0,
    confidence_threshold: float = 0.1,
) -> List[Dict]:
    """
    Aggregate per-detector DeepFace results using uncertainty-aware fusion.
    Output shape mirrors the legacy ``service._aggregate_emotions`` so that
    callers (routes, session_manager, report_generator) need no changes.
    """
    if not backend_results:
        raise ValueError("No emotion results available for ensemble aggregation")

    template = dict(backend_results[0])

    per_detector: List[Dict[str, float]] = []
    for r in backend_results:
        per_detector.append(dict(r.get("emotion") or {}))

    fusion = fuse_probabilities(
        per_detector,
        method=method,
        temperature=temperature,
        fixed_weights=fixed_weights,
        entropy_floor=entropy_floor,
        blend_with_fixed=blend_with_fixed,
    )

    fused = fusion["fused"]
    if not fused:
        raise ValueError("Missing emotion scores from ensemble backends")

    # Re-emit on the original 0..100 scale that downstream code expects
    fused_pct = {k: v * 100.0 for k, v in fused.items()}

    dominant = max(fused_pct, key=fused_pct.get)
    dominant_score = fused_pct[dominant]
    normalised_confidence = dominant_score / 100.0

    template["emotion"] = fused_pct
    template["dominant_emotion"] = dominant
    template["detector_backends"] = list(detectors)
    template["low_confidence"] = normalised_confidence < confidence_threshold
    template["ensemble"] = {
        "method": fusion["method"],
        "temperature": fusion["temperature"],
        "weights": {d: round(w, 4) for d, w in zip(detectors, fusion["weights"])},
        "per_entropy": {d: e for d, e in zip(detectors, fusion["per_entropy"])},
        "fused_entropy": fusion["fused_entropy"],
        "max_entropy": fusion["max_entropy"],
        "max_emotion_confidence": dominant_score,
        "confidence_threshold": confidence_threshold,
    }
    return [template]
