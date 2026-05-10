"""Unit tests for ``src.backend.uncertainty_fusion`` (Axis 1C)."""
from __future__ import annotations

import math

import pytest

from src.backend.uncertainty_fusion import (
    fuse_probabilities,
    aggregate_emotions,
    _entropy,
    _temperature_scale,
    _to_unit,
)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def test_entropy_near_zero_for_one_hot():
    # eps stabiliser leaves a tiny residual; assert with absolute tolerance
    assert abs(_entropy([1.0, 0.0, 0.0, 0.0])) < 1e-6


def test_entropy_max_for_uniform():
    n = 7
    p = [1.0 / n] * n
    assert math.isclose(_entropy(p), math.log(n), rel_tol=1e-3)


def test_temperature_softening_increases_entropy():
    p = {"a": 0.9, "b": 0.05, "c": 0.05}
    p_soft = _temperature_scale(p, T=2.0)
    assert _entropy(p.values()) < _entropy(p_soft.values())


def test_to_unit_rescales_pct_inputs():
    pct = {"happy": 90.0, "neutral": 10.0}
    unit = _to_unit(pct)
    assert math.isclose(sum(unit.values()), 1.0, rel_tol=1e-9)
    assert unit["happy"] > unit["neutral"]


# ---------------------------------------------------------------------------
# Fusion — semantic behaviour
# ---------------------------------------------------------------------------

CONFIDENT = {"angry": 0.05, "disgust": 0.05, "fear": 0.05, "happy": 0.05,
             "sad": 0.05, "surprise": 0.05, "neutral": 0.70}
UNCERTAIN = {"angry": 0.14, "disgust": 0.14, "fear": 0.14, "happy": 0.15,
             "sad": 0.14, "surprise": 0.14, "neutral": 0.15}


def test_uncertainty_weights_favour_confident_detector():
    fused = fuse_probabilities([CONFIDENT, UNCERTAIN], method="uncertainty")
    w_conf, w_unc = fused["weights"]
    assert w_conf > w_unc, f"expected confident > uncertain, got {fused['weights']}"
    assert math.isclose(sum(fused["weights"]), 1.0, abs_tol=1e-6)


def test_max_confidence_weights_favour_peaked_detector():
    fused = fuse_probabilities([CONFIDENT, UNCERTAIN], method="max_confidence")
    w_conf, w_unc = fused["weights"]
    assert w_conf > w_unc


def test_fixed_weights_with_provided_priors():
    fused = fuse_probabilities(
        [CONFIDENT, UNCERTAIN], method="fixed",
        fixed_weights=[0.7, 0.3],
    )
    assert math.isclose(fused["weights"][0], 0.7, abs_tol=1e-6)
    assert math.isclose(fused["weights"][1], 0.3, abs_tol=1e-6)


def test_blend_with_fixed_smooths_uncertainty_weights():
    f0 = fuse_probabilities([CONFIDENT, UNCERTAIN], method="uncertainty",
                            fixed_weights=[0.5, 0.5], blend_with_fixed=0.0)
    f1 = fuse_probabilities([CONFIDENT, UNCERTAIN], method="uncertainty",
                            fixed_weights=[0.5, 0.5], blend_with_fixed=1.0)
    # Full blending with [0.5, 0.5] should make weights more uniform
    spread_0 = abs(f0["weights"][0] - f0["weights"][1])
    spread_1 = abs(f1["weights"][0] - f1["weights"][1])
    assert spread_1 < spread_0


def test_fused_entropy_le_max():
    fused = fuse_probabilities([CONFIDENT, UNCERTAIN], method="uncertainty")
    assert fused["fused_entropy"] <= fused["max_entropy"] + 1e-6


def test_fused_distribution_normalises():
    fused = fuse_probabilities([CONFIDENT, UNCERTAIN], method="uncertainty")
    assert math.isclose(sum(fused["fused"].values()), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# aggregate_emotions — drop-in replacement shape
# ---------------------------------------------------------------------------

def test_aggregate_emotions_returns_legacy_shape():
    backend_results = [
        {"emotion": {k: v * 100 for k, v in CONFIDENT.items()}, "region": {"x": 0}},
        {"emotion": {k: v * 100 for k, v in UNCERTAIN.items()}, "region": {"x": 0}},
    ]
    out = aggregate_emotions(
        backend_results, ["retinaface", "mtcnn"], method="uncertainty"
    )
    assert isinstance(out, list) and len(out) == 1
    rec = out[0]
    assert "emotion" in rec
    assert "dominant_emotion" in rec
    assert "ensemble" in rec
    assert "weights" in rec["ensemble"]
    assert "per_entropy" in rec["ensemble"]
    assert math.isclose(sum(rec["emotion"].values()), 100.0, abs_tol=0.5)


def test_aggregate_emotions_low_confidence_flag():
    flat = {e: 100.0 / 7 for e in CONFIDENT}  # truly uniform
    out = aggregate_emotions([{"emotion": flat}], ["a"], method="uncertainty",
                             confidence_threshold=0.3)
    assert out[0]["low_confidence"] is True


def test_invalid_fusion_method_raises():
    with pytest.raises(ValueError):
        fuse_probabilities([CONFIDENT], method="bogus")
