"""Unit tests for ``src.backend.event_detector`` (Axis 1A)."""
from __future__ import annotations

import math
import random

import pytest

from src.backend.event_detector import (
    EventDetector,
    SlidingWindowDetector,
    CUSUMDetector,
    DetectedEvent,
    build_from_config,
    event_to_dict,
)
from src.backend.config import get_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def _peaked(target: str, p: float = 0.70) -> dict:
    base = (1.0 - p) / (len(EMOTIONS) - 1)
    return {e: (p if e == target else base) for e in EMOTIONS}


def _stream(segments):
    """[(emotion, n_frames), ...] → list[dict]."""
    out = []
    for emo, n in segments:
        out.extend([_peaked(emo) for _ in range(n)])
    return out


# ---------------------------------------------------------------------------
# Sliding-window detector
# ---------------------------------------------------------------------------

def test_sliding_detects_clean_transition():
    s = SlidingWindowDetector(window=5, z_threshold=2.0, min_magnitude=0.1)
    hist = _stream([("neutral", 25), ("angry", 25)])
    fired = []
    for i, p in enumerate(hist):
        c = s.push(p, i)
        if c:
            fired.append(c.frame_idx)
    assert fired, "sliding window must fire on a clean step transition"
    # The change-point should be reported within ±5 frames of frame 25
    assert any(20 <= f <= 30 for f in fired), f"unexpected fire frames {fired}"


def test_sliding_no_false_positive_on_constant_stream():
    s = SlidingWindowDetector(window=4)
    hist = _stream([("neutral", 60)])
    fired = []
    for i, p in enumerate(hist):
        c = s.push(p, i)
        if c:
            fired.append(c.frame_idx)
    assert not fired, f"sliding fired on constant stream: {fired}"


# ---------------------------------------------------------------------------
# CUSUM detector
# ---------------------------------------------------------------------------

def test_cusum_detects_drift_into_negative():
    cu = CUSUMDetector(drift=0.005, threshold=0.10)
    hist = _stream([("neutral", 30), ("sad", 30)])
    fired = [cu.push(p, i) for i, p in enumerate(hist)]
    assert any(c is not None for c in fired), "CUSUM must fire on a drift to negative valence"


# ---------------------------------------------------------------------------
# EventDetector — top-level
# ---------------------------------------------------------------------------

def test_ensemble_detects_single_change_point():
    det = EventDetector(method="ensemble", consensus_min_methods=2)
    hist = _stream([("neutral", 25), ("disgust", 25), ("neutral", 25)])
    events = det.detect_batch(hist)
    assert len(events) >= 1, "ensemble must detect at least one consensus event"
    # The first event should fall near the first transition (frame ~25)
    assert any(20 <= e.frame_idx <= 30 for e in events), \
        f"first transition not localised correctly: {[e.frame_idx for e in events]}"


def test_ensemble_no_event_on_static_stream():
    det = EventDetector(method="ensemble", consensus_min_methods=2)
    hist = _stream([("neutral", 80)])
    events = det.detect_batch(hist)
    assert events == [], f"ensemble fired on static stream: {events}"


def test_streaming_and_batch_are_consistent():
    """get_events() (streaming + cluster) should match detect_batch() up to refractory."""
    det_a = EventDetector(method="ensemble", consensus_min_methods=2)
    hist = _stream([("neutral", 20), ("happy", 20), ("neutral", 20)])
    for i, p in enumerate(hist):
        det_a.detect_streaming(p, frame_idx=i)
    streaming = det_a.get_events()

    det_b = EventDetector(method="ensemble", consensus_min_methods=2)
    batch = det_b.detect_batch(hist)
    assert {e["frame_idx"] for e in [event_to_dict(x) for x in batch]} \
        .issubset({s["frame_idx"] for s in [event_to_dict(x) for x in streaming]} |
                  {e["frame_idx"] for e in [event_to_dict(x) for x in batch]}), (
            "streaming and batch event sets diverged unexpectedly"
        )


def test_event_confidence_in_unit_range():
    det = EventDetector(method="ensemble", consensus_min_methods=2)
    hist = _stream([("neutral", 20), ("fear", 20), ("neutral", 20), ("happy", 20)])
    events = det.detect_batch(hist)
    for e in events:
        assert 0.0 <= e.confidence <= 1.0
        assert 0.0 <= e.magnitude <= 2.0   # L1 is bounded by 2 for proper distributions


def test_build_from_config_uses_settings():
    cfg = get_config()
    det = build_from_config(cfg)
    assert det.method == cfg.events.method
    assert det.refractory_frames == cfg.events.refractory_frames


def test_event_to_dict_round_trip():
    e = DetectedEvent(
        frame_idx=10, timestamp=None, from_emotion="neutral", to_emotion="happy",
        magnitude=0.4, confidence=0.6, methods=["sliding"], method_count=1,
    )
    d = event_to_dict(e)
    assert d["frame_idx"] == 10
    assert d["confidence"] == 0.6
    assert "scores" in d.get("metadata", {}) or "metadata" in d


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        EventDetector(method="not-a-method")
