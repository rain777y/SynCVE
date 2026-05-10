"""Unit tests for ``src.backend.clinical_metrics`` (Axis 1A research metrics)."""
from __future__ import annotations

import math

import pytest

from src.backend.clinical_metrics import (
    DEFAULT_VALENCE_MAP,
    compute_valence_trace,
    compute_valence_drift,
    compute_affect_blunting,
    compute_reactivity,
    compute_reaction_latencies,
    compute_suppression_index,
    compute_incongruence,
    compute_session_metrics,
    metrics_to_dict,
)

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def _peaked(emo: str, p: float = 0.7):
    base = (1.0 - p) / (len(EMOTIONS) - 1)
    return {e: (p if e == emo else base) for e in EMOTIONS}


# ---------------------------------------------------------------------------
# Valence
# ---------------------------------------------------------------------------

def test_valence_trace_handles_signed_emotions():
    history = [_peaked("happy"), _peaked("sad"), _peaked("neutral")]
    v = compute_valence_trace(history)
    assert v[0] > 0   # happy -> positive
    assert v[1] < 0   # sad -> negative
    # neutral with 0.05 mass on each non-neutral emotion has a small signed
    # valence (the default valence map has 4 negative vs 2 positive emotions),
    # so the neutral frame must lie strictly between happy and sad.
    assert v[1] < v[2] < v[0]


def test_valence_drift_negative_for_descending_trace():
    history = [_peaked("happy") for _ in range(20)] + [_peaked("sad") for _ in range(20)]
    v = compute_valence_trace(history)
    drift = compute_valence_drift(v, fps=2.0, bootstrap_samples=64)
    assert drift["slope_per_min"] is not None
    assert drift["slope_per_min"] < 0   # drifting negative


def test_valence_drift_returns_none_for_short_traces():
    drift = compute_valence_drift([0.1, 0.2], fps=1.0)
    assert drift["slope_per_min"] is None


# ---------------------------------------------------------------------------
# Affect blunting & reactivity
# ---------------------------------------------------------------------------

def test_affect_blunting_high_for_flat_session():
    history = [_peaked("neutral") for _ in range(60)]
    v = compute_valence_trace(history)
    bl = compute_affect_blunting(v, history, sigma_baseline=0.30, range_baseline=1.20)
    assert bl["abs"] is not None
    assert bl["abs"] > 0.95   # flat session → near-1 blunting


def test_affect_blunting_low_for_expressive_session():
    history = [_peaked(e) for e in ("happy", "sad", "fear", "happy", "angry") * 6]
    v = compute_valence_trace(history)
    bl = compute_affect_blunting(v, history, sigma_baseline=0.30, range_baseline=1.20)
    assert bl["abs"] is not None
    assert bl["abs"] < 0.5    # high variance → low blunting


def test_reactivity_handles_zero_duration():
    assert compute_reactivity(0, 0.0) is None
    assert compute_reactivity(5, 60.0) == 5.0   # 5 events / 1 min


# ---------------------------------------------------------------------------
# Reaction latency
# ---------------------------------------------------------------------------

def test_reaction_latency_matches_first_event_within_window():
    triggers = [{"word": "father", "frame_idx": 30}]
    events = [
        {"frame_idx": 35, "from_emotion": "neutral", "to_emotion": "fear",
         "confidence": 0.8, "magnitude": 0.4},
        {"frame_idx": 200, "from_emotion": "neutral", "to_emotion": "sad",
         "confidence": 0.6, "magnitude": 0.3},
    ]
    out = compute_reaction_latencies(triggers, events, fps=10.0, max_latency_sec=2.0)
    assert out["n"] == 1
    assert out["per_trigger"][0]["latency_ms"] == 500.0


def test_reaction_latency_skips_event_outside_window():
    triggers = [{"word": "father", "frame_idx": 30}]
    events = [
        {"frame_idx": 200, "from_emotion": "neutral", "to_emotion": "sad",
         "confidence": 0.6, "magnitude": 0.3},
    ]
    out = compute_reaction_latencies(triggers, events, fps=10.0, max_latency_sec=2.0)
    assert out["n"] == 0
    assert out["per_trigger"][0]["matched_event"] is None


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

def test_suppression_index_detects_return_to_neutral():
    # Event at frame 5; at frame 7 dominant returns to neutral with p>=0.5
    history = [_peaked("happy") for _ in range(5)] + [_peaked("disgust"), _peaked("neutral", p=0.6), _peaked("neutral", p=0.6)] + [_peaked("happy") for _ in range(5)]
    events = [{"frame_idx": 5}]
    si = compute_suppression_index(events, history, refractory_frames=4)
    assert si == 1.0


def test_suppression_index_zero_when_no_return_to_neutral():
    history = [_peaked("disgust") for _ in range(20)]
    events = [{"frame_idx": 5}]
    si = compute_suppression_index(events, history, refractory_frames=4)
    assert si == 0.0


# ---------------------------------------------------------------------------
# Incongruence
# ---------------------------------------------------------------------------

def test_incongruence_none_without_asr():
    v = [0.0, 0.1, -0.1]
    out = compute_incongruence(v, asr_segments=None, fps=1.0)
    assert out["index"] is None


def test_incongruence_high_when_face_neg_speech_pos():
    history = [_peaked("sad") for _ in range(20)]
    v = compute_valence_trace(history)
    asr = [{"t_start_sec": 0.0, "t_end_sec": 5.0, "valence": +1.0}]
    out = compute_incongruence(v, asr_segments=asr, fps=2.0, window_sec=5.0)
    assert out["index"] is not None
    assert out["index"] > 0.5


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def test_compute_session_metrics_round_trip_serialisable():
    history = [_peaked("happy") for _ in range(15)] + [_peaked("sad") for _ in range(15)]
    events = [{"frame_idx": 15, "confidence": 0.8, "magnitude": 0.5}]
    m = compute_session_metrics(history, events, fps=2.0)
    d = metrics_to_dict(m)
    assert d["samples"] == 30
    assert d["event_count"] == 1
    assert d["valence_mean"] is not None
    # All numeric fields must be JSON-safe
    import json
    json.dumps(d)
