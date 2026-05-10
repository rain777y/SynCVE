"""
Integration tests for the Axis 1A/4 endpoints in ``src.backend.routes``:

    GET  /session/<id>/events
    POST /session/<id>/clinical_metrics
    GET  /session/<id>/clinical_report?format=md|pdf

These tests construct a minimal Flask app from the blueprint (skipping
the DeepFace / GPU warmup that ``create_app`` runs in production),
monkey-patch the supabase client out of the picture, and seed an
in-memory ``TemporalAnalyzer`` so the endpoints have something to act on.
"""
from __future__ import annotations

import os
import random

import pytest
from flask import Flask

from src.backend import session_manager
from src.backend.event_detector import EventDetector
from src.backend.routes import blueprint
from src.backend.temporal_analysis import TemporalAnalyzer


EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_app():
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture
def seeded_session(monkeypatch):
    """Seed an in-memory analyzer with synthetic frames + transitions."""
    rng = random.Random(123)
    sid = "11111111-2222-3333-4444-555555555555"

    analyzer = TemporalAnalyzer(
        alpha=0.3,
        fps_estimate=2.0,
        event_detector=EventDetector(method="ensemble", consensus_min_methods=2),
    )

    def _peaked(emo: str, intensity: float = 0.7):
        base = (1.0 - intensity) / (len(EMOTIONS) - 1)
        return {e: (intensity if e == emo else base) for e in EMOTIONS}

    # neutral 25 frames → disgust 25 → neutral 25 → happy 25
    for emo, n in [("neutral", 25), ("disgust", 25), ("neutral", 25), ("happy", 25)]:
        for _ in range(n):
            d = _peaked(emo, intensity=0.7 + rng.uniform(-0.02, 0.02))
            analyzer.add_frame(
                d,
                ensemble_meta={
                    "weights": {"retinaface": 0.6, "mtcnn": 0.4},
                    "per_entropy": {"retinaface": 0.4, "mtcnn": 0.9},
                    "fused_entropy": 0.5,
                    "max_entropy": 1.95,
                },
            )

    # Inject the analyzer into the global registry that routes pull from
    monkeypatch.setitem(session_manager._temporal_analyzers, sid, analyzer)
    return sid, analyzer


# ---------------------------------------------------------------------------
# /session/<id>/events
# ---------------------------------------------------------------------------

def test_events_endpoint_returns_consensus_events(client, seeded_session):
    sid, analyzer = seeded_session
    resp = client.get(f"/session/{sid}/events")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert "events" in body
    assert "event_count" in body
    assert body["event_count"] == len(body["events"])
    # We planted 3 transitions; ensemble should pick up at least one.
    assert body["event_count"] >= 1
    for ev in body["events"]:
        assert "frame_idx" in ev and "from_emotion" in ev and "to_emotion" in ev
        assert "magnitude" in ev and "confidence" in ev
        assert 0 <= ev["confidence"] <= 1


def test_events_endpoint_honors_query_overrides(client, seeded_session):
    sid, _ = seeded_session
    # Tighten z_threshold and require more methods → expect fewer/zero events
    resp = client.get(
        f"/session/{sid}/events?method=ensemble&z_threshold=4.0"
        "&min_magnitude=0.50&consensus_min_methods=3"
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["z_threshold"] == 4.0
    assert body["min_magnitude"] == 0.50
    assert body["consensus_min_methods"] == 3


def test_events_endpoint_unknown_session_returns_empty(client):
    resp = client.get("/session/00000000-0000-0000-0000-000000000000/events")
    # Empty payload, not 404 — keeps the frontend simple
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["event_count"] == 0
    assert body["events"] == []


# ---------------------------------------------------------------------------
# /session/<id>/clinical_metrics
# ---------------------------------------------------------------------------

def test_clinical_metrics_endpoint(client, seeded_session):
    sid, _ = seeded_session
    resp = client.post(
        f"/session/{sid}/clinical_metrics",
        json={
            "triggers": [{"word": "father", "frame_idx": 28}],
        },
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["session_id"] == sid
    assert body["samples"] == 100
    assert body["valence_mean"] is not None
    assert body["valence_std"] is not None
    assert body["affect_blunting_score"] is not None
    assert "events" in body
    assert "reaction_latencies" in body
    rl = body["reaction_latencies"]
    assert "per_trigger" in rl


def test_clinical_metrics_rejects_non_list_triggers(client, seeded_session):
    sid, _ = seeded_session
    resp = client.post(
        f"/session/{sid}/clinical_metrics",
        json={"triggers": "not-a-list"},
    )
    assert resp.status_code == 422


def test_clinical_metrics_unknown_session_returns_error(client):
    resp = client.post(
        "/session/00000000-0000-0000-0000-000000000000/clinical_metrics",
        json={},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /session/<id>/clinical_report
# ---------------------------------------------------------------------------

def test_clinical_report_markdown(client, seeded_session):
    sid, _ = seeded_session
    resp = client.get(f"/session/{sid}/clinical_report?format=md")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["session_id"] == sid
    assert body["format"] == "md"
    md = body["report_markdown"]
    assert "# SynCVE Clinical Session Report" in md
    assert "Top Flagged Events" in md
    assert "Clinical Metrics" in md


def test_clinical_report_pdf_returns_binary(client, seeded_session, tmp_path):
    sid, _ = seeded_session
    resp = client.get(f"/session/{sid}/clinical_report?format=pdf")
    assert resp.status_code == 200
    assert resp.mimetype == "application/pdf"
    # PDF magic bytes
    assert resp.data[:4] == b"%PDF"
    assert len(resp.data) > 1000


def test_clinical_report_invalid_format(client, seeded_session):
    sid, _ = seeded_session
    resp = client.get(f"/session/{sid}/clinical_report?format=docx")
    assert resp.status_code == 422


def test_clinical_report_unknown_session(client):
    resp = client.get(
        "/session/00000000-0000-0000-0000-000000000000/clinical_report?format=md"
    )
    assert resp.status_code == 400
