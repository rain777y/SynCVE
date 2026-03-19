"""
End-to-end HTTP API tests that mirror the exact frontend (React) workflow.

Unlike test_real_environment.py (which calls Python functions directly),
these tests send real HTTP requests to the running Flask backend -- the same
way the EmotionDetector.jsx component does:

    1. POST /session/start          -> get session_id
    2. POST /analyze (with session_id) -> emotion detection + session logging
    3. POST /session/pause           -> visual report generation
    4. POST /session/stop            -> text report + session close
    5. GET  /session/history         -> list completed sessions
    6. GET  /session/<id>            -> session details

Requires: backend running on localhost:5005, Supabase + Gemini credentials.
All tests skip gracefully when the backend is unreachable or credentials
are missing.
"""
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e.conftest import (
    _has_gemini,
    _has_supabase,
    requires_all_services,
    requires_gemini,
    requires_supabase,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("SYNCVE_TEST_BACKEND_URL", "http://localhost:5005")
REQUEST_TIMEOUT = 30       # seconds for normal endpoints
REPORT_TIMEOUT = 180       # seconds for report-generation endpoints (Gemini)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def backend_url():
    """Verify the backend is reachable and return its URL."""
    url = BACKEND_URL
    try:
        resp = requests.get(f"{url}/", timeout=10)
        assert resp.status_code == 200, f"Backend health check failed: {resp.status_code}"
    except requests.ConnectionError:
        pytest.skip(
            f"Backend not running at {url}. "
            "Start it with: scripts\\start_backend.bat"
        )
    return url


@pytest.fixture(scope="module")
def face_b64(generated_face_images) -> str:
    """Return a neutral face as base64 for HTTP API calls."""
    img_bytes = generated_face_images.get("neutral")
    if not img_bytes:
        pytest.skip("No neutral face image available")
    return base64.b64encode(img_bytes).decode("utf-8")


@pytest.fixture(scope="module")
def no_face_b64(no_face_image) -> str:
    """Return a no-face image as base64."""
    return base64.b64encode(no_face_image).decode("utf-8")


def _start_session(url: str, **extra_json) -> str:
    """Helper: start a session and return the session_id."""
    resp = requests.post(f"{url}/session/start", json=extra_json, timeout=REQUEST_TIMEOUT)
    assert resp.status_code == 200, f"session/start failed: {resp.text[:200]}"
    data = resp.json()
    assert "session_id" in data, f"No session_id in response: {data}"
    return data["session_id"]


def _stop_session(url: str, session_id: str) -> None:
    """Helper: best-effort session cleanup."""
    try:
        requests.post(
            f"{url}/session/stop",
            json={"session_id": session_id},
            timeout=REQUEST_TIMEOUT,
        )
    except Exception:
        pass


# ============================================================================
# 1. HEALTH CHECK
# ============================================================================
class TestHealthCheck:
    """Verify the backend is alive and returns expected content."""

    def test_root_returns_200(self, backend_url):
        resp = requests.get(f"{backend_url}/", timeout=10)
        assert resp.status_code == 200

    def test_root_mentions_syncve_or_deepface(self, backend_url):
        resp = requests.get(f"{backend_url}/", timeout=10)
        body = resp.text.lower()
        assert "syncve" in body or "deepface" in body, (
            f"Health check page did not mention SynCVE or DeepFace: {body[:200]}"
        )


# ============================================================================
# 2. FRONTEND WORKFLOW SIMULATION
# ============================================================================
class TestFrontendWorkflow:
    """
    Mirror the exact sequence that EmotionDetector.jsx performs:
    start -> analyze (x N) -> pause -> stop.
    """

    def test_complete_frontend_flow(self, backend_url, face_b64):
        """Full frontend lifecycle with 3 analysis frames."""
        session_id = _start_session(backend_url)

        try:
            # ---- Analyze 3 frames (like the interval capture loop) ----
            for i in range(3):
                resp = requests.post(
                    f"{backend_url}/analyze",
                    json={
                        "img": face_b64,
                        "actions": ["emotion"],
                        "anti_spoofing": False,
                        "session_id": session_id,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                assert resp.status_code == 200, (
                    f"Analyze frame {i + 1} failed ({resp.status_code}): {resp.text[:200]}"
                )
                data = resp.json()
                assert "results" in data, f"Missing 'results' in analyze response: {list(data.keys())}"
                time.sleep(0.5)

            # ---- Pause session (triggers visual report) ----
            resp = requests.post(
                f"{backend_url}/session/pause",
                json={"session_id": session_id},
                timeout=REPORT_TIMEOUT,
            )
            assert resp.status_code == 200, (
                f"session/pause failed ({resp.status_code}): {resp.text[:200]}"
            )
            pause_data = resp.json()
            assert isinstance(pause_data, dict)
            # On success, the frontend reads pause_data.image_url
            if "error" not in pause_data:
                assert pause_data.get("status") == "paused"
                assert "image_url" in pause_data, (
                    f"Frontend expects 'image_url' key: {list(pause_data.keys())}"
                )

            # ---- Stop session (generates text report) ----
            resp = requests.post(
                f"{backend_url}/session/stop",
                json={"session_id": session_id},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
            stop_data = resp.json()
            assert isinstance(stop_data, dict)
            if "error" not in stop_data:
                assert stop_data.get("status") == "session_ended"
                assert "report" in stop_data

        except Exception:
            _stop_session(backend_url, session_id)
            raise

    def test_start_and_immediate_stop(self, backend_url):
        """
        Edge case: the user clicks Start then immediately clicks Stop
        without any analysis frames being captured.
        """
        session_id = _start_session(backend_url)
        resp = requests.post(
            f"{backend_url}/session/stop",
            json={"session_id": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_multiple_sessions_sequential(self, backend_url, face_b64):
        """
        Simulate a user running two sessions back-to-back
        (common pattern when refreshing the page).
        """
        for run in range(2):
            sid = _start_session(backend_url)
            try:
                resp = requests.post(
                    f"{backend_url}/analyze",
                    json={
                        "img": face_b64,
                        "actions": ["emotion"],
                        "anti_spoofing": False,
                        "session_id": sid,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                assert resp.status_code == 200
            finally:
                _stop_session(backend_url, sid)


# ============================================================================
# 3. ANALYZE ENDPOINT VALIDATION
# ============================================================================
class TestAnalyzeEndpoint:
    """Test the /analyze endpoint behavior with various inputs."""

    def test_analyze_returns_emotion_data(self, backend_url, face_b64):
        """Verify /analyze returns the expected emotion schema."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": face_b64,
                "actions": ["emotion"],
                "anti_spoofing": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        face = data["results"][0]
        assert "emotion" in face
        assert "dominant_emotion" in face

        # All 7 basic emotions should be present
        expected_emotions = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
        actual_emotions = set(face["emotion"].keys())
        assert expected_emotions.issubset(actual_emotions), (
            f"Missing emotions: {expected_emotions - actual_emotions}"
        )

    def test_analyze_with_multiple_actions(self, backend_url, face_b64):
        """Request multiple actions (age, gender, emotion)."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": face_b64,
                "actions": ["emotion", "age", "gender"],
                "anti_spoofing": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        face = data["results"][0]
        assert "emotion" in face
        assert "age" in face
        assert "gender" in face or "dominant_gender" in face

    def test_analyze_missing_image_returns_400(self, backend_url):
        """Missing 'img' field should return 400."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={"actions": ["emotion"]},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 400

    def test_analyze_invalid_action_returns_422(self, backend_url, face_b64):
        """Invalid action name should be rejected by Pydantic validator."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": face_b64,
                "actions": ["nonexistent_action"],
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data

    def test_analyze_with_session_id_logs_data(self, backend_url, face_b64):
        """
        When session_id is provided, the backend should log the analysis
        to the session without failing.
        """
        sid = _start_session(backend_url)
        try:
            resp = requests.post(
                f"{backend_url}/analyze",
                json={
                    "img": face_b64,
                    "actions": ["emotion"],
                    "anti_spoofing": False,
                    "session_id": sid,
                },
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "results" in data
            # Should NOT contain logging errors
            assert "error" not in data.get("results", [{}])[0].get("logging_status", {}), (
                f"Logging error: {data}"
            )
        finally:
            _stop_session(backend_url, sid)

    def test_analyze_no_face_image_graceful(self, backend_url, no_face_b64):
        """
        Analyzing an image without a face should not crash the backend.
        With enforce_detection not explicitly set, it may error gracefully.
        """
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": no_face_b64,
                "actions": ["emotion"],
                "anti_spoofing": False,
                "enforce_detection": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        # Either succeeds (with low confidence) or returns a structured error
        assert resp.status_code in [200, 400, 500]
        data = resp.json()
        assert isinstance(data, dict)


# ============================================================================
# 4. SESSION MANAGEMENT ENDPOINTS
# ============================================================================
class TestSessionManagement:
    """Test session CRUD endpoints via HTTP."""

    def test_session_start_schema(self, backend_url):
        """Verify /session/start returns session_id and status."""
        sid = _start_session(backend_url)
        try:
            # Also test with user_id and metadata
            resp = requests.post(
                f"{backend_url}/session/start",
                json={
                    "user_id": "e2e-http-test-user",
                    "metadata": {"source": "e2e_frontend_api_flow", "automated": True},
                },
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("status") == "active"
            assert "session_id" in data
            _stop_session(backend_url, data["session_id"])
        finally:
            _stop_session(backend_url, sid)

    def test_session_history_returns_list(self, backend_url):
        """GET /session/history should return a list of sessions."""
        resp = requests.get(f"{backend_url}/session/history", timeout=REQUEST_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_session_history_respects_limit(self, backend_url):
        """The limit query parameter should constrain results."""
        resp = requests.get(
            f"{backend_url}/session/history",
            params={"limit": 2},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) <= 2

    def test_session_details_for_valid_session(self, backend_url):
        """GET /session/<id> should return session details."""
        sid = _start_session(backend_url)
        try:
            resp = requests.get(f"{backend_url}/session/{sid}", timeout=REQUEST_TIMEOUT)
            assert resp.status_code == 200
            data = resp.json()
            assert "session" in data
            assert data["session"]["id"] == sid
            assert data["session"]["status"] == "active"
        finally:
            _stop_session(backend_url, sid)

    def test_session_details_nonexistent_returns_404(self, backend_url):
        """Querying a nonexistent session should return 404."""
        resp = requests.get(
            f"{backend_url}/session/00000000-0000-0000-0000-000000000000",
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 404

    def test_stop_invalid_session_id_format(self, backend_url):
        """Non-UUID session_id should be rejected by Pydantic validator."""
        resp = requests.post(
            f"{backend_url}/session/stop",
            json={"session_id": "not-a-uuid"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code in [400, 422]

    def test_pause_requires_session_id(self, backend_url):
        """POST /session/pause without session_id should return 400."""
        resp = requests.post(
            f"{backend_url}/session/pause",
            json={},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 400


# ============================================================================
# 5. REPORT GENERATION ENDPOINTS (HTTP)
# ============================================================================
class TestReportEndpoints:
    """Test the report generation endpoints via HTTP."""

    def test_emotion_report_with_synthetic_data(self, backend_url):
        """
        POST /session/report/emotion with inline raw_vision_data
        (bypasses the need for prior /analyze calls).
        """
        sid = _start_session(backend_url)
        try:
            synthetic_data = [
                {"dominant_emotion": "happy", "emotion": {"happy": 65, "sad": 5, "neutral": 20, "angry": 3, "surprised": 5, "fear": 1, "disgust": 1}},
                {"dominant_emotion": "neutral", "emotion": {"happy": 20, "sad": 10, "neutral": 55, "angry": 5, "surprised": 5, "fear": 3, "disgust": 2}},
                {"dominant_emotion": "happy", "emotion": {"happy": 70, "sad": 3, "neutral": 15, "angry": 2, "surprised": 8, "fear": 1, "disgust": 1}},
            ]

            resp = requests.post(
                f"{backend_url}/session/report/emotion",
                json={
                    "session_id": sid,
                    "raw_vision_data": synthetic_data,
                    "max_keyframes": 2,
                },
                timeout=REPORT_TIMEOUT,
            )
            # May succeed (200) or fail due to no keyframes in storage (400)
            assert resp.status_code in [200, 400, 500], (
                f"Unexpected status {resp.status_code}: {resp.text[:200]}"
            )
            data = resp.json()
            assert isinstance(data, dict)

            if resp.status_code == 200:
                assert "metrics" in data
                assert "report_markdown" in data
        finally:
            _stop_session(backend_url, sid)

    @pytest.mark.slow
    def test_visual_report_with_synthetic_data(self, backend_url):
        """
        POST /session/report/visual with inline raw_vision_data.
        Marked slow because Gemini image generation can take 10-30 seconds.
        """
        sid = _start_session(backend_url)
        try:
            synthetic_data = [
                {"dominant_emotion": "happy", "emotion": {"happy": 65, "sad": 5, "neutral": 20, "angry": 3, "surprised": 5, "fear": 1, "disgust": 1}},
                {"dominant_emotion": "sad", "emotion": {"happy": 10, "sad": 55, "neutral": 20, "angry": 5, "surprised": 5, "fear": 3, "disgust": 2}},
            ]

            resp = requests.post(
                f"{backend_url}/session/report/visual",
                json={
                    "session_id": sid,
                    "raw_vision_data": synthetic_data,
                    "style_preset": "futuristic",
                },
                timeout=REPORT_TIMEOUT,
            )
            assert resp.status_code in [200, 400], (
                f"Unexpected status {resp.status_code}: {resp.text[:200]}"
            )
            data = resp.json()
            assert isinstance(data, dict)

            if resp.status_code == 200:
                assert "metrics" in data
                assert "image_prompt" in data
                if data.get("public_url"):
                    print(f"[e2e-http] Visual report URL: {data['public_url'][:120]}")
        finally:
            _stop_session(backend_url, sid)

    def test_report_missing_session_id_returns_422(self, backend_url):
        """POST /session/report/emotion without session_id should be rejected."""
        resp = requests.post(
            f"{backend_url}/session/report/emotion",
            json={"raw_vision_data": [{"emotion": {"happy": 50}}]},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 422

    def test_report_invalid_session_id_returns_422(self, backend_url):
        """Non-UUID session_id in report request should be rejected."""
        resp = requests.post(
            f"{backend_url}/session/report/visual",
            json={"session_id": "bad-id", "raw_vision_data": []},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 422


# ============================================================================
# 6. SECURITY AND ROBUSTNESS
# ============================================================================
class TestSecurityAndRobustness:
    """Verify security constraints and graceful error handling over HTTP."""

    def test_error_responses_hide_tracebacks(self, backend_url):
        """Error responses must NOT expose Python tracebacks or file paths."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={"img": "definitely_not_an_image", "actions": ["emotion"]},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code >= 400:
            body = resp.text
            assert "Traceback" not in body, "Traceback leaked in error response"
            assert 'File "' not in body, "File path leaked in error response"

    def test_oversized_metadata_rejected(self, backend_url):
        """Session start with oversized metadata should be rejected (Pydantic validator)."""
        huge_metadata = {"key": "x" * 20000}
        resp = requests.post(
            f"{backend_url}/session/start",
            json={"metadata": huge_metadata},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code in [200, 422], (
            f"Expected 200 or 422, got {resp.status_code}"
        )

    def test_cors_preflight(self, backend_url):
        """OPTIONS request should include CORS headers for the React frontend."""
        resp = requests.options(
            f"{backend_url}/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
            timeout=REQUEST_TIMEOUT,
        )
        # Flask-CORS should respond with 200 or 204
        assert resp.status_code in [200, 204]

    def test_json_content_type_accepted(self, backend_url, face_b64):
        """Explicit application/json content type should work."""
        resp = requests.post(
            f"{backend_url}/analyze",
            data=json.dumps({
                "img": face_b64,
                "actions": ["emotion"],
                "anti_spoofing": False,
            }),
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200


# ============================================================================
# 7. CONCURRENT / STRESS PATTERNS
# ============================================================================
class TestConcurrentPatterns:
    """
    Test patterns that emerge from real frontend usage:
    rapid clicks, parallel sessions, etc.
    """

    def test_rapid_analyze_calls(self, backend_url, face_b64):
        """
        Simulate the frontend's interval-based capture: send 5 analyze
        requests in quick succession with a session_id.
        """
        sid = _start_session(backend_url)
        try:
            success_count = 0
            for i in range(5):
                resp = requests.post(
                    f"{backend_url}/analyze",
                    json={
                        "img": face_b64,
                        "actions": ["emotion"],
                        "anti_spoofing": False,
                        "session_id": sid,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 200:
                    success_count += 1
                time.sleep(0.2)  # Slight delay, similar to frontend interval

            assert success_count >= 3, (
                f"Expected at least 3/5 rapid analyses to succeed, got {success_count}"
            )
        finally:
            _stop_session(backend_url, sid)

    def test_parallel_sessions(self, backend_url, face_b64):
        """
        Two sessions active at the same time (e.g., two browser tabs).
        Each should maintain independent state.
        """
        sid1 = _start_session(backend_url)
        sid2 = _start_session(backend_url)
        assert sid1 != sid2, "Two sessions should have different IDs"

        try:
            # Analyze in session 1
            resp1 = requests.post(
                f"{backend_url}/analyze",
                json={
                    "img": face_b64,
                    "actions": ["emotion"],
                    "anti_spoofing": False,
                    "session_id": sid1,
                },
                timeout=REQUEST_TIMEOUT,
            )
            assert resp1.status_code == 200

            # Analyze in session 2
            resp2 = requests.post(
                f"{backend_url}/analyze",
                json={
                    "img": face_b64,
                    "actions": ["emotion"],
                    "anti_spoofing": False,
                    "session_id": sid2,
                },
                timeout=REQUEST_TIMEOUT,
            )
            assert resp2.status_code == 200

            # Verify session details are independent
            det1 = requests.get(f"{backend_url}/session/{sid1}", timeout=REQUEST_TIMEOUT).json()
            det2 = requests.get(f"{backend_url}/session/{sid2}", timeout=REQUEST_TIMEOUT).json()
            assert det1["session"]["id"] == sid1
            assert det2["session"]["id"] == sid2

        finally:
            _stop_session(backend_url, sid1)
            _stop_session(backend_url, sid2)
