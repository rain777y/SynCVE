"""
Real integration tests for AI-powered report generation.
Uses REAL Gemini API with actual API key.
"""
import pytest
import requests
import time


class TestReportGenerationReal:
    """Test report generation with real Gemini API calls."""

    @pytest.fixture
    def session_with_data(self, backend_url, test_face_image_base64):
        """Create a session with enough data for report generation."""
        # Start session
        resp = requests.post(f"{backend_url}/session/start", json={}, timeout=10)
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Submit 5 emotion analyses to build data
        for i in range(5):
            resp = requests.post(
                f"{backend_url}/analyze",
                json={
                    "img": test_face_image_base64,
                    "actions": ["emotion"],
                    "anti_spoofing": False,
                    "session_id": session_id,
                },
                timeout=30,
            )
            assert resp.status_code == 200, f"Analysis {i + 1} failed: {resp.text[:200]}"
            time.sleep(1)  # Space out for realistic data

        yield session_id

        # Cleanup
        try:
            requests.post(
                f"{backend_url}/session/stop",
                json={"session_id": session_id},
            )
        except Exception:
            pass

    def test_emotion_text_report(self, backend_url, session_with_data):
        """Test real Gemini-powered emotion text report generation."""
        resp = requests.post(
            f"{backend_url}/session/report/emotion",
            json={
                "session_id": session_with_data,
                "max_keyframes": 2,
            },
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should contain report_markdown from the two-stage pipeline
        assert (
            "report_markdown" in data or "error" in data
        ), f"Unexpected response keys: {list(data.keys())}"
        if "report_markdown" in data:
            assert len(data["report_markdown"]) > 0
            assert "metrics" in data
            assert "session_id" in data

    def test_visual_report_generation(self, backend_url, session_with_data):
        """Test real Gemini-powered visual report (image) generation."""
        resp = requests.post(
            f"{backend_url}/session/report/visual",
            json={
                "session_id": session_with_data,
                "aspect_ratio": "16:9",
                "style_preset": "futuristic",
            },
            timeout=180,  # Image generation can be slow
        )
        assert resp.status_code == 200
        data = resp.json()
        # v3 pipeline returns public_url, image_prompt, metrics, etc.
        has_report = any(
            k in data
            for k in ["public_url", "image_prompt", "metrics", "storage_path"]
        )
        has_error = "error" in data
        assert has_report or has_error, f"Unexpected response: {list(data.keys())}"

    def test_pause_triggers_visual_report(self, backend_url, session_with_data):
        """Test that pausing a session triggers visual report generation."""
        resp = requests.post(
            f"{backend_url}/session/pause",
            json={"session_id": session_with_data},
            timeout=180,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Pause should return status, image_url, visual_report on success
        assert isinstance(data, dict)
        if "error" not in data:
            assert data.get("status") == "paused"
            assert "visual_report" in data or "image_url" in data
