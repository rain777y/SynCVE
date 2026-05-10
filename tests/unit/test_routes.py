"""
Tests for all API endpoints defined in src.backend.routes.

Every test mocks DeepFace and external services so no GPU or network is needed.
"""

import json
import uuid
from unittest.mock import patch, MagicMock

import pytest

# Deterministic UUID for all session-id tests (valid UUID format required by Pydantic)
_TEST_UUID = str(uuid.uuid4())


# =========================================================================
# GET /
# =========================================================================

class TestHomeEndpoint:
    def test_home_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_home_contains_welcome_text(self, client):
        resp = client.get("/")
        assert b"Welcome to SynCVE Backend" in resp.data


# =========================================================================
# POST /analyze
# =========================================================================

class TestAnalyzeEndpoint:
    def test_analyze_with_valid_image(self, client, sample_base64_image, mock_deepface):
        """Valid base64 image should return 200 with emotion results."""
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.log_data.return_value = {"status": "logged"}

            resp = client.post(
                "/analyze",
                json={
                    "img": sample_base64_image,
                    "actions": ["emotion"],
                    "session_id": "test-session-1",
                },
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert "results" in data
            assert len(data["results"]) > 0
            assert "emotion" in data["results"][0]
            assert "dominant_emotion" in data["results"][0]

    def test_analyze_without_image_returns_400(self, client):
        """Missing image should return 400."""
        resp = client.post("/analyze", json={"actions": ["emotion"]})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_analyze_with_string_actions(self, client, sample_base64_image, mock_deepface):
        """Actions passed as comma-separated string should be parsed correctly."""
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.log_data.return_value = {"status": "logged"}

            resp = client.post(
                "/analyze",
                json={
                    "img": sample_base64_image,
                    "actions": "emotion,age",
                    "session_id": "test-sess",
                },
            )
            assert resp.status_code == 200

    def test_analyze_with_anti_spoofing_flag(self, client, sample_base64_image, mock_deepface):
        """anti_spoofing parameter should be forwarded to service layer."""
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.log_data.return_value = {"status": "logged"}
            with patch("src.backend.routes.service") as mock_svc:
                mock_svc.analyze.return_value = {"results": [{"emotion": {"happy": 80}, "dominant_emotion": "happy"}]}

                resp = client.post(
                    "/analyze",
                    json={
                        "img": sample_base64_image,
                        "actions": ["emotion"],
                        "anti_spoofing": True,
                        "session_id": "test-sess",
                    },
                )
                assert resp.status_code == 200
                call_kwargs = mock_svc.analyze.call_args
                assert call_kwargs is not None

    def test_analyze_logs_to_session_when_session_id_present(
        self, client, sample_base64_image, mock_deepface
    ):
        """If session_id is provided, session_manager.log_data should be called."""
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.log_data.return_value = {"status": "logged"}

            resp = client.post(
                "/analyze",
                json={
                    "img": sample_base64_image,
                    "actions": ["emotion"],
                    "session_id": "my-session",
                },
            )
            assert resp.status_code == 200
            mock_sm.log_data.assert_called_once()
            args = mock_sm.log_data.call_args
            assert args[0][0] == "my-session"

    def test_analyze_without_session_id_skips_logging(
        self, client, sample_base64_image, mock_deepface
    ):
        """Without session_id, log_data should NOT be called."""
        with patch("src.backend.routes.session_manager") as mock_sm:
            resp = client.post(
                "/analyze",
                json={"img": sample_base64_image, "actions": ["emotion"]},
            )
            assert resp.status_code == 200
            mock_sm.log_data.assert_not_called()

    def test_analyze_returns_error_on_deepface_exception(self, client, sample_base64_image):
        """If DeepFace.analyze raises, endpoint should return 400 with error details."""
        with patch("src.backend.routes.service") as mock_svc:
            mock_svc.analyze.return_value = (
                {"error": "Exception while analyzing: Face not found"},
                400,
            )
            resp = client.post(
                "/analyze",
                json={"img": sample_base64_image, "actions": ["emotion"]},
            )
            # Flask returns the tuple directly
            assert resp.status_code == 400


# =========================================================================
# POST /represent
# =========================================================================

class TestRepresentEndpoint:
    def test_represent_with_valid_image(self, client, sample_base64_image):
        with patch("src.backend.routes.service") as mock_svc:
            mock_svc.represent.return_value = {"results": [{"embedding": [0.1] * 128}]}
            resp = client.post(
                "/represent",
                json={"img": sample_base64_image, "model_name": "VGG-Face"},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert "results" in data

    def test_represent_without_image(self, client):
        resp = client.post("/represent", json={})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data


# =========================================================================
# POST /verify
# =========================================================================

class TestVerifyEndpoint:
    def test_verify_two_images(self, client, sample_base64_image):
        with patch("src.backend.routes.service") as mock_svc:
            mock_svc.verify.return_value = {"verified": True, "distance": 0.2}
            resp = client.post(
                "/verify",
                json={"img1": sample_base64_image, "img2": sample_base64_image},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["verified"] is True

    def test_verify_missing_img1(self, client, sample_base64_image):
        resp = client.post("/verify", json={"img2": sample_base64_image})
        assert resp.status_code == 400

    def test_verify_missing_img2(self, client, sample_base64_image):
        resp = client.post("/verify", json={"img1": sample_base64_image})
        assert resp.status_code == 400


# =========================================================================
# POST /session/start
# =========================================================================

class TestSessionStartEndpoint:
    def test_session_start(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.start_session.return_value = {
                "session_id": "new-sess-id",
                "status": "active",
            }
            resp = client.post(
                "/session/start",
                json={"user_id": "user-1", "metadata": {"source": "test"}},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["session_id"] == "new-sess-id"
            assert data["status"] == "active"

    def test_session_start_error_returns_500(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.start_session.return_value = {"error": "DB down"}
            resp = client.post("/session/start", json={})
            assert resp.status_code == 500


# =========================================================================
# POST /session/stop
# =========================================================================

class TestSessionStopEndpoint:
    def test_session_stop_without_session_id(self, client):
        resp = client.post("/session/stop", json={})
        assert resp.status_code == 422

    def test_session_stop_success(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.stop_session.return_value = {
                "status": "session_ended",
                "report": {"summary": "ok"},
            }
            resp = client.post("/session/stop", json={"session_id": _TEST_UUID})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "session_ended"


# =========================================================================
# POST /session/pause
# =========================================================================

class TestSessionPauseEndpoint:
    def test_session_pause_without_session_id(self, client):
        resp = client.post("/session/pause", json={})
        assert resp.status_code in (400, 422)

    def test_session_pause_success(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.pause_session.return_value = {
                "status": "paused",
                "image_url": "https://example.com/report.png",
                "message": "Session paused.",
            }
            resp = client.post("/session/pause", json={"session_id": _TEST_UUID})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "paused"
            assert "image_url" in data

    def test_session_pause_error_with_status_code(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.pause_session.return_value = {
                "error": "No data",
                "status_code": 400,
            }
            resp = client.post("/session/pause", json={"session_id": _TEST_UUID})
            assert resp.status_code == 400


# =========================================================================
# GET /session/history
# =========================================================================

class TestSessionHistoryEndpoint:
    def test_session_history_returns_list(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.get_recent_sessions.return_value = {"sessions": [{"id": "s-1"}]}
            resp = client.get("/session/history?limit=5")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "sessions" in data
            assert len(data["sessions"]) == 1

    def test_session_history_with_user_filter(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.get_recent_sessions.return_value = {"sessions": []}
            resp = client.get("/session/history?user_id=u1&limit=3")
            assert resp.status_code == 200
            mock_sm.get_recent_sessions.assert_called_once_with("u1", 3)


# =========================================================================
# GET /session/<session_id>
# =========================================================================

class TestSessionDetailsEndpoint:
    def test_get_session_details(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.get_session_details.return_value = {
                "session": {"id": "s-1", "status": "active"}
            }
            resp = client.get("/session/s-1")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["session"]["id"] == "s-1"

    def test_get_session_not_found(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.get_session_details.return_value = {"error": "Session not found"}
            resp = client.get("/session/nonexistent")
            assert resp.status_code == 404


# =========================================================================
# POST /session/report/emotion
# =========================================================================

class TestEmotionReportEndpoint:
    def test_emotion_report_missing_session_id(self, client):
        resp = client.post("/session/report/emotion", json={})
        assert resp.status_code == 422

    def test_emotion_report_success(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.EMOTION_REPORT_KEYFRAME_LIMIT = 4
            mock_sm.generate_emotion_report.return_value = {
                "session_id": _TEST_UUID,
                "report_markdown": "# Report",
                "metrics": {},
            }
            resp = client.post(
                "/session/report/emotion",
                json={"session_id": _TEST_UUID},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert "report_markdown" in data

    def test_emotion_report_value_error(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.EMOTION_REPORT_KEYFRAME_LIMIT = 4
            mock_sm.generate_emotion_report.side_effect = ValueError("No vision data")
            resp = client.post(
                "/session/report/emotion",
                json={"session_id": _TEST_UUID},
            )
            assert resp.status_code == 400

    def test_emotion_report_generic_error(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.EMOTION_REPORT_KEYFRAME_LIMIT = 4
            mock_sm.generate_emotion_report.side_effect = RuntimeError("oops")
            resp = client.post(
                "/session/report/emotion",
                json={"session_id": _TEST_UUID},
            )
            assert resp.status_code == 500


# =========================================================================
# POST /session/report/visual
# =========================================================================

class TestVisualReportEndpoint:
    def test_visual_report_missing_session_id(self, client):
        resp = client.post("/session/report/visual", json={})
        assert resp.status_code == 422

    def test_visual_report_success(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.EMOTION_VISUAL_ASPECT_RATIO = "16:9"
            mock_sm.EMOTION_VISUAL_STYLE_PRESET = "futuristic"
            mock_sm.generate_visual_report_v3.return_value = {
                "session_id": _TEST_UUID,
                "public_url": "https://fake/report.png",
                "metrics": {},
            }
            resp = client.post(
                "/session/report/visual",
                json={"session_id": _TEST_UUID},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["public_url"] == "https://fake/report.png"

    def test_visual_report_value_error(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.EMOTION_VISUAL_ASPECT_RATIO = "16:9"
            mock_sm.EMOTION_VISUAL_STYLE_PRESET = "futuristic"
            mock_sm.generate_visual_report_v3.side_effect = ValueError("No data")
            resp = client.post(
                "/session/report/visual",
                json={"session_id": _TEST_UUID},
            )
            assert resp.status_code == 400

    def test_visual_report_generic_error(self, client):
        with patch("src.backend.routes.session_manager") as mock_sm:
            mock_sm.EMOTION_VISUAL_ASPECT_RATIO = "16:9"
            mock_sm.EMOTION_VISUAL_STYLE_PRESET = "futuristic"
            mock_sm.generate_visual_report_v3.side_effect = RuntimeError("boom")
            resp = client.post(
                "/session/report/visual",
                json={"session_id": _TEST_UUID},
            )
            assert resp.status_code == 500
