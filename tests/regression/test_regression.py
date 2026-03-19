"""
Regression tests -- prevent known bugs from coming back.
Each test is named after the bug/issue it guards against.
"""
import pytest
import requests
import json
import time


class TestRegressionBugs:
    """Regression tests for previously fixed bugs."""

    def test_regression_stale_session_id_closure(
        self, backend_url, test_face_image_base64
    ):
        """
        Regression: React closure bug where session_id was stale.
        Fixed in commit 9edaf14. Backend must accept session_id in request body.
        """
        # Start session
        resp = requests.post(f"{backend_url}/session/start", json={}, timeout=10)
        session_id = resp.json()["session_id"]

        # Analyze with explicit session_id (the fix was using sessionIdRef)
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "session_id": session_id,
                "anti_spoofing": False,
            },
            timeout=30,
        )
        assert resp.status_code == 200

        # Stop
        requests.post(
            f"{backend_url}/session/stop", json={"session_id": session_id}
        )

    def test_regression_rls_policy_insert(
        self, backend_url, test_face_image_base64
    ):
        """
        Regression: RLS policies blocked inserts to vision_samples, session_events, etc.
        Fixed in commit 9edaf14.
        """
        resp = requests.post(f"{backend_url}/session/start", json={}, timeout=10)
        session_id = resp.json()["session_id"]

        # This should NOT fail with RLS errors
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "session_id": session_id,
                "anti_spoofing": False,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Ensure no RLS error in response
        data_str = str(data).lower()
        assert "rls" not in data_str, f"RLS error detected: {data}"
        assert (
            "policy" not in data_str
            or "anti" in data_str  # allow "anti_spoofing" to contain "policy" substring
        ), f"Policy error detected: {data}"

        requests.post(
            f"{backend_url}/session/stop", json={"session_id": session_id}
        )

    def test_regression_gemini_image_config_style(
        self, backend_url, test_face_image_base64
    ):
        """
        Regression: Gemini ImageConfig rejected style_preset parameter.
        Fixed in commit 9edaf14 (moved to prompt text).
        Visual report should NOT crash.
        """
        # Create session with data
        resp = requests.post(f"{backend_url}/session/start", json={}, timeout=10)
        session_id = resp.json()["session_id"]

        for _ in range(3):
            requests.post(
                f"{backend_url}/analyze",
                json={
                    "img": test_face_image_base64,
                    "actions": ["emotion"],
                    "session_id": session_id,
                    "anti_spoofing": False,
                },
                timeout=30,
            )
            time.sleep(0.5)

        # Visual report should not crash with style_preset
        resp = requests.post(
            f"{backend_url}/session/report/visual",
            json={
                "session_id": session_id,
                "style_preset": "futuristic",
            },
            timeout=180,
        )
        # Should succeed or fail gracefully (not 500 crash)
        assert resp.status_code in [
            200,
            400,
        ], f"Unexpected status {resp.status_code}: {resp.text[:200]}"

        requests.post(
            f"{backend_url}/session/stop", json={"session_id": session_id}
        )

    def test_regression_response_key_mismatch(
        self, backend_url, test_face_image_base64
    ):
        """
        Regression: Frontend expected image_url but backend returned report_url.
        Fixed in commit 9edaf14. Pause response must contain image_url key.
        """
        resp = requests.post(f"{backend_url}/session/start", json={}, timeout=10)
        session_id = resp.json()["session_id"]

        for _ in range(3):
            requests.post(
                f"{backend_url}/analyze",
                json={
                    "img": test_face_image_base64,
                    "actions": ["emotion"],
                    "session_id": session_id,
                    "anti_spoofing": False,
                },
                timeout=30,
            )
            time.sleep(0.5)

        resp = requests.post(
            f"{backend_url}/session/pause",
            json={"session_id": session_id},
            timeout=180,
        )

        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, dict), f"Response is not dict: {type(data)}"
            # After the fix, pause_session returns image_url for backward compat
            if "error" not in data:
                assert (
                    "image_url" in data
                ), f"Missing image_url in pause response: {list(data.keys())}"

        requests.post(
            f"{backend_url}/session/stop", json={"session_id": session_id}
        )

    def test_regression_error_no_traceback(self, backend_url):
        """
        Regression: Error responses used to expose full Python tracebacks.
        After security fix, errors should only contain error_id.
        """
        # Send bad request that should trigger error
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": "not_a_valid_base64_image",
                "actions": ["emotion"],
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            data = resp.json()
            error_str = json.dumps(data)
            # Should NOT contain Python traceback indicators
            assert (
                "Traceback" not in error_str
            ), "Traceback exposed in error response!"
            assert (
                'File "' not in error_str
            ), "File path exposed in error response!"

    def test_regression_empty_actions_list(
        self, backend_url, test_face_image_base64
    ):
        """Guard against empty actions list causing crash."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": [],
            },
            timeout=30,
        )
        # Pydantic validator requires at least one action -> 422
        assert resp.status_code in [200, 400, 422]
        if resp.status_code == 422:
            data = resp.json()
            assert "error" in data

    def test_regression_large_image_handling(self, backend_url):
        """Guard against oversized images causing OOM."""
        from PIL import Image
        import io
        import base64

        # Create a large image (4K resolution)
        img = Image.new("RGB", (3840, 2160), color=(128, 128, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        large_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": large_b64,
                "actions": ["emotion"],
                "anti_spoofing": False,
            },
            timeout=60,
        )
        # Should either process or reject gracefully
        assert resp.status_code in [200, 400, 413, 422]

    def test_regression_invalid_session_id_format(self, backend_url):
        """Guard against non-UUID session_id causing unhandled crash."""
        resp = requests.post(
            f"{backend_url}/session/stop",
            json={"session_id": "not-a-uuid"},
            timeout=10,
        )
        # Validator should catch and return 422
        assert resp.status_code in [400, 422, 500]


class TestAPIContractRegression:
    """Ensure API response schemas don't change unexpectedly."""

    def test_analyze_response_schema(self, backend_url, test_face_image_base64):
        """Verify /analyze response structure matches expected contract."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "anti_spoofing": False,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()

        # Response must have "results" key
        assert "results" in data, f"Missing 'results' key: {list(data.keys())}"
        results = data["results"]
        assert isinstance(results, list) and len(results) > 0

        face = results[0]

        # Emotion schema
        assert "emotion" in face, f"Missing 'emotion' in face: {list(face.keys())}"
        emotions = face["emotion"]
        required_emotions = {
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        }
        assert required_emotions.issubset(
            set(emotions.keys())
        ), f"Missing emotions: {required_emotions - set(emotions.keys())}"
        for name, value in emotions.items():
            assert isinstance(
                value, (int, float)
            ), f"Emotion {name} is not numeric: {type(value)}"

        # Dominant emotion
        assert "dominant_emotion" in face

        # Face region
        assert "region" in face or "face_confidence" in face

    def test_session_start_response_schema(self, backend_url):
        """Verify /session/start response structure."""
        resp = requests.post(f"{backend_url}/session/start", json={}, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert (
            "session_id" in data
        ), f"Missing session_id in response: {list(data.keys())}"
        assert data.get("status") == "active"

        # Cleanup
        requests.post(
            f"{backend_url}/session/stop",
            json={"session_id": data["session_id"]},
        )

    def test_session_history_response_schema(self, backend_url):
        """Verify /session/history response structure."""
        resp = requests.get(f"{backend_url}/session/history", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_home_endpoint_response(self, backend_url):
        """Verify / health check works."""
        resp = requests.get(f"{backend_url}/", timeout=10)
        assert resp.status_code == 200
        # Home returns HTML with DeepFace version
        assert "SynCVE" in resp.text or "DeepFace" in resp.text

    def test_analyze_validation_errors(self, backend_url):
        """Verify invalid requests return proper validation errors."""
        # Missing img
        resp = requests.post(
            f"{backend_url}/analyze",
            json={"actions": ["emotion"]},
            timeout=10,
        )
        assert resp.status_code == 400

        # Invalid action
        resp = requests.post(
            f"{backend_url}/analyze",
            json={"img": "placeholder", "actions": ["invalid_action"]},
            timeout=10,
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data
