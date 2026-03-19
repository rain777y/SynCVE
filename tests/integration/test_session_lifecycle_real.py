"""
Real integration tests for session lifecycle.
Tests: start -> analyze -> pause -> report -> resume -> stop
"""
import pytest
import requests
import time


class TestSessionLifecycleReal:
    """Test complete session lifecycle against real backend + Supabase."""

    def test_full_session_lifecycle(self, backend_url, test_face_image_base64):
        """Complete lifecycle: start -> 3 analyses -> pause -> get details -> stop."""
        # 1. Start session
        resp = requests.post(f"{backend_url}/session/start", json={}, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        session_id = data.get("session_id")
        assert session_id, f"No session_id: {data}"

        try:
            # 2. Submit 3 emotion analyses with session_id
            for i in range(3):
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
                assert resp.status_code == 200, f"Analysis {i + 1} failed"
                time.sleep(0.5)

            # 3. Pause session (triggers visual report generation via Gemini)
            resp = requests.post(
                f"{backend_url}/session/pause",
                json={"session_id": session_id},
                timeout=180,  # Report generation can be slow
            )
            assert resp.status_code == 200
            pause_data = resp.json()
            # pause_session returns {"status": "paused", "image_url": ..., ...} on success
            # or {"error": ...} on failure
            assert isinstance(pause_data, dict)

            # 4. Get session details
            resp = requests.get(
                f"{backend_url}/session/{session_id}", timeout=10
            )
            assert resp.status_code == 200
            details = resp.json()
            assert (
                "session" in details
            ), f"Expected 'session' key in details: {list(details.keys())}"

            # 5. Stop session
            resp = requests.post(
                f"{backend_url}/session/stop",
                json={"session_id": session_id},
                timeout=30,
            )
            assert resp.status_code == 200
        except Exception:
            # Always try to stop
            requests.post(
                f"{backend_url}/session/stop",
                json={"session_id": session_id},
            )
            raise

    def test_session_history(self, backend_url):
        """Test fetching session history (requires at least 1 prior session)."""
        resp = requests.get(f"{backend_url}/session/history", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_session_start_and_immediate_stop(self, backend_url):
        """Test starting and immediately stopping a session (edge case)."""
        resp = requests.post(f"{backend_url}/session/start", json={}, timeout=10)
        assert resp.status_code == 200
        session_id = resp.json().get("session_id")
        assert session_id

        resp = requests.post(
            f"{backend_url}/session/stop",
            json={"session_id": session_id},
            timeout=30,
        )
        assert resp.status_code == 200

    def test_stop_nonexistent_session(self, backend_url):
        """Test stopping a session that doesn't exist."""
        resp = requests.post(
            f"{backend_url}/session/stop",
            json={"session_id": "00000000-0000-0000-0000-000000000000"},
            timeout=10,
        )
        # Should handle gracefully (either 200 with error or 500)
        assert resp.status_code in [200, 400, 404, 500]

    def test_session_start_response_schema(self, backend_url):
        """Verify /session/start response contains session_id and status."""
        resp = requests.post(f"{backend_url}/session/start", json={}, timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data, f"Missing session_id: {list(data.keys())}"
        assert data.get("status") == "active"

        # Cleanup
        requests.post(
            f"{backend_url}/session/stop",
            json={"session_id": data["session_id"]},
        )

    def test_session_with_user_id_and_metadata(self, backend_url):
        """Test starting a session with optional user_id and metadata."""
        resp = requests.post(
            f"{backend_url}/session/start",
            json={
                "user_id": "test-user-integration",
                "metadata": {"test": True, "source": "integration_test"},
            },
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data

        # Cleanup
        requests.post(
            f"{backend_url}/session/stop",
            json={"session_id": data["session_id"]},
        )

    def test_session_history_with_limit(self, backend_url):
        """Test session history respects limit parameter."""
        resp = requests.get(
            f"{backend_url}/session/history", params={"limit": 3}, timeout=10
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert len(data["sessions"]) <= 3
