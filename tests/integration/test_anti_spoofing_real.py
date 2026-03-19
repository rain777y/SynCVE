"""
Real integration tests for anti-spoofing detection.
"""
import pytest
import requests


class TestAntiSpoofingReal:
    """Test anti-spoofing with real FasNet inference."""

    def test_anti_spoofing_enabled(self, backend_url, test_face_image_base64):
        """Test that anti-spoofing can be enabled without crashing."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "anti_spoofing": True,
                "enforce_detection": True,
            },
            timeout=30,
        )
        # Anti-spoofing may reject or accept - both are valid
        assert resp.status_code in [200, 400]
        data = resp.json()
        if resp.status_code == 400:
            # Should mention anti-spoofing in error
            error_str = str(data).lower()
            assert (
                "spoof" in error_str
                or "anti" in error_str
                or "error" in data
            ), f"Unexpected 400 response: {data}"

    def test_anti_spoofing_disabled(self, backend_url, test_face_image_base64):
        """Test that disabling anti-spoofing still returns results."""
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
        assert "results" in data

    def test_anti_spoofing_metadata_in_response(
        self, backend_url, test_face_image_base64
    ):
        """Check that spoof_check metadata is present in analysis results."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "anti_spoofing": False,
                "enable_ensemble": False,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        results = data.get("results", [])
        if results:
            face = results[0]
            # Service layer sets spoof_check metadata
            assert "spoof_check" in face, f"No spoof_check in face: {list(face.keys())}"
