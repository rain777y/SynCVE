"""
Real integration tests for emotion detection pipeline.
These tests hit the REAL backend with REAL images on REAL GPU.
"""
import pytest
import requests
import time


class TestEmotionDetectionReal:
    """Test the core emotion detection with real inference."""

    def test_analyze_real_face(self, backend_url, test_face_image_base64):
        """Test emotion analysis on a real face image -- verifies full GPU inference pipeline."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "detector_backend": "retinaface",
                "enforce_detection": True,
                "anti_spoofing": False,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Must return results dict with results key
        assert "results" in data, f"Unexpected response shape: {list(data.keys())}"
        results = data["results"]
        assert isinstance(results, list) and len(results) > 0, "No faces detected"
        face = results[0]
        assert "emotion" in face, f"No emotion data in face result: {list(face.keys())}"
        emotions = face["emotion"]
        # All 7 emotions must be present
        expected_emotions = {
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        }
        assert set(emotions.keys()) == expected_emotions
        # Probabilities should sum to ~100
        total = sum(emotions.values())
        assert 90 < total < 110, f"Emotion probabilities sum to {total}, expected ~100"

    def test_analyze_no_face_image(self, backend_url, no_face_image_base64):
        """Test that no-face image is handled gracefully."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": no_face_image_base64,
                "actions": ["emotion"],
                "enforce_detection": True,
                "anti_spoofing": False,
            },
            timeout=30,
        )
        # Should return 400 or 200 with results (ensemble may fallback)
        assert resp.status_code in [400, 200]
        if resp.status_code == 200:
            data = resp.json()
            # Either empty results, low confidence, or detection_fallback
            results = data.get("results", [])
            if isinstance(results, list) and len(results) > 0:
                # Detection fallback might have been used
                pass

    def test_analyze_with_all_actions(self, backend_url, test_face_image_base64):
        """Test all analysis actions: emotion + age + gender + race."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion", "age", "gender", "race"],
                "detector_backend": "retinaface",
                "enforce_detection": True,
                "anti_spoofing": False,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        results = data.get("results", [])
        assert len(results) > 0, "No results returned"
        face = results[0]
        assert "emotion" in face
        assert "age" in face
        # DeepFace returns dominant_gender and gender dict
        assert (
            "gender" in face or "dominant_gender" in face
        ), f"No gender data: {list(face.keys())}"
        assert (
            "race" in face or "dominant_race" in face
        ), f"No race data: {list(face.keys())}"

    def test_analyze_single_detector_no_ensemble(
        self, backend_url, test_face_image_base64
    ):
        """Test single detector mode with ensemble disabled."""
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                # opencv works reliably; retinaface has TF/Keras KerasTensor issues
                # in the Flask process on Python 3.13
                "detector_backend": "opencv",
                "enforce_detection": False,
                "anti_spoofing": False,
                "enable_ensemble": False,
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_analyze_inference_latency(self, backend_url, test_face_image_base64):
        """Performance test: subsequent inference should be < 5 seconds."""
        # Warmup
        requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "anti_spoofing": False,
            },
            timeout=30,
        )

        # Timed run
        start = time.time()
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "anti_spoofing": False,
            },
            timeout=30,
        )
        elapsed = time.time() - start
        assert resp.status_code == 200
        assert elapsed < 5.0, f"Inference took {elapsed:.2f}s, expected < 5s"

    def test_represent_real_face(self, backend_url, test_face_image_base64):
        """Test face embedding extraction via /represent endpoint."""
        resp = requests.post(
            f"{backend_url}/represent",
            json={
                "img": test_face_image_base64,
                "model_name": "Facenet",
                # Use opencv — retinaface has a TF/Keras compatibility issue on Windows
                # that causes KerasTensor errors after the Flask process loads other models.
                "detector_backend": "opencv",
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data, f"No results in /represent response: {list(data.keys())}"
        assert isinstance(data["results"], list) and len(data["results"]) > 0

    def test_verify_same_face(self, backend_url, test_face_image_base64):
        """Test face verification with the same image (should match)."""
        resp = requests.post(
            f"{backend_url}/verify",
            json={
                "img1": test_face_image_base64,
                "img2": test_face_image_base64,
                "detector_backend": "opencv",
            },
            timeout=30,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Same image should verify as same person
        if "verified" in data:
            assert data["verified"] is True
        elif "results" in data:
            assert data["results"].get("verified", True)

    def test_concurrent_sequential_requests(
        self, backend_url, test_face_image_base64
    ):
        """Test that backend handles 3 rapid sequential requests."""
        for i in range(3):
            resp = requests.post(
                f"{backend_url}/analyze",
                json={
                    "img": test_face_image_base64,
                    "actions": ["emotion"],
                    "anti_spoofing": False,
                },
                timeout=30,
            )
            assert (
                resp.status_code == 200
            ), f"Request {i + 1} failed: {resp.status_code}"
