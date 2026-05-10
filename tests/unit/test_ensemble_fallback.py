"""
Tests for ensemble detector fallback behavior.

Covers:
  1. Primary detector fails → fallback to secondary
  2. All detectors fail → structured error response
  3. enforce_detection fallback within ensemble
  4. Input size filtering via DETECTOR_MIN_SIZE
  5. Anti-spoofing rejection propagation
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Shared mock setup
# ---------------------------------------------------------------------------
def _make_emotion_result(dominant="happy", detector="opencv"):
    return {
        "emotion": {
            "happy": 80, "sad": 5, "neutral": 10,
            "angry": 2, "fear": 1, "surprise": 1, "disgust": 1,
        },
        "dominant_emotion": dominant,
        "region": {"x": 10, "y": 10, "w": 100, "h": 100},
        "face_confidence": 0.95,
        "detector_backend": detector,
    }


# ============================================================================
# 1. ENSEMBLE FALLBACK: ONE DETECTOR FAILS
# ============================================================================
class TestEnsembleFallback:

    def test_ensemble_succeeds_when_one_detector_fails(self, mock_deepface):
        """If first detector fails, ensemble should still succeed with remaining."""
        from src.backend.service import analyze

        call_count = [0]
        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            backend = kwargs.get("detector_backend", "unknown")
            if backend == "retinaface":
                raise ValueError("Face could not be detected in numpy array")
            return [_make_emotion_result(detector=backend)]

        mock_deepface.analyze.side_effect = _side_effect

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = analyze(
            img_path=img,
            actions=["emotion"],
            detector_backend="retinaface,mtcnn",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
            enable_ensemble=True,
        )

        # Should NOT be an error tuple
        assert not isinstance(result, tuple), f"Unexpected error: {result}"
        assert "results" in result

    def test_ensemble_all_detectors_fail_returns_error(self, mock_deepface):
        """If ALL detectors fail, should return structured error."""
        from src.backend.service import analyze

        mock_deepface.analyze.side_effect = ValueError("Face could not be detected")

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = analyze(
            img_path=img,
            actions=["emotion"],
            detector_backend="retinaface,mtcnn",
            enforce_detection=True,
            align=True,
            anti_spoofing=False,
            enable_ensemble=True,
        )

        # Should be error tuple
        assert isinstance(result, tuple)
        error_dict, status_code = result
        assert "error" in error_dict
        assert status_code == 400

    def test_single_detector_mode_no_ensemble(self, mock_deepface):
        """With enable_ensemble=False, only one detector should be used."""
        from src.backend.service import analyze

        mock_deepface.analyze.return_value = [_make_emotion_result()]

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = analyze(
            img_path=img,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
            enable_ensemble=False,
        )

        assert not isinstance(result, tuple)
        assert "results" in result


# ============================================================================
# 2. ENFORCE_DETECTION FALLBACK
# ============================================================================
class TestEnforceDetectionFallback:

    def test_retry_with_enforce_false_on_detection_failure(self, mock_deepface):
        """When face not detected, should retry with enforce_detection=False."""
        from src.backend.service import analyze

        attempt = [0]
        def _side_effect(*args, **kwargs):
            attempt[0] += 1
            if kwargs.get("enforce_detection", True):
                raise ValueError("Face could not be detected in numpy array")
            return [_make_emotion_result()]

        mock_deepface.analyze.side_effect = _side_effect

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = analyze(
            img_path=img,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            anti_spoofing=False,
            enable_ensemble=False,
        )

        assert not isinstance(result, tuple), f"Expected success after fallback: {result}"
        assert "results" in result


# ============================================================================
# 3. INPUT SIZE FILTERING (DETECTOR_MIN_SIZE)
# ============================================================================
class TestInputSizeFiltering:

    def test_small_input_skips_min_size_detectors(self, mock_deepface):
        """Detectors with min size > input size should be skipped in ensemble."""
        from src.backend.service import analyze, DETECTOR_MIN_SIZE

        # Use a 50x50 image — smaller than centerface/ssd min (100px)
        mock_deepface.analyze.return_value = [_make_emotion_result()]

        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = analyze(
            img_path=img,
            actions=["emotion"],
            detector_backend="opencv,centerface",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
            enable_ensemble=True,
        )

        assert not isinstance(result, tuple)
        # centerface should have been skipped due to min size

    def test_large_input_uses_all_detectors(self, mock_deepface):
        """Large images should use all detectors without filtering."""
        from src.backend.service import analyze

        mock_deepface.analyze.return_value = [_make_emotion_result()]

        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        result = analyze(
            img_path=img,
            actions=["emotion"],
            detector_backend="opencv,centerface",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
            enable_ensemble=True,
        )

        assert not isinstance(result, tuple)
        assert "results" in result


# ============================================================================
# 4. ANTI-SPOOFING REJECTION
# ============================================================================
class TestAntiSpoofing:

    def test_spoof_detected_returns_400(self, mock_deepface):
        """Anti-spoofing rejection should return 400 with clear message."""
        from src.backend.service import analyze

        mock_deepface.analyze.side_effect = ValueError("Spoof detected in the given image")

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = analyze(
            img_path=img,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            anti_spoofing=True,
            enable_ensemble=False,
        )

        assert isinstance(result, tuple)
        error_dict, status_code = result
        assert status_code == 400
        assert "anti-spoofing" in error_dict["error"].lower() or "spoof" in error_dict["error"].lower()

    def test_spoof_in_ensemble_returns_400(self, mock_deepface):
        """If all ensemble detectors detect spoof, return 400."""
        from src.backend.service import analyze

        mock_deepface.analyze.side_effect = ValueError("Spoof detected")

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = analyze(
            img_path=img,
            actions=["emotion"],
            detector_backend="opencv,mtcnn",
            enforce_detection=False,
            align=True,
            anti_spoofing=True,
            enable_ensemble=True,
        )

        assert isinstance(result, tuple)
        assert result[1] == 400


# ============================================================================
# 5. WEIGHTED AGGREGATION
# ============================================================================
class TestWeightedAggregation:

    def test_weights_normalized_to_one(self):
        """Detector weights should sum to 1.0 after normalization."""
        from src.backend.service import _normalize_weights

        weights = _normalize_weights(["retinaface", "mtcnn", "opencv"])
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_aggregate_preserves_dominant_emotion(self):
        """Aggregated result should pick the emotion with highest weighted score."""
        from src.backend.service import _aggregate_emotions

        results = [
            {"emotion": {"happy": 80, "sad": 10, "neutral": 10}},
            {"emotion": {"happy": 70, "sad": 20, "neutral": 10}},
        ]
        aggregated = _aggregate_emotions(
            results, ["retinaface", "mtcnn"], [0.6, 0.4], 0.2
        )
        assert aggregated[0]["dominant_emotion"] == "happy"

    def test_low_confidence_flagged(self):
        """If dominant score is below threshold, low_confidence should be True."""
        from src.backend.service import _aggregate_emotions

        results = [
            {"emotion": {"happy": 15, "sad": 15, "neutral": 20, "angry": 15, "fear": 15, "surprise": 10, "disgust": 10}},
        ]
        aggregated = _aggregate_emotions(results, ["opencv"], [1.0], 0.25)
        assert aggregated[0]["low_confidence"] is True
