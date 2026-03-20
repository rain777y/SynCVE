"""
Tests for the service layer (src.backend.service).

Covers image preprocessing, ensemble detection, weighted emotion aggregation,
represent, and verify. DeepFace calls are fully mocked.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_low_res_triggers_super_resolution(self, low_res_numpy_image):
        """Images smaller than 256px on their shortest side should be upscaled."""
        from src.backend.service import _maybe_super_resolve

        result = _maybe_super_resolve(low_res_numpy_image)
        h, w = result.shape[:2]
        # Minimum dimension should now be >= 256
        assert min(h, w) >= 256

    def test_normal_res_not_upscaled(self, sample_numpy_image):
        """Images already >= 256px should not have their dimensions inflated much."""
        from src.backend.service import _maybe_super_resolve

        # Use a 300x300 image
        big_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        result = _maybe_super_resolve(big_img)
        h, w = result.shape[:2]
        # Should remain the same dimensions (sharpening only)
        assert h == 300
        assert w == 300

    def test_super_resolve_handles_none(self):
        from src.backend.service import _maybe_super_resolve

        assert _maybe_super_resolve(None) is None

    def test_preprocess_image_with_numpy_input(self, sample_numpy_image):
        """_preprocess_image should accept a numpy array and return one."""
        from src.backend.service import _preprocess_image

        result = _preprocess_image(sample_numpy_image)
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # still 3-channel


class TestNormalizeLighting:
    def test_normalize_lighting_3channel(self, sample_numpy_image):
        from src.backend.service import _normalize_lighting

        result = _normalize_lighting(sample_numpy_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_numpy_image.shape

    def test_normalize_lighting_returns_input_for_grayscale(self):
        from src.backend.service import _normalize_lighting

        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        result = _normalize_lighting(gray)
        # Should return unchanged for non-3-channel images
        assert result is gray

    def test_normalize_lighting_handles_none(self):
        from src.backend.service import _normalize_lighting

        assert _normalize_lighting(None) is None


# ---------------------------------------------------------------------------
# Detector backend parsing
# ---------------------------------------------------------------------------

class TestParseDetectorBackends:
    def test_string_input(self):
        from src.backend.service import _parse_detector_backends

        result = _parse_detector_backends("retinaface", enable_ensemble=False)
        assert result == ["retinaface"]

    def test_comma_separated_string(self):
        from src.backend.service import _parse_detector_backends

        result = _parse_detector_backends("retinaface,mtcnn", enable_ensemble=True)
        assert "retinaface" in result
        assert "mtcnn" in result

    def test_list_input(self):
        from src.backend.service import _parse_detector_backends

        result = _parse_detector_backends(["retinaface", "mtcnn"], enable_ensemble=True)
        assert "retinaface" in result
        assert "mtcnn" in result

    def test_ensemble_disabled_uses_first_detector_only(self):
        from src.backend.service import _parse_detector_backends

        result = _parse_detector_backends(["retinaface", "mtcnn"], enable_ensemble=False)
        assert len(result) == 1
        assert result[0] == "retinaface"

    def test_ensemble_enabled_merges_defaults(self):
        from src.backend.service import _parse_detector_backends

        result = _parse_detector_backends("opencv", enable_ensemble=True)
        # Should merge in the configured ENSEMBLE_DETECTORS
        assert len(result) >= 2

    def test_none_input_with_ensemble(self):
        from src.backend.service import _parse_detector_backends

        result = _parse_detector_backends(None, enable_ensemble=True)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------

class TestNormalizeWeights:
    def test_known_detectors(self):
        from src.backend.service import _normalize_weights

        weights = _normalize_weights(["retinaface", "mtcnn"])
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-6
        # retinaface and mtcnn have equal weight (data-driven from ablation)
        assert abs(weights[0] - weights[1]) < 1e-6

    def test_unknown_detectors_get_equal_weight(self):
        from src.backend.service import _normalize_weights

        weights = _normalize_weights(["foo", "bar"])
        assert len(weights) == 2
        assert abs(weights[0] - weights[1]) < 1e-6

    def test_single_detector(self):
        from src.backend.service import _normalize_weights

        weights = _normalize_weights(["retinaface"])
        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Emotion aggregation (weighted fusion)
# ---------------------------------------------------------------------------

class TestAggregateEmotions:
    def test_single_backend_result(self):
        from src.backend.service import _aggregate_emotions

        backend_results = [
            {"emotion": {"happy": 80, "sad": 10, "angry": 10}, "dominant_emotion": "happy"}
        ]
        result = _aggregate_emotions(
            backend_results, detectors=["retinaface"], weights=[1.0], confidence_threshold=0.1
        )
        assert len(result) == 1
        assert result[0]["dominant_emotion"] == "happy"
        assert result[0]["emotion"]["happy"] == 80.0

    def test_weighted_fusion_two_backends(self):
        from src.backend.service import _aggregate_emotions

        results = [
            {"emotion": {"happy": 60, "sad": 40}, "dominant_emotion": "happy"},
            {"emotion": {"happy": 40, "sad": 60}, "dominant_emotion": "sad"},
        ]
        aggregated = _aggregate_emotions(
            results,
            detectors=["retinaface", "mtcnn"],
            weights=[0.6, 0.4],
            confidence_threshold=0.1,
        )
        # happy = 60*0.6 + 40*0.4 = 36 + 16 = 52
        # sad   = 40*0.6 + 60*0.4 = 24 + 24 = 48
        assert aggregated[0]["dominant_emotion"] == "happy"
        assert abs(aggregated[0]["emotion"]["happy"] - 52.0) < 1e-6
        assert abs(aggregated[0]["emotion"]["sad"] - 48.0) < 1e-6

    def test_low_confidence_flag(self):
        from src.backend.service import _aggregate_emotions

        results = [{"emotion": {"happy": 5, "sad": 5, "neutral": 5}, "dominant_emotion": "happy"}]
        aggregated = _aggregate_emotions(
            results, detectors=["retinaface"], weights=[1.0], confidence_threshold=0.5
        )
        assert aggregated[0]["low_confidence"] is True

    def test_high_confidence_no_flag(self):
        from src.backend.service import _aggregate_emotions

        results = [{"emotion": {"happy": 90, "sad": 5, "neutral": 5}, "dominant_emotion": "happy"}]
        aggregated = _aggregate_emotions(
            results, detectors=["retinaface"], weights=[1.0], confidence_threshold=0.1
        )
        assert aggregated[0]["low_confidence"] is False

    def test_ensemble_metadata_present(self):
        from src.backend.service import _aggregate_emotions

        results = [{"emotion": {"happy": 80}, "dominant_emotion": "happy"}]
        aggregated = _aggregate_emotions(
            results, detectors=["retinaface"], weights=[1.0], confidence_threshold=0.1
        )
        assert "ensemble" in aggregated[0]
        assert "detector_backends" in aggregated[0]

    def test_empty_results_raises_value_error(self):
        from src.backend.service import _aggregate_emotions

        with pytest.raises(ValueError, match="No emotion results"):
            _aggregate_emotions([], detectors=[], weights=[], confidence_threshold=0.1)


# ---------------------------------------------------------------------------
# represent()
# ---------------------------------------------------------------------------

class TestRepresent:
    def test_represent_success(self, sample_base64_image, mock_deepface):
        from src.backend.service import represent

        result = represent(
            img_path=sample_base64_image,
            model_name="VGG-Face",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
            anti_spoofing=False,
        )
        assert "results" in result
        assert len(result["results"]) > 0

    def test_represent_exception_returns_error(self, sample_base64_image, mock_deepface):
        mock_deepface.represent.side_effect = RuntimeError("Model load failed")
        from src.backend.service import represent

        result = represent(
            img_path=sample_base64_image,
            model_name="VGG-Face",
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
            anti_spoofing=False,
        )
        # Should be a tuple (dict, status_code)
        assert isinstance(result, tuple)
        assert result[1] == 500
        assert "error" in result[0]


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------

class TestVerify:
    def test_verify_success(self, sample_base64_image, mock_deepface):
        from src.backend.service import verify

        result = verify(
            img1_path=sample_base64_image,
            img2_path=sample_base64_image,
            model_name="VGG-Face",
            detector_backend="retinaface",
            distance_metric="cosine",
            enforce_detection=True,
            align=True,
            anti_spoofing=False,
        )
        assert result["verified"] is True

    def test_verify_exception_returns_error(self, sample_base64_image, mock_deepface):
        mock_deepface.verify.side_effect = RuntimeError("Verification error")
        from src.backend.service import verify

        result = verify(
            img1_path=sample_base64_image,
            img2_path=sample_base64_image,
            model_name="VGG-Face",
            detector_backend="retinaface",
            distance_metric="cosine",
            enforce_detection=True,
            align=True,
            anti_spoofing=False,
        )
        assert isinstance(result, tuple)
        assert result[1] == 500
        assert "error" in result[0]


# ---------------------------------------------------------------------------
# analyze() (full flow with ensemble)
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_analyze_single_detector(self, sample_base64_image, mock_deepface):
        from src.backend.service import analyze

        result = analyze(
            img_path=sample_base64_image,
            actions=["emotion"],
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
            anti_spoofing=False,
            enable_ensemble=False,
        )
        assert "results" in result
        assert result["results"][0]["dominant_emotion"] == "happy"

    def test_analyze_ensemble_detection(self, sample_base64_image, mock_deepface):
        from src.backend.service import analyze

        result = analyze(
            img_path=sample_base64_image,
            actions=["emotion"],
            detector_backend="retinaface,mtcnn",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
            enable_ensemble=True,
        )
        assert "results" in result
        first = result["results"][0]
        assert "ensemble" in first
        assert "detector_backends" in first

    def test_analyze_spoof_detection_returns_400(self, sample_base64_image, mock_deepface):
        mock_deepface.analyze.side_effect = ValueError("Spoof detected in the image")
        from src.backend.service import analyze

        result = analyze(
            img_path=sample_base64_image,
            actions=["emotion"],
            detector_backend="retinaface",
            enforce_detection=True,
            align=True,
            anti_spoofing=True,
            enable_ensemble=False,
        )
        # Single detector spoof should return tuple (error_dict, 400)
        assert isinstance(result, tuple)
        assert result[1] == 400
        assert "Anti-spoofing" in result[0]["error"]


# ---------------------------------------------------------------------------
# _to_bool() helper
# ---------------------------------------------------------------------------

class TestToBool:
    def test_none_uses_default(self):
        from src.backend.service import _to_bool

        assert _to_bool(None, default=True) is True
        assert _to_bool(None, default=False) is False

    def test_bool_passthrough(self):
        from src.backend.service import _to_bool

        assert _to_bool(True) is True
        assert _to_bool(False) is False

    def test_string_false_values(self):
        from src.backend.service import _to_bool

        for val in ("0", "false", "no", "off", "False", "NO"):
            assert _to_bool(val) is False

    def test_string_true_values(self):
        from src.backend.service import _to_bool

        for val in ("1", "true", "yes", "on"):
            assert _to_bool(val) is True

    def test_numeric_values(self):
        from src.backend.service import _to_bool

        assert _to_bool(1) is True
        assert _to_bool(0) is False
