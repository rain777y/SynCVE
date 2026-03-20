"""
Regression tests for base64 data-URI handling.

Root cause: DeepFace's image_utils.load_image() only accepts base64 with
`data:image/...;base64,` prefix. Raw base64 strings are treated as file
paths, raising ImgNotFound. This was the root cause of the
test_complete_session_lifecycle E2E failure.

These tests ensure the bug does not regress.
"""
import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(180, 140, 120))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(180, 140, 120))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================================
# 1. DATA-URI PREFIX HANDLING
# ============================================================================
class TestDataURIPrefixHandling:
    """Verify that base64 data-URI prefix is correctly handled."""

    def test_jpeg_data_uri_loads_successfully(self):
        """data:image/jpeg;base64,... should be decoded and returned as ndarray."""
        from deepface.commons import image_utils

        jpg_bytes = _make_jpeg_bytes()
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode()
        img, source = image_utils.load_image(data_uri)
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3  # BGR

    def test_png_data_uri_loads_successfully(self):
        """data:image/png;base64,... should also work (OpenCV decodes by content)."""
        from deepface.commons import image_utils

        png_bytes = _make_png_bytes()
        data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
        img, source = image_utils.load_image(data_uri)
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3

    def test_raw_base64_without_prefix_raises(self):
        """Raw base64 without data: prefix must NOT silently succeed as a file path."""
        from deepface.commons import image_utils

        jpg_bytes = _make_jpeg_bytes()
        raw_b64 = base64.b64encode(jpg_bytes).decode()

        # Raw base64 string is NOT a valid file path → should raise
        with pytest.raises(Exception):
            image_utils.load_image(raw_b64)

    def test_preprocess_image_with_data_uri(self):
        """service._preprocess_image should handle data-URI base64 strings."""
        from src.backend.service import _preprocess_image

        jpg_bytes = _make_jpeg_bytes(128, 128)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode()
        result = _preprocess_image(data_uri)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3

    def test_preprocess_image_with_numpy_array(self):
        """service._preprocess_image should accept numpy arrays directly."""
        from src.backend.service import _preprocess_image

        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = _preprocess_image(img)
        assert isinstance(result, np.ndarray)


# ============================================================================
# 2. PNG-AS-JPEG FORMAT MISMATCH (actual e2e artifact scenario)
# ============================================================================
class TestImageFormatMismatch:
    """
    The e2e face images are PNG format saved with .jpg extension.
    Verify this works when loaded via base64 data-URI.
    """

    def test_png_content_with_jpeg_mime_type(self):
        """PNG bytes wrapped in data:image/jpeg;base64,... should still decode."""
        from deepface.commons import image_utils

        png_bytes = _make_png_bytes(256, 256)
        # Intentionally use jpeg MIME for PNG content
        data_uri = "data:image/jpeg;base64," + base64.b64encode(png_bytes).decode()
        img, _ = image_utils.load_image(data_uri)
        assert isinstance(img, np.ndarray)
        # cv2.imdecode detects format by content, not MIME type
        assert img.shape[0] == 256
        assert img.shape[1] == 256

    def test_large_image_loads_without_truncation(self):
        """1024x1024 images (like e2e face images) should load fully."""
        from deepface.commons import image_utils

        large_bytes = _make_jpeg_bytes(1024, 1024)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(large_bytes).decode()
        img, _ = image_utils.load_image(data_uri)
        assert img.shape[0] == 1024
        assert img.shape[1] == 1024


# ============================================================================
# 3. SERVICE.ANALYZE WITH FULL PIPELINE (mocked DeepFace)
# ============================================================================
class TestAnalyzeDataURIPipeline:
    """Verify service.analyze() works end-to-end with data-URI base64."""

    def test_analyze_with_data_uri_returns_results(self, mock_deepface):
        """Full pipeline: data-URI → preprocess → DeepFace → results dict."""
        from src.backend.service import analyze

        jpg_bytes = _make_jpeg_bytes(256, 256)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode()

        result = analyze(
            img_path=data_uri,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
        )
        # Should NOT be an error tuple
        assert not isinstance(result, tuple), f"Got error: {result}"
        assert "results" in result
        assert len(result["results"]) >= 1
        assert "emotion" in result["results"][0]
        assert "dominant_emotion" in result["results"][0]

    def test_analyze_with_raw_base64_returns_error(self, mock_deepface):
        """Raw base64 (no prefix) should return an error, not crash."""
        from src.backend.service import analyze

        jpg_bytes = _make_jpeg_bytes()
        raw_b64 = base64.b64encode(jpg_bytes).decode()

        result = analyze(
            img_path=raw_b64,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
        )
        # Should return error tuple (dict, status_code)
        assert isinstance(result, tuple), "Raw base64 should trigger error path"
        assert "error" in result[0]


# ============================================================================
# 4. ROUTES: EXTRACT_IMAGE_FROM_REQUEST DATA-URI VALIDATION
# ============================================================================
class TestRouteImageExtraction:
    """Verify the /analyze endpoint rejects images without data-URI prefix."""

    def test_analyze_route_with_data_uri_succeeds(self, client, mock_supabase, mock_deepface):
        """POST /analyze with proper data-URI should return 200."""
        jpg_bytes = _make_jpeg_bytes()
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode()

        resp = client.post("/analyze", json={
            "img": data_uri,
            "actions": ["emotion"],
            "anti_spoofing": False,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data

    def test_analyze_route_missing_img_returns_400(self, client, mock_deepface):
        """POST /analyze without img field should return 400."""
        resp = client.post("/analyze", json={"actions": ["emotion"]})
        assert resp.status_code == 400
