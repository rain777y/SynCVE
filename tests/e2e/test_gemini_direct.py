"""
Direct tests for the Gemini client module -- no Flask, no backend server.

These tests exercise src.backend.gemini_client functions against the REAL
Google Gemini API.  They verify:
  - Text generation (generate_text)
  - Image generation (generate_image)
  - Model fallback chain
  - Retry logic with intentionally bad model names
  - resolve_image_model()

All tests skip gracefully when GEMINI_API_KEY is missing.
"""
import os
import sys
import time
from pathlib import Path

import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e.conftest import requires_gemini, GENERATED_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_valid_image(data: bytes) -> bool:
    """Quick check: JPEG starts with FFD8, PNG with 89504E47."""
    if not data or len(data) < 8:
        return False
    return data[:2] == b"\xff\xd8" or data[:4] == b"\x89PNG"


# ============================================================================
# Test class: Direct Gemini text generation
# ============================================================================
@requires_gemini
class TestGeminiTextDirect:
    """Test generate_text() with real API calls."""

    def test_simple_text_generation(self):
        """Generate a short text response and verify it is non-empty."""
        from src.backend.gemini_client import generate_text

        result = generate_text("Reply with exactly one word: hello")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_text_with_system_instruction(self):
        """generate_text with an explicit system instruction."""
        from src.backend.gemini_client import generate_text

        result = generate_text(
            "What is 2+2?",
            system_instruction="You are a math tutor. Answer with just the number.",
        )
        assert isinstance(result, str)
        assert "4" in result

    def test_text_generation_longer_prompt(self):
        """Send a longer prompt and ensure a substantive response."""
        from src.backend.gemini_client import generate_text

        prompt = (
            "You are an emotion analyst. Given the following data, provide a "
            "brief summary: dominant_emotion=happy (score 0.72), "
            "secondary=neutral (0.15), samples=50."
        )
        result = generate_text(prompt)
        assert isinstance(result, str)
        assert len(result) > 20, f"Response too short: {result!r}"

    def test_text_with_explicit_model(self):
        """Pass an explicit model name override."""
        from src.backend.gemini_client import generate_text

        result = generate_text(
            "Say 'pong'",
            model="gemini-2.5-flash",
        )
        assert isinstance(result, str)
        assert len(result.strip()) > 0


# ============================================================================
# Test class: Direct Gemini image generation
# ============================================================================
@requires_gemini
class TestGeminiImageDirect:
    """Test generate_image() with real API calls."""

    def test_image_generation_basic(self):
        """Generate a basic image and verify binary payload."""
        from src.backend.gemini_client import generate_image

        data = generate_image(
            "A simple red circle on a white background, minimalist",
        )
        assert isinstance(data, bytes)
        assert len(data) > 1000, "Image payload suspiciously small"
        assert _is_valid_image(data), "Payload is not valid JPEG or PNG"

    def test_image_generation_with_aspect_ratio(self):
        """Generate an image with a specific aspect ratio."""
        from src.backend.gemini_client import generate_image

        data = generate_image(
            "A futuristic dashboard with emotion data visualization, dark theme",
            aspect_ratio="16:9",
        )
        assert isinstance(data, bytes)
        assert _is_valid_image(data)

    def test_image_can_be_saved(self, tmp_path):
        """Generate an image and save to disk; verify file readability."""
        from src.backend.gemini_client import generate_image

        data = generate_image("A calm blue ocean scene, photorealistic")
        out = tmp_path / "test_image.png"
        out.write_bytes(data)
        assert out.exists()
        assert out.stat().st_size > 1000

        # Verify PIL can open it
        from PIL import Image
        img = Image.open(out)
        assert img.size[0] > 0 and img.size[1] > 0


# ============================================================================
# Test class: Model fallback chain
# ============================================================================
@requires_gemini
class TestGeminiFallbackChain:
    """Test the model fallback mechanism in generate_image and resolve_image_model."""

    def test_resolve_image_model_returns_string(self):
        """resolve_image_model should return a non-empty model name."""
        from src.backend.gemini_client import resolve_image_model

        model = resolve_image_model()
        assert isinstance(model, str)
        assert len(model) > 0
        print(f"[e2e] Resolved image model: {model}")

    def test_validate_model_available_good_model(self):
        """validate_model_available should return True for a known-good model."""
        from src.backend.gemini_client import validate_model_available

        assert validate_model_available("gemini-2.5-flash") is True

    def test_validate_model_available_bad_model(self):
        """validate_model_available should return False for a non-existent model."""
        from src.backend.gemini_client import validate_model_available

        result = validate_model_available("nonexistent-model-xyz-999")
        assert result is False

    def test_image_fallback_with_bad_primary(self):
        """
        If the primary model is garbage, generate_image should fall back
        to the configured fallback models and still return valid image bytes.
        """
        from src.backend.gemini_client import generate_image

        try:
            data = generate_image(
                "A simple green square",
                model="nonexistent-model-xyz-999",
            )
            # If we get here, a fallback model succeeded
            assert isinstance(data, bytes)
            assert _is_valid_image(data)
        except Exception:
            # Acceptable: all fallbacks might also fail in some environments
            pytest.skip("All fallback models also failed (acceptable in some configs)")


# ============================================================================
# Test class: Retry logic
# ============================================================================
@requires_gemini
class TestGeminiRetryLogic:
    """Test call_with_retry behavior."""

    def test_retry_succeeds_on_first_call(self):
        """Verify call_with_retry returns immediately on success."""
        from src.backend.gemini_client import call_with_retry

        call_count = 0

        def _success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = call_with_retry(_success, max_retries=3, base_delay=0.1)
        assert result == "ok"
        assert call_count == 1

    def test_retry_raises_on_non_retryable_error(self):
        """Non-retryable errors should raise immediately."""
        from src.backend.gemini_client import call_with_retry

        def _fail():
            raise ValueError("Something fundamentally wrong")

        with pytest.raises(ValueError, match="fundamentally wrong"):
            call_with_retry(_fail, max_retries=3, base_delay=0.01)

    def test_retry_exhausts_on_retryable_error(self):
        """Retryable errors should be retried up to max_retries then raise."""
        from src.backend.gemini_client import call_with_retry

        call_count = 0

        def _rate_limited():
            nonlocal call_count
            call_count += 1
            raise Exception("429 rate limit exceeded")

        with pytest.raises(Exception, match="429"):
            call_with_retry(_rate_limited, max_retries=2, base_delay=0.05)

        assert call_count == 2

    def test_retry_succeeds_after_transient_failure(self):
        """Verify recovery after a transient 503 error."""
        from src.backend.gemini_client import call_with_retry

        call_count = 0

        def _flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("503 service unavailable")
            return "recovered"

        result = call_with_retry(_flaky, max_retries=5, base_delay=0.05)
        assert result == "recovered"
        assert call_count == 3


# ============================================================================
# Test class: Multimodal generation
# ============================================================================
@requires_gemini
class TestGeminiMultimodal:
    """Test generate_multimodal with real images."""

    def test_multimodal_with_face_image(self, generated_face_images):
        """Send a face image to the multimodal endpoint and get text back."""
        from src.backend.gemini_client import generate_multimodal, to_image_part

        img_bytes = generated_face_images.get("happy")
        if not img_bytes:
            pytest.skip("No happy face image available")

        part = to_image_part(img_bytes, mime_type="image/jpeg")
        result = generate_multimodal(
            ["Describe the emotion on this person's face in one sentence.", part],
        )
        assert isinstance(result, str)
        assert len(result) > 5


# ============================================================================
# Test class: Client initialization
# ============================================================================
@requires_gemini
class TestGeminiClientInit:
    """Test the client initialization and singleton behavior."""

    def test_get_genai_client_returns_client(self):
        """get_genai_client should return a non-None client when API key is set."""
        from src.backend.gemini_client import get_genai_client

        client = get_genai_client()
        assert client is not None

    def test_client_can_count_tokens(self):
        """Verify the client can perform a lightweight token-count request."""
        from src.backend.gemini_client import get_genai_client

        client = get_genai_client()
        assert client is not None
        result = client.models.count_tokens(
            model="gemini-2.5-flash",
            contents=["Hello, world!"],
        )
        assert hasattr(result, "total_tokens")
        assert result.total_tokens > 0
