"""
Tests for Gemini API retry/rate-limit resilience.

Covers:
  1. call_with_retry exponential backoff logic
  2. Retryable vs non-retryable error classification
  3. Max retry exhaustion
  4. validate_model_available behavior
"""
import time
from unittest.mock import MagicMock, patch, call

import pytest


# ============================================================================
# 1. RETRY LOGIC
# ============================================================================
class TestCallWithRetry:
    """Test call_with_retry backoff and error classification."""

    def test_success_on_first_attempt(self):
        """No retries needed when call succeeds immediately."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(return_value="ok")
        result = call_with_retry(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 1

    def test_success_after_one_retry_on_429(self):
        """429 rate-limit error should be retried."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(side_effect=[
            Exception("429 RESOURCE_EXHAUSTED: rate limit exceeded"),
            "ok",
        ])
        result = call_with_retry(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 2

    def test_success_after_retry_on_503(self):
        """503 service unavailable should be retried."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(side_effect=[
            Exception("503 Service Unavailable"),
            "ok",
        ])
        result = call_with_retry(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 2

    def test_success_after_retry_on_timeout(self):
        """Timeout errors should be retried."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(side_effect=[
            Exception("Request timeout after 120s"),
            "ok",
        ])
        result = call_with_retry(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"

    def test_non_retryable_error_raises_immediately(self):
        """401 auth errors should not be retried."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(side_effect=Exception("401 Unauthorized: invalid API key"))
        with pytest.raises(Exception, match="401"):
            call_with_retry(fn, max_retries=3, base_delay=0.01)
        assert fn.call_count == 1

    def test_400_bad_request_not_retried(self):
        """Content policy / bad request errors should not be retried."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(side_effect=Exception("400 Bad Request: content policy violation"))
        with pytest.raises(Exception, match="400"):
            call_with_retry(fn, max_retries=3, base_delay=0.01)
        assert fn.call_count == 1

    def test_max_retries_exhausted_raises(self):
        """After max_retries, the last exception should be raised."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(side_effect=Exception("429 rate limit"))
        with pytest.raises(Exception, match="429"):
            call_with_retry(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 2

    def test_exponential_backoff_timing(self):
        """Delays should increase exponentially between retries.

        Uses base_delay=1.0 so the exponential doubling (1s→2s) clearly
        exceeds the random jitter (0-0.5s).
        """
        from src.backend.gemini_client import call_with_retry

        timestamps = []

        def _failing_fn():
            timestamps.append(time.monotonic())
            raise Exception("429 rate limit hit")

        with pytest.raises(Exception):
            call_with_retry(_failing_fn, max_retries=3, base_delay=1.0)

        assert len(timestamps) == 3
        # Gap between attempt 1→2 should be ~1.0-1.5s (1.0 * 2^0 + jitter)
        # Gap between attempt 2→3 should be ~2.0-2.5s (1.0 * 2^1 + jitter)
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]
        assert gap1 >= 0.8, f"First delay too short: {gap1:.3f}s"
        assert gap2 >= 1.5, f"Second delay too short: {gap2:.3f}s"
        assert gap2 > gap1, f"Backoff not increasing: gap1={gap1:.3f} gap2={gap2:.3f}"

    def test_mixed_retryable_then_non_retryable(self):
        """Retryable error followed by non-retryable should stop retrying."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(side_effect=[
            Exception("429 rate limit"),
            ValueError("Invalid input: empty prompt"),
        ])
        with pytest.raises(ValueError, match="Invalid input"):
            call_with_retry(fn, max_retries=5, base_delay=0.01)
        assert fn.call_count == 2


# ============================================================================
# 2. RETRYABLE ERROR CLASSIFICATION
# ============================================================================
class TestRetryableErrors:
    """Verify which error strings trigger retries."""

    @pytest.mark.parametrize("error_msg", [
        "429 RESOURCE_EXHAUSTED",
        "rate limit exceeded for project",
        "503 Service Unavailable",
        "500 Internal Server Error",
        "Request timeout",
        "Deadline exceeded",
        "Connection reset by peer",
        "Server overloaded, try again",
    ])
    def test_retryable_errors(self, error_msg):
        """These errors should trigger retries."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(side_effect=[Exception(error_msg), "ok"])
        result = call_with_retry(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 2

    @pytest.mark.parametrize("error_msg", [
        "401 Unauthorized",
        "403 Forbidden",
        "404 Not Found",
        "Invalid argument: prompt too long",
        "Permission denied",
    ])
    def test_non_retryable_errors(self, error_msg):
        """These errors should NOT trigger retries."""
        from src.backend.gemini_client import call_with_retry

        fn = MagicMock(side_effect=Exception(error_msg))
        with pytest.raises(Exception):
            call_with_retry(fn, max_retries=3, base_delay=0.01)
        assert fn.call_count == 1


# ============================================================================
# 3. VALIDATE MODEL AVAILABLE
# ============================================================================
class TestValidateModelAvailable:

    def test_returns_true_when_model_reachable(self, mock_genai):
        """validate_model_available should return True for reachable models."""
        from src.backend.gemini_client import validate_model_available

        assert validate_model_available("gemini-2.5-flash") is True
        mock_genai.models.count_tokens.assert_called_once()

    def test_returns_false_when_no_client(self, monkeypatch):
        """Without a client, should return False."""
        from src.backend.gemini_client import validate_model_available

        monkeypatch.setattr("src.backend.gemini_client._genai_client", None)
        monkeypatch.setattr("src.backend.gemini_client._initialized", True)
        assert validate_model_available("gemini-2.5-flash") is False

    def test_returns_false_on_empty_model_name(self, mock_genai):
        """Empty model name should return False."""
        from src.backend.gemini_client import validate_model_available

        assert validate_model_available("") is False

    def test_returns_false_on_api_error(self, mock_genai):
        """API error during validation should return False, not raise."""
        from src.backend.gemini_client import validate_model_available

        mock_genai.models.count_tokens.side_effect = Exception("Model not found")
        assert validate_model_available("nonexistent-model") is False
