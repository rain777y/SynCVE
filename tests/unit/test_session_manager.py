"""
Tests for the session management layer (src.backend.session_manager).

All Supabase and Gemini interactions are mocked. No network or database
required.
"""

import copy
import time
from unittest.mock import MagicMock, patch

import pytest

from src.backend import session_manager


# ---------------------------------------------------------------------------
# start_session
# ---------------------------------------------------------------------------

class TestStartSession:
    def test_start_session_creates_record(self, mock_supabase):
        result = session_manager.start_session(user_id="user-1", metadata={"test": True})
        assert result["session_id"] == "mock-session-id"
        assert result["status"] == "active"

    def test_start_session_without_user_id(self, mock_supabase):
        result = session_manager.start_session()
        assert "session_id" in result

    def test_start_session_no_supabase(self, monkeypatch):
        monkeypatch.setattr("src.backend.storage._supabase_client", None)
        monkeypatch.setattr("src.backend.storage._supabase_initialized", True)
        result = session_manager.start_session(user_id="u1")
        assert "error" in result
        assert "not configured" in result["error"].lower() or "Database" in result["error"]

    def test_start_session_registers_upload_throttle(self, mock_supabase):
        """After starting a session, the upload throttle timer should be initialized."""
        result = session_manager.start_session(user_id="user-1")
        sid = result["session_id"]
        assert sid in session_manager._last_upload_times


# ---------------------------------------------------------------------------
# stop_session
# ---------------------------------------------------------------------------

class TestStopSession:
    def test_stop_session_generates_report(self, mock_supabase, mock_genai):
        """stop_session should call generate_report and return the result."""
        # Seed the mock to return logs
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[{"dominant_emotion": "happy", "emotions": {"happy": 80}, "created_at": "2025-01-01T00:00:00"}]
        )
        result = session_manager.stop_session("test-sess-stop")
        assert result["status"] == "session_ended"
        assert "report" in result

    def test_stop_session_cleans_up_upload_timer(self, mock_supabase, mock_genai):
        # Pre-set the timer
        session_manager._last_upload_times["timer-sess"] = time.time()
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        session_manager.stop_session("timer-sess")
        assert "timer-sess" not in session_manager._last_upload_times

    def test_stop_session_no_supabase(self, monkeypatch):
        monkeypatch.setattr("src.backend.storage._supabase_client", None)
        monkeypatch.setattr("src.backend.storage._supabase_initialized", True)
        result = session_manager.stop_session("some-id")
        assert "error" in result


# ---------------------------------------------------------------------------
# pause_session
# ---------------------------------------------------------------------------

class TestPauseSession:
    def test_pause_session_no_session_id(self, mock_supabase):
        result = session_manager.pause_session("")
        assert "error" in result

    def test_pause_session_no_supabase(self, monkeypatch):
        monkeypatch.setattr("src.backend.storage._supabase_client", None)
        monkeypatch.setattr("src.backend.storage._supabase_initialized", True)
        result = session_manager.pause_session("some-id")
        assert "error" in result


# ---------------------------------------------------------------------------
# log_data
# ---------------------------------------------------------------------------

class TestLogData:
    def test_log_data_basic(self, mock_supabase, sample_analysis_result, sample_session_id):
        result = session_manager.log_data(sample_session_id, sample_analysis_result)
        assert result["status"] in ("logged", "cached_only")

    def test_log_data_without_session_id(self):
        result = session_manager.log_data("", {"results": []})
        assert "error" in result

    def test_log_data_with_error_result_skips_logging(self, mock_supabase):
        result = session_manager.log_data("s-1", {"error": "face not found"})
        assert "error" in result

    def test_log_data_caches_vision_sample(self, mock_supabase, sample_analysis_result, sample_session_id):
        # Clear any prior cache
        session_manager._vision_cache.pop(sample_session_id, None)

        session_manager.log_data(sample_session_id, sample_analysis_result)
        assert sample_session_id in session_manager._vision_cache
        assert len(session_manager._vision_cache[sample_session_id]) >= 1


# ---------------------------------------------------------------------------
# _cache_vision_sample
# ---------------------------------------------------------------------------

class TestCacheVisionSample:
    def test_basic_caching(self):
        sid = "cache-test-1"
        session_manager._vision_cache.pop(sid, None)
        session_manager._cache_vision_sample(sid, {"emotion": {"happy": 80}})
        assert len(session_manager._vision_cache[sid]) == 1

    def test_cache_limit_enforcement(self):
        """The cache should never exceed _VISION_CACHE_LIMIT entries."""
        sid = "cache-limit-test"
        session_manager._vision_cache.pop(sid, None)

        for i in range(session_manager._VISION_CACHE_LIMIT + 20):
            session_manager._cache_vision_sample(sid, {"idx": i})

        assert len(session_manager._vision_cache[sid]) == session_manager._VISION_CACHE_LIMIT
        # The oldest entries should have been evicted (FIFO)
        assert session_manager._vision_cache[sid][0]["idx"] == 20

    def test_cache_120_samples_max(self):
        """Explicit check that 120 is the current limit."""
        assert session_manager._VISION_CACHE_LIMIT == 120


# ---------------------------------------------------------------------------
# aggregate_emotion_metrics
# ---------------------------------------------------------------------------

class TestAggregateEmotionMetrics:
    def test_basic_aggregation(self):
        data = [
            {"emotions": {"happy": 80, "sad": 10, "angry": 10}},
            {"emotions": {"happy": 60, "sad": 30, "angry": 10}},
        ]
        metrics = session_manager.aggregate_emotion_metrics(data)
        assert metrics["samples"] == 2
        assert metrics["dominant"] == "happy"
        assert "averages" in metrics
        assert "peaks" in metrics

    def test_deepface_percentage_scores_normalized(self):
        """Scores > 1.5 should be treated as 0-100 and divided by 100."""
        data = [{"emotions": {"happy": 90, "sad": 5, "angry": 5}}]
        metrics = session_manager.aggregate_emotion_metrics(data)
        # 90/100 = 0.9
        assert metrics["averages"]["happy"] == pytest.approx(0.9, abs=0.01)

    def test_scores_already_normalized(self):
        """Scores <= 1.5 should be kept as-is."""
        data = [{"emotions": {"happy": 0.9, "sad": 0.05, "angry": 0.05}}]
        metrics = session_manager.aggregate_emotion_metrics(data)
        assert metrics["averages"]["happy"] == pytest.approx(0.9, abs=0.01)

    def test_noise_floor_filtering(self):
        """Emotions below the noise floor should be in filtered_out."""
        # Values > 1.5 are treated as 0-100 scale, so 2 -> 0.02, below noise_floor 0.05
        data = [
            {"emotions": {"happy": 80, "sad": 2, "angry": 2, "neutral": 2}},
        ]
        metrics = session_manager.aggregate_emotion_metrics(data, noise_floor=0.05)
        assert "sad" in metrics["filtered_out"] or metrics["averages"].get("sad", 0) < 0.05

    def test_noise_floor_keeps_at_least_one(self):
        """Even if all emotions are below the noise floor, the dominant one should survive."""
        # All scores > 1.5 so normalised to 0-100 scale: 3->0.03, 2->0.02, etc.
        # With noise_floor=0.99, all are filtered, but dominant (happy) should survive.
        data = [{"emotions": {"happy": 5, "sad": 3, "angry": 2}}]
        metrics = session_manager.aggregate_emotion_metrics(data, noise_floor=0.99)
        assert len(metrics["averages"]) >= 1
        assert "happy" in metrics["averages"]

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="No vision data"):
            session_manager.aggregate_emotion_metrics([])

    def test_unparseable_data_raises(self):
        """Data without any emotion keys should raise."""
        with pytest.raises(ValueError, match="Unable to parse"):
            session_manager.aggregate_emotion_metrics([{"foo": "bar"}])

    def test_peak_tracking(self):
        data = [
            {"emotions": {"happy": 50, "sad": 80}},
            {"emotions": {"happy": 90, "sad": 10}},
        ]
        metrics = session_manager.aggregate_emotion_metrics(data)
        # Peak for happy should be 0.9 (from second sample)
        assert metrics["peaks"]["happy"] == pytest.approx(0.9, abs=0.01)
        # Overall peak_emotion should be happy (0.9 > 0.8)
        assert metrics["peak_emotion"] == "happy"
        assert metrics["peak_score"] == pytest.approx(0.9, abs=0.01)

    def test_nested_results_format(self):
        """Handles DeepFace-style nested {'results': [{'emotion': {...}}]} format."""
        data = [
            {"results": [{"emotion": {"happy": 70, "sad": 30}}]},
        ]
        metrics = session_manager.aggregate_emotion_metrics(data)
        assert metrics["samples"] == 1
        assert "happy" in metrics["averages"]


# ---------------------------------------------------------------------------
# get_recent_sessions
# ---------------------------------------------------------------------------

class TestGetRecentSessions:
    def test_returns_sessions(self, mock_supabase):
        mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[{"id": "s-1"}, {"id": "s-2"}]
        )
        result = session_manager.get_recent_sessions(limit=10)
        assert "sessions" in result
        assert len(result["sessions"]) == 2

    def test_no_supabase_returns_error(self, monkeypatch):
        monkeypatch.setattr("src.backend.storage._supabase_client", None)
        monkeypatch.setattr("src.backend.storage._supabase_initialized", True)
        result = session_manager.get_recent_sessions()
        assert "error" in result


# ---------------------------------------------------------------------------
# get_session_details
# ---------------------------------------------------------------------------

class TestGetSessionDetails:
    def test_returns_session(self, mock_supabase):
        result = session_manager.get_session_details("mock-session-id")
        assert "session" in result

    def test_session_not_found(self, mock_supabase):
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = MagicMock(
            data=None
        )
        result = session_manager.get_session_details("nonexistent")
        assert "error" in result

    def test_no_supabase_returns_error(self, monkeypatch):
        monkeypatch.setattr("src.backend.storage._supabase_client", None)
        monkeypatch.setattr("src.backend.storage._supabase_initialized", True)
        result = session_manager.get_session_details("any")
        assert "error" in result


# ---------------------------------------------------------------------------
# upload_frame_to_storage
# ---------------------------------------------------------------------------

class TestUploadFrameToStorage:
    def test_upload_success(self, mock_supabase, sample_base64_image):
        # Strip the data-URI prefix
        raw_b64 = sample_base64_image.split("base64,")[1]
        result = session_manager.upload_frame_to_storage("s-1", raw_b64)
        assert result is not None
        assert "s-1/frames/" in result

    def test_upload_with_data_uri_prefix(self, mock_supabase, sample_base64_image):
        result = session_manager.upload_frame_to_storage("s-1", sample_base64_image)
        assert result is not None

    def test_upload_no_supabase(self, monkeypatch):
        monkeypatch.setattr("src.backend.storage._supabase_client", None)
        monkeypatch.setattr("src.backend.storage._supabase_initialized", True)
        result = session_manager.upload_frame_to_storage("s-1", "abc")
        assert result is None


# ---------------------------------------------------------------------------
# _summarize_for_art_direction
# ---------------------------------------------------------------------------

class TestSummarizeForArtDirection:
    def test_basic_summary(self):
        from src.backend.emotion_analytics import summarize_for_art_direction

        metrics = {
            "averages": {"happy": 0.7, "sad": 0.2, "angry": 0.1},
            "peak_emotion": "happy",
            "peak_score": 0.9,
        }
        summary = summarize_for_art_direction(metrics)
        assert summary["dominant"] == "happy"
        assert summary["score"] == 0.7
        assert summary["secondary"] == "sad"
        assert summary["peak_emotion"] == "happy"

    def test_single_emotion(self):
        from src.backend.emotion_analytics import summarize_for_art_direction

        metrics = {
            "averages": {"neutral": 0.5},
            "peak_emotion": "neutral",
            "peak_score": 0.5,
        }
        summary = summarize_for_art_direction(metrics)
        assert summary["dominant"] == "neutral"
        assert summary["secondary"] is None
