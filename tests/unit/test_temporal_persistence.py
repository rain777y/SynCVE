"""
Unit tests for temporal data persistence and smoothed score propagation.

Validates:
  1. stop_session() persists temporal_summary before cleanup
  2. pause_session() persists temporal_summary to sessions table
  3. log_data() returns smoothed_emotions when analyzer is active
  4. generate_report() includes temporal context in Gemini prompt
"""
import time
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed_temporal_analyzer(session_id: str, frame_count: int = 5):
    """Create and seed a TemporalAnalyzer with N frames of varied emotion data."""
    from src.backend.temporal_analysis import TemporalAnalyzer
    from src.backend import session_manager

    analyzer = TemporalAnalyzer(alpha=0.3, transition_threshold=0.15)

    # Alternate between happy-dominant and neutral-dominant frames
    frames = [
        {"happy": 80, "sad": 5, "neutral": 10, "angry": 2, "fear": 1, "surprise": 1, "disgust": 1},
        {"happy": 20, "sad": 5, "neutral": 60, "angry": 5, "fear": 3, "surprise": 5, "disgust": 2},
        {"happy": 70, "sad": 3, "neutral": 15, "angry": 2, "fear": 5, "surprise": 3, "disgust": 2},
        {"happy": 10, "sad": 50, "neutral": 20, "angry": 5, "fear": 8, "surprise": 5, "disgust": 2},
        {"happy": 60, "sad": 5, "neutral": 20, "angry": 3, "fear": 5, "surprise": 5, "disgust": 2},
        {"happy": 30, "sad": 10, "neutral": 40, "angry": 5, "fear": 5, "surprise": 5, "disgust": 5},
        {"happy": 75, "sad": 2, "neutral": 12, "angry": 3, "fear": 3, "surprise": 3, "disgust": 2},
        {"happy": 15, "sad": 40, "neutral": 30, "angry": 5, "fear": 5, "surprise": 3, "disgust": 2},
    ]

    for i in range(min(frame_count, len(frames))):
        analyzer.add_frame(frames[i])

    session_manager._temporal_analyzers[session_id] = analyzer
    session_manager._session_start_times[session_id] = time.time()
    return analyzer


# ============================================================================
# 1. STOP SESSION TEMPORAL PERSISTENCE
# ============================================================================
class TestStopSessionTemporalPersistence:
    """stop_session() should persist temporal_summary before cleanup."""

    def test_stop_persists_temporal_before_cleanup(self, mock_supabase, mock_genai):
        """temporal_summary should be included in the sessions update payload."""
        from src.backend import session_manager

        sid = "test-stop-temporal-001"
        _seed_temporal_analyzer(sid, frame_count=5)

        # Ensure analyzer exists before stop
        assert sid in session_manager._temporal_analyzers

        session_manager.stop_session(sid)

        # Analyzer should be cleaned up
        assert sid not in session_manager._temporal_analyzers

        # Check that the update call included temporal_summary
        update_calls = mock_supabase.table.return_value.update.call_args_list
        assert len(update_calls) > 0, "sessions.update should have been called"

        # Find the call that includes temporal_summary
        found_temporal = False
        for c in update_calls:
            payload = c[0][0] if c[0] else c[1]
            if isinstance(payload, dict) and "temporal_summary" in payload:
                ts = payload["temporal_summary"]
                assert ts is not None, "temporal_summary should not be None"
                assert "stability_score" in ts
                assert "frame_count" in ts
                assert ts["frame_count"] == 5
                found_temporal = True
                break

        assert found_temporal, (
            "No update call contained temporal_summary. "
            f"Calls: {update_calls}"
        )

    def test_stop_without_analyzer_graceful(self, mock_supabase, mock_genai):
        """stop_session without a temporal analyzer should not error."""
        from src.backend import session_manager

        sid = "test-stop-no-analyzer-001"
        # Do NOT seed analyzer
        session_manager._session_start_times[sid] = time.time()

        # Should not raise
        result = session_manager.stop_session(sid)
        assert isinstance(result, dict)

    def test_stop_with_empty_analyzer(self, mock_supabase, mock_genai):
        """Analyzer with 0 frames should result in None temporal_summary."""
        from src.backend import session_manager
        from src.backend.temporal_analysis import TemporalAnalyzer

        sid = "test-stop-empty-analyzer-001"
        session_manager._temporal_analyzers[sid] = TemporalAnalyzer()
        session_manager._session_start_times[sid] = time.time()

        session_manager.stop_session(sid)

        # temporal_summary should be None (get_temporal_summary returns None
        # when _frame_count == 0)
        update_calls = mock_supabase.table.return_value.update.call_args_list
        for c in update_calls:
            payload = c[0][0] if c[0] else c[1]
            if isinstance(payload, dict) and "temporal_summary" in payload:
                assert payload["temporal_summary"] is None
                break


# ============================================================================
# 2. PAUSE SESSION TEMPORAL PERSISTENCE
# ============================================================================
class TestPauseSessionTemporalPersistence:

    def test_pause_persists_temporal_to_sessions(self, mock_supabase, mock_genai):
        """pause_session should include temporal_summary in the update payload."""
        from src.backend import session_manager

        sid = "test-pause-temporal-001"
        _seed_temporal_analyzer(sid, frame_count=5)

        # Mock vision_samples fetch for fast_report
        vision_data = [
            {"dominant_emotion": "happy", "emotions": {"happy": 80, "sad": 5, "neutral": 10, "angry": 2, "fear": 1, "surprise": 1, "disgust": 1}},
            {"dominant_emotion": "neutral", "emotions": {"happy": 20, "sad": 5, "neutral": 60, "angry": 5, "fear": 3, "surprise": 5, "disgust": 2}},
            {"dominant_emotion": "happy", "emotions": {"happy": 70, "sad": 3, "neutral": 15, "angry": 2, "fear": 5, "surprise": 3, "disgust": 2}},
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=vision_data)

        result = session_manager.pause_session(sid)

        # Check update payload contains temporal_summary
        update_calls = mock_supabase.table.return_value.update.call_args_list
        found_temporal = False
        for c in update_calls:
            payload = c[0][0] if c[0] else c[1]
            if isinstance(payload, dict) and "temporal_summary" in payload:
                ts = payload["temporal_summary"]
                assert ts is not None
                assert "stability_score" in ts
                found_temporal = True
                break

        assert found_temporal, "pause_session should persist temporal_summary"

        # Cleanup
        session_manager._cleanup_session_cache(sid)


# ============================================================================
# 3. LOG_DATA RETURNS SMOOTHED SCORES
# ============================================================================
class TestLogDataSmoothedScores:
    """log_data() should return smoothed_emotions when analyzer exists."""

    def test_log_data_returns_smoothed_scores(self, mock_supabase):
        """When temporal analyzer is active, log_data should include smoothed_emotions."""
        from src.backend import session_manager

        sid = "test-logdata-smoothed-001"
        _seed_temporal_analyzer(sid, frame_count=3)

        analysis_result = {
            "results": [{
                "emotion": {
                    "happy": 65, "sad": 5, "neutral": 20,
                    "angry": 3, "fear": 2, "surprise": 3, "disgust": 2,
                },
                "dominant_emotion": "happy",
            }]
        }

        result = session_manager.log_data(sid, analysis_result)
        assert "smoothed_emotions" in result, (
            f"Expected smoothed_emotions in log_data result, got: {list(result.keys())}"
        )
        assert isinstance(result["smoothed_emotions"], dict)
        assert len(result["smoothed_emotions"]) >= 5

        # Cleanup
        session_manager._cleanup_session_cache(sid)

    def test_log_data_without_analyzer_no_smoothed(self, mock_supabase):
        """Without temporal analyzer, smoothed_emotions should be absent."""
        from src.backend import session_manager

        sid = "test-logdata-no-analyzer-001"
        # No analyzer seeded

        analysis_result = {
            "results": [{
                "emotion": {"happy": 65, "sad": 5, "neutral": 20, "angry": 3, "fear": 2, "surprise": 3, "disgust": 2},
                "dominant_emotion": "happy",
            }]
        }

        result = session_manager.log_data(sid, analysis_result)
        assert "smoothed_emotions" not in result or result.get("smoothed_emotions") is None


# ============================================================================
# 4. GENERATE REPORT WITH TEMPORAL CONTEXT
# ============================================================================
class TestGenerateReportWithTemporal:
    """generate_report() should include temporal data in the Gemini prompt."""

    def test_generate_report_includes_temporal_context(self, mock_supabase, mock_genai):
        """When temporal data exists, the Gemini prompt should mention it."""
        from src.backend import session_manager
        from src.backend.report_generator import generate_report

        sid = "test-report-temporal-001"
        _seed_temporal_analyzer(sid, frame_count=6)

        # Mock vision_samples for generate_report
        vision_data = [
            {"dominant_emotion": "happy", "created_at": "2026-01-01T00:00:00Z"},
            {"dominant_emotion": "neutral", "created_at": "2026-01-01T00:00:02Z"},
            {"dominant_emotion": "happy", "created_at": "2026-01-01T00:00:04Z"},
            {"dominant_emotion": "sad", "created_at": "2026-01-01T00:00:06Z"},
            {"dominant_emotion": "happy", "created_at": "2026-01-01T00:00:08Z"},
            {"dominant_emotion": "neutral", "created_at": "2026-01-01T00:00:10Z"},
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=vision_data)

        generate_report(sid)

        # Check that the Gemini prompt included temporal context
        gen_calls = mock_genai.models.generate_content.call_args_list
        assert len(gen_calls) > 0, "generate_text should have been called"

        # The prompt is passed as the first positional arg or in contents
        prompt_text = ""
        for c in gen_calls:
            args = c[0] if c[0] else []
            kwargs = c[1] if len(c) > 1 else {}
            contents = kwargs.get("contents", args[1] if len(args) > 1 else "")
            if isinstance(contents, str):
                prompt_text = contents
            elif isinstance(contents, list) and contents:
                prompt_text = str(contents[0]) if contents else ""

        assert "Stability Score" in prompt_text or "Temporal" in prompt_text, (
            f"Gemini prompt should reference temporal data. Prompt: {prompt_text[:500]}"
        )

        # Cleanup
        session_manager._cleanup_session_cache(sid)
