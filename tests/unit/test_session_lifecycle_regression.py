"""
Regression + real-world scenario tests for session lifecycle.

Covers:
  1. Session isolation (no cross-contamination between concurrent sessions)
  2. Temporal analyzer lifecycle (create → frames → summary → cleanup → no leak)
  3. Pause → stop sequence with temporal data
  4. Edge cases: 0 frames, 1 frame, minimal data
  5. Report generation with minimal/empty data
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from src.backend import session_manager
from src.backend.temporal_analysis import TemporalAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_EMOTION_FRAMES = [
    {"happy": 80, "sad": 5, "neutral": 10, "angry": 2, "fear": 1, "surprise": 1, "disgust": 1},
    {"happy": 20, "sad": 5, "neutral": 60, "angry": 5, "fear": 3, "surprise": 5, "disgust": 2},
    {"happy": 70, "sad": 3, "neutral": 15, "angry": 2, "fear": 5, "surprise": 3, "disgust": 2},
    {"happy": 10, "sad": 50, "neutral": 20, "angry": 5, "fear": 8, "surprise": 5, "disgust": 2},
    {"happy": 60, "sad": 5, "neutral": 20, "angry": 3, "fear": 5, "surprise": 5, "disgust": 2},
    {"happy": 30, "sad": 10, "neutral": 40, "angry": 5, "fear": 5, "surprise": 5, "disgust": 5},
]


def _seed_session(sid: str, frame_count: int = 5):
    """Set up a temporal analyzer with N frames for testing."""
    analyzer = TemporalAnalyzer(alpha=0.3, transition_threshold=0.15)
    for i in range(min(frame_count, len(_EMOTION_FRAMES))):
        analyzer.add_frame(_EMOTION_FRAMES[i])
    session_manager._temporal_analyzers[sid] = analyzer
    session_manager._session_start_times[sid] = time.time()
    session_manager._last_upload_times[sid] = 0
    return analyzer


def _cleanup(sid: str):
    session_manager._cleanup_session_cache(sid)


# ============================================================================
# 1. SESSION ISOLATION — NO CROSS-CONTAMINATION
# ============================================================================
class TestSessionIsolation:
    """Two concurrent sessions must not share state."""

    def test_vision_cache_isolated(self, mock_supabase):
        """Vision cache for session A must not bleed into session B."""
        sid_a, sid_b = "iso-a-001", "iso-b-001"
        try:
            session_manager._vision_cache.pop(sid_a, None)
            session_manager._vision_cache.pop(sid_b, None)

            session_manager._cache_vision_sample(sid_a, {"emotion": {"happy": 90}})
            session_manager._cache_vision_sample(sid_b, {"emotion": {"sad": 80}})

            assert len(session_manager._vision_cache[sid_a]) == 1
            assert len(session_manager._vision_cache[sid_b]) == 1
            assert session_manager._vision_cache[sid_a][0]["emotion"]["happy"] == 90
            assert session_manager._vision_cache[sid_b][0]["emotion"]["sad"] == 80
        finally:
            session_manager._vision_cache.pop(sid_a, None)
            session_manager._vision_cache.pop(sid_b, None)

    def test_temporal_analyzers_isolated(self):
        """Each session must have its own TemporalAnalyzer instance."""
        sid_a, sid_b = "iso-ta-001", "iso-tb-001"
        try:
            _seed_session(sid_a, frame_count=3)
            _seed_session(sid_b, frame_count=5)

            a = session_manager._temporal_analyzers[sid_a]
            b = session_manager._temporal_analyzers[sid_b]
            assert a is not b
            assert a._frame_count == 3
            assert b._frame_count == 5
        finally:
            _cleanup(sid_a)
            _cleanup(sid_b)

    def test_stop_one_session_preserves_other(self, mock_supabase, mock_genai):
        """Stopping session A must not affect session B's in-memory state."""
        sid_a, sid_b = "iso-stop-a", "iso-stop-b"
        try:
            _seed_session(sid_a, frame_count=3)
            _seed_session(sid_b, frame_count=5)

            session_manager.stop_session(sid_a)

            # A should be cleaned up
            assert sid_a not in session_manager._temporal_analyzers
            # B should be untouched
            assert sid_b in session_manager._temporal_analyzers
            assert session_manager._temporal_analyzers[sid_b]._frame_count == 5
        finally:
            _cleanup(sid_a)
            _cleanup(sid_b)


# ============================================================================
# 2. TEMPORAL ANALYZER LIFECYCLE
# ============================================================================
class TestTemporalAnalyzerLifecycle:
    """Full lifecycle: create → add frames → get summary → cleanup."""

    def test_summary_structure_complete(self):
        """get_session_summary must return all required fields."""
        analyzer = TemporalAnalyzer(alpha=0.3, transition_threshold=0.15)
        for frame in _EMOTION_FRAMES[:6]:
            analyzer.add_frame(frame)

        summary = analyzer.get_session_summary()
        required_keys = {
            "frame_count", "smoothed_timeline", "transitions",
            "transition_count", "durations", "trends",
            "volatility", "stability_score", "ema_alpha",
        }
        assert required_keys.issubset(set(summary.keys())), (
            f"Missing keys: {required_keys - set(summary.keys())}"
        )
        assert summary["frame_count"] == 6
        assert 0 <= summary["stability_score"] <= 1
        assert isinstance(summary["transitions"], list)
        assert isinstance(summary["durations"], list)
        assert isinstance(summary["volatility"], dict)

    def test_summary_with_one_frame(self):
        """Single frame should produce a valid (minimal) summary."""
        analyzer = TemporalAnalyzer()
        analyzer.add_frame(_EMOTION_FRAMES[0])

        summary = analyzer.get_session_summary()
        assert summary["frame_count"] == 1
        assert summary["transition_count"] == 0
        assert len(summary["transitions"]) == 0
        # Trends need ≥5 frames → empty
        assert len(summary["trends"]) == 0

    def test_summary_with_two_frames(self):
        """Two frames can detect a transition but not trends."""
        analyzer = TemporalAnalyzer(alpha=0.3, transition_threshold=0.10)
        # happy-dominant → neutral-dominant → should trigger transition
        analyzer.add_frame({"happy": 80, "sad": 5, "neutral": 10, "angry": 2, "fear": 1, "surprise": 1, "disgust": 1})
        analyzer.add_frame({"happy": 10, "sad": 5, "neutral": 70, "angry": 5, "fear": 3, "surprise": 5, "disgust": 2})

        summary = analyzer.get_session_summary()
        assert summary["frame_count"] == 2
        assert len(summary["trends"]) == 0  # <5 frames

    def test_cleanup_removes_all_state(self):
        """_cleanup_session_cache must remove analyzer, cache, timers."""
        sid = "lifecycle-cleanup-001"
        _seed_session(sid, frame_count=3)
        session_manager._vision_cache[sid] = [{"x": 1}]

        session_manager._cleanup_session_cache(sid)

        assert sid not in session_manager._temporal_analyzers
        assert sid not in session_manager._vision_cache
        assert sid not in session_manager._last_upload_times
        assert sid not in session_manager._session_start_times

    def test_no_memory_leak_after_cleanup(self):
        """After cleanup, analyzer reference should be garbage-collectable."""
        import weakref
        sid = "lifecycle-leak-001"
        analyzer = _seed_session(sid, frame_count=3)
        weak_ref = weakref.ref(analyzer)
        del analyzer  # drop local reference

        _cleanup(sid)
        import gc; gc.collect()
        # The weak reference should be dead now
        assert weak_ref() is None


# ============================================================================
# 3. PAUSE → STOP SEQUENCE WITH TEMPORAL DATA
# ============================================================================
class TestPauseStopSequence:
    """Realistic user flow: analyze → pause → more analyze → stop."""

    def test_pause_then_stop_both_persist_temporal(self, mock_supabase, mock_genai):
        """Both pause and stop should include temporal_summary in DB update."""
        sid = "pause-stop-001"
        _seed_session(sid, frame_count=5)

        # Mock vision data for fast_report
        vision_data = [
            {"dominant_emotion": "happy", "emotions": {"happy": 80}},
            {"dominant_emotion": "neutral", "emotions": {"neutral": 60}},
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=vision_data)

        # Step 1: Pause
        pause_result = session_manager.pause_session(sid)
        assert isinstance(pause_result, dict)

        # Step 2: Re-seed analyzer (simulating resume + more frames)
        _seed_session(sid, frame_count=6)

        # Step 3: Stop
        stop_result = session_manager.stop_session(sid)
        assert isinstance(stop_result, dict)

        # Both should have called update with temporal_summary
        update_calls = mock_supabase.table.return_value.update.call_args_list
        temporal_payloads = [
            c[0][0] for c in update_calls
            if c[0] and isinstance(c[0][0], dict) and "temporal_summary" in c[0][0]
        ]
        assert len(temporal_payloads) >= 2, (
            f"Expected ≥2 temporal_summary updates (pause+stop), got {len(temporal_payloads)}"
        )

    def test_stop_without_prior_pause(self, mock_supabase, mock_genai):
        """Direct stop (no pause) should still persist temporal_summary."""
        sid = "direct-stop-001"
        _seed_session(sid, frame_count=5)

        result = session_manager.stop_session(sid)
        assert "status" in result

        update_calls = mock_supabase.table.return_value.update.call_args_list
        found = any(
            "temporal_summary" in c[0][0]
            for c in update_calls
            if c[0] and isinstance(c[0][0], dict)
        )
        assert found, "stop_session should persist temporal_summary"


# ============================================================================
# 4. EDGE CASES: EMPTY / MINIMAL DATA
# ============================================================================
class TestEdgeCases:

    def test_stop_with_zero_frames_temporal_is_none(self, mock_supabase, mock_genai):
        """Session with 0 analysis frames → temporal_summary should be None."""
        sid = "zero-frames-001"
        session_manager._temporal_analyzers[sid] = TemporalAnalyzer()
        session_manager._session_start_times[sid] = time.time()

        session_manager.stop_session(sid)

        update_calls = mock_supabase.table.return_value.update.call_args_list
        for c in update_calls:
            payload = c[0][0] if c[0] else {}
            if isinstance(payload, dict) and "temporal_summary" in payload:
                assert payload["temporal_summary"] is None
                break

    def test_log_data_feeds_temporal_analyzer(self, mock_supabase):
        """Each log_data call should increment the temporal analyzer frame count."""
        sid = "logdata-feed-001"
        try:
            session_manager._temporal_analyzers[sid] = TemporalAnalyzer()
            session_manager._session_start_times[sid] = time.time()

            analysis = {
                "results": [{
                    "emotion": {"happy": 65, "sad": 5, "neutral": 20, "angry": 3, "fear": 2, "surprise": 3, "disgust": 2},
                    "dominant_emotion": "happy",
                }]
            }

            for _ in range(3):
                session_manager.log_data(sid, analysis)

            assert session_manager._temporal_analyzers[sid]._frame_count == 3
        finally:
            _cleanup(sid)

    def test_pause_with_no_frames_does_not_crash(self, mock_supabase, mock_genai):
        """Pausing immediately after start (0 frames) should not crash."""
        sid = "pause-empty-001"
        session_manager._temporal_analyzers[sid] = TemporalAnalyzer()
        session_manager._session_start_times[sid] = time.time()

        # Mock empty vision data
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])

        result = session_manager.pause_session(sid)
        assert isinstance(result, dict)
        # Should not raise, may have error about no data
        _cleanup(sid)

    def test_concurrent_log_data_different_sessions(self, mock_supabase):
        """log_data for session A and B should not mix up temporal analyzers."""
        sid_a, sid_b = "conc-a", "conc-b"
        try:
            session_manager._temporal_analyzers[sid_a] = TemporalAnalyzer()
            session_manager._temporal_analyzers[sid_b] = TemporalAnalyzer()

            happy_frame = {
                "results": [{
                    "emotion": {"happy": 90, "sad": 2, "neutral": 5, "angry": 1, "fear": 1, "surprise": 1, "disgust": 0},
                    "dominant_emotion": "happy",
                }]
            }
            sad_frame = {
                "results": [{
                    "emotion": {"happy": 5, "sad": 80, "neutral": 10, "angry": 2, "fear": 1, "surprise": 1, "disgust": 1},
                    "dominant_emotion": "sad",
                }]
            }

            session_manager.log_data(sid_a, happy_frame)
            session_manager.log_data(sid_b, sad_frame)

            assert session_manager._temporal_analyzers[sid_a]._frame_count == 1
            assert session_manager._temporal_analyzers[sid_b]._frame_count == 1
        finally:
            _cleanup(sid_a)
            _cleanup(sid_b)
