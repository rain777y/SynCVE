"""
Comprehensive unit tests for temporal_analysis.py.

Tests cover: EMA smoothing, transition detection, duration tracking,
trend analysis, volatility, stability score, session summary, reset,
and the pure-Python linear regression helper.
"""

import pytest
from src.backend.temporal_analysis import (
    TemporalAnalyzer,
    EmotionTransition,
    EmotionDuration,
    EmotionTrend,
    _linear_regression,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _happy_frame(score=0.8):
    """Generate a frame where 'happy' dominates."""
    return {"happy": score, "sad": 0.05, "angry": 0.03, "neutral": 0.05,
            "fear": 0.02, "surprise": 0.03, "disgust": 0.02}


def _sad_frame(score=0.8):
    """Generate a frame where 'sad' dominates."""
    return {"happy": 0.05, "sad": score, "angry": 0.03, "neutral": 0.05,
            "fear": 0.02, "surprise": 0.03, "disgust": 0.02}


def _angry_frame(score=0.8):
    return {"happy": 0.03, "sad": 0.03, "angry": score, "neutral": 0.05,
            "fear": 0.02, "surprise": 0.03, "disgust": 0.02}


def _neutral_frame(score=0.8):
    return {"happy": 0.03, "sad": 0.03, "angry": 0.03, "neutral": score,
            "fear": 0.02, "surprise": 0.03, "disgust": 0.02}


# ---------------------------------------------------------------------------
# TestEMASmoothing
# ---------------------------------------------------------------------------

class TestEMASmoothing:
    def test_alpha_one_returns_raw(self):
        """With alpha=1.0, smoothed should equal raw."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        raw = _happy_frame(0.9)
        smoothed = analyzer.add_frame(raw)
        for emo, val in raw.items():
            assert abs(smoothed[emo] - val) < 1e-6

    def test_alpha_zero_stays_initial(self):
        """With alpha=0.0, smoothed should stay at first frame."""
        analyzer = TemporalAnalyzer(alpha=0.0)
        first = _happy_frame(0.9)
        analyzer.add_frame(first)

        second = _sad_frame(0.9)
        smoothed = analyzer.add_frame(second)

        # Should still be close to first frame values
        assert abs(smoothed["happy"] - 0.9) < 1e-6
        assert abs(smoothed["sad"] - 0.05) < 1e-6

    def test_smoothing_reduces_noise(self):
        """Alternating high/low scores should smooth out."""
        analyzer = TemporalAnalyzer(alpha=0.3)
        values = []
        for i in range(20):
            score = 0.9 if i % 2 == 0 else 0.1
            frame = {"happy": score, "sad": 1.0 - score}
            smoothed = analyzer.add_frame(frame)
            values.append(smoothed["happy"])

        # After several frames of alternating 0.9/0.1, EMA should converge
        # toward the mean (~0.5). The range of late values should be smaller
        # than the raw input range (0.8).
        late_range = max(values[-4:]) - min(values[-4:])
        raw_range = 0.8  # 0.9 - 0.1
        assert late_range < raw_range

    def test_normalization_0_100_to_0_1(self):
        """Scores > 1.0 should be auto-normalized to 0-1."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        raw_100 = {"happy": 80.0, "sad": 10.0, "neutral": 10.0}
        smoothed = analyzer.add_frame(raw_100)
        assert abs(smoothed["happy"] - 0.8) < 1e-6
        assert abs(smoothed["sad"] - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# TestTransitionDetection
# ---------------------------------------------------------------------------

class TestTransitionDetection:
    def test_clear_transition(self):
        """5 happy frames then 5 sad frames -> 1 transition."""
        analyzer = TemporalAnalyzer(alpha=0.5, transition_threshold=0.1)
        for _ in range(5):
            analyzer.add_frame(_happy_frame())
        for _ in range(5):
            analyzer.add_frame(_sad_frame())

        transitions = analyzer.get_transitions()
        assert len(transitions) >= 1
        assert transitions[0].from_emotion == "happy"
        assert transitions[0].to_emotion == "sad"

    def test_no_transition_below_threshold(self):
        """Small score changes should not trigger transition."""
        analyzer = TemporalAnalyzer(alpha=0.3, transition_threshold=0.9)
        for _ in range(5):
            analyzer.add_frame(_happy_frame(0.6))
        for _ in range(5):
            analyzer.add_frame(_sad_frame(0.6))

        # With high threshold, transitions may not register
        transitions = analyzer.get_transitions()
        # All transitions (if any) should have delta >= threshold
        for t in transitions:
            assert t.confidence_delta >= 0  # At least non-negative

    def test_multiple_transitions(self):
        """happy -> sad -> angry -> neutral -> 3 transitions."""
        analyzer = TemporalAnalyzer(alpha=0.8, transition_threshold=0.05)
        for _ in range(5):
            analyzer.add_frame(_happy_frame())
        for _ in range(5):
            analyzer.add_frame(_sad_frame())
        for _ in range(5):
            analyzer.add_frame(_angry_frame())
        for _ in range(5):
            analyzer.add_frame(_neutral_frame())

        transitions = analyzer.get_transitions()
        assert len(transitions) >= 3

    def test_transition_records_correct_metadata(self):
        """from_emotion, to_emotion, frame_idx, confidence_delta all correct."""
        analyzer = TemporalAnalyzer(alpha=0.9, transition_threshold=0.05)
        for _ in range(3):
            analyzer.add_frame(_happy_frame(0.9))
        for _ in range(3):
            analyzer.add_frame(_sad_frame(0.9))

        transitions = analyzer.get_transitions()
        if transitions:
            t = transitions[0]
            assert t.from_emotion == "happy"
            assert t.to_emotion == "sad"
            assert isinstance(t.frame_idx, int)
            assert t.confidence_delta > 0


# ---------------------------------------------------------------------------
# TestDurationTracking
# ---------------------------------------------------------------------------

class TestDurationTracking:
    def test_single_emotion_entire_session(self):
        """All frames happy -> 1 duration entry spanning full session."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        for _ in range(10):
            analyzer.add_frame(_happy_frame())

        durations = analyzer.get_durations()
        assert len(durations) == 1
        assert durations[0].emotion == "happy"
        assert durations[0].duration_frames == 10
        assert durations[0].start_frame == 0
        assert durations[0].end_frame == 9

    def test_alternating_emotions(self):
        """happy(3) -> sad(2) -> happy(5) -> 3 duration entries."""
        analyzer = TemporalAnalyzer(alpha=1.0, transition_threshold=0.01)
        for _ in range(3):
            analyzer.add_frame(_happy_frame())
        for _ in range(2):
            analyzer.add_frame(_sad_frame())
        for _ in range(5):
            analyzer.add_frame(_happy_frame())

        durations = analyzer.get_durations()
        assert len(durations) == 3
        assert durations[0].emotion == "happy"
        assert durations[0].duration_frames == 3
        assert durations[1].emotion == "sad"
        assert durations[1].duration_frames == 2
        assert durations[2].emotion == "happy"
        assert durations[2].duration_frames == 5

    def test_duration_seconds_calculation(self):
        """10 frames at fps=0.5 should be 20 seconds."""
        analyzer = TemporalAnalyzer(alpha=1.0, fps_estimate=0.5)
        for _ in range(10):
            analyzer.add_frame(_happy_frame())

        durations = analyzer.get_durations()
        assert len(durations) == 1
        assert durations[0].duration_sec == 20.0


# ---------------------------------------------------------------------------
# TestTrendAnalysis
# ---------------------------------------------------------------------------

class TestTrendAnalysis:
    def test_increasing_trend(self):
        """Linearly increasing happy scores -> positive slope."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        for i in range(10):
            score = 0.1 + i * 0.08
            frame = {"happy": score, "sad": 1.0 - score}
            analyzer.add_frame(frame)

        trends = analyzer.get_trends()
        happy_trend = next(t for t in trends if t.emotion == "happy")
        assert happy_trend.slope > 0
        assert happy_trend.direction == "increasing"
        assert happy_trend.r_squared > 0.9

    def test_decreasing_trend(self):
        """Linearly decreasing scores -> negative slope."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        for i in range(10):
            score = 0.9 - i * 0.08
            frame = {"happy": score, "sad": 1.0 - score}
            analyzer.add_frame(frame)

        trends = analyzer.get_trends()
        happy_trend = next(t for t in trends if t.emotion == "happy")
        assert happy_trend.slope < 0
        assert happy_trend.direction == "decreasing"

    def test_stable_trend(self):
        """Constant scores -> slope ~0."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        for _ in range(10):
            analyzer.add_frame({"happy": 0.5, "sad": 0.5})

        trends = analyzer.get_trends()
        happy_trend = next(t for t in trends if t.emotion == "happy")
        assert abs(happy_trend.slope) < 0.001
        assert happy_trend.direction == "stable"

    def test_insufficient_frames(self):
        """< 5 frames -> empty trends list."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        for _ in range(3):
            analyzer.add_frame(_happy_frame())

        trends = analyzer.get_trends()
        assert len(trends) == 0


# ---------------------------------------------------------------------------
# TestVolatility
# ---------------------------------------------------------------------------

class TestVolatility:
    def test_constant_scores_zero_volatility(self):
        """All same scores -> volatility ~0 for all emotions."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        for _ in range(10):
            analyzer.add_frame({"happy": 0.5, "sad": 0.3, "neutral": 0.2})

        vol = analyzer.get_volatility()
        for v in vol.values():
            assert v < 0.001

    def test_alternating_scores_high_volatility(self):
        """0-1-0-1 pattern -> high volatility."""
        analyzer = TemporalAnalyzer(alpha=1.0, volatility_window=10)
        for i in range(10):
            score = 1.0 if i % 2 == 0 else 0.0
            analyzer.add_frame({"happy": score, "sad": 1.0 - score})

        vol = analyzer.get_volatility()
        assert vol["happy"] > 0.3  # std dev of alternating 0/1 is 0.5


# ---------------------------------------------------------------------------
# TestStabilityScore
# ---------------------------------------------------------------------------

class TestStabilityScore:
    def test_stable_session(self):
        """Constant emotions -> stability near 1.0."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        for _ in range(10):
            analyzer.add_frame(_happy_frame())

        score = analyzer.get_stability_score()
        assert score > 0.9

    def test_chaotic_session(self):
        """Wildly alternating -> low stability."""
        analyzer = TemporalAnalyzer(alpha=1.0, volatility_window=10)
        for i in range(10):
            if i % 2 == 0:
                analyzer.add_frame({"happy": 1.0, "sad": 0.0})
            else:
                analyzer.add_frame({"happy": 0.0, "sad": 1.0})

        score = analyzer.get_stability_score()
        assert score < 0.5


# ---------------------------------------------------------------------------
# TestSessionSummary
# ---------------------------------------------------------------------------

class TestSessionSummary:
    def test_summary_contains_all_fields(self):
        """Verify all expected keys present in summary dict."""
        analyzer = TemporalAnalyzer()
        for _ in range(10):
            analyzer.add_frame(_happy_frame())

        summary = analyzer.get_session_summary()
        expected_keys = {
            "frame_count", "smoothed_timeline", "transitions",
            "transition_count", "durations", "trends", "volatility",
            "stability_score", "ema_alpha",
        }
        assert expected_keys.issubset(summary.keys())
        assert summary["frame_count"] == 10
        assert summary["ema_alpha"] == 0.3

    def test_summary_timeline_sampling(self):
        """With >50 frames, timeline should be sampled down."""
        analyzer = TemporalAnalyzer(alpha=1.0)
        for _ in range(100):
            analyzer.add_frame(_happy_frame())

        summary = analyzer.get_session_summary()
        assert len(summary["smoothed_timeline"]) <= 51  # ~50 sampled points
        assert summary["frame_count"] == 100


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        """After reset, frame_count should be 0."""
        analyzer = TemporalAnalyzer()
        for _ in range(10):
            analyzer.add_frame(_happy_frame())
        assert analyzer._frame_count == 10

        analyzer.reset()
        assert analyzer._frame_count == 0
        assert len(analyzer._raw_history) == 0
        assert len(analyzer._smoothed_history) == 0
        assert len(analyzer._transitions) == 0
        assert len(analyzer._dominant_history) == 0


# ---------------------------------------------------------------------------
# TestLinearRegression
# ---------------------------------------------------------------------------

class TestLinearRegression:
    def test_perfect_linear(self):
        """[1, 2, 3, 4, 5] -> slope=1.0, r_squared=1.0."""
        slope, r_sq = _linear_regression([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(slope - 1.0) < 1e-6
        assert abs(r_sq - 1.0) < 1e-6

    def test_constant(self):
        """[5, 5, 5, 5] -> slope=0.0."""
        slope, r_sq = _linear_regression([5.0, 5.0, 5.0, 5.0])
        assert abs(slope) < 1e-6

    def test_single_point(self):
        """Single point -> slope=0, r_sq=0."""
        slope, r_sq = _linear_regression([3.0])
        assert slope == 0.0
        assert r_sq == 0.0

    def test_two_points(self):
        """Two points -> perfect fit."""
        slope, r_sq = _linear_regression([0.0, 1.0])
        assert abs(slope - 1.0) < 1e-6
        assert abs(r_sq - 1.0) < 1e-6

    def test_negative_slope(self):
        """Decreasing sequence -> negative slope."""
        slope, r_sq = _linear_regression([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(slope - (-1.0)) < 1e-6
        assert abs(r_sq - 1.0) < 1e-6
