"""
Temporal emotion analysis engine for SynCVE.

Transforms per-frame emotion snapshots into temporal intelligence:
  - EMA smoothing (noise reduction for real-time display)
  - Emotion transition detection (when dominant emotion shifts)
  - Duration tracking (how long each emotion persists)
  - Linear trend analysis (emotion intensity direction)
  - Volatility metrics (emotional stability measurement)

Pure Python — no numpy/scipy dependency.  Each session gets its own
TemporalAnalyzer instance keyed by session_id in session_manager.

Thread safety note: each analyzer is only accessed by its own session's
requests (session_id is the dictionary key), so no locks are required.
"""

import math
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, Deque, List, Optional, Tuple

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmotionTransition:
    from_emotion: str
    to_emotion: str
    frame_idx: int
    timestamp: Optional[str]
    confidence_delta: float


@dataclass
class EmotionDuration:
    emotion: str
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_sec: float


@dataclass
class EmotionTrend:
    emotion: str
    slope: float
    r_squared: float
    direction: str  # "increasing", "decreasing", "stable"


# ---------------------------------------------------------------------------
# Pure-Python linear regression
# ---------------------------------------------------------------------------

def _linear_regression(ys: List[float]) -> Tuple[float, float]:
    """
    Pure-Python OLS: y = mx + b.
    Returns (slope, r_squared).
    """
    n = len(ys)
    if n < 2:
        return 0.0, 0.0

    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))

    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return slope, max(0.0, min(1.0, r_squared))


# ---------------------------------------------------------------------------
# TemporalAnalyzer
# ---------------------------------------------------------------------------

class TemporalAnalyzer:
    """Stateful per-session temporal emotion analyzer."""

    def __init__(
        self,
        alpha: float = 0.3,
        transition_threshold: float = 0.15,
        volatility_window: int = 10,
        fps_estimate: float = 0.5,
    ):
        self.alpha = alpha
        self.transition_threshold = transition_threshold
        self.volatility_window = volatility_window
        self.fps_estimate = fps_estimate

        # Internal state — bounded to prevent memory creep in long sessions
        _MAX_HISTORY = 1000
        self._smoothed: Dict[str, float] = {}
        self._raw_history: Deque[Dict[str, float]] = deque(maxlen=_MAX_HISTORY)
        self._smoothed_history: Deque[Dict[str, float]] = deque(maxlen=_MAX_HISTORY)
        self._timestamps: Deque[Optional[str]] = deque(maxlen=_MAX_HISTORY)
        self._dominant_history: Deque[str] = deque(maxlen=_MAX_HISTORY)
        self._transitions: List[EmotionTransition] = []
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Core frame processing
    # ------------------------------------------------------------------

    def add_frame(
        self, scores: Dict[str, float], timestamp: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Process a new emotion frame.

        - Normalizes scores: if any score > 1.0, assume 0-100 scale -> divide by 100
        - Applies EMA: smoothed[e] = alpha * raw[e] + (1-alpha) * prev_smoothed[e]
        - Detects transition: if dominant emotion changed and delta >= threshold
        - Records all history
        - Returns the smoothed scores for this frame (for real-time display)
        """
        # Normalize to 0-1 if needed
        raw = {}
        needs_normalize = any(v > 1.0 for v in scores.values())
        for emo, val in scores.items():
            raw[emo] = val / 100.0 if needs_normalize else val

        self._raw_history.append(raw.copy())
        self._timestamps.append(timestamp)

        # EMA smoothing
        if not self._smoothed:
            # First frame: initialize smoothed to raw
            self._smoothed = raw.copy()
        else:
            for emo in raw:
                prev = self._smoothed.get(emo, 0.0)
                self._smoothed[emo] = self.alpha * raw[emo] + (1.0 - self.alpha) * prev

        smoothed_snapshot = self._smoothed.copy()
        self._smoothed_history.append(smoothed_snapshot)

        # Determine dominant from smoothed
        dominant = max(smoothed_snapshot, key=smoothed_snapshot.get)
        self._dominant_history.append(dominant)

        # Transition detection
        if len(self._dominant_history) >= 2:
            prev_dominant = self._dominant_history[-2]
            if dominant != prev_dominant:
                delta = abs(
                    smoothed_snapshot.get(dominant, 0.0)
                    - smoothed_snapshot.get(prev_dominant, 0.0)
                )
                if delta >= self.transition_threshold:
                    self._transitions.append(
                        EmotionTransition(
                            from_emotion=prev_dominant,
                            to_emotion=dominant,
                            frame_idx=self._frame_count,
                            timestamp=timestamp,
                            confidence_delta=round(delta, 4),
                        )
                    )

        self._frame_count += 1
        return smoothed_snapshot

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_smoothed_scores(self) -> Dict[str, float]:
        """Return current smoothed emotion scores (latest frame)."""
        return self._smoothed.copy() if self._smoothed else {}

    def get_smoothed_timeline(self) -> List[Dict]:
        """Return full smoothed timeline."""
        timeline = []
        for i, (scores, dom) in enumerate(
            zip(self._smoothed_history, self._dominant_history)
        ):
            timeline.append({
                "frame": i,
                "emotions": {k: round(v, 4) for k, v in scores.items()},
                "dominant": dom,
            })
        return timeline

    def get_transitions(self) -> List[EmotionTransition]:
        """Return all detected transitions."""
        return list(self._transitions)

    def get_durations(self) -> List[EmotionDuration]:
        """
        Scan dominant_history for contiguous runs.
        Convert frame count to seconds using fps_estimate.
        """
        if not self._dominant_history:
            return []

        durations: List[EmotionDuration] = []
        current_emo = self._dominant_history[0]
        start_frame = 0

        for i in range(1, len(self._dominant_history)):
            if self._dominant_history[i] != current_emo:
                count = i - start_frame
                durations.append(
                    EmotionDuration(
                        emotion=current_emo,
                        start_frame=start_frame,
                        end_frame=i - 1,
                        duration_frames=count,
                        duration_sec=round(count / self.fps_estimate, 1)
                        if self.fps_estimate > 0
                        else 0.0,
                    )
                )
                current_emo = self._dominant_history[i]
                start_frame = i

        # Final segment
        count = len(self._dominant_history) - start_frame
        durations.append(
            EmotionDuration(
                emotion=current_emo,
                start_frame=start_frame,
                end_frame=len(self._dominant_history) - 1,
                duration_frames=count,
                duration_sec=round(count / self.fps_estimate, 1)
                if self.fps_estimate > 0
                else 0.0,
            )
        )
        return durations

    def get_trends(self) -> List[EmotionTrend]:
        """
        For each emotion, run linear regression on smoothed scores over frame index.
        Requires at least 5 frames.
        """
        if self._frame_count < 5:
            return []

        trends: List[EmotionTrend] = []
        all_emotions = set()
        for snapshot in self._smoothed_history:
            all_emotions.update(snapshot.keys())

        for emo in sorted(all_emotions):
            ys = [snap.get(emo, 0.0) for snap in self._smoothed_history]
            slope, r_sq = _linear_regression(ys)

            if slope > 0.001:
                direction = "increasing"
            elif slope < -0.001:
                direction = "decreasing"
            else:
                direction = "stable"

            trends.append(
                EmotionTrend(
                    emotion=emo,
                    slope=round(slope, 6),
                    r_squared=round(r_sq, 4),
                    direction=direction,
                )
            )
        return trends

    def get_volatility(self) -> Dict[str, float]:
        """
        Per-emotion standard deviation over the last `volatility_window` frames.
        High value = rapid changes. Low value = stable.
        """
        if self._frame_count < 2:
            return {}

        window = self._smoothed_history[-self.volatility_window:]
        all_emotions = set()
        for snap in window:
            all_emotions.update(snap.keys())

        volatility: Dict[str, float] = {}
        for emo in sorted(all_emotions):
            vals = [snap.get(emo, 0.0) for snap in window]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            volatility[emo] = round(math.sqrt(variance), 4)

        return volatility

    def get_stability_score(self) -> float:
        """
        Aggregate stability: 1.0 - normalized_mean_volatility.
        Range 0-1 where 1 = perfectly stable, 0 = maximum chaos.
        """
        vol = self.get_volatility()
        if not vol:
            return 1.0
        mean_vol = sum(vol.values()) / len(vol)
        # Normalize: max possible std dev for 0-1 range is 0.5
        normalized = min(mean_vol / 0.5, 1.0)
        return round(1.0 - normalized, 4)

    def get_session_summary(self) -> Dict:
        """
        Aggregate all temporal metrics into a single dict for API response.
        """
        timeline = self.get_smoothed_timeline()

        # Sample timeline if > 50 frames
        if len(timeline) > 50:
            step = max(1, len(timeline) // 50)
            timeline = [timeline[i] for i in range(0, len(timeline), step)]

        return {
            "frame_count": self._frame_count,
            "smoothed_timeline": timeline,
            "transitions": [asdict(t) for t in self._transitions],
            "transition_count": len(self._transitions),
            "durations": [asdict(d) for d in self.get_durations()],
            "trends": [asdict(t) for t in self.get_trends()],
            "volatility": self.get_volatility(),
            "stability_score": self.get_stability_score(),
            "ema_alpha": self.alpha,
        }

    def reset(self):
        """Clear all state for reuse."""
        self._smoothed = {}
        self._raw_history = []
        self._smoothed_history = []
        self._timestamps = []
        self._dominant_history = []
        self._transitions = []
        self._frame_count = 0
