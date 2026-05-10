"""
Event-level change-point detection on per-frame emotion probability streams.

This module is the **core research contribution** (Axis 1A in
`dev/reference/roadmap.md`): instead of reporting only frame-by-frame
classifier outputs or naive dominant-emotion transitions, it detects
50-200 ms-class **affective events** (emotion shifts) by running three
complementary change-point detectors and confirming events when at least
``consensus_min_methods`` of them agree within a tolerance window.

Three detectors are implemented:

1. **Sliding-window z-score** — classic, fast, online-friendly.
   Compares the post-window mean to the pre-window mean of the smoothed
   probability vector and flags samples whose magnitude is at least
   ``min_magnitude`` and whose z-score exceeds ``z_threshold``.

2. **CUSUM** — Page (1954) cumulative-sum on the normalised valence trace.
   Detects persistent shifts the sliding-window method may miss.

3. **PELT** (optional, requires ``ruptures``) — Killick et al. 2012,
   exact pruned-linear-time change-point detection on the multivariate
   probability vector. Off-line, batch only; used during post-session
   refinement.

The ensemble layer takes the union of detected change-points from all
methods, clusters them within ``refractory_frames``, and emits a single
``DetectedEvent`` per cluster annotated with which methods agreed.

Public API:
    EventDetector(...)
        .detect_streaming(p_t)            # online: feed one frame at a time
        .detect_batch(p_history)          # off-line: post-session refinement
        .reset()
        .get_events() -> List[DetectedEvent]

The module has **no Flask / no I/O** — it is pure computation and is
exercised by ``tests/test_event_detector.py`` on synthetic ground-truth
change-points. See ``eval/event_eval.py`` for the formal evaluation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional, Sequence, Tuple

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectedEvent:
    """Single confirmed change-point event."""
    frame_idx: int
    timestamp: Optional[str]
    from_emotion: str
    to_emotion: str
    magnitude: float                # ||p_after - p_before||_1 / 2 in [0,1]
    confidence: float               # in [0, 1]
    methods: List[str] = field(default_factory=list)
    method_count: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class _Candidate:
    """Internal: a candidate change-point from a single method."""
    frame_idx: int
    method: str
    score: float                    # method-native score (z, cusum, etc.)
    magnitude: float


# ---------------------------------------------------------------------------
# Numeric helpers (no scipy dep for the hot streaming path)
# ---------------------------------------------------------------------------

def _entropy(probs: Sequence[float], eps: float = 1e-9) -> float:
    """Shannon entropy in nats."""
    return -sum(p * math.log(p + eps) for p in probs if p > 0)


def _l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """L1 distance between two probability dicts. In [0, 2]."""
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _vec(p: Dict[str, float]) -> List[float]:
    """Stable 7-vector representation of an emotion probability dict."""
    return [float(p.get(e, 0.0)) for e in EMOTIONS]


def _dominant(p: Dict[str, float]) -> str:
    """Return the dominant emotion key (max prob)."""
    if not p:
        return "neutral"
    return max(p, key=p.get)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


# ---------------------------------------------------------------------------
# Method 1: Sliding-window z-score on dominant-prob delta
# ---------------------------------------------------------------------------

class SlidingWindowDetector:
    """
    Online change-point detector. For each new frame `t`, compares the mean
    L1-distance of the most recent `window` frames to the previous `window`
    frames. Flags `t` when `(d_t - μ_pre) / (σ_pre + ε) >= z_threshold`
    AND `d_t >= min_magnitude`.

    The doubled-window design keeps the detector simple and online while
    matching the typical 50–200 ms micro-expression duration when ``window``
    is set to ~5 frames at 30 Hz capture (i.e. ~166 ms per side).
    """

    def __init__(
        self,
        window: int = 5,
        z_threshold: float = 2.5,
        min_magnitude: float = 0.10,
    ):
        if window < 2:
            raise ValueError("SlidingWindowDetector window must be >= 2")
        self.window = window
        self.z_threshold = z_threshold
        self.min_magnitude = min_magnitude
        self._buf: Deque[Dict[str, float]] = deque(maxlen=2 * window + 1)
        self._delta_history: Deque[float] = deque(maxlen=200)

    def push(self, p: Dict[str, float], frame_idx: int) -> Optional[_Candidate]:
        self._buf.append(p)
        n = len(self._buf)
        if n < 2 * self.window + 1:
            return None
        mid = self.window
        # Compare frame at the centre of the buffer to the pre/post windows
        pre = list(self._buf)[:mid]
        post = list(self._buf)[mid + 1:]
        # L1 distance between the means of pre and post windows
        avg_pre = {e: _mean([f.get(e, 0.0) for f in pre]) for e in EMOTIONS}
        avg_post = {e: _mean([f.get(e, 0.0) for f in post]) for e in EMOTIONS}
        delta = _l1_distance(avg_pre, avg_post) / 2.0
        self._delta_history.append(delta)

        if len(self._delta_history) < 8:
            return None

        # z-score of current delta against its own recent history
        history = list(self._delta_history)[:-1]
        mu = _mean(history)
        sigma = _std(history)
        if sigma < 1e-6:
            z = 0.0
        else:
            z = (delta - mu) / sigma

        if delta >= self.min_magnitude and z >= self.z_threshold:
            # Centre frame is `frame_idx - window`
            return _Candidate(
                frame_idx=frame_idx - self.window,
                method="sliding",
                score=float(z),
                magnitude=float(delta),
            )
        return None

    def reset(self):
        self._buf.clear()
        self._delta_history.clear()


# ---------------------------------------------------------------------------
# Method 2: CUSUM on the (signed) valence trace
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """
    Page (1954) two-sided cumulative sum on the valence trace.
    Cleaner detection of persistent shifts (e.g. drift into negative affect)
    that the sliding-window method, which thresholds local jumps, may miss.
    """

    DEFAULT_VALENCE = {
        "happy": 1.0,
        "surprise": 0.5,
        "neutral": 0.0,
        "fear": -0.7,
        "sad": -0.8,
        "disgust": -0.6,
        "angry": -0.9,
    }

    def __init__(
        self,
        drift: float = 0.005,
        threshold: float = 0.15,
        valence_map: Optional[Dict[str, float]] = None,
    ):
        self.drift = drift
        self.threshold = threshold
        self.valence_map = valence_map or self.DEFAULT_VALENCE
        self._g_pos = 0.0
        self._g_neg = 0.0
        self._mu = 0.0
        self._n = 0
        self._last_alarm_frame = -1

    def _valence(self, p: Dict[str, float]) -> float:
        return sum(self.valence_map.get(e, 0.0) * p.get(e, 0.0) for e in p)

    def push(self, p: Dict[str, float], frame_idx: int) -> Optional[_Candidate]:
        v = self._valence(p)
        # Welford running mean (used as the reference value for CUSUM)
        self._n += 1
        delta = v - self._mu
        self._mu += delta / self._n

        s = v - self._mu
        self._g_pos = max(0.0, self._g_pos + s - self.drift)
        self._g_neg = max(0.0, self._g_neg - s - self.drift)

        alarmed_score: Optional[float] = None
        if self._g_pos > self.threshold:
            alarmed_score = self._g_pos
            self._g_pos = 0.0
        elif self._g_neg > self.threshold:
            alarmed_score = self._g_neg
            self._g_neg = 0.0

        if alarmed_score is None:
            return None

        # Refractory: ignore alarms too close to the previous one
        if self._last_alarm_frame >= 0 and frame_idx - self._last_alarm_frame < 3:
            return None
        self._last_alarm_frame = frame_idx

        return _Candidate(
            frame_idx=frame_idx,
            method="cusum",
            score=float(alarmed_score),
            magnitude=min(1.0, float(abs(s) * 4.0)),  # rough magnitude proxy
        )

    def reset(self):
        self._g_pos = 0.0
        self._g_neg = 0.0
        self._mu = 0.0
        self._n = 0
        self._last_alarm_frame = -1


# ---------------------------------------------------------------------------
# Method 3: PELT (off-line, optional dependency)
# ---------------------------------------------------------------------------

class PELTDetector:
    """
    Killick / Fearnhead 2012 PELT change-point detection on the multivariate
    probability stream. Wrapped behind a ``ruptures`` import so the module
    keeps working when the optional dep is missing — the ensemble simply
    drops PELT votes in that case.
    """

    def __init__(
        self,
        model: str = "rbf",
        penalty: float = 3.0,
        min_size: int = 3,
    ):
        self.model = model
        self.penalty = penalty
        self.min_size = max(2, min_size)
        try:
            import ruptures  # type: ignore
            self._ruptures = ruptures
            self.available = True
        except ImportError:
            self._ruptures = None
            self.available = False

    def detect(self, p_history: List[Dict[str, float]]) -> List[_Candidate]:
        if not self.available or len(p_history) < 2 * self.min_size:
            return []

        try:
            # Lazy numpy import — keep module importable in numpy-less envs
            import numpy as np  # type: ignore
        except ImportError:
            return []

        signal = np.asarray([_vec(p) for p in p_history], dtype=float)
        try:
            algo = self._ruptures.Pelt(
                model=self.model, min_size=self.min_size, jump=1
            ).fit(signal)
            change_points = algo.predict(pen=self.penalty)
        except Exception:
            return []

        # PELT returns segment ends (1-based, last index = len). We need
        # change-point *positions* — drop the trailing one.
        if change_points and change_points[-1] == len(p_history):
            change_points = change_points[:-1]

        candidates: List[_Candidate] = []
        for cp in change_points:
            idx = max(0, min(len(p_history) - 1, cp))
            mag = 0.0
            if 1 <= idx < len(p_history) - 1:
                mag = _l1_distance(p_history[idx - 1], p_history[idx + 1]) / 2.0
            candidates.append(
                _Candidate(
                    frame_idx=int(idx),
                    method="pelt",
                    score=1.0,
                    magnitude=float(mag),
                )
            )
        return candidates


# ---------------------------------------------------------------------------
# Top-level EventDetector
# ---------------------------------------------------------------------------

class EventDetector:
    """
    Multi-method change-point detector for emotion probability streams.

    Streaming usage::

        det = EventDetector()
        for t, (p_t, ts) in enumerate(stream):
            det.detect_streaming(p_t, frame_idx=t, timestamp=ts)
        events = det.get_events()

    Off-line refinement (uses PELT in addition to streaming methods)::

        det = EventDetector()
        events = det.detect_batch(p_history, timestamps=ts_list)
    """

    def __init__(
        self,
        method: str = "ensemble",
        consensus_min_methods: int = 2,
        # sliding params
        window: int = 5,
        z_threshold: float = 2.5,
        min_magnitude: float = 0.10,
        # cusum params
        cusum_drift: float = 0.005,
        cusum_threshold: float = 0.15,
        valence_map: Optional[Dict[str, float]] = None,
        # pelt params
        pelt_model: str = "rbf",
        pelt_penalty: float = 3.0,
        pelt_min_size: int = 3,
        # consensus
        refractory_frames: int = 3,
        tolerance_frames: int = 2,
    ):
        if method not in ("sliding", "cusum", "pelt", "ensemble"):
            raise ValueError(f"Unknown event detection method: {method}")

        self.method = method
        # Single-method modes can only ever reach consensus = 1, so cap the
        # required votes accordingly. Only "ensemble" honours >= 2.
        if method == "ensemble":
            self.consensus_min_methods = max(1, consensus_min_methods)
        else:
            self.consensus_min_methods = 1
        self.refractory_frames = refractory_frames
        self.tolerance_frames = tolerance_frames

        self.sliding = SlidingWindowDetector(window, z_threshold, min_magnitude)
        self.cusum = CUSUMDetector(cusum_drift, cusum_threshold, valence_map)
        self.pelt = PELTDetector(pelt_model, pelt_penalty, pelt_min_size)

        self._candidates: List[_Candidate] = []
        self._frame_idx: int = -1
        self._history: List[Dict[str, float]] = []
        self._timestamps: List[Optional[str]] = []
        self._last_event_frame: int = -10_000

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def detect_streaming(
        self,
        p: Dict[str, float],
        frame_idx: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[DetectedEvent]:
        """Feed a single smoothed probability dict; may emit a confirmed event."""
        if frame_idx is None:
            frame_idx = self._frame_idx + 1
        self._frame_idx = frame_idx
        self._history.append(dict(p))
        self._timestamps.append(timestamp)

        if self.method in ("sliding", "ensemble"):
            cand = self.sliding.push(p, frame_idx)
            if cand:
                self._candidates.append(cand)

        if self.method in ("cusum", "ensemble"):
            cand = self.cusum.push(p, frame_idx)
            if cand:
                self._candidates.append(cand)

        # Try to confirm an event using the candidates accumulated so far.
        return self._maybe_confirm()

    # ------------------------------------------------------------------
    # Off-line / batch
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        p_history: List[Dict[str, float]],
        timestamps: Optional[List[Optional[str]]] = None,
    ) -> List[DetectedEvent]:
        """
        Run every available method on a full probability history. Returns the
        confirmed event list (does not mutate the streaming state).
        """
        if not p_history:
            return []
        ts = timestamps or [None] * len(p_history)

        local_candidates: List[_Candidate] = []

        if self.method in ("sliding", "ensemble"):
            sd = SlidingWindowDetector(
                self.sliding.window, self.sliding.z_threshold, self.sliding.min_magnitude
            )
            for i, p in enumerate(p_history):
                c = sd.push(p, i)
                if c:
                    local_candidates.append(c)

        if self.method in ("cusum", "ensemble"):
            cd = CUSUMDetector(self.cusum.drift, self.cusum.threshold, self.cusum.valence_map)
            for i, p in enumerate(p_history):
                c = cd.push(p, i)
                if c:
                    local_candidates.append(c)

        if self.method in ("pelt", "ensemble"):
            local_candidates.extend(self.pelt.detect(p_history))

        return self._cluster_candidates(local_candidates, p_history, ts)

    # ------------------------------------------------------------------
    # Internal: confirmation logic
    # ------------------------------------------------------------------

    def _maybe_confirm(self) -> Optional[DetectedEvent]:
        """Streaming consensus: confirm an event when consensus_min_methods agree."""
        if not self._candidates:
            return None

        latest = self._candidates[-1]
        # Find candidates within the tolerance window of `latest`
        nearby = [
            c for c in self._candidates
            if abs(c.frame_idx - latest.frame_idx) <= self.tolerance_frames
        ]
        unique_methods = {c.method for c in nearby}

        if len(unique_methods) < self.consensus_min_methods and self.method == "ensemble":
            return None
        if self.method != "ensemble" and len(unique_methods) < 1:
            return None

        # Refractory suppression
        if latest.frame_idx - self._last_event_frame < self.refractory_frames:
            return None

        ev = self._build_event(nearby, self._history, self._timestamps)
        self._last_event_frame = latest.frame_idx
        # Keep the candidates for visibility but mark the consumed cluster
        return ev

    def _cluster_candidates(
        self,
        cands: List[_Candidate],
        history: List[Dict[str, float]],
        timestamps: List[Optional[str]],
    ) -> List[DetectedEvent]:
        if not cands:
            return []
        cands = sorted(cands, key=lambda c: c.frame_idx)
        clusters: List[List[_Candidate]] = []
        current: List[_Candidate] = [cands[0]]
        for c in cands[1:]:
            if c.frame_idx - current[-1].frame_idx <= self.tolerance_frames:
                current.append(c)
            else:
                clusters.append(current)
                current = [c]
        clusters.append(current)

        events: List[DetectedEvent] = []
        last_emit = -10_000
        for cluster in clusters:
            unique_methods = {c.method for c in cluster}
            if len(unique_methods) < self.consensus_min_methods:
                continue
            centre_idx = int(round(_mean([c.frame_idx for c in cluster])))
            if centre_idx - last_emit < self.refractory_frames:
                continue
            last_emit = centre_idx
            ev = self._build_event(cluster, history, timestamps)
            events.append(ev)

        return events

    @staticmethod
    def _build_event(
        cluster: List[_Candidate],
        history: List[Dict[str, float]],
        timestamps: List[Optional[str]],
    ) -> DetectedEvent:
        centre = int(round(_mean([c.frame_idx for c in cluster])))
        before = max(0, centre - 1)
        after = min(len(history) - 1, centre + 1)
        from_emo = _dominant(history[before]) if history else "neutral"
        to_emo = _dominant(history[after]) if history else "neutral"
        if from_emo == to_emo and len(history) > 2:
            # Try to widen the lookback for a more meaningful from/to label
            far_before = max(0, centre - 3)
            far_after = min(len(history) - 1, centre + 3)
            from_emo = _dominant(history[far_before])
            to_emo = _dominant(history[far_after])

        magnitude = max(c.magnitude for c in cluster)
        method_count = len({c.method for c in cluster})
        n_methods_total = 3
        # Confidence: blend magnitude, method consensus, and inverse fused entropy
        H = _entropy(_vec(history[centre])) if history else 0.0
        H_norm = H / math.log(7)
        conf = (
            0.5 * min(1.0, magnitude * 1.5)
            + 0.3 * (method_count / n_methods_total)
            + 0.2 * max(0.0, 1.0 - H_norm)
        )
        ts = timestamps[centre] if 0 <= centre < len(timestamps) else None

        return DetectedEvent(
            frame_idx=centre,
            timestamp=ts,
            from_emotion=from_emo,
            to_emotion=to_emo,
            magnitude=round(magnitude, 4),
            confidence=round(min(1.0, max(0.0, conf)), 4),
            methods=sorted({c.method for c in cluster}),
            method_count=method_count,
            metadata={
                "scores": {c.method: round(c.score, 4) for c in cluster},
                "entropy": round(H, 4),
            },
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_events(self) -> List[DetectedEvent]:
        """Return the streaming-confirmed events to date."""
        # Re-run the cluster step on the accumulated candidates so that the
        # call is idempotent and consistent with the offline path.
        return self._cluster_candidates(
            list(self._candidates), self._history, self._timestamps
        )

    def reset(self) -> None:
        self.sliding.reset()
        self.cusum.reset()
        self._candidates.clear()
        self._frame_idx = -1
        self._history.clear()
        self._timestamps.clear()
        self._last_event_frame = -10_000


# ---------------------------------------------------------------------------
# Convenience: build from project AppConfig
# ---------------------------------------------------------------------------

def build_from_config(cfg) -> EventDetector:
    """
    Construct an EventDetector from an ``events`` config block. Accepts both
    the dataclass form (``cfg.events``) and a raw dict.
    """
    e = getattr(cfg, "events", cfg) if hasattr(cfg, "events") else cfg
    valence = getattr(cfg, "clinical", None)
    valence_map = getattr(valence, "valence_map", None) if valence else None

    return EventDetector(
        method=getattr(e, "method", "ensemble"),
        consensus_min_methods=int(getattr(e, "consensus_min_methods", 2)),
        window=int(getattr(getattr(e, "sliding", e), "window", 5)),
        z_threshold=float(getattr(getattr(e, "sliding", e), "z_threshold", 2.5)),
        min_magnitude=float(getattr(getattr(e, "sliding", e), "min_magnitude", 0.10)),
        cusum_drift=float(getattr(getattr(e, "cusum", e), "drift", 0.005)),
        cusum_threshold=float(getattr(getattr(e, "cusum", e), "threshold", 0.15)),
        valence_map=valence_map,
        pelt_model=str(getattr(getattr(e, "pelt", e), "model", "rbf")),
        pelt_penalty=float(getattr(getattr(e, "pelt", e), "penalty", 3.0)),
        pelt_min_size=int(getattr(getattr(e, "pelt", e), "min_size", 3)),
        refractory_frames=int(getattr(e, "refractory_frames", 3)),
    )


def event_to_dict(ev: DetectedEvent) -> Dict:
    """Serialise a DetectedEvent to a JSON-safe dict."""
    return asdict(ev)
