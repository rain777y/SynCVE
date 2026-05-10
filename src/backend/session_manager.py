"""
Session lifecycle management for SynCVE backend.

Handles session CRUD (start, stop, pause, resume), frame logging,
and in-memory caching.  Heavy-lifting is delegated to:
    - emotion_analytics   (aggregation / stats)
    - report_generator    (text & visual reports)
    - gemini_client       (LLM / image generation)
    - storage             (Supabase Storage I/O)
"""

import base64
import collections
import json
import re
import threading
import time
import traceback
import uuid as _uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Deque, Dict, List, Optional


_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _is_valid_uuid(s: str) -> bool:
    if not isinstance(s, str) or not _UUID_RE.match(s):
        return False
    try:
        _uuid.UUID(s)
        return True
    except (ValueError, AttributeError, TypeError):
        return False

from deepface.commons.logger import Logger

from src.backend.config import get_config
from src.backend.event_detector import build_from_config as build_event_detector
from src.backend.storage import get_supabase_client, upload_frame_to_storage
from src.backend.temporal_analysis import TemporalAnalyzer
from src.backend.clinical_metrics import (
    compute_session_metrics,
    metrics_to_dict,
)

# Re-export report/analytics functions so existing imports from
# ``session_manager`` continue to work (backward compatibility).
from src.backend.emotion_analytics import aggregate_emotion_metrics  # noqa: F401
from src.backend.report_generator import (  # noqa: F401
    generate_emotion_report,
    generate_fast_report,
    generate_report,
    generate_visual_report_v3,
)

logger = Logger()


def _to_json_safe(obj):
    """Recursively convert numpy scalars/arrays to Python native types for JSON serialization."""
    try:
        import numpy as _np
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, _np.bool_):
            return bool(obj)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Backward-compatible config constants accessed as module attributes
# (e.g. ``session_manager.EMOTION_REPORT_KEYFRAME_LIMIT`` in routes.py).
# We use module-level ``__getattr__`` so the config singleton is read lazily
# *after* .env has been loaded (which happens in app.py before any route
# handler runs).
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    _MAP = {
        "EMOTION_REPORT_KEYFRAME_LIMIT": lambda: get_config().gemini.keyframe_limit,
        "EMOTION_VISUAL_ASPECT_RATIO": lambda: get_config().gemini.visual_aspect_ratio,
        "EMOTION_VISUAL_STYLE_PRESET": lambda: get_config().gemini.visual_style_preset,
        "EMOTION_NOISE_FLOOR": lambda: get_config().gemini.noise_floor,
    }
    if name in _MAP:
        return _MAP[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ============================================================================
# In-memory session state
# ============================================================================

# Throttle per-session frame uploads
_last_upload_times: Dict[str, float] = {}
UPLOAD_INTERVAL = 5.0  # seconds

# Bounded cache for recent vision samples (fallback when Supabase is slow).
# Uses collections.deque(maxlen=N) for O(1) append+eviction instead of
# list.pop(0) which is O(n).
_vision_cache: Dict[str, Deque[Dict[str, Any]]] = {}
_VISION_CACHE_LIMIT = 120


# Per-session temporal analyzers
_temporal_analyzers: Dict[str, TemporalAnalyzer] = {}

# ---------------------------------------------------------------------------
# Lightweight LRU caches for /events and /clinical_metrics.
# Key includes analyzer._frame_count so growth in the session naturally
# invalidates stale entries. FIFO eviction once the cap is reached.
# ---------------------------------------------------------------------------
from collections import OrderedDict as _OrderedDict

_EVENTS_CACHE_MAX = 64
_CLINICAL_CACHE_MAX = 64
_events_cache: "_OrderedDict[tuple, tuple]" = _OrderedDict()
_clinical_cache: "_OrderedDict[tuple, tuple]" = _OrderedDict()
_events_cache_lock = threading.Lock()
_clinical_cache_lock = threading.Lock()


def _cache_get(store, lock, key):
    with lock:
        if key in store:
            # Touch to mark as recently used (still cheap O(1)).
            store.move_to_end(key)
            return store[key][1]
    return None


def _cache_put(store, lock, key, value, max_entries):
    with lock:
        store[key] = (time.time(), value)
        store.move_to_end(key)
        while len(store) > max_entries:
            store.popitem(last=False)


def _build_temporal_analyzer() -> TemporalAnalyzer:
    """Create a TemporalAnalyzer using the current runtime config."""
    cfg = get_config()
    tcfg = cfg.temporal
    event_det = None
    if cfg.events.enabled:
        try:
            event_det = build_event_detector(cfg)
        except Exception as e:
            logger.warn(f"Failed to build EventDetector: {e}")
    return TemporalAnalyzer(
        alpha=tcfg.ema_alpha,
        transition_threshold=tcfg.transition_threshold,
        volatility_window=tcfg.volatility_window,
        fps_estimate=tcfg.fps_estimate,
        event_detector=event_det,
    )


def _get_or_rebuild_temporal_analyzer(session_id: str) -> Optional[TemporalAnalyzer]:
    """Return an analyzer, rebuilding it from persisted vision_samples if needed."""
    analyzer = _temporal_analyzers.get(session_id)
    if analyzer and analyzer._frame_count > 0:
        return analyzer

    records = fetch_emotion_logs(session_id, limit=1000)
    if not records:
        return analyzer

    rebuilt = _build_temporal_analyzer()
    for record in reversed(records):
        emotions = record.get("emotions") or {}
        if not isinstance(emotions, dict) or not emotions:
            raw_payload = record.get("raw_payload") or {}
            if isinstance(raw_payload, dict):
                results = raw_payload.get("results") or []
                if results:
                    emotions = (results[0] or {}).get("emotion") or {}
        if not isinstance(emotions, dict) or not emotions:
            continue

        raw_payload = record.get("raw_payload") or {}
        ensemble_meta = None
        if isinstance(raw_payload, dict):
            results = raw_payload.get("results") or []
            if results:
                ensemble_meta = (results[0] or {}).get("ensemble")

        rebuilt.add_frame(
            emotions,
            timestamp=record.get("captured_at") or record.get("created_at"),
            ensemble_meta=ensemble_meta,
        )

    if rebuilt._frame_count > 0:
        _temporal_analyzers[session_id] = rebuilt
        return rebuilt
    return analyzer

# Session TTL: evict stale in-memory caches after 30 minutes
_SESSION_TTL_SECONDS = 30 * 60  # 30 minutes
_session_start_times: Dict[str, float] = {}
_session_frame_indices: Dict[str, int] = {}


def _cleanup_session_cache(session_id: str) -> None:
    """Remove ALL in-memory data for a session to prevent leaks."""
    _last_upload_times.pop(session_id, None)
    _vision_cache.pop(session_id, None)
    _temporal_analyzers.pop(session_id, None)
    _session_start_times.pop(session_id, None)
    _session_frame_indices.pop(session_id, None)


def _session_ttl_cleanup_loop():
    """Background thread: evict stale session caches every 5 minutes."""
    while True:
        try:
            now = time.time()
            stale = [sid for sid, t in _session_start_times.items()
                     if now - t > _SESSION_TTL_SECONDS]
            for sid in stale:
                _cleanup_session_cache(sid)
                _session_start_times.pop(sid, None)
                logger.info(f"TTL evicted session cache: {sid}")
        except Exception as exc:
            logger.warn(f"TTL cleanup error: {exc}")
        time.sleep(300)


threading.Thread(target=_session_ttl_cleanup_loop, daemon=True, name="session-ttl").start()


def _cache_vision_sample(session_id: str, sample: Dict[str, Any]) -> None:
    """Append a bounded sample to the in-memory vision cache."""
    if session_id not in _vision_cache:
        _vision_cache[session_id] = collections.deque(maxlen=_VISION_CACHE_LIMIT)
    _vision_cache[session_id].append(sample)


def _next_frame_idx(session_id: str) -> int:
    """Return a per-process monotonic frame index for the live session."""
    frame_idx = _session_frame_indices.get(session_id, 0)
    _session_frame_indices[session_id] = frame_idx + 1
    return frame_idx


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_emotion_scores(emotions: Dict[str, Any]) -> Dict[str, float]:
    """Convert either 0..100 DeepFace scores or 0..1 probabilities to 0..1."""
    if not isinstance(emotions, dict):
        return {}

    numeric: Dict[str, float] = {}
    for key, value in emotions.items():
        score = _coerce_float(value)
        if score is None:
            continue
        numeric[str(key)] = max(0.0, score)

    if not numeric:
        return {}

    values = list(numeric.values())
    scale = 100.0 if max(values) > 1.0 or sum(values) > 1.5 else 1.0
    return {
        emotion: max(0.0, min(1.0, score / scale))
        for emotion, score in numeric.items()
    }


def _dominant_score(scores: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    numeric = {
        str(key): _coerce_float(value, 0.0) or 0.0
        for key, value in (scores or {}).items()
    }
    if not numeric:
        return None, None
    emotion = max(numeric, key=numeric.get)
    return emotion, numeric[emotion]


def _first_face_payload(payload_root: Any) -> Dict[str, Any]:
    if isinstance(payload_root, dict):
        results = payload_root.get("results")
        if isinstance(results, list) and results:
            return results[0] or {}
        return payload_root
    return {}


def _as_string_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


# ============================================================================
# Database helpers: session events / aggregates / report metadata
# ============================================================================

def _record_session_event(
    session_id: str,
    to_status: str,
    from_status: Optional[str] = None,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a lifecycle transition. Best-effort."""
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        supabase.table("session_events").insert({
            "session_id": session_id,
            "from_status": from_status,
            "to_status": to_status,
            "reason": reason,
            "metadata": metadata or {},
        }).execute()
    except Exception as e:
        logger.warn(f"Failed to record session event for {session_id}: {e}")


def persist_aggregate_snapshot(session_id: str, metrics: Dict[str, Any]) -> None:
    """Store aggregated emotion metrics for analytics. Best-effort."""
    supabase = get_supabase_client()
    if not supabase or not metrics:
        return
    try:
        payload = _to_json_safe({
            "session_id": session_id,
            "sample_count": metrics.get("samples"),
            "averages": metrics.get("averages"),
            "peaks": metrics.get("peaks"),
            "filtered_out": metrics.get("filtered_out"),
            "dominant_emotion": metrics.get("dominant"),
            "dominant_score": metrics.get("dominant_score"),
            "peak_emotion": metrics.get("peak_emotion"),
            "peak_score": metrics.get("peak_score"),
            "noise_floor": metrics.get("noise_floor"),
        })
        supabase.table("session_aggregates").insert(payload).execute()
    except Exception as e:
        logger.warn(f"Failed to persist aggregate snapshot: {e}")


def _store_report_metadata(
    session_id: str,
    report_type: str,
    *,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    prompt: Optional[str] = None,
    public_url: Optional[str] = None,
    storage_path: Optional[str] = None,
    status: str = "completed",
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist report generation lineage. Best-effort."""
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        supabase.table("reports").insert({
            "session_id": session_id,
            "report_type": report_type,
            "model_name": model_name,
            "model_version": model_version,
            "prompt": prompt,
            "public_url": public_url,
            "storage_path": storage_path,
            "status": status,
            "error": error,
            "metadata": metadata or {},
        }).execute()
    except Exception as e:
        logger.warn(f"Failed to store report metadata for {session_id}: {e}")


# ============================================================================
# Fetch helpers (used by report_generator via public accessor)
# ============================================================================

def fetch_emotion_logs(session_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch recent emotion logs for a session.
    Falls back to the in-memory cache when remote queries fail.
    """
    records: List[Dict[str, Any]] = []
    supabase = get_supabase_client()

    if supabase:
        try:
            response = (
                supabase.table("vision_samples")
                .select("*")
                .eq("session_id", session_id)
                .order("captured_at", desc=True)
                .limit(limit)
                .execute()
            )
            records = response.data or []
        except Exception as e:
            logger.warn(f"Failed to fetch vision_samples for {session_id}: {e}")

    if not records:
        cached = _vision_cache.get(session_id)
        if cached:
            return list(cached)[-limit:]

    return records


def get_temporal_summary(session_id: str) -> Optional[Dict]:
    """Get temporal analysis summary for a session."""
    analyzer = _get_or_rebuild_temporal_analyzer(session_id)
    if analyzer and analyzer._frame_count > 0:
        return analyzer.get_session_summary()
    return None


def get_session_events(
    session_id: str,
    *,
    method: Optional[str] = None,
    z_threshold: Optional[float] = None,
    min_magnitude: Optional[float] = None,
    consensus_min_methods: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Re-run the event detector against the smoothed history with the supplied
    overrides. The base session detector is preserved; an ad-hoc detector is
    constructed for the override-driven re-run so the sensitivity panel can
    explore parameters without disturbing live state.
    """
    analyzer = _get_or_rebuild_temporal_analyzer(session_id)
    if analyzer is None or analyzer._frame_count == 0:
        return {"events": [], "event_count": 0, "samples": 0}

    cfg = get_config()
    events_cfg = cfg.events
    method_eff = method or events_cfg.method
    consensus_eff = consensus_min_methods or events_cfg.consensus_min_methods
    z_eff = z_threshold if z_threshold is not None else events_cfg.sliding.z_threshold
    mag_eff = min_magnitude if min_magnitude is not None else events_cfg.sliding.min_magnitude

    cache_key = (
        session_id,
        analyzer._frame_count,
        method_eff,
        z_eff,
        mag_eff,
        consensus_eff,
    )
    cached = _cache_get(_events_cache, _events_cache_lock, cache_key)
    if cached is not None:
        return cached

    from src.backend.event_detector import EventDetector
    det = EventDetector(
        method=method_eff,
        consensus_min_methods=consensus_eff,
        window=events_cfg.sliding.window,
        z_threshold=z_eff,
        min_magnitude=mag_eff,
        cusum_drift=events_cfg.cusum.drift,
        cusum_threshold=events_cfg.cusum.threshold,
        valence_map=cfg.clinical.valence_map,
        pelt_model=events_cfg.pelt.model,
        pelt_penalty=events_cfg.pelt.penalty,
        pelt_min_size=events_cfg.pelt.min_size,
        refractory_frames=events_cfg.refractory_frames,
    )
    history = analyzer.get_smoothed_history_dicts()
    timestamps = list(analyzer._timestamps)
    evs = det.detect_batch(history, timestamps=timestamps)
    samples = len(history)
    sliding_min_frames = 2 * events_cfg.sliding.window + 8
    pelt_min_frames = 2 * events_cfg.pelt.min_size
    min_required_frames = {
        "sliding": sliding_min_frames,
        "cusum": 2,
        "pelt": pelt_min_frames,
        "ensemble": (
            max(2, min(sliding_min_frames, pelt_min_frames))
            if det.pelt.available
            else sliding_min_frames
        ),
    }
    selected_min_frames = min_required_frames.get(method_eff, 2)

    from src.backend.event_detector import event_to_dict
    result = {
        "events": [event_to_dict(e) for e in evs],
        "event_count": len(evs),
        "samples": samples,
        "method": method_eff,
        "z_threshold": z_eff,
        "min_magnitude": mag_eff,
        "consensus_min_methods": consensus_eff,
        "fps_estimate": cfg.temporal.fps_estimate,
        "diagnostics": {
            "pelt_available": det.pelt.available,
            "sliding_window": events_cfg.sliding.window,
            "pelt_min_size": events_cfg.pelt.min_size,
            "min_required_frames": min_required_frames,
            "selected_min_required_frames": selected_min_frames,
            "sample_count": samples,
            "no_event_reason": (
                f"{method_eff} requires at least {selected_min_frames} frames at current settings"
                if samples < selected_min_frames
                else "No candidate exceeded the current sensitivity thresholds"
            ),
        },
    }
    _cache_put(_events_cache, _events_cache_lock, cache_key, result, _EVENTS_CACHE_MAX)
    return result


def get_clinical_metrics(
    session_id: str,
    *,
    triggers: Optional[List[Dict[str, Any]]] = None,
    asr_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute Axis 1A clinical metrics over the current session: valence trace,
    drift, affect blunting, reactivity, suppression, incongruence, etc.
    """
    triggers_key = json.dumps(triggers or [], sort_keys=True, default=str)
    asr_key = json.dumps(asr_segments or [], sort_keys=True, default=str)

    analyzer = _get_or_rebuild_temporal_analyzer(session_id)
    if analyzer is None or analyzer._frame_count == 0:
        # Even after a session has been stopped (analyzer cleaned up), the
        # warm-cache populated by stop_session may still hold a result.
        # Try a frame-count-agnostic lookup so /clinical_report is fast.
        with _clinical_cache_lock:
            for k, (_ts, val) in reversed(_clinical_cache.items()):
                if k[0] == session_id and k[2] == triggers_key and k[3] == asr_key:
                    return val
        # Distinguish "session does not exist at all" (→ 404) from
        # "session exists but no frames yet" (→ 400).
        details = get_session_details(session_id)
        if details.get("code") == "not_found":
            return {"error": "Session not found", "code": "not_found"}
        return {"error": "No smoothed history for this session yet.", "code": "empty_session"}

    cache_key = (
        session_id,
        analyzer._frame_count,
        triggers_key,
        asr_key,
    )
    cached = _cache_get(_clinical_cache, _clinical_cache_lock, cache_key)
    if cached is not None:
        return cached

    cfg = get_config()
    history = analyzer.get_smoothed_history_dicts()
    events = analyzer.get_events()
    ensemble_history = analyzer.get_ensemble_history()
    detectors = list(cfg.deepface.ensemble_detectors)

    metrics = compute_session_metrics(
        history,
        events,
        fps=cfg.temporal.fps_estimate,
        valence_map=cfg.clinical.valence_map,
        sigma_baseline=cfg.clinical.sigma_baseline,
        range_baseline=cfg.clinical.range_baseline,
        triggers=triggers,
        asr_segments=asr_segments,
        refractory_frames=max(2, int(cfg.temporal.fps_estimate * 1.5) or 2),
        incongruence_window_sec=cfg.clinical.incongruence_window_sec,
        reaction_latency_max_sec=cfg.clinical.reaction_latency_max_sec,
        per_frame_ensemble=ensemble_history,
        detectors=detectors,
    )

    out = metrics_to_dict(metrics)
    out["session_id"] = session_id
    out["events"] = events

    # Reaction latencies — surface as a top-level field for clinician reports
    if triggers:
        from src.backend.clinical_metrics import compute_reaction_latencies
        latencies = compute_reaction_latencies(
            triggers, events, cfg.temporal.fps_estimate,
            max_latency_sec=cfg.clinical.reaction_latency_max_sec,
        )
        out["reaction_latencies"] = latencies

    _cache_put(_clinical_cache, _clinical_cache_lock, cache_key, out, _CLINICAL_CACHE_MAX)
    return out


# ============================================================================
# Session CRUD
# ============================================================================

def start_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Start a new monitoring session."""
    supabase = get_supabase_client()
    if not supabase:
        return {"error": "Database not configured"}

    try:
        payload: Dict[str, Any] = {"status": "active", "metadata": metadata or {}}
        if user_id:
            payload["user_id"] = user_id

        response = supabase.table("sessions").insert(payload).execute()

        if response.data and len(response.data) > 0:
            session_id = response.data[0]["id"]
            _last_upload_times[session_id] = 0
            _session_start_times[session_id] = time.time()
            _session_frame_indices[session_id] = 0
            _temporal_analyzers[session_id] = _build_temporal_analyzer()
            _record_session_event(session_id, "active", None, "Session started", metadata)
            return {"session_id": session_id, "status": "active"}
        else:
            return {"error": "Failed to create session, no data returned"}

    except Exception as e:
        logger.error(f"Error starting session: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


def stop_session(session_id: str) -> Dict[str, Any]:
    """End the session and generate the final text report."""
    supabase = get_supabase_client()
    if not supabase:
        return {"error": "Database not configured"}

    cfg = get_config().gemini

    try:
        # Capture temporal summary BEFORE cleanup destroys the analyzer
        temporal_summary = get_temporal_summary(session_id)

        report = generate_report(session_id)

        update_payload = _to_json_safe({
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "summary": report.get("summary", ""),
            "recommendations": report.get("recommendations", ""),
            "last_event_at": datetime.now(timezone.utc).isoformat(),
            "temporal_summary": temporal_summary,
        })

        supabase.table("sessions").update(update_payload).eq("id", session_id).execute()
        _record_session_event(session_id, "completed", reason="stop endpoint")
        _store_report_metadata(
            session_id,
            "text_summary",
            model_name=cfg.text_model,
            prompt="session text summary",
            metadata={"report": report},
        )

        # Pre-compute clinical metrics so the first /clinical_report request is fast.
        # Must run BEFORE _cleanup_session_cache (which drops the temporal analyzer).
        try:
            get_clinical_metrics(session_id)
        except Exception as warm_err:
            logger.warn(f"Clinical metrics warm-cache failed (non-fatal): {warm_err}")

        _cleanup_session_cache(session_id)

        return {"status": "session_ended", "report": report}

    except Exception as e:
        logger.error(f"Error stopping session: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


def pause_session(session_id: str) -> Dict[str, Any]:
    """Pause the session and generate a report.

    report.mode="fast"  → structured JSON only (instant, 0 LLM calls)
    report.mode="full"  → fast report + AI dashboard image (slow, 2 LLM calls)
    """
    try:
        if not session_id:
            return {"error": "session_id is required", "status_code": 400}

        supabase = get_supabase_client()
        if not supabase:
            raise ValueError("Database not configured")

        cfg = get_config().gemini
        mode = cfg.report_mode  # "fast" or "full"

        # --- Always: instant structured report ---
        fast_result = generate_fast_report(session_id)

        report_entry: Dict[str, Any] = {
            "report_mode": mode,
            "dominant_emotion": fast_result.get("stats_summary", {}).get("dominant"),
            "metrics": fast_result.get("metrics"),
            "stats_summary": fast_result.get("stats_summary"),
            "text_summary": fast_result.get("text_summary"),
            "emotion_ranking": fast_result.get("emotion_ranking"),
            "emotion_timeline": fast_result.get("emotion_timeline"),
            "temporal": fast_result.get("temporal"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # --- Optional: AI-generated dashboard image ---
        if mode == "full" and get_genai_client():
            try:
                v3_result = generate_visual_report_v3(
                    session_id=session_id,
                    aspect_ratio=cfg.visual_aspect_ratio,
                    style_preset=cfg.visual_style_preset,
                )
                report_entry["visual_report_url"] = v3_result.get("public_url")
                report_entry["image_prompt"] = v3_result.get("image_prompt")
                report_entry["storage_path"] = v3_result.get("storage_path")
            except Exception as img_err:
                logger.warn(f"Visual report generation failed (non-fatal): {img_err}")
                report_entry["visual_report_error"] = str(img_err)

        # --- Update session in DB ---
        existing_meta: Dict[str, Any] = {}
        try:
            meta_resp = (
                supabase.table("sessions")
                .select("metadata")
                .eq("id", session_id)
                .single()
                .execute()
            )
            if getattr(meta_resp, "data", None):
                existing_meta = meta_resp.data.get("metadata") or {}
        except Exception as e:
            logger.warn(f"Could not fetch existing session metadata: {e}")

        merged_meta = _to_json_safe({**existing_meta, "pause_report": report_entry})
        update_payload = _to_json_safe({
            "status": "paused",
            "metadata": merged_meta,
            "summary": fast_result.get("text_summary", ""),
            "last_event_at": datetime.now(timezone.utc).isoformat(),
            "temporal_summary": fast_result.get("temporal"),
        })
        supabase.table("sessions").update(update_payload).eq("id", session_id).execute()
        _record_session_event(
            session_id, "paused",
            reason="pause endpoint",
            metadata={"report_mode": mode, "dominant": report_entry.get("dominant_emotion")},
        )

        response = {
            "status": "paused",
            "report_mode": mode,
            "message": "Session paused. Report generated.",
            "report": report_entry,
            # Backward compat: frontend reads image_url
            "image_url": report_entry.get("visual_report_url"),
        }
        return response

    except ValueError as e:
        logger.warn(f"Pause session aborted: {e}")
        return {
            "error": str(e),
            "status_code": 400,
            "hint": "Ensure /analyze is called with a valid session_id so frames are logged before pausing.",
        }
    except Exception as e:
        logger.error(f"Error pausing session: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


def log_data(
    session_id: str,
    analysis_result: Dict[str, Any],
    metadata: Optional[Dict] = None,
    image_data=None,
) -> Dict[str, Any]:
    """
    Log a single analysis frame to the DB and optionally upload
    the raw image (throttled).
    """
    try:
        if not session_id:
            return {"error": "session_id is required for logging"}

        payload_root = analysis_result
        if isinstance(analysis_result, tuple) and len(analysis_result) == 2:
            payload_root = analysis_result[0]
        if isinstance(payload_root, dict) and payload_root.get("error"):
            return {"error": "analysis_result contains error; skipping log"}

        metadata_payload: Dict[str, Any] = dict(metadata or {})
        frame_idx = _next_frame_idx(session_id)
        face_data = _first_face_payload(payload_root)
        analysis_meta = payload_root.get("analysis_meta", {}) if isinstance(payload_root, dict) else {}
        request_meta = metadata_payload.get("request") or {}

        dominant_emotion = face_data.get("dominant_emotion")
        emotions_data: Dict[str, Any] = face_data.get("emotion", {}) or {}
        normalized_emotions = _normalize_emotion_scores(emotions_data)
        normalized_dominant, normalized_confidence = _dominant_score(normalized_emotions)
        if not dominant_emotion:
            dominant_emotion = normalized_dominant

        confidence = 0.0
        if dominant_emotion and isinstance(emotions_data, dict):
            confidence = float(_coerce_float(emotions_data.get(dominant_emotion), 0.0) or 0.0)

        spoof_check = face_data.get("spoof_check") or {}
        threshold = _coerce_float(
            request_meta.get("confidence_threshold")
            or analysis_meta.get("confidence_threshold")
        )
        low_confidence = bool(face_data.get("low_confidence", False))
        if threshold is not None and normalized_confidence is not None:
            low_confidence = low_confidence or normalized_confidence < threshold

        quality_flags = {
            "face_detected": bool(face_data.get("region") or face_data.get("face_confidence") or emotions_data),
            "low_confidence": low_confidence,
            "detection_fallback": bool(face_data.get("detection_fallback", False)),
            "anti_spoofing_triggered": bool(spoof_check.get("triggered", False)),
            "anti_spoofing_bypassed": bool(spoof_check.get("bypassed", False)),
        }

        detector_backends = _as_string_list(
            face_data.get("detector_backends")
            or analysis_meta.get("detectors_used")
            or analysis_meta.get("detectors_requested")
            or face_data.get("detector_backend")
            or request_meta.get("detector_backend")
        )

        # Single client lookup for both frame upload and DB inserts
        supabase = get_supabase_client()

        # Upload frame (throttled)
        if image_data is not None:
            import numpy as np

            if isinstance(image_data, np.ndarray):
                try:
                    import cv2
                    success, encoded_img = cv2.imencode(".jpg", image_data)
                    if success:
                        image_data = base64.b64encode(encoded_img).decode("utf-8")
                    else:
                        logger.warn("Failed to encode numpy image for upload.")
                        image_data = None
                except ImportError:
                    try:
                        from PIL import Image
                        pil_img = Image.fromarray(image_data)
                        buf = BytesIO()
                        pil_img.save(buf, format="JPEG")
                        image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                    except Exception as e:
                        logger.warn(f"Failed to convert numpy image: {e}")
                        image_data = None

            now = time.time()
            last_upload = _last_upload_times.get(session_id, 0)

            if supabase is None:
                logger.warn("Supabase not configured; skipping frame upload to storage.")
            elif image_data and (now - last_upload > UPLOAD_INTERVAL):
                logger.info(f"Attempting to upload frame for session {session_id}. Type: {type(image_data)}")
                upload_start = time.time()
                frame_ref = upload_frame_to_storage(session_id, image_data)
                upload_latency_ms = int((time.time() - upload_start) * 1000)
                if frame_ref:
                    _last_upload_times[session_id] = now
                    metadata_payload = {
                        **metadata_payload,
                        "frame_ref": frame_ref,
                        "frame_upload_latency_ms": upload_latency_ms,
                    }

        # Feed to temporal analyzer and capture smoothed scores
        smoothed_emotions = None
        analyzer = _temporal_analyzers.get(session_id)
        if analyzer and emotions_data:
            ensemble_meta = None
            if "results" in analysis_result and isinstance(analysis_result["results"], list):
                ensemble_meta = (analysis_result["results"][0] or {}).get("ensemble")
            smoothed_emotions = analyzer.add_frame(
                emotions_data, ensemble_meta=ensemble_meta
            )
        smoothed_dominant, smoothed_confidence = _dominant_score(smoothed_emotions or {})

        frame_ledger = {
            "schema_version": 1,
            "session_id": session_id,
            "frame_idx": frame_idx,
            "client_frame_id": metadata_payload.get("client_frame_id"),
            "timing": {
                "client_capture_ts": metadata_payload.get("client_capture_ts"),
                "server_received_ts": metadata_payload.get("server_received_ts"),
                "server_completed_ts": metadata_payload.get("server_completed_ts"),
                "server_logged_ts": datetime.now(timezone.utc).isoformat(),
                "inference_latency_ms": metadata_payload.get("inference_latency_ms"),
                "frame_upload_latency_ms": metadata_payload.get("frame_upload_latency_ms"),
            },
            "raw": {
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "confidence_norm": normalized_confidence,
                "emotions": emotions_data,
                "emotions_norm": normalized_emotions,
            },
            "smoothed": (
                {
                    "dominant_emotion": smoothed_dominant,
                    "confidence_norm": smoothed_confidence,
                    "emotions": smoothed_emotions,
                }
                if smoothed_emotions
                else None
            ),
            "model": {
                "detector_backend": face_data.get("detector_backend") or request_meta.get("detector_backend"),
                "detector_backends": detector_backends,
                "ensemble_enabled": bool(analysis_meta.get("ensemble_enabled") or face_data.get("ensemble")),
                "confidence_threshold": threshold,
                "analysis_meta": analysis_meta,
            },
            "quality_flags": quality_flags,
            "storage": {
                "frame_ref": metadata_payload.get("frame_ref"),
            },
        }

        metadata_payload = _to_json_safe({
            **metadata_payload,
            "frame_ledger": frame_ledger,
            "quality_flags": quality_flags,
        })

        payload: Dict[str, Any] = {
            "session_id": session_id,
            "dominant_emotion": dominant_emotion,
            "emotions": emotions_data,
            "confidence": confidence,
            "metadata": metadata_payload,
            "frame_ref": metadata_payload.get("frame_ref"),
        }

        # Cache locally
        _cache_vision_sample(session_id, {
            "session_id": session_id,
            "dominant_emotion": dominant_emotion,
            "emotions": emotions_data,
            "confidence": confidence,
            "metadata": metadata_payload,
            "frame_ref": metadata_payload.get("frame_ref"),
            "raw_payload": {
                **(payload_root if isinstance(payload_root, dict) else {"result": str(payload_root)}),
                "frame_ledger": frame_ledger,
            },
        })

        if supabase:
            raw_payload = (
                {**payload_root, "frame_ledger": frame_ledger}
                if isinstance(payload_root, dict)
                else {"result": str(payload_root), "frame_ledger": frame_ledger}
            )
            vision_payload = _to_json_safe({
                **payload,
                "raw_payload": raw_payload,
            })
            supabase.table("vision_samples").insert(vision_payload).execute()
            try:
                supabase.table("sessions").update({
                    "last_event_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", session_id).execute()
            except Exception as e:
                logger.warn(f"Could not update session heartbeat: {e}")

        status = "logged" if supabase else "cached_only"
        result = {
            "status": status,
            "frame_ref": metadata_payload.get("frame_ref"),
            "frame_ledger": frame_ledger,
            "data_quality": {
                "frame_idx": frame_idx,
                "quality_flags": quality_flags,
                "inference_latency_ms": metadata_payload.get("inference_latency_ms"),
            },
        }
        if smoothed_emotions:
            result["smoothed_emotions"] = smoothed_emotions
        return result

    except Exception as e:
        logger.error(f"Error logging data: {e}")
        return {"error": str(e)}


def get_recent_sessions(
    user_id: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Retrieve the most recent sessions."""
    supabase = get_supabase_client()
    if not supabase:
        return {"error": "Database not configured"}

    try:
        query = (
            supabase.table("sessions")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
        )
        if user_id:
            query = query.eq("user_id", user_id)

        response = query.execute()
        return {"sessions": response.data}
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return {"error": str(e)}


def get_session_details(session_id: str) -> Dict[str, Any]:
    """Retrieve full details for a single session."""
    if not _is_valid_uuid(session_id):
        return {"error": "Invalid session id format", "code": "bad_id"}

    supabase = get_supabase_client()
    if not supabase:
        return {"error": "Database not configured", "code": "db_unavailable"}

    try:
        sess_resp = (
            supabase.table("sessions")
            .select("*")
            .eq("id", session_id)
            .single()
            .execute()
        )
        if not sess_resp.data:
            return {"error": "Session not found", "code": "not_found"}
        return {"session": sess_resp.data}
    except Exception as e:
        # supabase-py raises APIError with PGRST116 when .single() finds no rows.
        msg = str(e)
        if "PGRST116" in msg or "0 rows" in msg or "Cannot coerce" in msg:
            return {"error": "Session not found", "code": "not_found"}
        logger.error(f"Error fetching session details: {e}")
        return {"error": "Database error while fetching session", "code": "db_error"}
