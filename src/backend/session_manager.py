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
import threading
import time
import traceback
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Deque, Dict, List, Optional

from deepface.commons.logger import Logger

from src.backend.config import get_config
from src.backend.storage import get_supabase_client, upload_frame_to_storage
from src.backend.temporal_analysis import TemporalAnalyzer

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

# Session TTL: evict stale in-memory caches after 30 minutes
_SESSION_TTL_SECONDS = 30 * 60  # 30 minutes
_session_start_times: Dict[str, float] = {}


def _cleanup_session_cache(session_id: str) -> None:
    """Remove ALL in-memory data for a session to prevent leaks."""
    _last_upload_times.pop(session_id, None)
    _vision_cache.pop(session_id, None)
    _temporal_analyzers.pop(session_id, None)
    _session_start_times.pop(session_id, None)


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
    analyzer = _temporal_analyzers.get(session_id)
    if analyzer and analyzer._frame_count > 0:
        return analyzer.get_session_summary()
    return None


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
            # Initialize temporal analyzer for this session
            tcfg = get_config().temporal
            _temporal_analyzers[session_id] = TemporalAnalyzer(
                alpha=tcfg.ema_alpha,
                transition_threshold=tcfg.transition_threshold,
                volatility_window=tcfg.volatility_window,
                fps_estimate=tcfg.fps_estimate,
            )
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

        # Extract emotion data
        dominant_emotion = None
        emotions_data: Dict[str, Any] = {}
        confidence = 0.0

        if "results" in analysis_result and isinstance(analysis_result["results"], list):
            face_data = analysis_result["results"][0]
            dominant_emotion = face_data.get("dominant_emotion")
            emotions_data = face_data.get("emotion", {})
        else:
            dominant_emotion = analysis_result.get("dominant_emotion")
            emotions_data = analysis_result.get("emotion", {})

        if dominant_emotion and emotions_data:
            confidence = float(emotions_data.get(dominant_emotion, 0.0))

        payload: Dict[str, Any] = {
            "session_id": session_id,
            "dominant_emotion": dominant_emotion,
            "emotions": emotions_data,
            "confidence": confidence,
        }

        metadata_payload = metadata or {}

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

        payload["metadata"] = metadata_payload

        # Cache locally
        _cache_vision_sample(session_id, {
            "session_id": session_id,
            "dominant_emotion": dominant_emotion,
            "emotions": emotions_data,
            "confidence": confidence,
            "metadata": metadata_payload,
        })

        # Feed to temporal analyzer and capture smoothed scores
        smoothed_emotions = None
        analyzer = _temporal_analyzers.get(session_id)
        if analyzer and emotions_data:
            smoothed_emotions = analyzer.add_frame(emotions_data)

        if supabase:
            vision_payload = _to_json_safe({
                **payload,
                "raw_payload": payload_root if isinstance(payload_root, dict) else {"result": str(payload_root)},
            })
            supabase.table("vision_samples").insert(vision_payload).execute()
            try:
                supabase.table("sessions").update({
                    "last_event_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", session_id).execute()
            except Exception as e:
                logger.warn(f"Could not update session heartbeat: {e}")

        status = "logged" if supabase else "cached_only"
        result = {"status": status, "frame_ref": metadata_payload.get("frame_ref")}
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
    supabase = get_supabase_client()
    if not supabase:
        return {"error": "Database not configured"}

    try:
        sess_resp = (
            supabase.table("sessions")
            .select("*")
            .eq("id", session_id)
            .single()
            .execute()
        )
        if not sess_resp.data:
            return {"error": "Session not found"}
        return {"session": sess_resp.data}
    except Exception as e:
        logger.error(f"Error fetching session details: {e}")
        return {"error": str(e)}
