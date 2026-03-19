"""
Report generation pipelines for SynCVE backend.

Two modes controlled by settings.yml ``report.mode``:
  - "fast"  — Structured JSON report from aggregated metrics. Zero LLM calls
              on pause; one LLM call on stop (text summary). Sub-second response.
  - "full"  — Fast report + AI-generated dashboard image (2 LLM calls + storage upload).

Public API:
  generate_fast_report(session_id, ...)   → always available, instant
  generate_visual_report_v3(session_id, ...)  → only when mode="full"
  generate_report(session_id)             → text summary on stop (1 LLM call)
  generate_emotion_report(session_id, ...)  → two-stage keyframe pipeline
"""
import json
import time
from typing import Any, Dict, List, Optional

from deepface.commons.logger import Logger

from src.backend.config import get_config
from src.backend.emotion_analytics import aggregate_emotion_metrics, summarize_for_art_direction
from src.backend.gemini_client import (
    generate_image,
    generate_text,
    generate_multimodal,
    get_genai_client,
    to_image_part,
)
from src.backend.storage import (
    download_from_supabase,
    get_public_url,
    list_files,
    upload_to_supabase,
    get_supabase_client,
)

logger = Logger()


# ============================================================================
# Fast Report (zero LLM calls — instant structured data)
# ============================================================================

def generate_fast_report(
    session_id: str,
    raw_vision_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generate a structured report from aggregated emotion metrics.
    No LLM calls — returns in milliseconds.

    Returns:
        Dict with session_id, metrics, stats_summary, emotion_timeline,
        and a human-readable text_summary built from the data.
    """
    from src.backend.session_manager import fetch_emotion_logs, persist_aggregate_snapshot, get_temporal_summary

    if not session_id:
        raise ValueError("session_id is required")

    vision_data = raw_vision_data or fetch_emotion_logs(session_id)
    if not vision_data:
        raise ValueError(
            f"No vision data for session {session_id}; "
            "call /analyze with session_id before generating a report."
        )

    metrics = aggregate_emotion_metrics(vision_data)
    stats = summarize_for_art_direction(metrics)

    # Temporal analysis data (None if no temporal data available)
    temporal = get_temporal_summary(session_id)

    # Build a readable summary without any LLM call
    dominant = metrics.get("dominant", "unknown")
    dom_score = metrics.get("dominant_score", 0)
    peak = metrics.get("peak_emotion", dominant)
    peak_score = metrics.get("peak_score", 0)
    samples = metrics.get("samples", 0)
    averages = metrics.get("averages", {})

    # Emotion timeline from raw data (sampled)
    timeline = []
    step = max(1, len(vision_data) // 20)
    for i, entry in enumerate(vision_data):
        if i % step == 0:
            emo = entry.get("dominant_emotion") or ""
            if not emo:
                scores = entry.get("emotions") or entry.get("emotion") or {}
                if scores:
                    emo = max(scores, key=scores.get)
            ts = entry.get("captured_at") or entry.get("created_at") or ""
            timeline.append({"t": ts, "emotion": emo})

    # Top emotions ranked
    ranked = sorted(averages.items(), key=lambda kv: kv[1], reverse=True)

    text_summary = (
        f"Over {samples} frames, the dominant emotion was {dominant} "
        f"({dom_score:.0%}). Peak expression: {peak} ({peak_score:.0%}). "
        f"Distribution: {', '.join(f'{e} {s:.0%}' for e, s in ranked[:4])}."
    )

    persist_aggregate_snapshot(session_id, metrics)

    return {
        "session_id": session_id,
        "report_mode": "fast",
        "metrics": metrics,
        "stats_summary": stats,
        "text_summary": text_summary,
        "emotion_ranking": [{"emotion": e, "score": round(s, 4)} for e, s in ranked],
        "emotion_timeline": timeline,
        "samples": samples,
        "temporal": temporal,
    }


# ============================================================================
# Visual Dashboard Image (optional, mode="full" only)
# ============================================================================

def generate_visual_report_v3(
    session_id: str,
    raw_vision_data: Optional[List[Dict[str, Any]]] = None,
    *,
    aspect_ratio: Optional[str] = None,
    style_preset: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Data-to-Visual Image Pipeline (v3.0):
      1) Generate fast report (instant)
      2) Flash-Lite (Art Director) writes an image prompt
      3) Image model renders the final image
      4) Image is uploaded to storage; URL is returned

    This is expensive (2 LLM calls + upload). Only used when report.mode="full".
    """
    from src.backend.session_manager import fetch_emotion_logs, persist_aggregate_snapshot

    if not session_id:
        raise ValueError("session_id is required")

    supabase = get_supabase_client()
    if not supabase:
        raise ValueError("Database not configured")
    if not get_genai_client():
        raise ValueError("Gemini client not configured")

    cfg = get_config().gemini
    ar = aspect_ratio or cfg.visual_aspect_ratio
    style = style_preset or cfg.visual_style_preset

    vision_data = raw_vision_data or fetch_emotion_logs(session_id)
    if not vision_data:
        raise ValueError(
            f"No vision data available for session {session_id}; "
            "ensure /analyze is called with a session_id before pausing."
        )

    metrics = aggregate_emotion_metrics(vision_data)
    stats = summarize_for_art_direction(metrics)
    art_prompt = _run_flash_art_director(stats)

    styled_prompt = f"[Style: {style}] {art_prompt}" if style else art_prompt
    image_bytes = generate_image(styled_prompt, aspect_ratio=ar)

    persist_aggregate_snapshot(session_id, metrics)

    # Upload
    timestamp = int(time.time() * 1000)
    report_path = f"{session_id}/reports/image_v3_{timestamp}.png"
    upload_to_supabase(report_path, image_bytes, content_type="image/png")
    public_url = get_public_url(report_path)

    return {
        "session_id": session_id,
        "report_mode": "full",
        "metrics": metrics,
        "stats_summary": stats,
        "image_prompt": art_prompt,
        "storage_path": report_path,
        "public_url": public_url,
    }


# ============================================================================
# Text Report (stop-session flow — 1 LLM call)
# ============================================================================

def generate_report(session_id: str) -> Dict[str, str]:
    """
    Text summary report on session stop. Single LLM call.
    """
    from src.backend.session_manager import fetch_emotion_logs

    supabase = get_supabase_client()
    if not supabase:
        return {"summary": "Error: DB not connected", "recommendations": ""}

    cfg = get_config().gemini

    try:
        logs = fetch_emotion_logs(session_id)
        if not logs:
            return {"summary": "No data recorded.", "recommendations": "N/A"}

        total_frames = len(logs)
        emotion_counts: Dict[str, int] = {}
        timeline: List[str] = []
        step = max(1, total_frames // 50)

        for i, log in enumerate(logs):
            emo = log.get("dominant_emotion", "unknown")
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            if i % step == 0:
                created_at = log.get("created_at", "")
                timeline.append(f"{created_at}: {emo}")

        stats_str = ", ".join(
            f"{k}: {v} ({v / total_frames * 100:.1f}%)" for k, v in emotion_counts.items()
        )
        timeline_str = "\n".join(timeline[:30])  # cap timeline length

        prompt = f"""Analyze this emotion session data and return JSON only.

Session: {total_frames} frames analyzed.
Distribution: {stats_str}
Timeline (sampled): {timeline_str}

Return exactly this JSON (no markdown fences):
{{"summary": "<2-3 sentence emotional journey summary>", "recommendations": "<3 actionable bullet points>"}}"""

        text_response = generate_text(prompt, model=cfg.text_model)
        clean_text = text_response.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(clean_text)
        except Exception:
            return {"summary": text_response[:500], "recommendations": ""}

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return {"summary": "Error generating report.", "recommendations": str(e)}


# ============================================================================
# Two-stage Emotion Report (Aggregator -> Flash -> Pro Vision)
# ============================================================================

def generate_emotion_report(
    session_id: str,
    raw_vision_data: Optional[List[Dict[str, Any]]] = None,
    max_keyframes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full two-stage pipeline: aggregate → flash prompt → keyframes → vision model.
    """
    from src.backend.session_manager import fetch_emotion_logs, persist_aggregate_snapshot

    if not session_id:
        raise ValueError("session_id is required")

    supabase = get_supabase_client()
    if not supabase:
        raise ValueError("Database not configured")
    if not get_genai_client():
        raise ValueError("Gemini client not configured")

    cfg = get_config().gemini
    kf_limit = max_keyframes or cfg.keyframe_limit

    vision_data = raw_vision_data or fetch_emotion_logs(session_id)
    metrics = aggregate_emotion_metrics(vision_data)
    context_prompt = _run_flash_prompt(metrics)
    keyframes = _fetch_session_keyframes(session_id, limit=kf_limit)
    final_report = _run_pro_vision_report(context_prompt, keyframes)
    persist_aggregate_snapshot(session_id, metrics)

    return {
        "session_id": session_id,
        "metrics": metrics,
        "context_prompt": context_prompt,
        "keyframes_used": [frame["path"] for frame in keyframes],
        "report_markdown": final_report,
    }


# ============================================================================
# Internal pipeline helpers
# ============================================================================

def _run_flash_art_director(stats: Dict[str, Any]) -> str:
    """Turn numeric stats into an art-direction prompt for the image model."""
    dominant = stats.get("dominant", "neutral")
    score = stats.get("score", 0)
    secondary = stats.get("secondary")
    sec_score = stats.get("secondary_score", 0)
    peak = stats.get("peak_emotion", dominant)

    data_block = (
        f"Dominant: {dominant} ({score:.0%}) | "
        f"Secondary: {secondary or 'none'} ({sec_score:.0%}) | "
        f"Peak: {peak}"
    )

    prompt = f"""
Role: Expert AI Art Director for a real-time emotion analytics dashboard.
Data: {data_block}

Task: Write a single, precise image-generation prompt (no commentary).

Mandatory constraints:
- Dark background (#0d0d0d to #1a1a2e) so the image renders cleanly on a dark UI modal.
- Central radial gauge or arc meter displaying "{dominant.upper()}" with its percentage.
- A smaller secondary indicator for "{secondary or 'N/A'}" if present.
- Glassmorphism / frosted-glass panels with subtle neon glow edges.
- Emotion-driven accent palette:
    Sad/Fear  -> deep indigo + amber warning accents
    Happy/Surprise -> cyan + warm orange highlights
    Angry/Disgust -> crimson + dark steel
    Neutral -> cool silver-blue + soft teal
- Particle / waveform background element reflecting emotional intensity ({score:.0%}).
- A short HUD headline text (e.g., "STATUS: {dominant.upper()}").
- 16:9 landscape, no watermarks, no human faces, no real photographs.

Return ONLY the prompt string.
""".strip()

    return generate_text(prompt)


def _run_flash_prompt(metrics: Dict[str, Any]) -> str:
    """Convert metrics into a contextual prompt for the vision model."""
    dominant = metrics.get("dominant")
    dominant_score = metrics.get("dominant_score", 0.0)
    peak_emotion = metrics.get("peak_emotion")
    peak_score = metrics.get("peak_score", 0.0)
    averages = metrics.get("averages", {})

    top_pairs = sorted(averages.items(), key=lambda kv: kv[1], reverse=True)
    top_summary = ", ".join(f"{emo}: {score:.2f}" for emo, score in top_pairs[:4])

    prompt = f"""
You are crafting a focused instruction for a vision model that WILL receive images separately.
Do NOT assume or fabricate visuals.

Context metrics (normalized 0-1):
- Dominant: {dominant} ({dominant_score:.2f})
- Peak: {peak_emotion} ({peak_score:.2f})
- Distribution: {top_summary}
- Samples analyzed: {metrics.get('samples')}
- Noise floor applied: >= {metrics.get('noise_floor')}

Write a concise, actionable request for the vision model to validate or refute these emotional cues,
including what subtle facial regions/micro-expressions to check. Keep it under 120 words.
Return only the instruction text.
""".strip()

    return generate_text(prompt)


def _fetch_session_keyframes(
    session_id: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Download keyframe images from Supabase Storage for the vision model."""
    cfg = get_config().gemini
    kf_limit = limit or cfg.keyframe_limit

    folder_path = f"{session_id}/frames"
    files = list_files(folder_path, limit=kf_limit)

    keyframes: List[Dict[str, Any]] = []
    for file_obj in files:
        name = getattr(file_obj, "name", None)
        if not name and isinstance(file_obj, dict):
            name = file_obj.get("name")
        if not name:
            continue

        file_path = f"{folder_path}/{name}"
        data = download_from_supabase(file_path)
        if data:
            keyframes.append({"path": file_path, "bytes": data})

        if len(keyframes) >= kf_limit:
            break

    return keyframes


def _run_pro_vision_report(context_prompt: str, keyframes: List[Dict[str, Any]]) -> str:
    """Combine context prompt + keyframe images via the vision model for a Markdown report."""
    if not keyframes:
        raise ValueError("No keyframes available for vision report")

    contents: list = [context_prompt]
    for frame in keyframes:
        data = frame.get("bytes")
        if not isinstance(data, (bytes, bytearray)):
            continue
        contents.append(to_image_part(data))

    return generate_multimodal(
        contents,
        system_instruction=(
            "You are an empathetic visual analyst. Use the provided instruction as the primary lens. "
            "Summarize emotional evidence from the supplied keyframes, highlight agreement or conflicts "
            "with the metrics, and keep the output as concise Markdown with bullet points and a closing "
            "reassurance note. Avoid hallucinating scenes outside the provided frames."
        ),
    )
