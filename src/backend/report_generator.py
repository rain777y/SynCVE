"""
Report generation pipelines for SynCVE backend.

- Text summary report  (generate_report / generate_emotion_report)
- Visual dashboard image (generate_visual_report_v3)
- Two-stage Gemini pipelines  (aggregation -> prompt -> render)
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
# Text Report (legacy stop-session flow)
# ============================================================================

def generate_report(session_id: str) -> Dict[str, str]:
    """
    Standard text report: fetch logs, summarise, and ask Gemini for analysis.

    Returns:
        Dict with ``summary`` and ``recommendations`` keys.
    """
    from src.backend.session_manager import fetch_emotion_logs  # avoid circular at module level

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
        step = max(1, total_frames // 100)

        for i, log in enumerate(logs):
            emo = log.get("dominant_emotion", "unknown")
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            if i % step == 0:
                created_at = log.get("created_at", "")
                timeline.append(f"{created_at}: {emo}")

        stats_str = ", ".join(
            [f"{k}: {v} ({v / total_frames * 100:.1f}%)" for k, v in emotion_counts.items()]
        )
        timeline_str = "\n".join(timeline)

        prompt = f"""
        You are an AI Emotional Assistant analyzing a user's emotional state from a video session.

        **Session Statistics:**
        Total Duration: (Derived from logs)
        Emotion Distribution: {stats_str}

        **Timeline Samples:**
        {timeline_str}

        **Task:**
        1. Provide a brief 'Summary' of the user's emotional journey. Was it stable? Did it fluctuate?
        2. Provide 3 specific, actionable 'Recommendations' to help the user improve their mood or maintain positivity.

        **Output Format (JSON):**
        {{
            "summary": "...",
            "recommendations": "..."
        }}
        """

        if not cfg.api_key:
            return {"summary": "Gemini API Key missing.", "recommendations": "Cannot generate report."}

        text_response = generate_text(prompt, model=cfg.text_model)
        clean_text = text_response.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(clean_text)
        except Exception:
            result = {
                "summary": text_response[:500],
                "recommendations": "Raw output: " + text_response[:500],
            }

        return result

    except Exception as e:
        logger.error(f"Error calling Gemini: {e}")
        return {"summary": "Error generating report.", "recommendations": str(e)}


# ============================================================================
# Visual Dashboard Image (v3.0 Data-to-Visual pipeline)
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
      1) Aggregate emotion stats
      2) Flash-Lite (Art Director) writes an image prompt
      3) Image model renders the final image
      4) Image is uploaded to storage; URL is returned
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
    logger.info(
        f"Aggregated {metrics.get('samples')} samples for session {session_id}. "
        f"Dominant={metrics.get('dominant')} ({metrics.get('dominant_score', 0.0):.2f}) "
        f"Peak={metrics.get('peak_emotion')} ({metrics.get('peak_score', 0.0):.2f})"
    )

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
        "metrics": metrics,
        "stats_summary": stats,
        "image_prompt": art_prompt,
        "storage_path": report_path,
        "public_url": public_url,
    }


# ============================================================================
# Two-stage Emotion Reporting (Aggregator -> Flash -> Pro Vision)
# ============================================================================

def generate_emotion_report(
    session_id: str,
    raw_vision_data: Optional[List[Dict[str, Any]]] = None,
    max_keyframes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full two-stage pipeline:
      1) Aggregate raw emotion scores
      2) Flash-lite text model crafts a context prompt
      3) Pull keyframes from storage + run the pro vision model for a Markdown report
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
# Deprecated legacy visual report (kept for backward compat)
# ============================================================================

def generate_visual_report_auto(session_id: str) -> str:
    """
    DEPRECATED: legacy visual report (text-to-image only).
    Use ``generate_visual_report_v3`` instead.
    """
    logger.info(f"Generating visual report for session {session_id}...")

    try:
        text_report = generate_report(session_id)
        summary = text_report.get("summary", "User session analysis.")

        prompt_generation_prompt = f"""
        Based on this emotional summary: "{summary}", create a detailed prompt for an AI image generator to create a "Futuristic Emotion Analytics Dashboard".

        The image should feature:
        1.  Sleek data visualizations (waveforms, circular heatmaps, or connection graphs).
        2.  A color palette reflecting the dominant emotion (e.g., Golden/Yellow for Happy, Blue for Sad, Red for Angry).
        3.  High-tech, sci-fi UI elements (HUD style).
        4.  No text or minimal abstract text.

        The goal is to visualize the "Emotional Journey" as a piece of sophisticated technology monitoring.
        Output ONLY the prompt string.
        """

        visual_prompt = generate_text(prompt_generation_prompt)
        visual_prompt = visual_prompt.strip()
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        visual_prompt = "A visual report of the user's emotional session."

    try:
        image_bytes = generate_image(visual_prompt, aspect_ratio="16:9")

        timestamp = int(time.time() * 1000)
        report_path = f"{session_id}/reports/report_{timestamp}.png"
        upload_to_supabase(report_path, image_bytes, content_type="image/png")
        return get_public_url(report_path)
    except Exception as e:
        logger.error(f"Error generating visual report: {e}")
        return f"Error: {str(e)}"


# ============================================================================
# Internal pipeline helpers
# ============================================================================

def _run_flash_art_director(stats: Dict[str, Any]) -> str:
    """Turn numeric stats into an art-direction prompt for the image model."""
    prompt = f"""
Role: You are an expert AI Art Director.
Input: User emotion telemetry (JSON): {json.dumps(stats)}
Task: Convert this data into a precise "Image Generation Prompt" for a high-end generative AI.

Design Guidelines:
1. Mood & Color:
   - If Sad/Fear: Use Cool Blues, Deep Indigos, with Amber/Red warning accents. Dark, moody atmosphere.
   - If Happy/Surprised: Use Bright Cyans, Oranges, high brightness.
2. Layout: A futuristic HUD (Heads-Up Display) or Glassmorphism dashboard.
3. Data Visualization:
   - Describe a central gauge showing the Dominant Emotion with its percentage.
   - Describe a secondary indicator for the Secondary Emotion with its percentage.
4. Text Elements: Include a short, punchy headline text to be rendered (e.g., "STATUS: ANXIOUS DISTRESS").

Output Format:
Return ONLY the prompt string.
""".strip()

    return generate_text(prompt)


def _run_flash_prompt(metrics: Dict[str, Any]) -> str:
    """Use the fast/cheap text model to convert metrics into a contextual prompt for the vision model."""
    dominant = metrics.get("dominant")
    dominant_score = metrics.get("dominant_score", 0.0)
    peak_emotion = metrics.get("peak_emotion")
    peak_score = metrics.get("peak_score", 0.0)
    averages = metrics.get("averages", {})

    top_pairs = sorted(averages.items(), key=lambda kv: kv[1], reverse=True)
    top_summary = ", ".join([f"{emo}: {score:.2f}" for emo, score in top_pairs[:4]])

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
    """Combine context prompt + keyframe images via the vision model for the final Markdown report."""
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
