"""
Clinical report assembler (Axis 4).

Produces either a Markdown string or a PDF (via reportlab) summarising a
session in clinician-readable form:

    1. Header — session id, duration, dominant emotion, disclaimer.
    2. Top-K events — magnitude × confidence ranking, transition labels.
    3. Clinical metric block — valence trace summary, ABS, reactivity,
       suppression index, incongruence (if available), per-detector
       reliability.
    4. Limitations & ethical reminder.

The Markdown form is the source of truth; the PDF is rendered from a
flat structured representation derived from the same data.
"""

from __future__ import annotations

import os
import tempfile
from html import escape
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.backend.config import get_config


DISCLAIMER = (
    "Emotion predictions are inferred from facial expression patterns using "
    "machine learning. They do not represent definitive measures of internal "
    "emotional states. Results should be interpreted as probabilistic estimates "
    "and must not be used as a stand-alone diagnostic instrument."
)


def _fmt(v: Any, digits: int = 4, default: str = "—") -> str:
    if v is None:
        return default
    if isinstance(v, float):
        return f"{round(v, digits)}"
    return str(v)


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "—"
    return f"{round(float(v) * 100, 1)}%"


def _pdf_text(value: Any) -> str:
    """Return ReportLab Paragraph-safe text using the built-in Helvetica font."""
    text = str(value)
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2026": "...",
        "\u2022": "*",
        "\u2192": "->",
        "\u0394": "Delta",
        "`": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return escape(text)


def _rank_events(events: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    def score(ev: Dict[str, Any]) -> float:
        return float(ev.get("magnitude", 0.0)) * float(ev.get("confidence", 0.0))
    return sorted(events, key=score, reverse=True)[:top_k]


def _extract_frame_ledger(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = record.get("metadata") or {}
    raw_payload = record.get("raw_payload") or {}
    ledger = None
    if isinstance(metadata, dict):
        ledger = metadata.get("frame_ledger")
    if not ledger and isinstance(raw_payload, dict):
        ledger = raw_payload.get("frame_ledger")
    return ledger if isinstance(ledger, dict) else None


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _frame_quality_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ledgers = [_extract_frame_ledger(record) for record in records]
    ledgers = [ledger for ledger in ledgers if ledger]

    detector_counts: Counter = Counter()
    frame_refs: List[str] = []
    latencies: List[float] = []
    capture_times: List[datetime] = []
    low_confidence = fallback = face_missing = spoof_triggered = spoof_bypassed = 0
    smoothed_count = raw_smoothed_mismatch = 0

    for record, ledger in zip(records, [_extract_frame_ledger(r) for r in records]):
        if not isinstance(ledger, dict):
            ref = record.get("frame_ref")
            if ref:
                frame_refs.append(str(ref))
            continue

        flags = ledger.get("quality_flags") or {}
        low_confidence += int(bool(flags.get("low_confidence")))
        fallback += int(bool(flags.get("detection_fallback")))
        face_missing += int(not bool(flags.get("face_detected", True)))
        spoof_triggered += int(bool(flags.get("anti_spoofing_triggered")))
        spoof_bypassed += int(bool(flags.get("anti_spoofing_bypassed")))

        storage = ledger.get("storage") or {}
        ref = storage.get("frame_ref") or record.get("frame_ref")
        if ref:
            frame_refs.append(str(ref))

        timing = ledger.get("timing") or {}
        latency = timing.get("inference_latency_ms")
        if latency is not None:
            try:
                latencies.append(float(latency))
            except (TypeError, ValueError):
                pass

        ts = _parse_ts(timing.get("client_capture_ts")) or _parse_ts(record.get("captured_at"))
        if ts:
            capture_times.append(ts)

        model = ledger.get("model") or {}
        for detector in model.get("detector_backends") or []:
            detector_counts[str(detector)] += 1

        raw = ledger.get("raw") or {}
        smoothed = ledger.get("smoothed") or {}
        if smoothed:
            smoothed_count += 1
            raw_dom = raw.get("dominant_emotion")
            smooth_dom = smoothed.get("dominant_emotion")
            raw_smoothed_mismatch += int(bool(raw_dom and smooth_dom and raw_dom != smooth_dom))

    effective_fps = None
    if len(capture_times) >= 2:
        ordered = sorted(capture_times)
        duration = (ordered[-1] - ordered[0]).total_seconds()
        if duration > 0:
            effective_fps = (len(ordered) - 1) / duration

    return {
        "frame_rows": len(records),
        "ledger_rows": len(ledgers),
        "uploaded_keyframes": len(set(frame_refs)),
        "sample_keyframes": list(dict.fromkeys(frame_refs))[:5],
        "low_confidence_frames": low_confidence,
        "fallback_frames": fallback,
        "face_missing_frames": face_missing,
        "spoof_triggered_frames": spoof_triggered,
        "spoof_bypassed_frames": spoof_bypassed,
        "mean_inference_latency_ms": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "max_inference_latency_ms": max(latencies) if latencies else None,
        "effective_fps": effective_fps,
        "detector_counts": dict(detector_counts),
        "smoothed_frames": smoothed_count,
        "raw_smoothed_mismatch": raw_smoothed_mismatch,
    }


def build_markdown(report_data: Dict[str, Any]) -> str:
    """Render the structured report dict into a clinician-readable Markdown."""
    sid = report_data.get("session_id", "—")
    duration = report_data.get("duration_sec", 0.0)
    samples = report_data.get("samples", 0)
    metrics = report_data.get("clinical_metrics") or {}
    events = report_data.get("events") or []
    frame_quality = report_data.get("frame_quality") or {}
    fps = report_data.get("fps_estimate", 0.5)
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    md = []
    md.append(f"# SynCVE Clinical Session Report")
    md.append("")
    md.append(f"**Session:** `{sid}`")
    md.append(f"**Generated:** {generated_at}")
    md.append(f"**Duration:** {round(float(duration), 1)} s ({samples} smoothed frames @ {fps} fps)")
    md.append("")
    md.append("> " + DISCLAIMER)
    md.append("")

    md.append("## 1. Top Flagged Events")
    if not events:
        md.append("_No consensus events detected — the session was either short, very stable, or detection sensitivity is too strict._")
    else:
        md.append("| # | Frame | Time (s) | Transition | Δp | Conf. | Methods |")
        md.append("|---|-------|----------|------------|-----|-------|---------|")
        for i, ev in enumerate(_rank_events(events, top_k=10), start=1):
            f = ev.get("frame_idx", 0)
            t = round(f / max(fps, 1e-6), 2)
            transition = f"{ev.get('from_emotion', '?')} → {ev.get('to_emotion', '?')}"
            mag = _fmt(ev.get("magnitude"), 3)
            conf = _fmt(ev.get("confidence"), 3)
            methods = ", ".join(ev.get("methods", []) or [])
            md.append(f"| {i} | {f} | {t} | {transition} | {mag} | {conf} | {methods} |")
    md.append("")

    md.append("## 2. Clinical Metrics")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Valence mean | {_fmt(metrics.get('valence_mean'))} |")
    md.append(f"| Valence std | {_fmt(metrics.get('valence_std'))} |")
    drift = metrics.get('valence_drift_per_min')
    ci = metrics.get('valence_drift_ci95') or [None, None]
    drift_str = _fmt(drift)
    if ci[0] is not None and ci[1] is not None:
        drift_str += f"  (95% CI {_fmt(ci[0])} … {_fmt(ci[1])})"
    md.append(f"| Valence drift / min | {drift_str} |")
    md.append(f"| Affect blunting score | {_fmt(metrics.get('affect_blunting_score'))} |")
    md.append(f"| Reactivity (events/min) | {_fmt(metrics.get('reactivity_events_per_min'))} |")
    md.append(f"| Suppression index | {_fmt(metrics.get('suppression_index'))} |")
    md.append(f"| Incongruence index | {_fmt(metrics.get('incongruence_index'))} |")
    md.append(f"| High-confidence events | {metrics.get('high_confidence_event_count', 0)} / {metrics.get('event_count', 0)} |")
    md.append("")

    if metrics.get("expressive_range_per_emotion"):
        md.append("### Expressive range per emotion")
        md.append("")
        md.append("| Emotion | Range |")
        md.append("|---------|-------|")
        for e, r in sorted(metrics["expressive_range_per_emotion"].items()):
            md.append(f"| {e} | {_fmt(r)} |")
        md.append("")

    if metrics.get("affect_blunting_per_emotion"):
        md.append("### Per-emotion blunting")
        md.append("")
        md.append("| Emotion | Blunting (0=full range, 1=flat) |")
        md.append("|---------|---|")
        for e, b in sorted(metrics["affect_blunting_per_emotion"].items()):
            md.append(f"| {e} | {_fmt(b)} |")
        md.append("")

    if metrics.get("detector_reliability"):
        md.append("### Per-detector reliability (Axis 1C diagnostic)")
        md.append("")
        md.append("| Detector | Reliability (0..1) |")
        md.append("|----------|---|")
        for d, r in sorted(metrics["detector_reliability"].items()):
            md.append(f"| {d} | {_fmt(r)} |")
        md.append("")

    md.append("## 3. Data Quality & Provenance")
    md.append("")
    md.append("| Data signal | Value |")
    md.append("|-------------|-------|")
    md.append(f"| Persisted vision rows | {frame_quality.get('frame_rows', 0)} |")
    md.append(f"| Frame ledger rows | {frame_quality.get('ledger_rows', 0)} |")
    md.append(f"| Uploaded keyframes | {frame_quality.get('uploaded_keyframes', 0)} |")
    md.append(f"| Effective capture FPS | {_fmt(frame_quality.get('effective_fps'), 3)} |")
    md.append(f"| Mean inference latency | {_fmt(frame_quality.get('mean_inference_latency_ms'), 1)} ms |")
    md.append(f"| Max inference latency | {_fmt(frame_quality.get('max_inference_latency_ms'), 1)} ms |")
    md.append(f"| Low-confidence frames | {frame_quality.get('low_confidence_frames', 0)} |")
    md.append(f"| Detection fallback frames | {frame_quality.get('fallback_frames', 0)} |")
    md.append(f"| Face-missing frames | {frame_quality.get('face_missing_frames', 0)} |")
    md.append(f"| Raw/smoothed dominant mismatches | {frame_quality.get('raw_smoothed_mismatch', 0)} / {frame_quality.get('smoothed_frames', 0)} |")
    detectors = frame_quality.get("detector_counts") or {}
    if detectors:
        detector_text = ", ".join(f"{name}: {count}" for name, count in sorted(detectors.items()))
        md.append(f"| Detector backends observed | {detector_text} |")
    keyframes = frame_quality.get("sample_keyframes") or []
    if keyframes:
        md.append("")
        md.append("Uploaded keyframe refs: " + ", ".join(f"`{ref}`" for ref in keyframes))
    md.append("")

    md.append("## 4. Reaction Latencies (if triggers were supplied)")
    rl = report_data.get("reaction_latencies") or {}
    if rl.get("n", 0) > 0:
        md.append("")
        md.append(f"- N matched: {rl['n']}")
        md.append(f"- Mean latency: {rl.get('mean_ms', '—')} ms")
        md.append(f"- Std latency: {rl.get('std_ms', '—')} ms")
        md.append("")
        md.append("| Trigger | t_sec | Latency (ms) | Matched event |")
        md.append("|---------|-------|---|---|")
        for tr in rl.get("per_trigger", []):
            mt = tr.get("matched_event") or {}
            transition = ""
            if mt:
                transition = f"{mt.get('from_emotion','?')} → {mt.get('to_emotion','?')}"
            md.append(
                f"| {tr.get('word', tr.get('label', '—'))} | "
                f"{tr.get('t_sec', '—')} | "
                f"{tr.get('latency_ms', '—')} | "
                f"{transition} |"
            )
    else:
        md.append("_No triggers supplied for this report._")

    md.append("")
    md.append("## 5. Limitations & Ethics")
    md.append("- Cohort calibration: results are valid only for cohorts whose distribution matches the calibration set.")
    md.append("- Consent: all data was processed under the user-controlled session lifecycle; raw frames are evicted on TTL.")
    md.append("- This report is decision-support, not a diagnostic instrument.")

    md.append("")
    md.append("---")
    md.append(f"_Generated by SynCVE clinical_report at {generated_at}._")
    return "\n".join(md)


def build_pdf(markdown_text: str, *, output_dir: Optional[str] = None) -> str:
    """
    Render the Markdown report to a PDF using reportlab. Returns the path
    on disk (under ``output_dir`` or a tmp dir).

    The rendering is intentionally simple — clinical reports prioritise
    legibility and unambiguous structure over typographic polish.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
        )
        from reportlab.lib import colors
    except ImportError as e:
        raise RuntimeError(
            "reportlab is required for PDF clinical reports. "
            "Add `reportlab>=4.0,<5.0` to requirements.txt and install."
        ) from e

    out_dir = output_dir or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    # Include a uuid suffix so concurrent requests don't collide on the same path.
    import uuid as _uuid
    _tag = f"{int(datetime.utcnow().timestamp())}_{_uuid.uuid4().hex[:8]}"
    out_path = os.path.join(out_dir, f"syncve_clinical_{_tag}.pdf")

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=12, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, spaceAfter=4)
    note = ParagraphStyle("note", parent=body, textColor=colors.HexColor("#555555"))

    story = []
    table_rows: List[List[str]] = []
    in_table = False
    skip_separator = False

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                # Markdown header separator row
                skip_separator = True
                continue
            in_table = True
            table_rows.append(cells)
            continue
        else:
            if in_table and table_rows:
                story.append(_render_table(table_rows, Table, TableStyle, colors))
                story.append(Spacer(1, 4 * mm))
                table_rows = []
            in_table = False

        if line.startswith("# "):
            story.append(Paragraph(_pdf_text(line[2:].strip()), h1))
        elif line.startswith("## "):
            story.append(Paragraph(_pdf_text(line[3:].strip()), h2))
        elif line.startswith("### "):
            story.append(Paragraph(_pdf_text(line[4:].strip()), h3))
        elif line.startswith("> "):
            story.append(Paragraph("<i>" + _pdf_text(line[2:].strip()) + "</i>", note))
        elif line.startswith("- "):
            story.append(Paragraph("* " + _pdf_text(line[2:].strip()), body))
        elif line.strip() == "---":
            story.append(Spacer(1, 4 * mm))
        elif line.strip() == "":
            story.append(Spacer(1, 2 * mm))
        else:
            story.append(Paragraph(_pdf_text(line), body))

    if in_table and table_rows:
        story.append(_render_table(table_rows, Table, TableStyle, colors))

    doc.build(story)
    return out_path


def _render_table(rows, Table, TableStyle, colors):
    if not rows:
        return None
    safe_rows = [[_pdf_text(cell) for cell in row] for row in rows]
    t = Table(safe_rows, hAlign="LEFT")
    t.setStyle(
        TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#888888")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    return t


def build_clinical_report(session_id: str, *, fmt: str = "md") -> Dict[str, Any]:
    """
    Top-level builder used by routes. ``fmt`` is "md" (returns rendered text)
    or "pdf" (returns the on-disk path under ``pdf_path``).
    """
    from src.backend.session_manager import (
        fetch_emotion_logs,
        get_clinical_metrics,
        get_temporal_summary,
    )

    cfg = get_config()
    metrics_payload = get_clinical_metrics(session_id)
    if isinstance(metrics_payload, dict) and metrics_payload.get("error"):
        raise ValueError(metrics_payload["error"])

    temporal = get_temporal_summary(session_id) or {}
    samples = temporal.get("frame_count", metrics_payload.get("samples", 0))
    duration = metrics_payload.get("duration_sec") or (
        samples / max(cfg.temporal.fps_estimate, 1e-6)
    )

    report_data = {
        "session_id": session_id,
        "samples": samples,
        "duration_sec": duration,
        "fps_estimate": cfg.temporal.fps_estimate,
        "clinical_metrics": {
            k: v for k, v in metrics_payload.items()
            if k not in {"events", "session_id", "reaction_latencies", "valence_trace"}
        },
        "events": metrics_payload.get("events") or [],
        "reaction_latencies": metrics_payload.get("reaction_latencies"),
        "frame_quality": _frame_quality_summary(fetch_emotion_logs(session_id, limit=1000)),
    }

    md = build_markdown(report_data)
    if fmt == "md":
        return {
            "session_id": session_id,
            "format": "md",
            "report_markdown": md,
            "report_data": report_data,
        }
    pdf_path = build_pdf(md)
    return {
        "session_id": session_id,
        "format": "pdf",
        "pdf_path": pdf_path,
        "report_data": report_data,
    }
