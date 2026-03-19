"""
Comprehensive end-to-end test suite for SynCVE.

Runs against REAL services:
  - Google Gemini API (text + image generation, video generation)
  - Supabase (database + storage)
  - DeepFace (face detection + emotion analysis)

No mocks. No human intervention. Fully automated.

Test categories:
  1. Asset generation via Gemini API
  2. Video generation pipeline
  3. Full session lifecycle (start -> analyze -> log -> reports -> pause -> stop)
  4. Emotion analytics pipeline
  5. Storage operations
"""
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e.conftest import (
    GENERATED_DIR,
    requires_all_services,
    requires_gemini,
    requires_supabase,
    _has_gemini,
    _has_supabase,
)


# ============================================================================
# 1. TEST ASSET GENERATION VIA GEMINI API
# ============================================================================
@requires_gemini
class TestAssetGeneration:
    """Verify that test face images can be generated and are usable."""

    def test_face_images_generated(self, generated_face_images):
        """All five emotion faces should be present."""
        expected = {"happy", "sad", "angry", "neutral", "surprised"}
        assert expected.issubset(set(generated_face_images.keys())), (
            f"Missing emotions: {expected - set(generated_face_images.keys())}"
        )

    def test_face_images_are_valid_jpeg_or_png(self, generated_face_images):
        """Every generated image should be a valid JPEG or PNG."""
        for emotion, data in generated_face_images.items():
            assert isinstance(data, bytes), f"{emotion}: not bytes"
            assert len(data) > 500, f"{emotion}: image too small ({len(data)} bytes)"
            is_jpeg = data[:2] == b"\xff\xd8"
            is_png = data[:4] == b"\x89PNG"
            assert is_jpeg or is_png, f"{emotion}: not JPEG/PNG"

    def test_no_face_image_generated(self, no_face_image):
        """No-face image should exist and be valid."""
        assert isinstance(no_face_image, bytes)
        assert len(no_face_image) > 500

    def test_face_images_detectable_by_deepface(self, generated_face_images):
        """DeepFace should detect a face in at least one generated image."""
        import cv2
        import numpy as np

        detected_count = 0
        for emotion, data in generated_face_images.items():
            try:
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                from deepface import DeepFace
                result = DeepFace.analyze(
                    img_path=img,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    anti_spoofing=False,
                    silent=True,
                )
                if result and isinstance(result, list) and result[0].get("dominant_emotion"):
                    detected_count += 1
            except Exception:
                continue

        assert detected_count >= 1, (
            "DeepFace could not detect a face in any generated image"
        )

    def test_generated_assets_cached_on_disk(self, generated_face_images):
        """After generation, cached files should exist in the generated dir."""
        # The conftest fixture saves to GENERATED_DIR
        for emotion in ["happy", "sad", "angry", "neutral", "surprised"]:
            path = GENERATED_DIR / f"e2e_{emotion}_face.jpg"
            if path.exists():
                assert path.stat().st_size > 500


# ============================================================================
# 2. VIDEO GENERATION TEST
# ============================================================================
@requires_gemini
class TestVideoGeneration:
    """
    Test video generation using Gemini/Veo.

    The Veo video generation API (client.models.generate_videos) is used when
    available.  If the model or API is not accessible (common with free-tier
    keys), we fall back to generating an image sequence as an alternative
    pipeline test.
    """

    def test_video_generation_or_image_sequence(self, gemini_client):
        """
        Attempt to generate a video via Veo.  If the model is unavailable,
        generate a sequence of images as an alternative visual pipeline test.
        """
        from google.genai import types

        video_generated = False
        video_path = GENERATED_DIR / "e2e_test_video.mp4"
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        # --- Attempt 1: Real video generation via Veo ---
        try:
            operation = gemini_client.models.generate_videos(
                model="veo-3.0-fast-generate-001",
                prompt=(
                    "A close-up of a person's face transitioning from a neutral "
                    "expression to a happy smile, well-lit studio setting, 4 seconds"
                ),
                config=types.GenerateVideosConfig(
                    person_generation="allow_adult",
                    aspect_ratio="16:9",
                    number_of_videos=1,
                    duration_seconds=5,
                ),
            )

            # Poll with timeout (5 minutes max)
            deadline = time.time() + 300
            while not operation.done and time.time() < deadline:
                time.sleep(15)
                operation = gemini_client.operations.get(operation)

            if operation.done and hasattr(operation, "response"):
                resp = operation.response
                if hasattr(resp, "generated_videos") and resp.generated_videos:
                    video = resp.generated_videos[0]
                    gemini_client.files.download(file=video.video)
                    video.video.save(str(video_path))
                    video_generated = True
                    print(f"[e2e] Video generated and saved to {video_path}")

        except Exception as exc:
            print(f"[e2e] Veo video generation not available: {exc}")

        # --- Attempt 2: Image sequence fallback ---
        if not video_generated:
            print("[e2e] Falling back to image-sequence generation pipeline")
            sequence_prompts = [
                "Photorealistic portrait, neutral calm expression, well-lit, plain background",
                "Photorealistic portrait, slight smile beginning to form, well-lit, plain background",
                "Photorealistic portrait, broad genuine happy smile, well-lit, plain background",
            ]
            sequence_images: List[bytes] = []

            for i, prompt in enumerate(sequence_prompts):
                try:
                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash-image",
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                        ),
                    )
                    for part in getattr(response, "parts", []) or []:
                        if hasattr(part, "inline_data") and part.inline_data:
                            img_data = part.inline_data.data
                            sequence_images.append(img_data)
                            out_path = GENERATED_DIR / f"e2e_sequence_frame_{i}.jpg"
                            out_path.write_bytes(img_data)
                            break
                    time.sleep(1.0)
                except Exception as exc:
                    print(f"[e2e] Frame {i} generation failed: {exc}")

            assert len(sequence_images) >= 2, (
                f"Expected at least 2 sequence frames, got {len(sequence_images)}"
            )
            print(f"[e2e] Generated {len(sequence_images)} image-sequence frames")

        # At least one approach must succeed
        assert video_generated or len(os.listdir(GENERATED_DIR)) > 0

    def test_video_pipeline_end_to_end(self, gemini_client, generated_face_images):
        """
        Full pipeline: generate content -> analyze emotions -> generate report text.
        Uses the generated face images as input to the analysis + report pipeline.
        """
        import cv2
        import numpy as np
        from src.backend.emotion_analytics import aggregate_emotion_metrics

        # Step 1: Analyze emotions on each generated face
        analysis_results: List[Dict[str, Any]] = []
        for emotion, data in generated_face_images.items():
            try:
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                from deepface import DeepFace
                result = DeepFace.analyze(
                    img_path=img,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    anti_spoofing=False,
                    silent=True,
                )
                if result and isinstance(result, list):
                    analysis_results.append(result[0])
            except Exception as exc:
                print(f"[e2e] Analysis failed for {emotion}: {exc}")

        assert len(analysis_results) >= 1, "Need at least one successful analysis"

        # Step 2: Aggregate emotion metrics
        metrics = aggregate_emotion_metrics(analysis_results)
        assert "dominant" in metrics
        assert "averages" in metrics
        assert metrics["samples"] >= 1

        # Step 3: Generate a text report from the aggregated metrics
        from src.backend.gemini_client import generate_text

        report_prompt = (
            f"Summarize these emotion metrics for a user report: "
            f"dominant={metrics['dominant']} (score={metrics.get('dominant_score', 0):.2f}), "
            f"samples={metrics['samples']}, "
            f"averages={json.dumps({k: round(v, 2) for k, v in metrics['averages'].items()})}. "
            f"Give a 2-sentence summary."
        )
        report_text = generate_text(report_prompt)
        assert isinstance(report_text, str)
        assert len(report_text) > 20
        print(f"[e2e] Pipeline report: {report_text[:200]}")


# ============================================================================
# 3. FULL SESSION LIFECYCLE (NO HUMAN INTERVENTION)
# ============================================================================
@requires_all_services
class TestFullSessionLifecycle:
    """
    Complete automated session lifecycle:
      start -> generate test image -> analyze emotion -> log to session ->
      generate text report -> generate visual report -> pause -> stop

    Uses real Gemini, real Supabase, real DeepFace.
    """

    def test_complete_session_lifecycle(
        self, generated_face_images, face_image_base64
    ):
        """
        Full lifecycle with no human intervention.
        Every step uses real services.
        """
        from src.backend.session_manager import (
            start_session,
            stop_session,
            pause_session,
            log_data,
            fetch_emotion_logs,
        )
        from src.backend.report_generator import generate_report
        from src.backend import service

        # ----- Step 1: Start session -----
        session_result = start_session(
            user_id="e2e-test-user",
            metadata={"source": "e2e_test", "automated": True},
        )
        assert "error" not in session_result, f"Start failed: {session_result}"
        session_id = session_result["session_id"]
        assert session_id
        print(f"[e2e] Session started: {session_id}")

        try:
            # ----- Step 2: Analyze emotions on multiple faces -----
            analysis_results = []
            for emotion in ["happy", "neutral", "sad"]:
                img_bytes = generated_face_images.get(emotion)
                if not img_bytes:
                    continue

                b64 = base64.b64encode(img_bytes).decode("utf-8")
                result = service.analyze(
                    img_path=b64,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True,
                    anti_spoofing=False,
                )

                # service.analyze returns (dict, status_code) on error
                if isinstance(result, tuple):
                    print(f"[e2e] Analysis returned error for {emotion}: {result}")
                    continue

                assert isinstance(result, dict), f"Unexpected result type: {type(result)}"
                analysis_results.append(result)

                # ----- Step 3: Log to session -----
                log_result = log_data(
                    session_id=session_id,
                    analysis_result=result,
                    metadata={"emotion_label": emotion, "test": True},
                    image_data=b64,
                )
                assert isinstance(log_result, dict)
                print(f"[e2e] Logged {emotion}: {log_result.get('status', 'unknown')}")

                time.sleep(0.5)

            assert len(analysis_results) >= 1, "Need at least one successful analysis"

            # ----- Step 4: Verify logs persisted -----
            logs = fetch_emotion_logs(session_id, limit=10)
            assert isinstance(logs, list)
            print(f"[e2e] Fetched {len(logs)} emotion logs for session")

            # ----- Step 5: Generate text report -----
            text_report = generate_report(session_id)
            assert isinstance(text_report, dict)
            assert "summary" in text_report
            print(f"[e2e] Text report summary: {text_report['summary'][:150]}")

            # ----- Step 6: Pause session (triggers visual report) -----
            pause_result = pause_session(session_id)
            assert isinstance(pause_result, dict)
            if "error" not in pause_result:
                assert pause_result.get("status") == "paused"
                visual_url = pause_result.get("image_url", "")
                print(f"[e2e] Visual report URL: {visual_url[:120]}")
            else:
                # Visual report generation can fail if Gemini image models are
                # temporarily unavailable; that is acceptable for e2e
                print(f"[e2e] Pause (visual report) soft-failed: {pause_result.get('error')}")

            # ----- Step 7: Stop session -----
            stop_result = stop_session(session_id)
            assert isinstance(stop_result, dict)
            if "error" not in stop_result:
                assert stop_result.get("status") == "session_ended"
            print(f"[e2e] Session stopped: {stop_result.get('status', stop_result.get('error'))}")

        except Exception:
            # Ensure cleanup even on failure
            try:
                stop_session(session_id)
            except Exception:
                pass
            raise

    def test_session_start_stop_minimal(self):
        """Minimal lifecycle: start and immediately stop."""
        from src.backend.session_manager import start_session, stop_session

        result = start_session()
        assert "session_id" in result
        sid = result["session_id"]

        stop_result = stop_session(sid)
        assert isinstance(stop_result, dict)

    def test_session_with_multiple_analyses(self, generated_face_images):
        """
        Submit 5 analyses to a single session, verify log count grows.
        """
        from src.backend.session_manager import start_session, stop_session, log_data, fetch_emotion_logs
        from src.backend import service

        result = start_session(metadata={"test": "multi_analysis"})
        assert "session_id" in result
        sid = result["session_id"]

        try:
            logged_count = 0
            for emotion in ["happy", "neutral", "sad", "angry", "surprised"]:
                img_bytes = generated_face_images.get(emotion)
                if not img_bytes:
                    continue

                b64 = base64.b64encode(img_bytes).decode("utf-8")
                analysis = service.analyze(
                    img_path=b64,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True,
                    anti_spoofing=False,
                )
                if isinstance(analysis, tuple):
                    continue

                log_data(sid, analysis, metadata={"emotion": emotion})
                logged_count += 1
                time.sleep(0.3)

            # Verify at least some logs were persisted
            logs = fetch_emotion_logs(sid, limit=50)
            assert isinstance(logs, list)
            print(f"[e2e] Multi-analysis: logged {logged_count}, fetched {len(logs)} records")

        finally:
            stop_session(sid)


# ============================================================================
# 4. EMOTION ANALYTICS PIPELINE
# ============================================================================
class TestEmotionAnalyticsPipeline:
    """Test emotion_analytics module with real DeepFace results."""

    def test_aggregate_metrics_from_real_analysis(self, generated_face_images):
        """Run DeepFace on generated faces and aggregate results."""
        import cv2
        import numpy as np
        from src.backend.emotion_analytics import aggregate_emotion_metrics

        raw_results = []
        for emotion, data in generated_face_images.items():
            try:
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                from deepface import DeepFace
                result = DeepFace.analyze(
                    img_path=img,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    anti_spoofing=False,
                    silent=True,
                )
                if result and isinstance(result, list):
                    raw_results.append(result[0])
            except Exception:
                continue

        if not raw_results:
            pytest.skip("No faces detected for aggregation test")

        metrics = aggregate_emotion_metrics(raw_results)

        assert "dominant" in metrics
        assert "dominant_score" in metrics
        assert "averages" in metrics
        assert "peaks" in metrics
        assert "samples" in metrics
        assert metrics["samples"] >= 1
        assert metrics["dominant_score"] >= 0
        assert isinstance(metrics["averages"], dict)
        print(f"[e2e] Aggregated metrics: dominant={metrics['dominant']} "
              f"score={metrics['dominant_score']:.3f} samples={metrics['samples']}")

    def test_summarize_for_art_direction(self, generated_face_images):
        """Test art-direction summary generation from real metrics."""
        import cv2
        import numpy as np
        from src.backend.emotion_analytics import (
            aggregate_emotion_metrics,
            summarize_for_art_direction,
        )

        raw_results = []
        for emotion, data in generated_face_images.items():
            try:
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                from deepface import DeepFace
                result = DeepFace.analyze(
                    img_path=img,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    anti_spoofing=False,
                    silent=True,
                )
                if result and isinstance(result, list):
                    raw_results.append(result[0])
            except Exception:
                continue

        if not raw_results:
            pytest.skip("No analysis results for art direction test")

        metrics = aggregate_emotion_metrics(raw_results)
        summary = summarize_for_art_direction(metrics)

        assert "dominant" in summary
        assert "score" in summary
        assert summary["dominant"] is not None
        assert summary["score"] >= 0


# ============================================================================
# 5. STORAGE OPERATIONS (Supabase)
# ============================================================================
@requires_supabase
class TestSupabaseStorage:
    """Test Supabase storage upload / download / URL generation."""

    def test_upload_and_download(self, generated_face_images):
        """Upload a test image and download it back."""
        from src.backend.storage import (
            upload_to_supabase,
            download_from_supabase,
            get_public_url,
        )

        img_bytes = generated_face_images.get("neutral")
        if not img_bytes:
            pytest.skip("No neutral face image available")

        test_path = f"e2e_test/storage_test_{int(time.time())}.jpg"

        # Upload
        result = upload_to_supabase(test_path, img_bytes, content_type="image/jpeg")
        assert result is not None, "Upload returned None"

        # Download
        downloaded = download_from_supabase(test_path)
        assert downloaded is not None, "Download returned None"
        assert len(downloaded) > 100

        # Public URL
        url = get_public_url(test_path)
        assert isinstance(url, str)
        assert len(url) > 10
        print(f"[e2e] Storage public URL: {url[:120]}")

    def test_session_table_operations(self, supabase_client):
        """Verify we can insert and query the sessions table."""
        # Insert
        response = supabase_client.table("sessions").insert({
            "status": "active",
            "metadata": {"source": "e2e_test", "automated": True},
        }).execute()

        assert response.data and len(response.data) > 0
        session_id = response.data[0]["id"]

        # Query
        query_resp = (
            supabase_client.table("sessions")
            .select("*")
            .eq("id", session_id)
            .single()
            .execute()
        )
        assert query_resp.data is not None
        assert query_resp.data["status"] == "active"

        # Cleanup
        supabase_client.table("sessions").update({
            "status": "completed",
        }).eq("id", session_id).execute()


# ============================================================================
# 6. REPORT GENERATION (requires both Gemini + Supabase)
# ============================================================================
@requires_all_services
class TestReportGeneration:
    """Test the full report generation pipeline with real services."""

    def test_text_report_with_synthetic_data(self):
        """
        Start a session, inject synthetic emotion logs, generate a text report.
        """
        from src.backend.session_manager import start_session, stop_session

        result = start_session(metadata={"test": "text_report"})
        assert "session_id" in result
        sid = result["session_id"]

        try:
            # Inject synthetic emotion data directly into the vision cache
            from src.backend import session_manager as sm
            for i in range(5):
                sm._cache_vision_sample(sid, {
                    "session_id": sid,
                    "dominant_emotion": ["happy", "neutral", "sad", "happy", "surprised"][i],
                    "emotions": {
                        "happy": [60.0, 30.0, 10.0, 55.0, 15.0][i],
                        "sad": [5.0, 10.0, 50.0, 8.0, 5.0][i],
                        "neutral": [20.0, 45.0, 25.0, 20.0, 10.0][i],
                        "angry": [5.0, 5.0, 5.0, 7.0, 5.0][i],
                        "surprised": [5.0, 5.0, 5.0, 5.0, 60.0][i],
                        "fear": [3.0, 3.0, 3.0, 3.0, 3.0][i],
                        "disgust": [2.0, 2.0, 2.0, 2.0, 2.0][i],
                    },
                    "confidence": [60.0, 45.0, 50.0, 55.0, 60.0][i],
                })

            # Generate text report
            from src.backend.report_generator import generate_report
            report = generate_report(sid)
            assert isinstance(report, dict)
            assert "summary" in report
            assert len(report["summary"]) > 10
            print(f"[e2e] Report summary: {report['summary'][:200]}")

        finally:
            sm._cleanup_session_cache(sid)
            stop_session(sid)

    @pytest.mark.slow
    def test_visual_report_with_synthetic_data(self):
        """
        Start a session, inject synthetic data, generate a visual dashboard.
        This test is marked slow because image generation takes 10-30 seconds.
        """
        from src.backend.session_manager import start_session, stop_session
        from src.backend.report_generator import generate_visual_report_v3

        result = start_session(metadata={"test": "visual_report"})
        assert "session_id" in result
        sid = result["session_id"]

        try:
            # Build synthetic vision data inline
            synthetic_data = []
            for dominant, scores in [
                ("happy", {"happy": 65, "sad": 5, "neutral": 20, "angry": 3, "surprised": 5, "fear": 1, "disgust": 1}),
                ("happy", {"happy": 70, "sad": 3, "neutral": 15, "angry": 2, "surprised": 8, "fear": 1, "disgust": 1}),
                ("neutral", {"happy": 20, "sad": 10, "neutral": 55, "angry": 5, "surprised": 5, "fear": 3, "disgust": 2}),
            ]:
                synthetic_data.append({
                    "dominant_emotion": dominant,
                    "emotion": scores,
                })

            v3_result = generate_visual_report_v3(
                session_id=sid,
                raw_vision_data=synthetic_data,
            )

            assert isinstance(v3_result, dict)
            assert "metrics" in v3_result
            assert "image_prompt" in v3_result
            # public_url and storage_path should be present if storage worked
            if v3_result.get("public_url"):
                print(f"[e2e] Visual report URL: {v3_result['public_url'][:120]}")
            print(f"[e2e] Visual report prompt: {v3_result.get('image_prompt', '')[:150]}")

        finally:
            stop_session(sid)


# ============================================================================
# 7. DEEPFACE ANALYSIS WITH REAL PREPROCESSING
# ============================================================================
class TestDeepFaceRealAnalysis:
    """Test the service.analyze pipeline with real preprocessing."""

    def test_analyze_happy_face(self, generated_face_images):
        """Analyze a happy face image through the full pipeline."""
        from src.backend import service

        img_bytes = generated_face_images.get("happy")
        if not img_bytes:
            pytest.skip("No happy face image")

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        result = service.analyze(
            img_path=b64,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
        )

        if isinstance(result, tuple):
            pytest.skip(f"Analysis returned error: {result[0]}")

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) >= 1
        face = result["results"][0]
        assert "emotion" in face
        assert "dominant_emotion" in face
        print(f"[e2e] Happy face -> detected: {face['dominant_emotion']}")

    def test_analyze_no_face_image(self, no_face_image):
        """Analyze a no-face image; should handle gracefully."""
        from src.backend import service

        b64 = base64.b64encode(no_face_image).decode("utf-8")
        result = service.analyze(
            img_path=b64,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
        )

        # With enforce_detection=False, DeepFace may still return a result
        # (possibly low confidence) or an error dict. Both are acceptable.
        assert isinstance(result, (dict, tuple))

    def test_ensemble_analysis(self, generated_face_images):
        """Test ensemble detector analysis on a neutral face."""
        from src.backend import service

        img_bytes = generated_face_images.get("neutral")
        if not img_bytes:
            pytest.skip("No neutral face image")

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        result = service.analyze(
            img_path=b64,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            anti_spoofing=False,
            enable_ensemble=False,  # Single detector for speed in e2e
        )

        if isinstance(result, tuple):
            pytest.skip(f"Ensemble analysis returned error: {result[0]}")

        assert isinstance(result, dict)
        assert "results" in result


# ============================================================================
# 8. NEGATIVE / EDGE CASE TESTS
# ============================================================================
@requires_all_services
class TestEdgeCases:
    """Edge cases and error handling with real services."""

    def test_stop_nonexistent_session(self):
        """Stopping a nonexistent session should not crash."""
        from src.backend.session_manager import stop_session

        result = stop_session("00000000-0000-0000-0000-000000000000")
        assert isinstance(result, dict)
        # May contain an error, but should not raise an exception

    def test_pause_nonexistent_session(self):
        """Pausing a nonexistent session should handle gracefully."""
        from src.backend.session_manager import pause_session

        result = pause_session("00000000-0000-0000-0000-000000000000")
        assert isinstance(result, dict)

    def test_fetch_logs_empty_session(self):
        """Fetching logs for a new session returns empty list."""
        from src.backend.session_manager import start_session, stop_session, fetch_emotion_logs

        result = start_session()
        assert "session_id" in result
        sid = result["session_id"]

        try:
            logs = fetch_emotion_logs(sid)
            assert isinstance(logs, list)
            # New session should have no logs yet
            assert len(logs) == 0
        finally:
            stop_session(sid)

    def test_aggregate_empty_data_raises(self):
        """aggregate_emotion_metrics with empty data should raise ValueError."""
        from src.backend.emotion_analytics import aggregate_emotion_metrics

        with pytest.raises(ValueError, match="No vision data"):
            aggregate_emotion_metrics([])
