"""
E2E tests for video generation and video-based emotion analysis.

These tests use the Veo API to generate test videos, then extract frames
and run them through the emotion detection pipeline.

Requires: GEMINI_API_KEY environment variable.
Reference: dev/reference/libraries/google-genai/docs/12-gemini-video-understanding.md
"""

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "tests" / "artifacts"
VIDEOS_DIR = ARTIFACTS_DIR / "videos"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
SA_JSON = PROJECT_ROOT / "src" / "backend" / "sightline-backend-sa.json"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def veo_client():
    """Return a genai Client configured for video generation (Vertex AI or API key)."""
    from google import genai

    # Prefer Vertex AI with service account
    if SA_JSON.exists():
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(SA_JSON))
        return genai.Client(
            vertexai=True,
            project=os.getenv("GCP_PROJECT", "sightline-hackathon"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
        )
    # Fallback to API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        pytest.skip("No Vertex AI SA or GEMINI_API_KEY available")
    return genai.Client(api_key=api_key)


@pytest.fixture(scope="module")
def generated_video(veo_client) -> Path:
    """
    Generate a single short test video (or use cached).
    Returns path to .mp4 file.
    """
    from google.genai import types

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Use pre-generated artifact if available
    for candidate in ["happy_person.mp4", "e2e_test_happy.mp4"]:
        video_path = VIDEOS_DIR / candidate
        if video_path.exists() and video_path.stat().st_size > 10_000:
            return video_path

    # Generate on the fly
    video_path = VIDEOS_DIR / "e2e_test_happy.mp4"
    operation = veo_client.models.generate_videos(
        model="veo-3.0-fast-generate-001",
        prompt=(
            "Close-up portrait of a young woman with a genuine happy smile, "
            "warm golden-hour lighting, soft bokeh background. "
            "Camera: slow dolly forward."
        ),
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            duration_seconds=6,
            enhance_prompt=True,
            person_generation="allow_adult",
            aspect_ratio="16:9",
        ),
    )

    start = time.time()
    while not operation.done:
        if time.time() - start > 600:
            pytest.skip("Video generation timed out (>10min)")
        time.sleep(20)
        operation = veo_client.operations.get(operation)

    result = operation.response or getattr(operation, 'result', None)
    if not result or not getattr(result, 'generated_videos', None):
        pytest.fail(f"No generated videos in response: {operation}")
    generated = result.generated_videos[0]
    try:
        veo_client.files.download(file=generated.video)
        generated.video.save(str(video_path))
    except (ValueError, AttributeError):
        video_obj = generated.video
        if hasattr(video_obj, 'uri') and video_obj.uri:
            import urllib.request
            urllib.request.urlretrieve(video_obj.uri, str(video_path))
        elif hasattr(video_obj, 'save'):
            video_obj.save(str(video_path))
        else:
            pytest.fail(f"Cannot download video: {dir(video_obj)}")
    return video_path


@pytest.fixture(scope="module")
def video_frames(generated_video) -> list:
    """
    Extract frames from generated video using OpenCV.
    Returns list of (frame_index, numpy_array) tuples.
    """
    import cv2

    cap = cv2.VideoCapture(str(generated_video))
    if not cap.isOpened():
        pytest.fail(f"Cannot open video: {generated_video}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample 1 frame per second
    sample_interval = max(1, int(fps))

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval == 0:
            frames.append((idx, frame))
        idx += 1

    cap.release()
    assert len(frames) > 0, f"No frames extracted from {generated_video}"
    return frames


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVideoGeneration:
    """Test that Veo API produces valid video files."""

    @pytest.mark.slow
    def test_video_file_valid(self, generated_video):
        """Generated video file exists and has substantial size."""
        assert generated_video.exists()
        size = generated_video.stat().st_size
        assert size > 10_000, f"Video too small ({size} bytes), likely corrupt"

    @pytest.mark.slow
    def test_video_opencv_readable(self, generated_video):
        """OpenCV can open and read frames from the video."""
        import cv2
        cap = cv2.VideoCapture(str(generated_video))
        assert cap.isOpened(), "OpenCV cannot open the video"
        ret, frame = cap.read()
        assert ret, "Cannot read first frame"
        assert frame is not None
        assert frame.shape[0] > 100 and frame.shape[1] > 100, f"Frame too small: {frame.shape}"
        cap.release()

    @pytest.mark.slow
    def test_video_has_multiple_frames(self, generated_video):
        """Video contains at least 100 frames (~4s at 24fps)."""
        import cv2
        cap = cv2.VideoCapture(str(generated_video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert total >= 96, f"Expected >= 96 frames for 4s@24fps, got {total}"


class TestVideoEmotionPipeline:
    """Test emotion detection on video frames."""

    @pytest.mark.slow
    def test_frame_extraction(self, video_frames):
        """Frames are extracted at ~1fps from the video."""
        assert len(video_frames) >= 3, f"Expected >= 3 sampled frames, got {len(video_frames)}"

    @pytest.mark.slow
    def test_deepface_on_video_frame(self, video_frames):
        """DeepFace can detect a face and emotion in at least one video frame."""
        from deepface import DeepFace

        detected = False
        for idx, frame in video_frames[:5]:  # Try first 5 frames
            try:
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    detector_backend="retinaface",
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(result, list) and len(result) > 0:
                    emotions = result[0].get("emotion", {})
                    if emotions:
                        detected = True
                        print(f"  Frame {idx}: dominant={result[0].get('dominant_emotion')}")
                        break
            except Exception as e:
                print(f"  Frame {idx}: detection failed - {e}")
                continue

        assert detected, "DeepFace could not detect emotion in any video frame"

    @pytest.mark.slow
    def test_full_video_emotion_aggregate(self, video_frames):
        """Run emotion detection on all frames and aggregate results."""
        from deepface import DeepFace
        from src.backend.emotion_analytics import aggregate_emotion_metrics

        raw_results = []
        for idx, frame in video_frames:
            try:
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    detector_backend="retinaface",
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(result, list) and len(result) > 0:
                    raw_results.append({"emotion": result[0].get("emotion", {})})
            except Exception:
                continue

        assert len(raw_results) >= 1, "No frames produced emotion data"

        metrics = aggregate_emotion_metrics(raw_results)
        assert "dominant" in metrics
        assert "averages" in metrics
        assert metrics["samples"] >= 1
        print(f"  Aggregated {metrics['samples']} frames: dominant={metrics['dominant']}")

        # Save report artifact
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / "video_emotion_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Video Emotion Analysis Report\n")
            f.write(f"Frames analyzed: {metrics['samples']}\n")
            f.write(f"Dominant: {metrics['dominant']} ({metrics.get('dominant_score', 0):.2%})\n")
            f.write(f"Peak: {metrics.get('peak_emotion')} ({metrics.get('peak_score', 0):.2%})\n")
            f.write(f"Averages: {metrics.get('averages')}\n")
        print(f"  Report saved to {report_path}")


class TestVideoReportGeneration:
    """Test Gemini-powered report generation from video emotion data."""

    @pytest.mark.slow
    def test_visual_report_from_video_data(self, video_frames, veo_client):
        """Generate a visual report image from video emotion data."""
        from deepface import DeepFace
        from src.backend.emotion_analytics import aggregate_emotion_metrics, summarize_for_art_direction
        from src.backend.gemini_client import generate_text, generate_image

        # Collect emotion data from frames
        raw_results = []
        for idx, frame in video_frames[:5]:
            try:
                result = DeepFace.analyze(
                    img_path=frame, actions=["emotion"],
                    detector_backend="retinaface", enforce_detection=False, silent=True,
                )
                if isinstance(result, list) and len(result) > 0:
                    raw_results.append({"emotion": result[0].get("emotion", {})})
            except Exception:
                continue

        if not raw_results:
            pytest.skip("No emotion data from video frames")

        metrics = aggregate_emotion_metrics(raw_results)
        stats = summarize_for_art_direction(metrics)

        # Generate art direction prompt
        import json
        art_prompt = generate_text(
            f"Create an image generation prompt for a futuristic HUD dashboard "
            f"showing emotion data: {json.dumps(stats)}. "
            f"Dark background, glassmorphism, neon accents. "
            f"Return ONLY the prompt string."
        )
        assert len(art_prompt) > 20, f"Art prompt too short: {art_prompt}"

        # Generate the visual report image
        image_bytes = generate_image(art_prompt, aspect_ratio="16:9")
        assert len(image_bytes) > 1000, "Generated image too small"

        # Save artifact
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_img = REPORTS_DIR / "video_visual_report.png"
        report_img.write_bytes(image_bytes)
        print(f"  Visual report saved to {report_img} ({len(image_bytes):,} bytes)")
