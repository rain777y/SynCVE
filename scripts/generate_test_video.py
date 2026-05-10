"""
Generate test videos via Gemini Veo API for SynCVE e2e testing.

Usage:
    python -m tests.artifacts.generate_test_video          # generate all
    python -m tests.artifacts.generate_test_video --list    # list scenarios only

Reference: dev/reference/libraries/google-genai/docs/12-gemini-video-understanding.md
           dev/reference/libraries/google-genai/docs/14-veo-video-generation-prompt-guide.md

Requires: GEMINI_API_KEY environment variable.
Output:   tests/artifacts/videos/
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=False)
    except ImportError:
        with open(_env_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GCP_PROJECT = os.getenv("GCP_PROJECT", "sightline-hackathon")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
SA_JSON = PROJECT_ROOT / "src" / "backend" / "sightline-backend-sa.json"
VIDEO_MODEL = os.getenv("VEO_MODEL", "veo-3.0-fast-generate-001")
VIDEOS_DIR = PROJECT_ROOT / "tests" / "artifacts" / "videos"
IMAGES_DIR = PROJECT_ROOT / "tests" / "artifacts" / "images"


def _make_client():
    """Create a genai Client using Vertex AI (service account) or API key."""
    from google import genai

    # Prefer Vertex AI with service account
    if SA_JSON.exists():
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(SA_JSON))
        return genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
        )
    # Fallback to API key
    if GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)
    return None

# ---------------------------------------------------------------------------
# Video scenarios: emotion-themed clips for testing the analysis pipeline
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "name": "happy_person",
        "prompt": (
            "Close-up portrait shot of a young woman with a genuine happy smile, "
            "warm golden-hour lighting, soft bokeh background. She laughs naturally. "
            "Camera: slow dolly forward. Audio: gentle laughter, birds chirping."
        ),
        "duration": 6,
        "description": "Happy emotion - for testing emotion detection on video frames",
    },
    {
        "name": "sad_person",
        "prompt": (
            "Medium close-up of a young man looking sad and contemplative, "
            "sitting by a rain-streaked window. Cool blue lighting, "
            "slight camera drift. Audio: soft rain, distant piano."
        ),
        "duration": 6,
        "description": "Sad emotion - for testing sadness detection pipeline",
    },
    {
        "name": "angry_person",
        "prompt": (
            "Close-up portrait of a person with a tense, angry expression, "
            "furrowed brows and clenched jaw. Dramatic side lighting with "
            "warm red/orange tones. Camera: static, slow zoom in. "
            "Audio: tense atmospheric hum."
        ),
        "duration": 6,
        "description": "Angry emotion - for testing anger detection",
    },
    {
        "name": "neutral_talking",
        "prompt": (
            "Medium shot of a calm person with neutral expression speaking to camera, "
            "well-lit office environment, natural daylight. Professional setting. "
            "Camera: static. Audio: calm speaking voice."
        ),
        "duration": 6,
        "description": "Neutral emotion - baseline for emotion detection",
    },
    {
        "name": "emotion_transition",
        "prompt": (
            "Close-up portrait of a person transitioning from a neutral expression "
            "to a surprised look, with widening eyes and raised eyebrows. "
            "Well-lit, plain background. Camera: static close-up. "
            "Audio: soft gasp of surprise."
        ),
        "duration": 8,
        "description": "Emotion transition - tests temporal emotion tracking",
    },
    {
        "name": "no_face_landscape",
        "prompt": (
            "Cinematic aerial drone shot of mountains and a flowing river at sunset. "
            "No people, no faces. Golden light reflecting on water. "
            "Camera: slow pan right. Audio: wind, flowing water."
        ),
        "duration": 6,
        "description": "No face - negative test for face detection on video",
    },
]


def generate_video(client, scenario: dict, output_dir: Path) -> Path:
    """
    Generate a single test video using Veo API.

    Returns the path to the saved .mp4 file.
    Raises on failure.
    """
    from google.genai import types

    name = scenario["name"]
    output_path = output_dir / f"{name}.mp4"

    # Skip if already generated and non-trivial
    if output_path.exists() and output_path.stat().st_size > 10_000:
        print(f"  [CACHED] {name} ({output_path.stat().st_size:,} bytes)")
        return output_path

    print(f"  [GENERATING] {name} ({scenario['duration']}s)...")
    start_time = time.time()

    operation = client.models.generate_videos(
        model=VIDEO_MODEL,
        prompt=scenario["prompt"],
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            duration_seconds=scenario["duration"],
            enhance_prompt=True,
            person_generation="allow_adult",
            aspect_ratio="16:9",
        ),
    )

    # Poll for completion (Veo is async, can take 11s-6min)
    poll_count = 0
    while not operation.done:
        poll_count += 1
        wait = min(20, 10 + poll_count * 2)  # progressive backoff
        print(f"    polling... (attempt {poll_count}, waiting {wait}s)")
        time.sleep(wait)
        operation = client.operations.get(operation)

        # Safety timeout: 10 minutes
        if time.time() - start_time > 600:
            raise TimeoutError(f"Video generation timed out after 10 minutes for {name}")

    elapsed = time.time() - start_time

    # Download and save — SDK uses .response or .result depending on version
    result = operation.response or getattr(operation, 'result', None)
    if not result or not getattr(result, 'generated_videos', None):
        raise RuntimeError(f"No generated videos in response: {operation}")
    generated = result.generated_videos[0]

    # Vertex AI: client.files.download not supported, use video URI directly
    try:
        client.files.download(file=generated.video)
        generated.video.save(str(output_path))
    except (ValueError, AttributeError):
        # Vertex AI path: download via video URI or raw bytes
        video_obj = generated.video
        uri = getattr(video_obj, 'uri', None)
        if uri:
            import urllib.request
            urllib.request.urlretrieve(uri, str(output_path))
        elif hasattr(video_obj, 'video_bytes') and video_obj.video_bytes:
            output_path.write_bytes(video_obj.video_bytes)
        elif hasattr(video_obj, 'save'):
            video_obj.save(str(output_path))
        else:
            raise RuntimeError(f"Cannot download video. Object attrs: {dir(video_obj)}")

    size = output_path.stat().st_size
    print(f"  [OK] {name}: {size:,} bytes, {elapsed:.1f}s")
    return output_path


def generate_all(skip_existing: bool = True) -> dict:
    """
    Generate all test video scenarios.

    Returns dict mapping scenario name to output Path.
    """
    client = _make_client()
    if not client:
        print("ERROR: No credentials. Need service account JSON or GEMINI_API_KEY.")
        sys.exit(1)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    failed = []

    print(f"Generating {len(SCENARIOS)} test videos to {VIDEOS_DIR}/")
    print(f"Model: {VIDEO_MODEL}")
    print(f"Auth: {'Vertex AI (SA)' if SA_JSON.exists() else 'API Key'}")
    print()

    for scenario in SCENARIOS:
        try:
            path = generate_video(client, scenario, VIDEOS_DIR)
            results[scenario["name"]] = path
        except Exception as e:
            print(f"  [FAIL] {scenario['name']}: {e}")
            failed.append(scenario["name"])
            # Continue with remaining scenarios
            continue

        # Rate limit courtesy delay
        time.sleep(3)

    print()
    print(f"Results: {len(results)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    return results


def list_scenarios():
    """Print all available scenarios."""
    print(f"{'Name':<25} {'Duration':<10} {'Description'}")
    print("-" * 80)
    for s in SCENARIOS:
        print(f"{s['name']:<25} {s['duration']}s{'':<7} {s['description']}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SynCVE test videos via Veo API")
    parser.add_argument("--list", action="store_true", help="List scenarios without generating")
    parser.add_argument("--only", type=str, help="Generate only this scenario (by name)")
    args = parser.parse_args()

    if args.list:
        list_scenarios()
    elif args.only:
        matching = [s for s in SCENARIOS if s["name"] == args.only]
        if not matching:
            print(f"Unknown scenario: {args.only}")
            list_scenarios()
            sys.exit(1)
        client = _make_client()
        if not client:
            print("ERROR: No credentials.")
            sys.exit(1)
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        generate_video(client, matching[0], VIDEOS_DIR)
    else:
        generate_all()
