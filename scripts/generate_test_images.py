"""
Generate test face images via Gemini API for SynCVE e2e testing.

Usage:
    python -m tests.artifacts.generate_test_images

Reference: dev/reference/libraries/google-genai/docs/01-gemini-image-generation-guide.md

Requires: GEMINI_API_KEY environment variable.
Output:   tests/artifacts/images/
"""

import os
import sys
import time
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
GCP_PROJECT = os.getenv("GCP_PROJECT", "sightline-hackathon")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
SA_JSON = PROJECT_ROOT / "src" / "backend" / "sightline-backend-sa.json"
IMAGES_DIR = PROJECT_ROOT / "tests" / "artifacts" / "images"


def _make_client():
    """Create genai Client: Vertex AI (service account) or API key."""
    from google import genai
    if SA_JSON.exists():
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(SA_JSON))
        return genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    if GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)
    return None

SCENARIOS = [
    ("happy_face", "Photorealistic portrait, genuine happy smile, well-lit, facing camera, plain white background"),
    ("sad_face", "Photorealistic portrait, person looking sad with downturned mouth, well-lit, facing camera, plain background"),
    ("angry_face", "Photorealistic portrait, angry expression, furrowed brows, well-lit, facing camera, plain background"),
    ("surprised_face", "Photorealistic portrait, surprised expression, wide eyes and open mouth, well-lit, plain background"),
    ("neutral_face", "Photorealistic portrait, calm neutral expression, well-lit, facing camera, plain white background"),
    ("fear_face", "Photorealistic portrait, fearful expression, wide eyes, tense, well-lit, facing camera, plain background"),
    ("disgust_face", "Photorealistic portrait, disgusted expression, wrinkled nose, well-lit, facing camera, plain background"),
    ("low_light_face", "Photorealistic portrait in dim low lighting, facing camera, warm shadows"),
    ("side_angle_face", "Photorealistic portrait at 30 degree side angle, well-lit, neutral expression"),
    ("multiple_faces", "Photorealistic photo of three people standing together facing camera, different expressions, well-lit"),
    ("no_face", "Photorealistic landscape with mountains, lake, and trees at sunset. No people, no faces."),
    ("small_face", "Photorealistic full body photo of a person standing far away, small in frame, large room"),
]


def _pil_fallback(name: str, output_path: Path):
    """Create a simple fallback image using PIL."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 480), color=(200, 180, 160))
    draw = ImageDraw.Draw(img)
    draw.ellipse([220, 80, 420, 400], fill=(220, 195, 175), outline=(180, 160, 140), width=2)
    draw.ellipse([270, 180, 310, 210], fill=(255, 255, 255))
    draw.ellipse([280, 188, 300, 202], fill=(60, 40, 20))
    draw.ellipse([330, 180, 370, 210], fill=(255, 255, 255))
    draw.ellipse([340, 188, 360, 202], fill=(60, 40, 20))
    draw.arc([285, 290, 355, 340], 0, 180, fill=(180, 80, 80), width=2)
    draw.text((10, 10), f"Fallback: {name}", fill=(255, 255, 255))
    img.save(str(output_path), "JPEG", quality=90)


def generate_all():
    client = _make_client()
    if not client:
        print("ERROR: No credentials. Need service account JSON or GEMINI_API_KEY.")
        sys.exit(1)

    from google.genai import types
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(SCENARIOS)} test images to {IMAGES_DIR}/")
    print(f"Model: {IMAGE_MODEL}")
    print()

    for name, prompt in SCENARIOS:
        output_path = IMAGES_DIR / f"{name}.jpg"
        if output_path.exists() and output_path.stat().st_size > 1000:
            print(f"  [CACHED] {name}")
            continue

        try:
            response = client.models.generate_content(
                model=IMAGE_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
            saved = False
            for part in getattr(response, "parts", []) or []:
                if hasattr(part, "inline_data") and part.inline_data:
                    output_path.write_bytes(part.inline_data.data)
                    print(f"  [OK] {name} ({output_path.stat().st_size:,} bytes)")
                    saved = True
                    break
            if not saved:
                print(f"  [FALLBACK] {name} - no image in response")
                _pil_fallback(name, output_path)
        except Exception as e:
            print(f"  [FALLBACK] {name} - {e}")
            _pil_fallback(name, output_path)

        time.sleep(2)  # rate limit

    print(f"\nDone. Images in {IMAGES_DIR}/")


if __name__ == "__main__":
    generate_all()
