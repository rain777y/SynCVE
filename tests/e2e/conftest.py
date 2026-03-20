"""
Shared fixtures for SynCVE end-to-end tests.

These tests hit REAL external services (Gemini API, Supabase, DeepFace).
Required environment variables:
    GEMINI_API_KEY   - Google Gemini API key
    SUPABASE_URL     - Supabase project URL
    SUPABASE_KEY     - Supabase anon/service key

All tests skip gracefully when credentials are missing.
"""
import base64
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Auto-skip Vertex AI / Gemini 429 rate-limit errors instead of failing
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def skip_on_rate_limit():
    """Convert Vertex AI 429 RESOURCE_EXHAUSTED into a skip, not a failure."""
    try:
        yield
    except Exception as exc:
        exc_str = str(exc)
        if "429" in exc_str and ("RESOURCE_EXHAUSTED" in exc_str or "rate" in exc_str.lower()):
            pytest.skip(f"Vertex AI / Gemini rate limit (429) — {exc_str[:120]}")
        raise

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `from src.backend.*` works
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load secrets from .env into os.environ BEFORE importing project code
# ---------------------------------------------------------------------------
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=False)
    except ImportError:
        # Manual fallback if python-dotenv is unavailable
        with open(_env_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

# ---------------------------------------------------------------------------
# Credential accessors
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

ASSETS_DIR = PROJECT_ROOT / "tests" / "artifacts"
GENERATED_DIR = ASSETS_DIR / "images"


def _has_gemini() -> bool:
    """True if Gemini can be used: either an API key is set, or a service account JSON exists."""
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        return True
    # Also accept Vertex AI service account authentication
    sa_candidates = [
        PROJECT_ROOT / "sightline-backend-sa.json",
        PROJECT_ROOT / "src" / "backend" / "sightline-backend-sa.json",
    ]
    return any(p.exists() for p in sa_candidates)


def _has_supabase() -> bool:
    return bool(
        SUPABASE_URL
        and SUPABASE_KEY
        and "your-project-id" not in SUPABASE_URL
        and SUPABASE_KEY != "your_supabase_anon_key_here"
    )


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
requires_gemini = pytest.mark.skipif(
    not _has_gemini(), reason="GEMINI_API_KEY not set or placeholder"
)
requires_supabase = pytest.mark.skipif(
    not _has_supabase(), reason="SUPABASE_URL / SUPABASE_KEY not set or placeholder"
)
requires_all_services = pytest.mark.skipif(
    not (_has_gemini() and _has_supabase()),
    reason="Both GEMINI_API_KEY and SUPABASE credentials required",
)


# ---------------------------------------------------------------------------
# PIL fallback face generator
# ---------------------------------------------------------------------------
def _generate_pil_face(emotion: str = "neutral") -> bytes:
    """
    Generate a simple face image using PIL.
    Returns JPEG bytes that DeepFace can detect as a face.
    """
    from PIL import Image, ImageDraw

    width, height = 640, 480
    # Emotion-specific background tints
    tint_map = {
        "happy": (255, 240, 200),
        "sad": (180, 200, 230),
        "angry": (230, 180, 180),
        "surprised": (230, 230, 180),
        "neutral": (220, 220, 220),
        "fear": (200, 190, 220),
        "disgust": (200, 220, 190),
    }
    bg = tint_map.get(emotion, (220, 220, 220))
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)

    # Skin-tone ellipse for face outline
    skin = (210, 180, 140)
    draw.ellipse([180, 60, 460, 420], fill=skin, outline=(120, 90, 60), width=2)

    # Eyes
    eye_white = (255, 255, 255)
    pupil = (50, 50, 50)
    # Left eye
    draw.ellipse([240, 160, 300, 210], fill=eye_white, outline=(80, 60, 40))
    draw.ellipse([260, 170, 285, 200], fill=pupil)
    # Right eye
    draw.ellipse([340, 160, 400, 210], fill=eye_white, outline=(80, 60, 40))
    draw.ellipse([360, 170, 385, 200], fill=pupil)

    # Nose
    draw.polygon([(315, 230), (300, 290), (330, 290)], fill=(190, 155, 120))

    # Mouth varies by emotion
    if emotion == "happy":
        draw.arc([270, 290, 370, 370], 0, 180, fill=(180, 60, 60), width=3)
    elif emotion == "sad":
        draw.arc([270, 320, 370, 380], 180, 360, fill=(180, 60, 60), width=3)
    elif emotion == "angry":
        draw.line([(270, 340), (370, 340)], fill=(180, 60, 60), width=3)
        # Angry eyebrows
        draw.line([(240, 150), (300, 160)], fill=(80, 40, 20), width=3)
        draw.line([(400, 150), (340, 160)], fill=(80, 40, 20), width=3)
    elif emotion == "surprised":
        draw.ellipse([290, 310, 350, 370], outline=(180, 60, 60), width=3)
        # Wider eyes
        draw.ellipse([235, 150, 305, 215], fill=eye_white, outline=(80, 60, 40))
        draw.ellipse([335, 150, 405, 215], fill=eye_white, outline=(80, 60, 40))
    else:
        draw.line([(280, 340), (360, 340)], fill=(180, 60, 60), width=2)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _generate_pil_no_face() -> bytes:
    """Generate a landscape image with no face."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), color=(100, 160, 220))
    draw = ImageDraw.Draw(img)
    # Sky gradient
    for y in range(240):
        r = 100 + int(y * 0.3)
        g = 160 + int(y * 0.2)
        b = 220 - int(y * 0.1)
        draw.line([(0, y), (639, y)], fill=(r, g, b))
    # Ground
    draw.rectangle([0, 240, 639, 479], fill=(60, 130, 60))
    # Mountains
    draw.polygon([(100, 240), (200, 120), (300, 240)], fill=(90, 90, 100))
    draw.polygon([(250, 240), (380, 80), (500, 240)], fill=(110, 100, 110))
    # Sun
    draw.ellipse([500, 40, 560, 100], fill=(255, 220, 80))

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Session-scoped: Gemini-generated (or PIL-fallback) test assets
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def generated_face_images() -> Dict[str, bytes]:
    """
    Generate test face images via Gemini API. Falls back to PIL if API fails.
    Returns a dict mapping emotion name to JPEG bytes.
    """
    emotions = ["happy", "sad", "angry", "neutral", "surprised"]
    prompts = {
        "happy": (
            "A photorealistic portrait of a person with a genuine happy smile, "
            "well-lit, facing camera directly, plain white background"
        ),
        "sad": (
            "A photorealistic portrait of a person looking sad with downturned mouth, "
            "well-lit, facing camera, plain white background"
        ),
        "angry": (
            "A photorealistic portrait of a person with an angry expression and "
            "furrowed brows, well-lit, facing camera, plain white background"
        ),
        "neutral": (
            "A photorealistic portrait of a person with a calm neutral expression, "
            "well-lit, facing camera, plain white background"
        ),
        "surprised": (
            "A photorealistic portrait of a person with a surprised expression, "
            "wide eyes and open mouth, well-lit, facing camera, plain white background"
        ),
    }

    results: Dict[str, bytes] = {}
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Static fallback: use real e2e face images from artifacts (detectable by DeepFace)
    _artifacts_dir = PROJECT_ROOT / "tests" / "artifacts" / "images"

    def _static_or_pil(emotion_name: str) -> bytes:
        # Prefer emotion-specific real face images (these are DeepFace-detectable)
        for candidate in [
            _artifacts_dir / f"e2e_{emotion_name}_face.jpg",
            _artifacts_dir / f"e2e_neutral_face.jpg",
            _artifacts_dir / "test_face_basic.jpg",
        ]:
            if candidate.exists():
                return candidate.read_bytes()
        return _generate_pil_face(emotion_name)

    if _has_gemini():
        # Build a genai client: prefer Vertex AI SA, fall back to API key
        def _make_genai_client():
            from google import genai
            sa_candidates = [
                PROJECT_ROOT / "sightline-backend-sa.json",
                PROJECT_ROOT / "src" / "backend" / "sightline-backend-sa.json",
            ]
            for sa_path in sa_candidates:
                if sa_path.exists():
                    import os as _os
                    _os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(sa_path))
                    return genai.Client(vertexai=True, project="sightline-hackathon", location="us-central1")
            if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
                return genai.Client(api_key=GEMINI_API_KEY)
            return None

        try:
            from google.genai import types
            client = _make_genai_client()

            for emotion in emotions:
                cache_path = GENERATED_DIR / f"e2e_{emotion}_face.jpg"
                # Re-use cached file if it exists and is non-trivial
                if cache_path.exists() and cache_path.stat().st_size > 1000:
                    results[emotion] = cache_path.read_bytes()
                    continue

                if client is None:
                    results[emotion] = _static_or_pil(emotion)
                    continue

                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-image",
                        contents=[prompts[emotion]],
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                        ),
                    )
                    img_bytes = None
                    for part in getattr(response, "parts", []) or []:
                        if hasattr(part, "inline_data") and part.inline_data:
                            img_bytes = part.inline_data.data
                            break
                    if img_bytes:
                        cache_path.write_bytes(img_bytes)
                        results[emotion] = img_bytes
                    else:
                        results[emotion] = _static_or_pil(emotion)
                except Exception as exc:
                    print(f"[e2e] Gemini image gen failed for {emotion}: {exc}")
                    results[emotion] = _static_or_pil(emotion)

                # Small delay to avoid rate limits
                time.sleep(1.0)
        except Exception as exc:
            print(f"[e2e] Gemini client init failed: {exc}. Using static fallback.")
            for emotion in emotions:
                results[emotion] = _static_or_pil(emotion)
    else:
        for emotion in emotions:
            results[emotion] = _static_or_pil(emotion)

    return results


@pytest.fixture(scope="session")
def no_face_image() -> bytes:
    """Generate or load a no-face image for negative testing."""
    cache_path = GENERATED_DIR / "e2e_no_face.jpg"
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        return cache_path.read_bytes()

    # Use static no-face test image if available
    _static_no_face = PROJECT_ROOT / "tests" / "artifacts" / "images" / "test_no_face.jpg"
    if _static_no_face.exists():
        return _static_no_face.read_bytes()

    if _has_gemini():
        try:
            from google import genai
            from google.genai import types
            sa_candidates = [
                PROJECT_ROOT / "sightline-backend-sa.json",
                PROJECT_ROOT / "src" / "backend" / "sightline-backend-sa.json",
            ]
            _sa = next((str(p) for p in sa_candidates if p.exists()), None)
            if _sa:
                import os as _os
                _os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", _sa)
                client = genai.Client(vertexai=True, project="sightline-hackathon", location="us-central1")
            elif GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
                client = genai.Client(api_key=GEMINI_API_KEY)
            else:
                client = None
            if client:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=[
                        "A photorealistic landscape photo of mountains and a lake at sunset, "
                        "no people, no faces, nature only"
                    ],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )
                for part in getattr(response, "parts", []) or []:
                    if hasattr(part, "inline_data") and part.inline_data:
                        data = part.inline_data.data
                        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
                        cache_path.write_bytes(data)
                        return data
        except Exception as exc:
            print(f"[e2e] Gemini no-face gen failed: {exc}")

    data = _generate_pil_no_face()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


@pytest.fixture(scope="session")
def face_image_base64(generated_face_images) -> str:
    """Return a neutral face as data-URI base64 string for API calls."""
    img_bytes = generated_face_images.get("neutral") or _generate_pil_face("neutral")
    return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")


@pytest.fixture(scope="session")
def no_face_image_base64(no_face_image) -> str:
    """Return a no-face image as data-URI base64 string for API calls."""
    return "data:image/jpeg;base64," + base64.b64encode(no_face_image).decode("utf-8")


# ---------------------------------------------------------------------------
# Gemini client fixture (direct, not through Flask)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def gemini_client():
    """Return a real google.genai.Client instance (Vertex AI SA preferred, API key fallback)."""
    if not _has_gemini():
        pytest.skip("Neither GEMINI_API_KEY nor service account JSON found")
    from google import genai
    import os as _os
    sa_candidates = [
        PROJECT_ROOT / "sightline-backend-sa.json",
        PROJECT_ROOT / "src" / "backend" / "sightline-backend-sa.json",
    ]
    for sa_path in sa_candidates:
        if sa_path.exists():
            _os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(sa_path))
            return genai.Client(vertexai=True, project="sightline-hackathon", location="us-central1")
    # Fallback: API key
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Supabase client fixture (direct)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def supabase_client():
    """Return a real Supabase client. Skips if no credentials."""
    if not _has_supabase():
        pytest.skip("Supabase credentials not available")
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Reset config singletons so e2e tests get fresh state
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _reset_singletons():
    """Reset module-level singletons before the test session."""
    import src.backend.config as config_mod
    import src.backend.storage as storage_mod
    import src.backend.gemini_client as gemini_mod

    config_mod._config = None
    storage_mod._supabase_client = None
    storage_mod._supabase_initialized = False
    gemini_mod._genai_client = None
    gemini_mod._initialized = False
    yield
