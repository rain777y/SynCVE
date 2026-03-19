"""
Integration test configuration.
These tests require:
- Backend running on localhost:5005
- GPU available (or CPU fallback)
- Supabase connection (from backend.env)
- Gemini API key (for report tests)
"""
import pytest
import requests
import os
import base64
from pathlib import Path

BACKEND_URL = os.getenv("SYNCVE_TEST_BACKEND_URL", "http://localhost:5005")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TEST_ASSETS_DIR = Path(__file__).parent.parent / "assets"


@pytest.fixture(scope="session")
def backend_url():
    """Verify backend is running and return URL."""
    url = BACKEND_URL
    try:
        resp = requests.get(f"{url}/", timeout=10)
        assert resp.status_code == 200, f"Backend not healthy: {resp.status_code}"
    except requests.ConnectionError:
        pytest.skip(
            f"Backend not running at {url}. Start it first with scripts\\start_backend.bat"
        )
    return url


@pytest.fixture(scope="session")
def test_face_image_base64():
    """Load a real test face image as base64."""
    # Try generated images first, then static fallback
    image_paths = [
        TEST_ASSETS_DIR / "generated" / "neutral_face.jpg",
        TEST_ASSETS_DIR / "generated" / "happy_face.jpg",
        TEST_ASSETS_DIR / "static" / "test_face_basic.jpg",
    ]
    for path in image_paths:
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    pytest.skip("No test face image available. Run create_test_images.py first.")


@pytest.fixture
def no_face_image_base64():
    """Image with no face for negative testing."""
    paths = [
        TEST_ASSETS_DIR / "generated" / "no_face.jpg",
        TEST_ASSETS_DIR / "static" / "test_no_face.jpg",
    ]
    for path in paths:
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    # Create a simple no-face image on the fly
    from PIL import Image
    import io

    img = Image.new("RGB", (640, 480), color=(0, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def session_id(backend_url, test_face_image_base64):
    """Create a real session and return its ID. Clean up after test."""
    resp = requests.post(f"{backend_url}/session/start", json={})
    assert resp.status_code == 200
    data = resp.json()
    sid = data.get("session_id")
    assert sid, f"No session_id in response: {data}"
    yield sid
    # Cleanup: stop session
    try:
        requests.post(f"{backend_url}/session/stop", json={"session_id": sid})
    except Exception:
        pass
