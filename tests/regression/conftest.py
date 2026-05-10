"""
Regression test configuration.
Shares fixtures with integration tests but can be run independently.
"""
import pytest
import requests
import os
import base64
from pathlib import Path

BACKEND_URL = os.getenv("SYNCVE_TEST_BACKEND_URL", "http://localhost:5005")
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


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
    """Load a real test face image as data-URI base64."""
    image_paths = [
        ARTIFACTS_DIR / "images" / "neutral_face.jpg",
        ARTIFACTS_DIR / "images" / "happy_face.jpg",
        ARTIFACTS_DIR / "images" / "e2e_neutral_face.jpg",
        ARTIFACTS_DIR / "images" / "e2e_happy_face.jpg",
        ARTIFACTS_DIR / "images" / "test_face_basic.jpg",
    ]
    for path in image_paths:
        if path.exists():
            with open(path, "rb") as f:
                return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    pytest.skip("No test face image available. Run create_test_images.py first.")
