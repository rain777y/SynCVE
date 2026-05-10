"""
Shared fixtures for SynCVE backend test suite.

All external services (Supabase, DeepFace, TensorFlow, Gemini) are mocked so
tests run without a GPU, database, or network access.
"""

import base64
import os
import sys
import types as _types
from unittest.mock import MagicMock, patch
from io import BytesIO

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Environment stubs – must be set BEFORE any application module is imported
# ---------------------------------------------------------------------------
os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "fake-key")
os.environ.setdefault("GEMINI_API_KEY", "fake-gemini-key")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# Lightweight deepface stubs injected into sys.modules BEFORE any application
# module is imported.  This prevents the real deepface import chain from
# cascading into heavyweight detector backends (RetinaFace -> tf-keras, etc.)
# which may not be available in the test environment.
# ---------------------------------------------------------------------------

def _install_deepface_stubs():
    """Create minimal mock modules for deepface so imports resolve without
    pulling in TensorFlow, RetinaFace, or any other heavy dependency."""

    # Only install stubs if the real deepface cannot be fully imported.
    # This lets the suite run both in lightweight CI and in a full dev env.
    try:
        from deepface import DeepFace as _real  # noqa: F401
        # If we got here the real deepface is importable -- nothing to do.
        return
    except Exception:
        pass

    def _make_module(name, attrs=None):
        mod = _types.ModuleType(name)
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        mod.__path__ = []
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    # -- deepface (top-level) ------------------------------------------------
    _make_module("deepface")

    # -- deepface.commons & deepface.commons.logger --------------------------
    _make_module("deepface.commons")

    # Provide a Logger class that behaves like the real one (callable, returns
    # an object with info/warn/error/debug methods).
    class _StubLogger:
        def __init__(self, *a, **kw):
            pass
        def info(self, *a, **kw): pass
        def warn(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def debug(self, *a, **kw): pass

    _make_module("deepface.commons.logger", {"Logger": _StubLogger})

    # -- deepface.commons.image_utils ----------------------------------------
    def _load_image(img, *a, **kw):
        """Stub: return a tiny numpy array when asked to load an image."""
        if isinstance(img, np.ndarray):
            return img, "numpy"
        return np.zeros((64, 64, 3), dtype=np.uint8), "base64"

    def _load_image_from_file_storage(file):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    _make_module("deepface.commons.image_utils", {
        "load_image": _load_image,
        "load_image_from_file_storage": _load_image_from_file_storage,
    })

    # -- deepface.DeepFace ---------------------------------------------------
    _mock_deepface_obj = MagicMock()
    _mock_deepface_obj.__version__ = "0.0.99"
    _make_module("deepface.DeepFace", {
        "__version__": "0.0.99",
        "analyze": _mock_deepface_obj.analyze,
        "represent": _mock_deepface_obj.represent,
        "verify": _mock_deepface_obj.verify,
    })
    # Also expose DeepFace as an attribute on the top-level deepface module.
    sys.modules["deepface"].DeepFace = sys.modules["deepface.DeepFace"]

    # -- deepface.modules & deepface.modules.modeling (for gpu_utils) --------
    _make_module("deepface.modules")
    _make_module("deepface.modules.modeling", {"cached_models": {}})


_install_deepface_stubs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_jpeg_base64() -> str:
    """Return a valid base64-encoded 8x8 JPEG string (data-URI format)."""
    from PIL import Image

    img = Image.new("RGB", (8, 8), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    raw = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{raw}"


def _make_numpy_image(height: int = 64, width: int = 64, channels: int = 3) -> np.ndarray:
    """Return a small BGR numpy image (uint8)."""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_base64_image() -> str:
    """A tiny valid JPEG as a base64 data-URI."""
    return _make_tiny_jpeg_base64()


@pytest.fixture()
def sample_numpy_image() -> np.ndarray:
    """A small 64x64 BGR numpy image."""
    return _make_numpy_image(64, 64)


@pytest.fixture()
def low_res_numpy_image() -> np.ndarray:
    """A tiny 32x32 image that should trigger super-resolution upscaling."""
    return _make_numpy_image(32, 32)


@pytest.fixture()
def sample_session_id() -> str:
    return "test-session-abc123"


@pytest.fixture()
def sample_analysis_result():
    """A realistic DeepFace-style analysis result."""
    return {
        "results": [
            {
                "emotion": {
                    "angry": 2.5,
                    "disgust": 0.3,
                    "fear": 1.2,
                    "happy": 85.0,
                    "sad": 3.0,
                    "surprise": 5.0,
                    "neutral": 3.0,
                },
                "dominant_emotion": "happy",
                "region": {"x": 10, "y": 10, "w": 100, "h": 100},
                "face_confidence": 0.99,
            }
        ]
    }


@pytest.fixture()
def mock_deepface(monkeypatch):
    """Patch DeepFace.analyze / represent / verify everywhere they are imported."""
    mock = MagicMock()
    mock.analyze.return_value = [
        {
            "emotion": {
                "angry": 2.5,
                "disgust": 0.3,
                "fear": 1.2,
                "happy": 85.0,
                "sad": 3.0,
                "surprise": 5.0,
                "neutral": 3.0,
            },
            "dominant_emotion": "happy",
            "region": {"x": 10, "y": 10, "w": 100, "h": 100},
            "face_confidence": 0.99,
        }
    ]
    mock.represent.return_value = [{"embedding": [0.1] * 128, "face_confidence": 0.99}]
    mock.verify.return_value = {
        "verified": True,
        "distance": 0.25,
        "threshold": 0.40,
        "model": "VGG-Face",
        "similarity_metric": "cosine",
    }
    mock.__version__ = "0.0.99"

    monkeypatch.setattr("deepface.DeepFace.analyze", mock.analyze)
    monkeypatch.setattr("deepface.DeepFace.represent", mock.represent)
    monkeypatch.setattr("deepface.DeepFace.verify", mock.verify)
    # Also patch inside service module in case it imported directly
    monkeypatch.setattr("src.backend.service.DeepFace", mock)
    return mock


@pytest.fixture()
def mock_supabase(monkeypatch):
    """Replace the Supabase client singleton inside the storage module."""
    mock_client = MagicMock()

    # Default .execute() returns an object with .data attribute
    def _make_response(data=None):
        resp = MagicMock()
        resp.data = data if data is not None else []
        return resp

    # sessions table
    mock_table = MagicMock()
    mock_table.insert.return_value.execute.return_value = _make_response(
        [{"id": "mock-session-id", "status": "active"}]
    )
    mock_table.update.return_value.eq.return_value.execute.return_value = _make_response()
    mock_table.select.return_value.order.return_value.limit.return_value.execute.return_value = _make_response([])
    mock_table.select.return_value.eq.return_value.single.return_value.execute.return_value = _make_response(
        {"id": "mock-session-id", "status": "active", "metadata": {}}
    )

    mock_client.table.return_value = mock_table

    # Storage
    mock_storage = MagicMock()
    mock_storage.from_.return_value.upload.return_value = True
    mock_storage.from_.return_value.get_public_url.return_value = "https://fake.supabase.co/storage/v1/object/public/report.png"
    mock_storage.from_.return_value.list.return_value = []
    mock_storage.from_.return_value.download.return_value = b"\x89PNG"
    mock_client.storage = mock_storage

    monkeypatch.setattr("src.backend.storage._supabase_client", mock_client)
    monkeypatch.setattr("src.backend.storage._supabase_initialized", True)
    return mock_client


@pytest.fixture()
def mock_genai(monkeypatch):
    """Stub out all Gemini/genai calls inside session_manager.

    Uses the unified google-genai SDK only (legacy google-generativeai removed).
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"summary": "Mock summary", "recommendations": "Mock recommendations"}'
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock()
    mock_part.inline_data.data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    mock_response.parts = [mock_part]
    mock_client.models.generate_content.return_value = mock_response
    mock_client.models.count_tokens.return_value = MagicMock()

    # Patch the genai client in the gemini_client module (where it actually lives)
    monkeypatch.setattr("src.backend.gemini_client._genai_client", mock_client)
    monkeypatch.setattr("src.backend.gemini_client._initialized", True)
    return mock_client


@pytest.fixture()
def app(mock_deepface):
    """Flask application configured for testing.

    Imports are deferred so that patches are applied first.
    """
    # We need to patch DeepFace.__version__ for the home route and create_app
    with patch("deepface.DeepFace.__version__", "0.0.99"):
        from src.backend.routes import blueprint
        from flask import Flask
        from flask_cors import CORS

        test_app = Flask(__name__)
        test_app.config["TESTING"] = True
        CORS(test_app)
        test_app.register_blueprint(blueprint)
        return test_app


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()
