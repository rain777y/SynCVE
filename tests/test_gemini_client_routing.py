"""
Lightweight unit tests for gemini_client routing logic.

These do **not** make any network calls — they only verify that the SDK
client is constructed with the correct (vertexai/api_key/project) tuple
based on the credential format and the GENAI_MODE env var. Any real call
is covered by the e2e suite under ``tests/e2e/``.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

import src.backend.gemini_client as gc
from src.backend.config import (
    AppConfig, ServerConfig, GPUConfig, DeepFaceConfig, SupabaseConfig,
    GeminiConfig, ClientConfig, TemporalConfig, PreprocessConfig,
    EventsConfig, FusionConfig, ClinicalConfig,
)


def _stub_cfg(api_key: str = "", sa_path: str = "") -> AppConfig:
    return AppConfig(
        server=ServerConfig(),
        gpu=GPUConfig(),
        deepface=DeepFaceConfig(),
        supabase=SupabaseConfig(),
        gemini=GeminiConfig(api_key=api_key, service_account_path=sa_path),
        client=ClientConfig(),
        temporal=TemporalConfig(),
        preprocess=PreprocessConfig(),
        events=EventsConfig(),
        fusion=FusionConfig(),
        clinical=ClinicalConfig(),
    )


@pytest.fixture(autouse=True)
def _reset_singleton():
    gc._initialized = False
    gc._genai_client = None
    yield
    gc._initialized = False
    gc._genai_client = None


def _capture_kwargs(kwarg_log: list):
    """Return a fake genai.Client constructor that records kwargs."""
    class _FakeClient:
        def __init__(self, **kwargs):
            kwarg_log.append(dict(kwargs))
    return _FakeClient


def test_vertex_express_key_uses_vertexai_true(monkeypatch):
    monkeypatch.delenv("GENAI_MODE", raising=False)
    cfg = _stub_cfg(api_key="AQ.Ab8RN6_test_key")
    monkeypatch.setattr(gc, "get_config", lambda: cfg)

    captured: list = []
    fake = _capture_kwargs(captured)
    monkeypatch.setattr("google.genai.Client", fake)
    gc._ensure_initialized()
    assert captured, "genai.Client must be invoked"
    kwargs = captured[0]
    assert kwargs.get("vertexai") is True
    assert kwargs.get("api_key") == "AQ.Ab8RN6_test_key"
    # Vertex Express does not accept project/location alongside api_key
    assert "project" not in kwargs
    assert "location" not in kwargs


def test_ai_studio_key_skips_vertexai(monkeypatch):
    monkeypatch.delenv("GENAI_MODE", raising=False)
    cfg = _stub_cfg(api_key="AIzaSyTEST_legacy_studio_key")
    monkeypatch.setattr(gc, "get_config", lambda: cfg)
    captured: list = []
    fake = _capture_kwargs(captured)
    monkeypatch.setattr("google.genai.Client", fake)
    gc._ensure_initialized()
    kwargs = captured[0]
    assert "vertexai" not in kwargs or kwargs.get("vertexai") is None
    assert kwargs.get("api_key") == "AIzaSyTEST_legacy_studio_key"


def test_genai_mode_force_ai_studio_overrides_prefix(monkeypatch):
    monkeypatch.setenv("GENAI_MODE", "ai_studio")
    cfg = _stub_cfg(api_key="AQ.Ab8_TEST_force_studio")
    monkeypatch.setattr(gc, "get_config", lambda: cfg)
    captured: list = []
    fake = _capture_kwargs(captured)
    monkeypatch.setattr("google.genai.Client", fake)
    gc._ensure_initialized()
    kwargs = captured[0]
    assert "vertexai" not in kwargs or kwargs.get("vertexai") is None


def test_no_credentials_logs_warning_and_does_not_raise(monkeypatch):
    monkeypatch.delenv("GENAI_MODE", raising=False)
    cfg = _stub_cfg(api_key="", sa_path="")
    monkeypatch.setattr(gc, "get_config", lambda: cfg)
    # Initialization must be tolerant of missing creds.
    gc._ensure_initialized()
    assert gc._genai_client is None
