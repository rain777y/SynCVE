"""
GeminiClient: wrapper around Google Generative AI SDK calls.
Encapsulates text generation, image generation, model fallback,
retry logic with exponential backoff, and error handling.
"""
import random
import time
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

from deepface.commons.logger import Logger

from src.backend.config import get_config

logger = Logger()

# ---------------------------------------------------------------------------
# Lazy SDK imports and client singleton
# ---------------------------------------------------------------------------

_genai_client = None
_initialized = False


def _ensure_initialized():
    """One-time initialization of the Google GenAI SDK.

    Auth priority:
    1. Vertex AI with service account JSON (GOOGLE_APPLICATION_CREDENTIALS)
    2. API key (GEMINI_API_KEY)
    """
    global _genai_client, _initialized
    if _initialized:
        return

    _initialized = True
    cfg = get_config().gemini

    try:
        import os
        from google import genai
        from pathlib import Path

        sa_path = cfg.service_account_path
        # Auto-detect SA JSON next to backend code
        if not sa_path:
            candidate = Path(__file__).parent / "sightline-backend-sa.json"
            if candidate.exists():
                sa_path = str(candidate)

        if sa_path and Path(sa_path).exists():
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", sa_path)
            project = cfg.gcp_project or "sightline-hackathon"
            location = cfg.gcp_location or "us-central1"
            _genai_client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
            logger.info(f"GenAI client initialized via Vertex AI (project={project}).")
        elif cfg.api_key:
            _genai_client = genai.Client(api_key=cfg.api_key)
            logger.info("GenAI client initialized via API key.")
        else:
            logger.warn("No Gemini credentials found (no SA JSON, no API key).")
    except Exception as e:
        logger.error(f"Failed to initialize GenAI client: {e}")


def get_genai_client():
    """Return the ``google.genai.Client`` singleton."""
    _ensure_initialized()
    return _genai_client


# ---------------------------------------------------------------------------
# Retry / resilience helpers
# ---------------------------------------------------------------------------

# Defaults read from env at import time via config; callers can override.
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_DELAY = 1.0


def call_with_retry(
    call_fn: Callable,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_delay: float = _DEFAULT_RETRY_BASE_DELAY,
    operation_name: str = "Gemini API call",
):
    """
    Execute a Gemini API call with exponential-backoff retry logic.

    ``call_fn`` is a zero-argument callable that performs the actual request.
    Retries on transient errors (rate limits, server errors, timeouts).
    """
    last_exception: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_fn()
        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            is_retryable = any(kw in err_str for kw in [
                "429", "rate limit", "resource exhausted",
                "503", "service unavailable", "500", "internal",
                "timeout", "deadline exceeded", "connection",
                "overloaded",
            ])
            if not is_retryable or attempt == max_retries:
                logger.error(
                    f"{operation_name} failed after {attempt} attempt(s): {e}"
                )
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warn(
                f"{operation_name} attempt {attempt}/{max_retries} failed "
                f"({type(e).__name__}). Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
    raise last_exception  # safety net


def validate_model_available(model_name: str) -> bool:
    """
    Lightweight check: attempt a minimal request to verify the model is reachable.
    Returns True if the model responds, False otherwise.
    """
    client = get_genai_client()
    if not client or not model_name:
        return False
    try:
        client.models.count_tokens(model=model_name, contents=["ping"])
        return True
    except Exception as e:
        logger.warn(f"Model '{model_name}' availability check failed: {e}")
        return False


def resolve_image_model() -> str:
    """
    Resolve the best available image model from primary + fallback chain.
    Returns the first model that passes validation, or the primary as last resort.
    """
    cfg = get_config().gemini
    primary = cfg.image_model

    if validate_model_available(primary):
        return primary

    fallbacks = [
        fb.strip() if isinstance(fb, str) else fb
        for fb in cfg.fallback_image_models
        if (fb.strip() if isinstance(fb, str) else fb) != primary
    ]
    for fallback in fallbacks:
        logger.info(f"Trying fallback image model: {fallback}")
        if validate_model_available(fallback):
            logger.info(f"Using fallback image model: {fallback}")
            return fallback

    logger.warn(
        f"No image models validated successfully. "
        f"Falling back to primary '{primary}' and hoping for the best."
    )
    return primary


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _get_types():
    """Lazy import of ``google.genai.types``."""
    from google.genai import types
    return types


def generate_text(
    prompt: str,
    *,
    model: Optional[str] = None,
    system_instruction: Optional[str] = None,
) -> str:
    """
    Generate text using the GenAI SDK (with retry).

    Args:
        prompt: The text prompt.
        model: Override model name (defaults to ``gemini.text_model``).
        system_instruction: Optional system instruction for the model.

    Returns:
        The generated text string.

    Raises:
        ValueError: If the client is not configured or the response is empty.
    """
    client = get_genai_client()
    if not client:
        raise ValueError("Gemini client not configured")

    cfg = get_config().gemini
    model_name = model or cfg.text_model
    types = _get_types()

    config_kwargs: Dict[str, Any] = {"response_modalities": ["TEXT"]}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    def _call():
        return client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(**config_kwargs),
        )

    response = call_with_retry(_call, operation_name=f"generate_text ({model_name})")
    text = _extract_text(response)
    if not text:
        raise ValueError("Text generation returned no text")
    return text


def generate_image(
    prompt: str,
    *,
    model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
) -> bytes:
    """
    Generate an image via the GenAI image model (with retry + fallback chain).

    Returns:
        Raw PNG/JPEG bytes.

    Raises:
        ValueError: If no image data is returned.
    """
    client = get_genai_client()
    if not client:
        raise ValueError("Gemini client not configured")

    cfg = get_config().gemini
    model_name = model or cfg.image_model
    fallback_list = fallback_model or cfg.fallback_image_models
    ar = aspect_ratio or cfg.visual_aspect_ratio
    types = _get_types()

    def _render(selected_model: str) -> bytes:
        def _call():
            return client.models.generate_content(
                model=selected_model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=ar),
                ),
            )
        response = call_with_retry(_call, operation_name=f"generate_image ({selected_model})")
        return _extract_image_bytes(response)

    try:
        return _render(model_name)
    except Exception as e:
        logger.error(f"Gemini image generation failed with model '{model_name}': {e}")
        # Try each fallback in the chain
        if isinstance(fallback_list, str):
            fallback_list = [fb.strip() for fb in fallback_list.split(",") if fb.strip()]
        fallbacks = [
            fb for fb in fallback_list
            if fb and fb != model_name
        ]
        for fb in fallbacks:
            try:
                logger.warn(f"Retrying image generation with fallback model '{fb}'")
                return _render(fb)
            except Exception as fb_err:
                logger.error(f"Fallback model '{fb}' also failed: {fb_err}")
                continue
        raise


def generate_multimodal(
    contents: list,
    *,
    model: Optional[str] = None,
    system_instruction: Optional[str] = None,
) -> str:
    """
    Send mixed text+image content to the model and return text output (with retry).

    Args:
        contents: List of text strings and ``types.Part`` image parts.
        model: Override model name.
        system_instruction: Optional system instruction.

    Returns:
        The generated text string.
    """
    client = get_genai_client()
    if not client:
        raise ValueError("Gemini client not configured")

    cfg = get_config().gemini
    model_name = model or cfg.image_model
    types = _get_types()

    config_kwargs: Dict[str, Any] = {"response_modalities": ["TEXT"]}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    def _call():
        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    response = call_with_retry(_call, operation_name=f"generate_multimodal ({model_name})")
    text = _extract_text(response)
    if not text:
        raise ValueError("Multimodal generation returned no text")
    return text


def to_image_part(image_bytes: bytes, mime_type: str = "image/jpeg"):
    """
    Create a ``google.genai.types.Part`` from raw image bytes.
    Handles minor SDK version differences.
    """
    types = _get_types()
    try:
        return types.Part.from_bytes(image_bytes, mime_type=mime_type)
    except AttributeError:
        return types.Part.from_bytes_data(data=image_bytes, mime_type=mime_type)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text(response) -> Optional[str]:
    """Pull the first text string from a GenAI response object."""
    if hasattr(response, "text") and response.text:
        return response.text.strip()

    if hasattr(response, "candidates") and response.candidates:
        for cand in response.candidates:
            if hasattr(cand, "content") and cand.content and getattr(cand.content, "parts", None):
                texts = [
                    getattr(part, "text", "")
                    for part in cand.content.parts
                    if getattr(part, "text", "")
                ]
                joined = " ".join(t.strip() for t in texts if t)
                if joined:
                    return joined

    if getattr(response, "parts", None):
        texts = [
            getattr(part, "text", "")
            for part in response.parts
            if getattr(part, "text", "")
        ]
        if texts:
            return " ".join(texts).strip()

    return None


def _extract_image_bytes(response) -> bytes:
    """Pull raw image bytes from a GenAI response object."""
    for part in getattr(response, "parts", []) or []:
        if hasattr(part, "inline_data") and part.inline_data:
            return part.inline_data.data
        if hasattr(part, "image") and part.image:
            img_obj = part.as_image()
            buf = BytesIO()
            img_obj.save(buf, format="PNG")
            return buf.getvalue()

    try:
        logger.error(f"Image renderer returned empty payload. Raw response: {response}")
    except Exception:
        logger.error("Image renderer returned empty payload.")
    raise ValueError("Renderer returned no image data")
