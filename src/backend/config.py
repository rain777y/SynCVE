"""
Centralized configuration management for SynCVE backend.
Loads from environment with validation, defaults, and type safety.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_list(key: str, default: str = "", sep: str = ",") -> list:
    raw = os.getenv(key, default)
    if not raw:
        return []
    return [item.strip() for item in raw.split(sep) if item.strip()]


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5005
    debug: bool = False
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000"])


@dataclass(frozen=True)
class GPUConfig:
    cuda_visible_devices: str = "0"
    tf_memory_fraction: float = 0.8
    tf_allow_growth: bool = True
    omp_num_threads: int = 12
    model_cache_limit: int = 5
    log_gpu_memory: bool = True


@dataclass(frozen=True)
class DeepFaceConfig:
    detector_backend: str = "retinaface"
    model_name: str = "Facenet"
    distance_metric: str = "cosine"
    anti_spoofing: bool = True
    confidence_threshold: float = 0.1
    enable_ensemble: bool = True
    # DeepFace 0.0.99+ supports: opencv, ssd, dlib, mtcnn, fastmtcnn,
    # retinaface, mediapipe, yolov8n/m/l, yolov11n/s/m/l, yolov12n/s/m/l,
    # yunet, centerface
    ensemble_detectors: list = field(
        default_factory=lambda: ["retinaface", "mtcnn", "centerface"]
    )
    ensemble_weights: dict = field(
        default_factory=lambda: {
            "retinaface": 0.50,
            "mtcnn": 0.30,
            "centerface": 0.20,
        }
    )


@dataclass(frozen=True)
class SupabaseConfig:
    url: str = ""
    key: str = ""
    bucket_name: str = "syn_cve_assets"


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str = ""
    text_model: str = "gemini-2.5-flash"
    image_model: str = "gemini-2.5-flash-image"
    fallback_image_model: str = "gemini-3.1-flash-image-preview,gemini-3-pro-image-preview"
    request_timeout: int = 120  # seconds
    max_retries: int = 3
    retry_base_delay: float = 1.0  # seconds
    noise_floor: float = 0.10
    keyframe_limit: int = 4
    visual_aspect_ratio: str = "16:9"
    visual_style_preset: str = "futuristic"


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig
    gpu: GPUConfig
    deepface: DeepFaceConfig
    supabase: SupabaseConfig
    gemini: GeminiConfig


def load_config() -> AppConfig:
    """Load configuration from environment variables with validation."""
    server = ServerConfig(
        host=_env_str("BACKEND_HOST", "0.0.0.0"),
        port=_env_int("BACKEND_PORT", 5005),
        debug=_env_bool("DEBUG", False),
        max_content_length=_env_int("MAX_CONTENT_LENGTH", 16 * 1024 * 1024),
        cors_origins=_env_list("CORS_ORIGINS", "http://localhost:3000"),
    )

    gpu = GPUConfig(
        cuda_visible_devices=_env_str("CUDA_VISIBLE_DEVICES", "0"),
        tf_memory_fraction=_env_float("TF_GPU_MEMORY_FRACTION", 0.8),
        tf_allow_growth=_env_bool("TF_FORCE_GPU_ALLOW_GROWTH", True),
        omp_num_threads=_env_int("OMP_NUM_THREADS", 12),
        model_cache_limit=_env_int("MODEL_CACHE_LIMIT", 5),
        log_gpu_memory=_env_bool("LOG_GPU_MEMORY", True),
    )

    # Parse ensemble detectors list
    ensemble_detectors = _env_list(
        "EMOTION_ENSEMBLE_DETECTORS", "retinaface,mtcnn,centerface"
    )
    if not ensemble_detectors:
        ensemble_detectors = ["retinaface", "mtcnn", "centerface"]

    deepface = DeepFaceConfig(
        detector_backend=_env_str("DEEPFACE_DETECTOR_BACKEND", "retinaface"),
        model_name=_env_str("DEEPFACE_MODEL_NAME", "Facenet"),
        distance_metric=_env_str("DEEPFACE_DISTANCE_METRIC", "cosine"),
        anti_spoofing=_env_bool("DEEPFACE_ANTI_SPOOFING", True),
        confidence_threshold=_env_float("EMOTION_CONFIDENCE_THRESHOLD", 0.1),
        enable_ensemble=_env_bool("ENABLE_EMOTION_ENSEMBLE", True),
        ensemble_detectors=ensemble_detectors,
        ensemble_weights={
            "retinaface": 0.50,
            "mtcnn": 0.30,
            "centerface": 0.20,
        },
    )

    supabase = SupabaseConfig(
        url=_env_str("SUPABASE_URL", ""),
        key=_env_str("SUPABASE_KEY", ""),
        bucket_name=_env_str("ASSETS_BUCKET", "syn_cve_assets"),
    )

    gemini = GeminiConfig(
        api_key=_env_str("GEMINI_API_KEY", ""),
        text_model=_env_str("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
        image_model=_env_str("GEMINI_IMAGE_MODEL_NAME", "gemini-2.5-flash-image"),
        fallback_image_model=_env_str(
            "GEMINI_IMAGE_MODEL_FALLBACK",
            "gemini-3.1-flash-image-preview,gemini-3-pro-image-preview",
        ),
        request_timeout=_env_int("GEMINI_REQUEST_TIMEOUT", 120),
        max_retries=_env_int("GEMINI_MAX_RETRIES", 3),
        retry_base_delay=_env_float("GEMINI_RETRY_BASE_DELAY", 1.0),
        noise_floor=_env_float("EMOTION_NOISE_FLOOR", 0.10),
        keyframe_limit=_env_int("EMOTION_REPORT_KEYFRAME_LIMIT", 4),
        visual_aspect_ratio=_env_str("EMOTION_VISUAL_ASPECT_RATIO", "16:9"),
        visual_style_preset=_env_str("EMOTION_VISUAL_STYLE_PRESET", "futuristic"),
    )

    return AppConfig(
        server=server,
        gpu=gpu,
        deepface=deepface,
        supabase=supabase,
        gemini=gemini,
    )


# Module-level singleton: import and use directly.
# Created once when this module is first imported (after .env loading in app.py).
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the global AppConfig singleton, creating it on first access."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
