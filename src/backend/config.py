"""
Centralized configuration for SynCVE backend.

  - Application settings  → settings.yml   (project root, tracked in git)
  - Secrets (API keys)     → .env           (project root, gitignored)
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains .git/)."""
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _find_project_root()


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict if missing or invalid."""
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        # Fallback: minimal YAML parser for simple key: value files
        return _parse_simple_yaml(path)
    except Exception:
        return {}


def _parse_simple_yaml(path: Path) -> dict:
    """Minimal YAML parser for flat/nested key-value configs (no PyYAML needed)."""
    result = {}
    stack = [result]
    indent_stack = [-1]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.lstrip().startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())
            content = stripped.lstrip()

            # Pop stack back to correct nesting level
            while indent <= indent_stack[-1] and len(stack) > 1:
                stack.pop()
                indent_stack.pop()

            if ":" in content:
                key, _, val = content.partition(":")
                key = key.strip()
                val = val.strip()

                if val == "" or val.startswith("#"):
                    # Nested section
                    new_dict = {}
                    stack[-1][key] = new_dict
                    stack.append(new_dict)
                    indent_stack.append(indent)
                elif val.startswith("- "):
                    # Inline list start
                    stack[-1][key] = [_coerce(val[2:].strip())]
                else:
                    stack[-1][key] = _coerce(val)
            elif content.startswith("- "):
                # List continuation
                val = _coerce(content[2:].strip())
                parent = stack[-1]
                # Find the last key that holds a list
                for k in reversed(list(parent.keys())):
                    if isinstance(parent[k], list):
                        parent[k].append(val)
                        break

    return result


def _coerce(val: str) -> Any:
    """Convert YAML string values to Python types."""
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    low = val.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~", ""):
        return None
    # Remove inline comments
    if " #" in val:
        val = val[:val.index(" #")].strip()
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _get(data: dict, *keys, default=None):
    """Safely traverse nested dict."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current if current is not None else default


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5005
    debug: bool = False
    max_content_length: int = 16 * 1024 * 1024
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000"])


@dataclass(frozen=True)
class GPUConfig:
    cuda_visible_devices: str = "0"
    tf_memory_fraction: float = 0.8
    tf_allow_growth: bool = True
    tf_log_level: int = 2
    omp_num_threads: int = 0


@dataclass(frozen=True)
class PreprocessConfig:
    enable_sr: bool = True
    sr_min_size: int = 256
    enable_clahe: bool = True
    enable_unsharp: bool = True
    adaptive_threshold: int = 128  # skip CLAHE+unsharp when original min dim < this


@dataclass(frozen=True)
class DeepFaceConfig:
    detector_backend: str = "retinaface"
    model_name: str = "Facenet"
    distance_metric: str = "cosine"
    anti_spoofing: bool = True
    confidence_threshold: float = 0.1
    enable_ensemble: bool = True
    ensemble_detectors: list = field(
        default_factory=lambda: ["retinaface", "mtcnn"]
    )
    ensemble_weights: dict = field(
        default_factory=lambda: {"retinaface": 0.50, "mtcnn": 0.50}
    )


@dataclass(frozen=True)
class SupabaseConfig:
    url: str = ""
    key: str = ""
    bucket_name: str = "syn_cve_assets"


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str = ""
    service_account_path: str = ""
    gcp_project: str = ""
    gcp_location: str = "us-central1"
    text_model: str = "gemini-2.5-flash"
    image_model: str = "gemini-2.5-flash-image"
    fallback_image_models: list = field(
        default_factory=lambda: ["gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview"]
    )
    request_timeout: int = 120
    max_retries: int = 3
    retry_base_delay: float = 1.0
    report_mode: str = "fast"           # "fast" = structured data only, "full" = fast + AI image
    noise_floor: float = 0.0
    keyframe_limit: int = 4
    visual_aspect_ratio: str = "16:9"
    visual_style_preset: str = "futuristic"


@dataclass(frozen=True)
class TemporalConfig:
    ema_alpha: float = 0.2
    transition_threshold: float = 0.15
    volatility_window: int = 10
    fps_estimate: float = 0.5


@dataclass(frozen=True)
class EventsSlidingConfig:
    window: int = 5
    z_threshold: float = 2.5
    min_magnitude: float = 0.10


@dataclass(frozen=True)
class EventsCusumConfig:
    drift: float = 0.005
    threshold: float = 0.15


@dataclass(frozen=True)
class EventsPeltConfig:
    model: str = "rbf"
    penalty: float = 3.0
    min_size: int = 3


@dataclass(frozen=True)
class EventsConfig:
    enabled: bool = True
    method: str = "ensemble"
    consensus_min_methods: int = 2
    refractory_frames: int = 3
    sliding: EventsSlidingConfig = field(default_factory=EventsSlidingConfig)
    cusum: EventsCusumConfig = field(default_factory=EventsCusumConfig)
    pelt: EventsPeltConfig = field(default_factory=EventsPeltConfig)


@dataclass(frozen=True)
class FusionConfig:
    method: str = "uncertainty"        # "fixed" | "uncertainty" | "max_confidence"
    temperature: float = 1.0
    entropy_floor: float = 0.05
    blend_with_fixed: float = 0.0


@dataclass(frozen=True)
class ClinicalConfig:
    enabled: bool = True
    valence_map: dict = field(default_factory=lambda: {
        "happy": 1.0,
        "surprise": 0.5,
        "neutral": 0.0,
        "fear": -0.7,
        "sad": -0.8,
        "disgust": -0.6,
        "angry": -0.9,
    })
    affect_blunting_window: int = 30
    reaction_latency_max_sec: float = 3.0
    incongruence_window_sec: float = 2.0
    sigma_baseline: float = 0.30
    range_baseline: float = 1.20


@dataclass(frozen=True)
class ClientConfig:
    detection_interval: int = 2000


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig
    gpu: GPUConfig
    deepface: DeepFaceConfig
    supabase: SupabaseConfig
    gemini: GeminiConfig
    client: ClientConfig = field(default_factory=ClientConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_dotenv_secrets():
    """Load secrets from .env file into os.environ."""
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


def load_config() -> AppConfig:
    """Load settings from YAML, secrets from .env."""
    settings_path = PROJECT_ROOT / "settings.yml"
    cfg = _load_yaml(settings_path)

    # Load secrets from .env into os.environ
    _load_dotenv_secrets()

    # Auto-detect CPU threads
    omp = _get(cfg, "gpu", "omp_num_threads", default=0)
    if not omp or omp <= 0:
        omp = os.cpu_count() or 4

    # Ensemble config
    ens = _get(cfg, "deepface", "ensemble", default={})
    ens_detectors = _get(ens, "detectors", default=["retinaface", "mtcnn"])
    ens_weights = _get(ens, "weights", default={"retinaface": 0.50, "mtcnn": 0.50})

    # Fallback models can be a list or comma-separated string
    fallback_raw = _get(cfg, "gemini", "fallback_image_models", default=[])
    if isinstance(fallback_raw, str):
        fallback_raw = [m.strip() for m in fallback_raw.split(",") if m.strip()]

    return AppConfig(
        server=ServerConfig(
            host=_get(cfg, "server", "host", default="0.0.0.0"),
            port=int(_get(cfg, "server", "port", default=5005)),
            debug=bool(_get(cfg, "server", "debug", default=False)),
            max_content_length=int(_get(cfg, "server", "max_content_length", default=16777216)),
            cors_origins=_get(cfg, "server", "cors_origins", default=["http://localhost:3000"]),
        ),
        gpu=GPUConfig(
            cuda_visible_devices=str(_get(cfg, "gpu", "cuda_visible_devices", default="0")),
            tf_memory_fraction=float(_get(cfg, "gpu", "tf_memory_fraction", default=0.8)),
            tf_allow_growth=bool(_get(cfg, "gpu", "tf_allow_growth", default=True)),
            tf_log_level=int(_get(cfg, "gpu", "tf_log_level", default=2)),
            omp_num_threads=omp,
        ),
        deepface=DeepFaceConfig(
            detector_backend=_get(cfg, "deepface", "detector_backend", default="retinaface"),
            model_name=_get(cfg, "deepface", "model_name", default="Facenet"),
            distance_metric=_get(cfg, "deepface", "distance_metric", default="cosine"),
            anti_spoofing=bool(_get(cfg, "deepface", "anti_spoofing", default=True)),
            confidence_threshold=float(_get(cfg, "deepface", "confidence_threshold", default=0.1)),
            enable_ensemble=bool(_get(ens, "enabled", default=True)),
            ensemble_detectors=ens_detectors,
            ensemble_weights=ens_weights,
        ),
        supabase=SupabaseConfig(
            url=os.getenv("SUPABASE_URL", ""),
            key=os.getenv("SUPABASE_KEY", ""),
            bucket_name=_get(cfg, "supabase", "bucket_name", default="syn_cve_assets"),
        ),
        gemini=GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            service_account_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
            gcp_project=os.getenv("GCP_PROJECT", _get(cfg, "gemini", "gcp_project", default="")),
            gcp_location=os.getenv("GCP_LOCATION", _get(cfg, "gemini", "gcp_location", default="us-central1")),
            text_model=_get(cfg, "gemini", "text_model", default="gemini-2.5-flash"),
            image_model=_get(cfg, "gemini", "image_model", default="gemini-2.5-flash-image"),
            fallback_image_models=fallback_raw,
            request_timeout=int(_get(cfg, "gemini", "request_timeout", default=120)),
            max_retries=int(_get(cfg, "gemini", "max_retries", default=3)),
            retry_base_delay=float(_get(cfg, "gemini", "retry_base_delay", default=1.0)),
            report_mode=_get(cfg, "report", "mode", default="fast"),
            noise_floor=float(_get(cfg, "report", "noise_floor", default=0.0)),
            keyframe_limit=int(_get(cfg, "report", "keyframe_limit", default=4)),
            visual_aspect_ratio=_get(cfg, "report", "visual_aspect_ratio", default="16:9"),
            visual_style_preset=_get(cfg, "report", "visual_style_preset", default="futuristic"),
        ),
        client=ClientConfig(
            detection_interval=int(_get(cfg, "client", "detection_interval", default=2000)),
        ),
        temporal=TemporalConfig(
            ema_alpha=float(_get(cfg, "temporal", "ema_alpha", default=0.2)),
            transition_threshold=float(_get(cfg, "temporal", "transition_threshold", default=0.15)),
            volatility_window=int(_get(cfg, "temporal", "volatility_window", default=10)),
            fps_estimate=float(_get(cfg, "temporal", "fps_estimate", default=0.5)),
        ),
        preprocess=PreprocessConfig(
            enable_sr=bool(_get(cfg, "preprocess", "enable_sr", default=True)),
            sr_min_size=int(_get(cfg, "preprocess", "sr_min_size", default=256)),
            enable_clahe=bool(_get(cfg, "preprocess", "enable_clahe", default=True)),
            enable_unsharp=bool(_get(cfg, "preprocess", "enable_unsharp", default=True)),
            adaptive_threshold=int(_get(cfg, "preprocess", "adaptive_threshold", default=128)),
        ),
        events=EventsConfig(
            enabled=bool(_get(cfg, "events", "enabled", default=True)),
            method=str(_get(cfg, "events", "method", default="ensemble")),
            consensus_min_methods=int(_get(cfg, "events", "consensus_min_methods", default=2)),
            refractory_frames=int(_get(cfg, "events", "refractory_frames", default=3)),
            sliding=EventsSlidingConfig(
                window=int(_get(cfg, "events", "sliding", "window", default=5)),
                z_threshold=float(_get(cfg, "events", "sliding", "z_threshold", default=2.5)),
                min_magnitude=float(_get(cfg, "events", "sliding", "min_magnitude", default=0.10)),
            ),
            cusum=EventsCusumConfig(
                drift=float(_get(cfg, "events", "cusum", "drift", default=0.005)),
                threshold=float(_get(cfg, "events", "cusum", "threshold", default=0.15)),
            ),
            pelt=EventsPeltConfig(
                model=str(_get(cfg, "events", "pelt", "model", default="rbf")),
                penalty=float(_get(cfg, "events", "pelt", "penalty", default=3.0)),
                min_size=int(_get(cfg, "events", "pelt", "min_size", default=3)),
            ),
        ),
        fusion=FusionConfig(
            method=str(_get(cfg, "fusion", "method", default="uncertainty")),
            temperature=float(_get(cfg, "fusion", "temperature", default=1.0)),
            entropy_floor=float(_get(cfg, "fusion", "entropy_floor", default=0.05)),
            blend_with_fixed=float(_get(cfg, "fusion", "blend_with_fixed", default=0.0)),
        ),
        clinical=ClinicalConfig(
            enabled=bool(_get(cfg, "clinical", "enabled", default=True)),
            valence_map=_get(cfg, "clinical", "valence_map", default={
                "happy": 1.0, "surprise": 0.5, "neutral": 0.0,
                "fear": -0.7, "sad": -0.8, "disgust": -0.6, "angry": -0.9,
            }),
            affect_blunting_window=int(_get(cfg, "clinical", "affect_blunting_window", default=30)),
            reaction_latency_max_sec=float(_get(cfg, "clinical", "reaction_latency_max_sec", default=3.0)),
            incongruence_window_sec=float(_get(cfg, "clinical", "incongruence_window_sec", default=2.0)),
            sigma_baseline=float(_get(cfg, "clinical", "sigma_baseline", default=0.30)),
            range_baseline=float(_get(cfg, "clinical", "range_baseline", default=1.20)),
        ),
    )


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the global AppConfig singleton, creating it on first access."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
