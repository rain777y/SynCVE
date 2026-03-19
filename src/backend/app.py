# GPU Configuration - Load before importing TensorFlow
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Load environment variables from config directory first, then legacy locations
try:
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent.parent
    config_env_new = project_root / 'config' / 'backend' / 'backend.env'
    config_env_legacy = project_root / 'config' / 'backend.env'
    src_backend_env = project_root / 'src' / 'backend' / 'backend.env'
    legacy_backend_env = project_root / 'backend' / '.env'
    root_env = project_root / '.env'

    if src_backend_env.exists():
        load_dotenv(src_backend_env)
        print(f"[OK] Loaded environment configuration from {src_backend_env}")
    elif config_env_new.exists():
        load_dotenv(config_env_new)
        print(f"[OK] Loaded environment configuration from {config_env_new}")
    elif config_env_legacy.exists():
        load_dotenv(config_env_legacy)
        print(f"[OK] Loaded environment configuration from {config_env_legacy}")
    elif legacy_backend_env.exists():
        load_dotenv(legacy_backend_env)
        print(f"[OK] Loaded environment configuration from {legacy_backend_env}")
    elif root_env.exists():
        load_dotenv(root_env)
        print(f"[OK] Loaded environment configuration from {root_env}")
    else:
        print("[WARN] No .env file found; using system environment variables")
except ImportError:
    print("[WARN] python-dotenv not installed, using system environment variables")

# Configure GPU before importing TensorFlow
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

# Torch must be imported BEFORE TensorFlow to avoid DLL conflicts (WinError 127)
try:
    import torch
except ImportError:
    pass

# Import TensorFlow and configure GPU
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU Acceleration Enabled: {len(gpus)} GPU(s) detected")
        print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"   TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')}")
    except RuntimeError as e:
        print(f"[WARN] GPU Configuration Error: {e}")
else:
    print("[WARN] No GPU detected, using CPU")

# 3rd party dependencies
from flask import Flask
from flask_cors import CORS

# project dependencies
from deepface import DeepFace
from src.backend.config import get_config
from src.backend.routes import blueprint
from deepface.commons.logger import Logger

logger = Logger()


def create_app():
    cfg = get_config()

    app = Flask(__name__)
    CORS(app, origins=cfg.server.cors_origins)
    app.config["MAX_CONTENT_LENGTH"] = cfg.server.max_content_length

    # Rate limiting (in-memory storage for simplicity)
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["60/minute"],
        storage_uri="memory://",
    )

    # Store limiter on app so routes can import and use it
    app.limiter = limiter

    app.register_blueprint(blueprint)

    # Apply per-endpoint rate limits after blueprint registration
    for rule in app.url_map.iter_rules():
        endpoint = rule.endpoint
        view_func = app.view_functions.get(endpoint)
        if not view_func:
            continue
        if rule.rule == "/analyze":
            app.view_functions[endpoint] = limiter.limit("30/minute")(view_func)
        elif rule.rule.startswith("/session/report/"):
            app.view_functions[endpoint] = limiter.limit("10/minute")(view_func)

    logger.info(f"Welcome to SynCVE Backend (DeepFace API v{DeepFace.__version__})!")
    return app


if __name__ == "__main__":
    cfg = get_config()
    app = create_app()
    app.run(host=cfg.server.host, port=cfg.server.port, debug=cfg.server.debug)
