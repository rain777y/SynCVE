# GPU Configuration — must run before importing TensorFlow
import os
import sys

# UTF-8 for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load config: settings from settings.yml, secrets from .env
from src.backend.config import get_config

cfg = get_config()

# Apply GPU environment variables BEFORE TensorFlow import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.gpu.cuda_visible_devices)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", str(cfg.gpu.tf_log_level))
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", str(cfg.gpu.tf_allow_growth).lower())
os.environ["OMP_NUM_THREADS"] = str(cfg.gpu.omp_num_threads)

# Protobuf compatibility: TF (old) vs Gemini SDK (new)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Torch must be imported BEFORE TensorFlow to avoid DLL conflicts on Windows
try:
    import torch
except ImportError:
    pass

# Import TensorFlow and configure GPU memory
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU Enabled: {len(gpus)} device(s), "
              f"VRAM fraction={cfg.gpu.tf_memory_fraction}, "
              f"CPU threads={cfg.gpu.omp_num_threads}")
    except RuntimeError as e:
        print(f"[WARN] GPU config error: {e}")
else:
    print(f"[WARN] No GPU detected, CPU mode (threads={cfg.gpu.omp_num_threads})")

# Flask app
from flask import Flask
from flask_cors import CORS
from deepface import DeepFace
from src.backend.routes import blueprint
from deepface.commons.logger import Logger

logger = Logger()


def create_app():
    app = Flask(__name__)
    CORS(app, origins=cfg.server.cors_origins)
    app.config["MAX_CONTENT_LENGTH"] = cfg.server.max_content_length

    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["60/minute"],
        storage_uri="memory://",
    )
    app.limiter = limiter
    app.register_blueprint(blueprint)

    # Per-endpoint rate limits
    for rule in app.url_map.iter_rules():
        view_func = app.view_functions.get(rule.endpoint)
        if not view_func:
            continue
        if rule.rule == "/analyze":
            app.view_functions[rule.endpoint] = limiter.limit("30/minute")(view_func)
        elif rule.rule.startswith("/session/report/"):
            app.view_functions[rule.endpoint] = limiter.limit("10/minute")(view_func)

    logger.info(f"SynCVE Backend v{DeepFace.__version__} ready")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=cfg.server.host, port=cfg.server.port, debug=cfg.server.debug)
