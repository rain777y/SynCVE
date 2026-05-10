"""
GPU initialization for eval scripts on Windows.
Import this BEFORE tensorflow/deepface to ensure cuDNN DLLs are found.

Usage (at the very top of eval scripts):
    from eval._gpu_init import init_gpu
    init_gpu()
"""
import os
import sys


def init_gpu():
    """Add conda env Library/bin to PATH so TF can find cuDNN DLLs."""
    # Suppress TF/Keras verbose logging (MTCNN step logs crash output buffers)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    # Protobuf compatibility: TF 2.10 + newer protobuf
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    if sys.platform != "win32":
        return

    # Infer conda env from Python executable (most reliable)
    # e.g. E:/conda/envs/SynCVE/python.exe → E:/conda/envs/SynCVE
    env_root = os.path.dirname(sys.executable)
    dll_dir = os.path.join(env_root, "Library", "bin")

    if os.path.isdir(dll_dir):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)

    # Suppress Keras step-by-step progress bars (MTCNN outputs 12 lines per image)
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        tf.keras.utils.disable_interactive_logging()
    except Exception:
        pass
