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
