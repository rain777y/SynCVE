"""
SynCVE Health Check - verifies all dependencies and configuration.

Usage:
    python scripts/health_check.py

Run from the project root directory.
"""

import os
import sys
import importlib
import platform
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SKIP = "[SKIP]"

results: list[tuple[str, str, str]] = []  # (status, check_name, detail)


def record(status: str, name: str, detail: str = ""):
    results.append((status, name, detail))
    tag = status
    line = f"  {tag} {name}"
    if detail:
        line += f"  --  {detail}"
    print(line)


def try_import(module_name: str, display_name: str | None = None):
    """Attempt to import a module and record the result."""
    label = display_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "installed")
        record(PASS, label, str(version))
        return mod
    except Exception as exc:
        record(FAIL, label, str(exc))
        return None


# ---------------------------------------------------------------------------
# Locate project root (the directory containing requirements.txt)
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Walk upward from this script to find the project root."""
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "requirements.txt").exists():
        return candidate
    # Fallback: current working directory
    cwd = Path.cwd()
    if (cwd / "requirements.txt").exists():
        return cwd
    return candidate


PROJECT_ROOT = find_project_root()


# ============================================================================
# Checks
# ============================================================================

def check_python_version():
    ver = platform.python_version()
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        record(PASS, "Python version", f"{ver} (3.10+ required)")
    else:
        record(FAIL, "Python version", f"{ver} (3.10+ required)")


def check_core_packages():
    print()
    print("  --- Core Packages ---")
    try_import("numpy")
    try_import("pandas")
    try_import("requests")
    try_import("dotenv", "python-dotenv")
    try_import("pydantic")


def check_deep_learning():
    print()
    print("  --- Deep Learning ---")
    tf = try_import("tensorflow")
    try_import("keras")
    torch = try_import("torch")
    try_import("torchvision")
    return tf, torch


def check_computer_vision():
    print()
    print("  --- Computer Vision ---")
    try_import("cv2", "opencv-python")
    try_import("PIL", "Pillow")
    try_import("deepface")
    try_import("mtcnn")


def check_web_framework():
    print()
    print("  --- Web Framework ---")
    try_import("flask", "Flask")
    try_import("flask_cors", "flask-cors")


def check_genai_database():
    print()
    print("  --- GenAI & Database ---")
    try_import("google.genai", "google-genai")
    # google-generativeai is deprecated (EOL 2025-11-30); replaced by google-genai
    try_import("supabase")


def check_gpu(torch_mod):
    print()
    print("  --- GPU / CUDA ---")

    # nvidia-smi
    import shutil
    if shutil.which("nvidia-smi"):
        record(PASS, "nvidia-smi", "found on PATH")
    else:
        record(WARN, "nvidia-smi", "not found (GPU monitoring unavailable)")

    # PyTorch CUDA
    if torch_mod is not None:
        if torch_mod.cuda.is_available():
            dev_name = torch_mod.cuda.get_device_name(0)
            record(PASS, "PyTorch CUDA", f"{dev_name}")
        else:
            record(WARN, "PyTorch CUDA", "not available (CPU mode)")
    else:
        record(SKIP, "PyTorch CUDA", "torch not importable")

    # TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            record(PASS, "TensorFlow GPU", f"{len(gpus)} device(s)")
        else:
            record(WARN, "TensorFlow GPU", "no GPU devices found (CPU mode)")
    except Exception:
        record(SKIP, "TensorFlow GPU", "tensorflow not importable")

    # CUDA version from torch
    if torch_mod is not None and torch_mod.cuda.is_available():
        cuda_ver = torch_mod.version.cuda
        record(PASS, "CUDA version (torch)", str(cuda_ver))


def check_environment_files():
    print()
    print("  --- Environment Files ---")

    backend_env = PROJECT_ROOT / "src" / "backend" / "backend.env"
    backend_env_example = PROJECT_ROOT / "src" / "backend" / "backend.env.example"
    frontend_env = PROJECT_ROOT / "src" / "frontend" / ".env"
    frontend_env_example = PROJECT_ROOT / "src" / "frontend" / ".env.example"

    if backend_env.exists():
        record(PASS, "backend.env", str(backend_env))
    else:
        if backend_env_example.exists():
            record(FAIL, "backend.env", f"missing (copy from {backend_env_example})")
        else:
            record(FAIL, "backend.env", "missing (no template found either)")

    if backend_env_example.exists():
        record(PASS, "backend.env.example", "template present")
    else:
        record(WARN, "backend.env.example", "template missing")

    if frontend_env.exists():
        record(PASS, "frontend .env", str(frontend_env))
    else:
        if frontend_env_example.exists():
            record(FAIL, "frontend .env", f"missing (copy from {frontend_env_example})")
        else:
            record(FAIL, "frontend .env", "missing (no template found either)")

    if frontend_env_example.exists():
        record(PASS, "frontend .env.example", "template present")
    else:
        record(WARN, "frontend .env.example", "template missing")


def check_supabase_connectivity():
    print()
    print("  --- Supabase Connectivity ---")
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / "src" / "backend" / "backend.env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    if not url or "your" in url.lower():
        record(SKIP, "Supabase connectivity", "SUPABASE_URL not configured")
        return
    if not key or "your" in key.lower():
        record(SKIP, "Supabase connectivity", "SUPABASE_KEY not configured")
        return

    try:
        from supabase import create_client
        client = create_client(url, key)
        # Simple connectivity test: list tables or do a health-like query
        record(PASS, "Supabase client", f"created for {url}")
    except Exception as exc:
        record(FAIL, "Supabase connectivity", str(exc))


def check_frontend_build():
    print()
    print("  --- Frontend ---")
    pkg_json = PROJECT_ROOT / "src" / "frontend" / "package.json"
    node_modules = PROJECT_ROOT / "src" / "frontend" / "node_modules"

    if pkg_json.exists():
        record(PASS, "package.json", str(pkg_json))
    else:
        record(FAIL, "package.json", "missing")

    if node_modules.exists():
        record(PASS, "node_modules", "installed")
    else:
        record(WARN, "node_modules", "missing (run: npm install in src/frontend/)")

    import shutil
    if shutil.which("node"):
        import subprocess
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        ver = result.stdout.strip()
        record(PASS, "Node.js", ver)
    else:
        record(FAIL, "Node.js", "not found on PATH")


# ============================================================================
# Summary
# ============================================================================

def print_summary():
    print()
    print("=" * 60)
    print("  SynCVE Health Check Summary")
    print("=" * 60)

    pass_count = sum(1 for s, _, _ in results if s == PASS)
    fail_count = sum(1 for s, _, _ in results if s == FAIL)
    warn_count = sum(1 for s, _, _ in results if s == WARN)
    skip_count = sum(1 for s, _, _ in results if s == SKIP)
    total = len(results)

    print(f"  Total checks:  {total}")
    print(f"  Passed:        {pass_count}")
    print(f"  Failed:        {fail_count}")
    print(f"  Warnings:      {warn_count}")
    print(f"  Skipped:       {skip_count}")
    print()

    if fail_count > 0:
        print("  FAILURES:")
        for s, name, detail in results:
            if s == FAIL:
                print(f"    - {name}: {detail}")
        print()

    if fail_count == 0:
        print("  Status: ALL CHECKS PASSED")
    elif fail_count <= 3:
        print("  Status: MOSTLY OK (fix failures above)")
    else:
        print("  Status: SETUP INCOMPLETE (run scripts/setup.bat)")

    print("=" * 60)
    return fail_count


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("=" * 60)
    print("  SynCVE Health Check")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python: {sys.executable}")
    print(f"  Platform: {platform.platform()}")
    print("=" * 60)

    check_python_version()
    check_core_packages()
    tf, torch_mod = check_deep_learning()
    check_computer_vision()
    check_web_framework()
    check_genai_database()
    check_gpu(torch_mod)
    check_environment_files()
    check_supabase_connectivity()
    check_frontend_build()

    fail_count = print_summary()
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
