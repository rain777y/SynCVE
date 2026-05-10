#!/bin/bash
# =============================================================================
# SynCVE One-Click Environment Setup (Linux/macOS/Git Bash)
#
# Usage:
#   bash setup_env.sh
#
# Prerequisites:
#   - Anaconda/Miniconda installed
#   - NVIDIA GPU with CUDA support (optional, CPU fallback available)
# =============================================================================

set -e

echo "============================================================"
echo "  SynCVE Environment Setup"
echo "============================================================"
echo ""

# --- Check conda ---
if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found. Please install Anaconda or Miniconda first."
    echo "  Download: https://docs.anaconda.com/miniconda/"
    exit 1
fi

# --- Init conda for this shell ---
eval "$(conda shell.bash hook)"

# --- Check if env exists ---
if conda info --envs 2>/dev/null | grep -q "SynCVE"; then
    echo "[INFO] SynCVE conda env already exists. Updating..."
else
    echo "[1/4] Creating conda env 'SynCVE' with Python 3.10..."
    conda create -n SynCVE python=3.10 -y
fi

# --- Activate ---
echo "[2/4] Activating SynCVE env..."
conda activate SynCVE

# --- Install PyTorch ---
echo "[3/4] Installing PyTorch with CUDA 11.8..."
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# --- Install dependencies ---
echo "[4/4] Installing project dependencies..."
pip install --no-cache-dir \
    "numpy>=1.22.0,<2.0" \
    "pandas>=2.0,<3.0" \
    "requests>=2.28,<3.0" \
    "python-dotenv>=1.0,<2.0" \
    "pydantic>=2.0,<3.0" \
    "PyYAML>=6.0,<7.0" \
    "setuptools<81" \
    "tensorflow==2.10.1" \
    "opencv-python>=4.8,<5.0" \
    "Pillow>=10.0,<12.0" \
    "deepface>=0.0.99,<0.1.0" \
    "mtcnn==0.1.1" \
    "retina-face>=0.0.14,<0.0.18" \
    "Flask>=3.0,<4.0" \
    "flask-cors>=4.0,<5.0" \
    "flask-limiter>=3.0,<4.0" \
    "google-genai>=1.65,<2.0" \
    "supabase>=2.5,<3.0" \
    "scikit-learn>=1.3,<2.0" \
    "matplotlib>=3.7,<4.0" \
    "seaborn>=0.13,<1.0" \
    "tqdm>=4.65,<5.0"

# --- Setup .env ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example. Please edit it with your API keys."
fi

# --- Verify ---
echo ""
echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"
python -c "import tensorflow as tf; print(f'  TensorFlow: {tf.__version__}, GPU: {len(tf.config.list_physical_devices(\"GPU\"))} device(s)')"
python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import deepface; print(f'  DeepFace: {deepface.__version__}')"
python -c "import flask; print(f'  Flask: {flask.__version__}')"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  To start the backend:"
echo "    conda activate SynCVE"
echo "    python -m src.backend.app"
echo ""
echo "  Or use run.sh:"
echo "    ./run.sh -m src.backend.app"
echo ""
