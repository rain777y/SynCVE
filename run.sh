#!/bin/bash
# =============================================================================
# SynCVE Runner — Ensures correct Python environment
#
# Usage:
#   ./run.sh -m eval.benchmark_fer2013 --limit 100
#   ./run.sh -m eval.optimize_ensemble_weights --train-limit 2000
#   ./run.sh -m src.backend.app
#   ./run.sh -m pytest tests/
#
# Resolves: conda run instability, global env pollution, version mismatches
# =============================================================================
set -e

# --- Locate Python ---
# Priority: 1) Docker  2) Conda SynCVE env  3) Error
if [ -f "/.dockerenv" ]; then
    PYTHON="python"
elif [ -f "E:/conda/envs/SynCVE/python.exe" ]; then
    PYTHON="E:/conda/envs/SynCVE/python.exe"
elif [ -f "$HOME/conda/envs/SynCVE/bin/python" ]; then
    PYTHON="$HOME/conda/envs/SynCVE/bin/python"
elif command -v conda &>/dev/null; then
    CONDA_PREFIX=$(conda info --envs 2>/dev/null | grep SynCVE | awk '{print $NF}')
    if [ -n "$CONDA_PREFIX" ]; then
        PYTHON="$CONDA_PREFIX/python.exe"
        [ -f "$PYTHON" ] || PYTHON="$CONDA_PREFIX/bin/python"
    fi
fi

if [ -z "$PYTHON" ] || [ ! -f "$PYTHON" ] && [ "$PYTHON" != "python" ]; then
    echo "ERROR: SynCVE Python not found."
    echo "  Options:"
    echo "    1. conda env create -f environment.yml"
    echo "    2. docker compose up backend"
    exit 1
fi

# --- Verify version ---
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$PY_VER" != "3.10" ] && [ ! -f "/.dockerenv" ]; then
    echo "WARNING: Expected Python 3.10, got $PY_VER"
fi

# --- Suppress TF noise ---
export TF_CPP_MIN_LOG_LEVEL=2
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# --- Run ---
cd "$(dirname "$0")"
exec "$PYTHON" "$@"
