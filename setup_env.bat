@echo off
REM =============================================================================
REM SynCVE One-Click Environment Setup (Windows + Conda)
REM
REM Usage: Right-click -> Run as Administrator (recommended)
REM        Or just double-click
REM
REM Prerequisites:
REM   - Anaconda/Miniconda installed
REM   - NVIDIA GPU with CUDA support (optional, CPU fallback available)
REM
REM This script will:
REM   1. Create conda env "SynCVE" with Python 3.10
REM   2. Install all pip dependencies (TF 2.10.1, PyTorch cu118, DeepFace, etc.)
REM   3. Copy .env.example to .env if not exists
REM   4. Print instructions to start the app
REM =============================================================================

echo ============================================================
echo   SynCVE Environment Setup
echo ============================================================
echo.

REM --- Check conda ---
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] conda not found. Please install Anaconda or Miniconda first.
    echo   Download: https://docs.anaconda.com/miniconda/
    pause
    exit /b 1
)

REM --- Check if env already exists ---
conda info --envs 2>nul | findstr "SynCVE" >nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] SynCVE conda env already exists. Updating...
    goto :install_deps
)

REM --- Create conda env ---
echo [1/4] Creating conda env "SynCVE" with Python 3.10...
call conda create -n SynCVE python=3.10 -y
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create conda env.
    pause
    exit /b 1
)

:install_deps
REM --- Activate env ---
echo [2/4] Activating SynCVE env...
call conda activate SynCVE
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate SynCVE env.
    pause
    exit /b 1
)

REM --- Verify Python version ---
python -c "import sys; assert sys.version_info[:2] == (3, 10), f'Expected 3.10, got {sys.version_info.major}.{sys.version_info.minor}'"
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Python version mismatch. Expected 3.10.
)

REM --- Install PyTorch with CUDA 11.8 ---
echo [3/4] Installing PyTorch with CUDA 11.8 (this may take a while)...
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

REM --- Install all other dependencies ---
echo [4/4] Installing project dependencies...
pip install --no-cache-dir ^
    "numpy>=1.22.0,<2.0" ^
    "pandas>=2.0,<3.0" ^
    "requests>=2.28,<3.0" ^
    "python-dotenv>=1.0,<2.0" ^
    "pydantic>=2.0,<3.0" ^
    "PyYAML>=6.0,<7.0" ^
    "setuptools<81" ^
    "tensorflow==2.10.1" ^
    "opencv-python>=4.8,<5.0" ^
    "Pillow>=10.0,<12.0" ^
    "deepface>=0.0.99,<0.1.0" ^
    "mtcnn==0.1.1" ^
    "retina-face>=0.0.14,<0.0.18" ^
    "Flask>=3.0,<4.0" ^
    "flask-cors>=4.0,<5.0" ^
    "flask-limiter>=3.0,<4.0" ^
    "google-genai>=1.65,<2.0" ^
    "supabase>=2.5,<3.0" ^
    "scikit-learn>=1.3,<2.0" ^
    "matplotlib>=3.7,<4.0" ^
    "seaborn>=0.13,<1.0" ^
    "tqdm>=4.65,<5.0"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Some packages failed to install. Check the output above.
    pause
    exit /b 1
)

REM --- Setup .env ---
cd /d "%~dp0"
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo [INFO] Created .env from .env.example. Please edit it with your API keys.
    )
)

REM --- Verify installation ---
echo.
echo ============================================================
echo   Verifying installation...
echo ============================================================
python -c "import tensorflow as tf; print(f'  TensorFlow: {tf.__version__}, GPU: {len(tf.config.list_physical_devices(\"GPU\"))} device(s)')"
python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import deepface; print(f'  DeepFace: {deepface.__version__}')"
python -c "import flask; print(f'  Flask: {flask.__version__}')"

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo   To start the backend:
echo     conda activate SynCVE
echo     python -m src.backend.app
echo.
echo   To start the frontend:
echo     cd src\frontend
echo     npm install --legacy-peer-deps
echo     npm start
echo.
pause
