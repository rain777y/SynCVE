@echo off
setlocal EnableExtensions EnableDelayedExpansion

for /f "delims=" %%E in ('echo prompt $E^| cmd') do set "ESC=%%E"
set "OK=%ESC%[92m[OK]%ESC%[0m"
set "WARN=%ESC%[93m[WARN]%ESC%[0m"
set "ERR=%ESC%[91m[ERROR]%ESC%[0m"
set "INFO=%ESC%[96m[INFO]%ESC%[0m"

set "ROOT=%~dp0..\"
set "ENV_NAME=SynCVE"

cd /d "%ROOT%" || (
    echo %ERR% Failed to change to project root "%ROOT%".
    pause
    exit /b 1
)

echo.
echo %ESC%[96m====================================%ESC%[0m
echo %ESC%[96m  SynCVE - First Time Setup%ESC%[0m
echo %ESC%[96m====================================%ESC%[0m
echo.

:: [1/8] Check prerequisites
echo %INFO% [1/8] Checking prerequisites...

where conda >nul 2>&1 || (
    echo %ERR% Conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)
for /f "delims=" %%C in ('where conda 2^>nul') do (
    set "CONDA_PATH=%%C"
    goto :condaDone
)
:condaDone
echo   %OK% Conda found at "!CONDA_PATH!"

where node >nul 2>&1 || (
    echo %ERR% Node.js not found. Install Node.js 18+: https://nodejs.org/
    pause
    exit /b 1
)
for /f "delims=" %%N in ('node --version 2^>nul') do set "NODEVER=%%N"
echo   %OK% Node.js found: %NODEVER%

where npm >nul 2>&1 || (
    echo %ERR% npm not found. Ensure Node.js is properly installed.
    pause
    exit /b 1
)
for /f "delims=" %%M in ('call npm --version 2^>nul') do set "NPMVER=%%M"
echo   %OK% npm found: %NPMVER%
echo.

:: [2/8] Create conda environment (Python + CUDA + cuDNN)
echo %INFO% [2/8] Setting up Python environment with GPU support...

call conda env list 2>nul | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   Creating conda environment "%ENV_NAME%" from environment.yml...
    call conda env create -f "%ROOT%environment.yml"
    if errorlevel 1 (
        echo %ERR% Failed to create conda environment.
        pause
        exit /b 1
    )
    echo   %OK% Conda environment "%ENV_NAME%" created.
) else (
    echo   %OK% Conda environment "%ENV_NAME%" already exists.
    echo   Updating from environment.yml...
    call conda env update -n %ENV_NAME% -f "%ROOT%environment.yml" --prune
)
echo.

:: [3/8] Activate environment
echo %INFO% [3/8] Activating environment...
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo %ERR% Failed to activate conda environment "%ENV_NAME%".
    echo %INFO% Try running: conda init cmd.exe   then restart your terminal.
    pause
    exit /b 1
)
echo   %OK% Activated conda environment "%ENV_NAME%".

for /f "delims=" %%P in ('python --version 2^>nul') do set "PYVER=%%P"
echo   %OK% Python: %PYVER%
echo.

:: [4/8] Install PyTorch with CUDA support (special index URL)
echo %INFO% [4/8] Installing PyTorch with CUDA 11.8 support...

python -c "import torch; print(torch.cuda.is_available())" 2>nul | findstr "True" >nul 2>&1
if errorlevel 1 (
    echo   Installing PyTorch CUDA from pytorch.org...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo %WARN% PyTorch CUDA install failed. GPU will not be available.
    ) else (
        echo   %OK% PyTorch CUDA installed.
    )
) else (
    echo   %OK% PyTorch CUDA already available.
)
echo.

:: [5/8] Check GPU availability
echo %INFO% [5/8] Checking GPU availability...

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo   %OK% NVIDIA GPU driver detected.
    for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
        echo   %OK% GPU: %%G
    )
) else (
    echo   %WARN% nvidia-smi not found. GPU acceleration may not be available.
)

python -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else 'N/A'; print(f'  PyTorch CUDA: {gpu}, Device: {name}')" 2>nul
if errorlevel 1 (
    echo   %WARN% Could not check PyTorch CUDA.
)

python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'  TensorFlow GPU: {len(gpus)} device(s)')" 2>nul
echo.

:: [6/8] Install frontend dependencies
echo %INFO% [6/8] Installing frontend dependencies...

if not exist "%ROOT%src\frontend\package.json" (
    echo %ERR% Frontend package.json not found.
    pause
    exit /b 1
)

pushd "%ROOT%src\frontend" >nul
call npm install --no-fund --no-audit
set "NPM_EXIT=!ERRORLEVEL!"
popd >nul

if not "!NPM_EXIT!"=="0" (
    echo %ERR% npm install failed.
    pause
    exit /b 1
)
echo   %OK% Frontend dependencies installed.
echo.

:: [7/8] Set up environment files
echo %INFO% [7/8] Setting up environment files...

if not exist "%ROOT%.env" (
    if exist "%ROOT%.env.example" (
        copy "%ROOT%.env.example" "%ROOT%.env" >nul
        echo   %OK% Created .env from template.
        echo   %WARN% EDIT .env with your API keys (Supabase, Gemini)!
    )
) else (
    echo   %OK% .env already exists.
)
echo   %OK% Application settings in settings.yml (no edits needed).
echo.

:: [8/8] Verify setup
echo %INFO% [8/8] Verifying setup...

set "VERIFY_OK=1"
python -c "import flask; print(f'  Flask {flask.__version__}')" 2>nul || set "VERIFY_OK=0"
python -c "import deepface; print(f'  DeepFace {deepface.__version__}')" 2>nul || set "VERIFY_OK=0"
python -c "import tensorflow as tf; print(f'  TensorFlow {tf.__version__}')" 2>nul || set "VERIFY_OK=0"
python -c "import torch; print(f'  PyTorch {torch.__version__} (CUDA={torch.cuda.is_available()})')" 2>nul || set "VERIFY_OK=0"
python -c "import cv2; print(f'  OpenCV {cv2.__version__}')" 2>nul || set "VERIFY_OK=0"
python -c "import sklearn; print(f'  scikit-learn {sklearn.__version__}')" 2>nul || set "VERIFY_OK=0"

echo.
echo %ESC%[92m====================================%ESC%[0m
echo %ESC%[92m  Setup Complete!%ESC%[0m
echo %ESC%[92m====================================%ESC%[0m
echo.
echo   Next steps:
echo     1. Edit .env with your API keys
echo     2. Run scripts\start_backend.bat
echo     3. Run scripts\start_frontend.bat
echo     4. Open http://localhost:3000
echo.
echo   For evaluation:
echo     conda activate SynCVE
echo     python eval/run_all.py --limit 500
echo.
pause
