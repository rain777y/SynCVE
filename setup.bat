@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Enable ANSI escape support for colored output
for /f "delims=" %%E in ('echo prompt $E^| cmd') do set "ESC=%%E"
set "OK=%ESC%[92m[OK]%ESC%[0m"
set "WARN=%ESC%[93m[WARN]%ESC%[0m"
set "ERR=%ESC%[91m[ERROR]%ESC%[0m"
set "INFO=%ESC%[96m[INFO]%ESC%[0m"

set "ROOT=%~dp0"
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

:: ========================================================================
:: [1/7] Check prerequisites
:: ========================================================================
echo %INFO% [1/7] Checking prerequisites...

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

:: ========================================================================
:: [2/7] Set up Python environment
:: ========================================================================
echo %INFO% [2/7] Setting up Python environment...

call conda env list 2>nul | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   Creating conda environment "%ENV_NAME%" with Python 3.10...
    call conda create -n %ENV_NAME% python=3.10 -y
    if errorlevel 1 (
        echo %ERR% Failed to create conda environment.
        pause
        exit /b 1
    )
    echo   %OK% Conda environment "%ENV_NAME%" created.
) else (
    echo   %OK% Conda environment "%ENV_NAME%" already exists.
)
echo.

:: ========================================================================
:: [3/7] Install Python dependencies
:: ========================================================================
echo %INFO% [3/7] Installing Python dependencies...

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

echo   Installing pip packages from requirements.txt...
pip install -r requirements.txt --upgrade --no-cache-dir
if errorlevel 1 (
    echo %WARN% Some pip packages may have failed. Check output above.
) else (
    echo   %OK% Python dependencies installed.
)
echo.

:: ========================================================================
:: [4/7] Check GPU availability
:: ========================================================================
echo %INFO% [4/7] Checking GPU availability...

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
    echo   %WARN% Could not check PyTorch CUDA. torch may not be installed yet.
)
echo.

:: ========================================================================
:: [5/7] Install frontend dependencies
:: ========================================================================
echo %INFO% [5/7] Installing frontend dependencies...

if not exist "%ROOT%src\frontend\package.json" (
    echo %ERR% Frontend package.json not found at "%ROOT%src\frontend\package.json".
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

:: ========================================================================
:: [6/7] Set up environment files
:: ========================================================================
echo %INFO% [6/7] Setting up environment files...

if not exist "%ROOT%src\backend\backend.env" (
    if exist "%ROOT%src\backend\backend.env.example" (
        copy "%ROOT%src\backend\backend.env.example" "%ROOT%src\backend\backend.env" >nul
        echo   %OK% Created src\backend\backend.env from template.
        echo   %WARN% EDIT src\backend\backend.env with your Supabase and Gemini API keys!
    ) else (
        echo   %WARN% No backend.env.example found. Create src\backend\backend.env manually.
    )
) else (
    echo   %OK% src\backend\backend.env already exists.
)

if not exist "%ROOT%src\frontend\.env" (
    if exist "%ROOT%src\frontend\.env.example" (
        copy "%ROOT%src\frontend\.env.example" "%ROOT%src\frontend\.env" >nul
        echo   %OK% Created src\frontend\.env from template.
        echo   %WARN% EDIT src\frontend\.env with your Supabase keys!
    ) else (
        echo   %WARN% No frontend .env.example found. Create src\frontend\.env manually.
    )
) else (
    echo   %OK% src\frontend\.env already exists.
)
echo.

:: ========================================================================
:: [7/7] Verify setup
:: ========================================================================
echo %INFO% [7/7] Verifying setup...

set "VERIFY_OK=1"

python -c "import flask; print(f'  Flask {flask.__version__}')" 2>nul
if errorlevel 1 (
    echo   %ERR% Flask not importable.
    set "VERIFY_OK=0"
)

python -c "import deepface; print(f'  DeepFace {deepface.__version__}')" 2>nul
if errorlevel 1 (
    echo   %ERR% DeepFace not importable.
    set "VERIFY_OK=0"
)

python -c "import tensorflow as tf; print(f'  TensorFlow {tf.__version__}')" 2>nul
if errorlevel 1 (
    echo   %ERR% TensorFlow not importable.
    set "VERIFY_OK=0"
)

python -c "import torch; print(f'  PyTorch {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo   %ERR% PyTorch not importable.
    set "VERIFY_OK=0"
)

python -c "import cv2; print(f'  OpenCV {cv2.__version__}')" 2>nul
if errorlevel 1 (
    echo   %ERR% OpenCV not importable.
    set "VERIFY_OK=0"
)

if "!VERIFY_OK!"=="1" (
    echo   %OK% All core packages verified.
) else (
    echo   %WARN% Some packages could not be verified. See errors above.
)

echo.
echo %ESC%[92m====================================%ESC%[0m
echo %ESC%[92m  Setup Complete!%ESC%[0m
echo %ESC%[92m====================================%ESC%[0m
echo.
echo   Next steps:
echo     1. Edit src\backend\backend.env with your API keys (Supabase, Gemini)
echo     2. Edit src\frontend\.env with your Supabase keys
echo     3. Run start_backend.bat to start the backend
echo     4. Run start_frontend.bat to start the frontend
echo     5. Open http://localhost:3000 in your browser
echo.
echo   Optional:
echo     - Run: python scripts\health_check.py   for a full health check
echo     - Run: pip install -r requirements-dev.txt   for dev tools
echo.
pause
