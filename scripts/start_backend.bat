@echo off
setlocal EnableExtensions EnableDelayedExpansion

for /f "delims=" %%E in ('echo prompt $E^| cmd') do set "ESC=%%E"
set "OK=%ESC%[92m[OK]%ESC%[0m"
set "WARN=%ESC%[93m[WARN]%ESC%[0m"
set "ERR=%ESC%[91m[ERROR]%ESC%[0m"
set "INFO=%ESC%[96m[INFO]%ESC%[0m"

set "ROOT=%~dp0..\"
set "ENV_NAME=SynCVE"
set "BACKEND_DIR=%ROOT%src\backend"
set "BACKEND_ENV_FILE=%ROOT%.env"
set "BACKEND_MODULE=src.backend.app"
set "BACKEND_PORT=5005"

cd /d "%ROOT%" || (
    echo %ERR% Failed to change to project root "%ROOT%".
    pause
    exit /b 1
)

echo.
echo %INFO% SynCVE Backend Launcher
echo %INFO% Project root: %ROOT%
echo.

call :checkConda || goto :fail
call :activateEnv || goto :fail
call :checkDependencies || goto :fail
call :checkEnvFile
call :checkPortAvailable
call :checkGPU

echo.
echo %INFO% Starting Backend module: %BACKEND_MODULE% on port %BACKEND_PORT%
echo %INFO% Press Ctrl+C to stop the server.
echo.
python -m %BACKEND_MODULE%

if errorlevel 1 (
    echo.
    echo %ERR% Backend crashed or exited with error.
    goto :fail
)

echo %OK% Backend stopped normally.
pause
exit /b 0

:fail
echo.
echo %ERR% Backend startup failed.
pause
exit /b 1

:checkConda
where conda >nul 2>&1 || (
    echo %ERR% Conda not found. Install Anaconda/Miniconda and ensure it is on PATH.
    exit /b 1
)
for /f "delims=" %%C in ('where conda 2^>nul') do (
    set "CONDA_PATH=%%C"
    goto :condaFound
)
:condaFound
echo %OK% Conda found at "!CONDA_PATH!"
exit /b 0

:activateEnv
call conda activate %ENV_NAME% >nul 2>&1
if not errorlevel 1 (
    echo %OK% Conda environment "%ENV_NAME%" activated.
    exit /b 0
)
echo %ERR% Conda environment "%ENV_NAME%" not found.
echo %INFO% Run scripts\setup.bat first, or: conda create -n %ENV_NAME% python=3.10 -y
exit /b 1

:checkDependencies
python --version >nul 2>&1 || (
    echo %ERR% Python not found after conda activation.
    exit /b 1
)
for /f "delims=" %%P in ('python --version 2^>nul') do set "PYVER=%%P"
echo %OK% Python detected: %PYVER%
exit /b 0

:checkEnvFile
if not exist "%BACKEND_ENV_FILE%" (
    echo %WARN% .env not found at "%BACKEND_ENV_FILE%".
    if exist "%ROOT%.env.example" (
        echo %INFO% Copy template: copy .env.example .env
    )
    echo %WARN% Backend will start with system environment variables only.
) else (
    echo %OK% Environment file found: %BACKEND_ENV_FILE%
)
exit /b 0

:checkPortAvailable
netstat -ano 2>nul | findstr /R ":%BACKEND_PORT% " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo %WARN% Port %BACKEND_PORT% is already in use!
    echo %INFO% Stop it first with: scripts\stop_service.bat
) else (
    echo %OK% Port %BACKEND_PORT% is available.
)
exit /b 0

:checkGPU
set "GPU_STATUS=None"
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set "GPU_STATUS=CUDA"
    echo %OK% GPU detected via nvidia-smi.
    for /f "tokens=*" %%G in ('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2^>nul') do (
        echo       %%G
    )
)
if "!GPU_STATUS!"=="None" (
    echo %WARN% No GPU detected; CPU mode will be used.
)
exit /b 0
