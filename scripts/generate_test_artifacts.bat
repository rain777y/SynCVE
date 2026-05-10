@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\"
cd /d "%ROOT%"

echo ========================================
echo   SynCVE Test Artifact Generator
echo ========================================
echo.

call conda activate SynCVE

REM Load env vars from .env
if exist ".env" (
    echo Loading environment from .env...
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" (
            if not "%%B"=="" set "%%A=%%B"
        )
    )
)

set "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"

echo.
echo [1/2] Generating test images...
echo ----------------------------------------
python scripts\generate_test_images.py
set IMG_EXIT=%ERRORLEVEL%

echo.
echo [2/2] Generating test videos...
echo ----------------------------------------
python scripts\generate_test_video.py
set VID_EXIT=%ERRORLEVEL%

echo.
echo ========================================
echo   Results
echo ========================================
if %IMG_EXIT% equ 0 (echo   Images: DONE) else (echo   Images: FAILED)
if %VID_EXIT% equ 0 (echo   Videos: DONE) else (echo   Videos: FAILED)
echo ========================================
echo.

echo Artifacts directory:
dir /b tests\artifacts\images\ 2>nul
echo ---
dir /b tests\artifacts\videos\ 2>nul

echo.
pause
