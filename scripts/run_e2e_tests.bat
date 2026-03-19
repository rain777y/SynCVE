@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\"
cd /d "%ROOT%"

echo ========================================
echo   SynCVE E2E Test Suite
echo   Real APIs, Real Database, Real Inference
echo ========================================
echo.

:: ---- Activate conda environment ----
call conda activate SynCVE 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Could not activate conda env 'SynCVE'. Trying current Python...
)

:: ---- Load secrets from .env ----
set "ENV_FILE=%ROOT%.env"
if exist "%ENV_FILE%" (
    echo [INFO] Loading environment variables from .env
    for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
        set "LINE=%%A"
        :: Skip comments and empty lines
        if not "!LINE:~0,1!"=="#" (
            if not "%%A"=="" (
                set "%%A=%%B"
            )
        )
    )
) else (
    echo [WARN] .env not found at %ENV_FILE%
    echo        E2E tests will skip if GEMINI_API_KEY / SUPABASE_URL / SUPABASE_KEY are not set.
)

:: ---- Display credential status ----
echo.
if defined GEMINI_API_KEY (
    echo   GEMINI_API_KEY:  set
) else (
    echo   GEMINI_API_KEY:  NOT SET
)
if defined SUPABASE_URL (
    echo   SUPABASE_URL:    set
) else (
    echo   SUPABASE_URL:    NOT SET
)
if defined SUPABASE_KEY (
    echo   SUPABASE_KEY:    set
) else (
    echo   SUPABASE_KEY:    NOT SET
)
echo.

:: ---- Protobuf compatibility ----
set "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"

:: ---- Parse arguments ----
set "EXTRA_ARGS="
set "VERBOSE=-v"
set "MARKER="

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--fast" (
    set "MARKER=-m \"not slow\""
    shift
    goto parse_args
)
if /i "%~1"=="--verbose" (
    set "VERBOSE=-v -s"
    shift
    goto parse_args
)
if /i "%~1"=="--gemini-only" (
    set "EXTRA_ARGS=tests/e2e/test_gemini_direct.py"
    shift
    goto parse_args
)
if /i "%~1"=="--pipeline-only" (
    set "EXTRA_ARGS=tests/e2e/test_real_environment.py"
    shift
    goto parse_args
)
if /i "%~1"=="--http-only" (
    set "EXTRA_ARGS=tests/e2e/test_frontend_api_flow.py"
    shift
    goto parse_args
)
set "EXTRA_ARGS=%EXTRA_ARGS% %~1"
shift
goto parse_args
:done_args

:: ---- Default test path ----
if "%EXTRA_ARGS%"=="" (
    set "EXTRA_ARGS=tests/e2e/"
)

:: ---- Run tests ----
echo ========================================
echo   Running E2E Tests
echo ========================================
echo.

python -m pytest %EXTRA_ARGS% %VERBOSE% --tb=short %MARKER% -x --timeout=300
set E2E_EXIT=%ERRORLEVEL%

echo.
echo ========================================
echo   E2E Test Results
echo ========================================
if %E2E_EXIT% equ 0 (
    echo   Status: ALL PASSED
) else if %E2E_EXIT% equ 5 (
    echo   Status: NO TESTS COLLECTED (all skipped -- check API keys)
) else (
    echo   Status: FAILED (exit code %E2E_EXIT%)
)
echo ========================================
echo.

:: ---- Show generated assets ----
if exist "tests\assets\generated\" (
    echo   Generated test assets:
    dir /b tests\assets\generated\e2e_* 2>nul
    echo.
)

pause
exit /b %E2E_EXIT%
