@echo off
REM ============================================================================
REM SynCVE Test Runner
REM Runs both backend (pytest) and frontend (Jest) test suites.
REM ============================================================================

echo ============================================
echo  SynCVE Test Suite
echo ============================================

echo.
echo --- Backend Tests (pytest) ---
echo.
cd /d "%~dp0"
python -m pytest src/backend/tests -v --tb=short
set BACKEND_EXIT=%ERRORLEVEL%

echo.
echo --- Frontend Tests (Jest) ---
echo.
cd /d "%~dp0src\frontend"
call npx react-scripts test --watchAll=false --ci
set FRONTEND_EXIT=%ERRORLEVEL%

echo.
echo ============================================
echo  Results
echo ============================================
if %BACKEND_EXIT% equ 0 (
    echo  Backend:  PASSED
) else (
    echo  Backend:  FAILED (exit code %BACKEND_EXIT%)
)
if %FRONTEND_EXIT% equ 0 (
    echo  Frontend: PASSED
) else (
    echo  Frontend: FAILED (exit code %FRONTEND_EXIT%)
)
echo ============================================

cd /d "%~dp0"
if %BACKEND_EXIT% neq 0 exit /b %BACKEND_EXIT%
if %FRONTEND_EXIT% neq 0 exit /b %FRONTEND_EXIT%
exit /b 0
