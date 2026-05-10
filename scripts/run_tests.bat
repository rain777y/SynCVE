@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0..\"
cd /d "%ROOT%"

echo ========================================
echo   SynCVE Test Suite
echo ========================================

call conda activate SynCVE

echo.
echo [1/3] Unit Tests (no backend required)...
echo ----------------------------------------
python -m pytest tests/unit/ -v --tb=short
set UNIT_EXIT=%ERRORLEVEL%

echo.
echo [2/3] Integration Tests (backend required)...
echo ----------------------------------------
curl -s http://localhost:5005/ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python -m pytest tests/integration/ -v --tb=short -s
    set INTEG_EXIT=!ERRORLEVEL!
) else (
    echo SKIPPED: Backend not running at localhost:5005
    set INTEG_EXIT=-1
)

echo.
echo [3/3] Regression Tests (backend required)...
echo ----------------------------------------
curl -s http://localhost:5005/ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python -m pytest tests/regression/ -v --tb=short -s
    set REGRESS_EXIT=!ERRORLEVEL!
) else (
    echo SKIPPED: Backend not running
    set REGRESS_EXIT=-1
)

echo.
echo ========================================
echo   Results
echo ========================================
if %UNIT_EXIT% equ 0 (echo   Unit:        PASSED) else (echo   Unit:        FAILED)
if !INTEG_EXIT! equ -1 (echo   Integration: SKIPPED) else if !INTEG_EXIT! equ 0 (echo   Integration: PASSED) else (echo   Integration: FAILED)
if !REGRESS_EXIT! equ -1 (echo   Regression:  SKIPPED) else if !REGRESS_EXIT! equ 0 (echo   Regression:  PASSED) else (echo   Regression:  FAILED)
echo ========================================

echo.
echo   Frontend Tests: cd src\frontend ^&^& npx react-scripts test --watchAll=false
echo.
pause
exit /b %UNIT_EXIT%
