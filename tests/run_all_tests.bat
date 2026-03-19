@echo off
echo ========================================
echo   SynCVE Complete Test Suite
echo ========================================

call conda activate SynCVE

echo.
echo [1/3] Unit Tests (no backend required)...
python -m pytest src/backend/tests/ -v --tb=short 2>&1

echo.
echo [2/3] Integration Tests (backend required)...
curl -s http://localhost:5005/ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python -m pytest tests/integration/ -v --tb=short -s 2>&1
) else (
    echo SKIPPED: Backend not running
)

echo.
echo [3/3] Regression Tests (backend required)...
curl -s http://localhost:5005/ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python -m pytest tests/regression/ -v --tb=short -s 2>&1
) else (
    echo SKIPPED: Backend not running
)

echo.
echo ========================================
echo   All Tests Complete
echo ========================================
pause
