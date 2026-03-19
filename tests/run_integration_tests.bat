@echo off
echo ========================================
echo   SynCVE Integration Test Suite
echo ========================================
echo.
echo PREREQUISITES:
echo   - Backend must be running (start_backend.bat)
echo   - GPU should be available for performance tests
echo.
echo Checking backend availability...
curl -s http://localhost:5005/ >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Backend not running at localhost:5005
    echo Please start the backend first: start_backend.bat
    pause
    exit /b 1
)
echo Backend is running.
echo.

call conda activate SynCVE

echo --- Running Integration Tests ---
python -m pytest tests/integration/ -v --tb=short -s 2>&1

echo.
echo --- Running Regression Tests ---
python -m pytest tests/regression/ -v --tb=short -s 2>&1

echo.
echo ========================================
echo   Tests Complete
echo ========================================
pause
