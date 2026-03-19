@echo off
setlocal EnableExtensions EnableDelayedExpansion

for /f "delims=" %%E in ('echo prompt $E^| cmd') do set "ESC=%%E"
set "OK=%ESC%[92m[OK]%ESC%[0m"
set "WARN=%ESC%[93m[WARN]%ESC%[0m"
set "ERR=%ESC%[91m[ERROR]%ESC%[0m"
set "INFO=%ESC%[96m[INFO]%ESC%[0m"

set "ROOT=%~dp0..\"
set "FRONTEND_DIR=%ROOT%src\frontend"
set "FRONTEND_PORT=3000"

cd /d "%ROOT%" || (
    echo %ERR% Failed to change to project root "%ROOT%".
    pause
    exit /b 1
)

echo.
echo %INFO% SynCVE Frontend Launcher
echo %INFO% Project root: %ROOT%
echo.

call :checkNode || goto :fail
call :checkEnvFile
call :ensureFrontendDeps || goto :fail

echo.
echo %INFO% Starting Frontend in: %FRONTEND_DIR%
echo %INFO% Browser should open at http://localhost:%FRONTEND_PORT%
echo %INFO% Press Ctrl+C to stop.
echo.

cd /d "%FRONTEND_DIR%"
npm start

if errorlevel 1 (
    echo.
    echo %ERR% Frontend exited with error.
    goto :fail
)

echo %OK% Frontend stopped.
pause
exit /b 0

:fail
echo.
echo %ERR% Frontend startup failed.
pause
exit /b 1

:checkNode
node --version >nul 2>&1 || (
    echo %ERR% Node.js not found. Install Node.js 18+: https://nodejs.org/
    exit /b 1
)
for /f "delims=" %%N in ('node --version 2^>nul') do set "NODEVER=%%N"
set "NODEMAJOR=!NODEVER:~1!"
for /f "delims=." %%V in ("!NODEMAJOR!") do set "NODEMAJOR=%%V"
if !NODEMAJOR! LSS 18 (
    echo %WARN% Node.js %NODEVER% detected. Version 18+ recommended.
) else (
    echo %OK% Node.js %NODEVER% detected.
)
call npm --version >nul 2>&1 || (
    echo %ERR% npm not found.
    exit /b 1
)
for /f "delims=" %%M in ('call npm --version 2^>nul') do set "NPMVER=%%M"
echo %OK% npm %NPMVER% detected.
exit /b 0

:checkEnvFile
if not exist "%ROOT%.env" (
    echo %WARN% Root .env not found.
    if exist "%ROOT%.env.example" (
        echo %INFO% Copy template: copy .env.example .env
    )
) else (
    echo %OK% Root .env found.
)
exit /b 0

:ensureFrontendDeps
if not exist "%FRONTEND_DIR%\package.json" (
    echo %ERR% package.json missing.
    exit /b 1
)
if exist "%FRONTEND_DIR%\node_modules" (
    echo %OK% Frontend dependencies installed.
    exit /b 0
)
echo %INFO% node_modules not found. Installing...
pushd "%FRONTEND_DIR%" >nul
call npm install --no-fund --no-audit
set "NPM_EXIT=!ERRORLEVEL!"
popd >nul
if not "!NPM_EXIT!"=="0" (
    echo %ERR% npm install failed.
    exit /b 1
)
echo %OK% Frontend dependencies installed.
exit /b 0
