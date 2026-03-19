@echo off
setlocal EnableExtensions EnableDelayedExpansion

for /f "delims=" %%E in ('echo prompt $E^| cmd') do set "ESC=%%E"
set "OK=%ESC%[92m[OK]%ESC%[0m"
set "WARN=%ESC%[93m[WARN]%ESC%[0m"
set "ERR=%ESC%[91m[ERROR]%ESC%[0m"
set "INFO=%ESC%[96m[INFO]%ESC%[0m"

set "BACKEND_PORT=5005"
set "FRONTEND_PORT=3000"

echo %INFO% SynCVE service shutdown starting...
echo.

rem === Backend shutdown ===
call :findPidsOnPort %BACKEND_PORT% BACKEND_INITIAL_PIDS
call :reportStatus "Backend before shutdown" "%BACKEND_INITIAL_PIDS%" %BACKEND_PORT%

if defined BACKEND_INITIAL_PIDS (
    echo %INFO% Stopping backend...
    call :terminatePids NORMAL "%BACKEND_INITIAL_PIDS%"
    timeout /t 3 >nul 2>&1
    call :findPidsOnPort %BACKEND_PORT% BACKEND_REMAINING_PIDS
    if defined BACKEND_REMAINING_PIDS (
        echo %WARN% Backend still running; forcing...
        call :terminatePids FORCE "%BACKEND_REMAINING_PIDS%"
        timeout /t 2 >nul 2>&1
    )
) else (
    echo %WARN% Backend not running.
)
call :findPidsOnPort %BACKEND_PORT% BACKEND_FINAL_PIDS
call :reportStatus "Backend after shutdown" "%BACKEND_FINAL_PIDS%" %BACKEND_PORT%
echo.

rem === Frontend shutdown ===
call :findPidsOnPort %FRONTEND_PORT% FRONTEND_INITIAL_PIDS
call :reportStatus "Frontend before shutdown" "%FRONTEND_INITIAL_PIDS%" %FRONTEND_PORT%

if defined FRONTEND_INITIAL_PIDS (
    echo %INFO% Stopping frontend...
    call :terminatePids NORMAL "%FRONTEND_INITIAL_PIDS%"
    timeout /t 3 >nul 2>&1
    call :findPidsOnPort %FRONTEND_PORT% FRONTEND_REMAINING_PIDS
    if defined FRONTEND_REMAINING_PIDS (
        echo %WARN% Frontend still running; forcing...
        call :terminatePids FORCE "%FRONTEND_REMAINING_PIDS%"
        timeout /t 2 >nul 2>&1
    )
) else (
    echo %WARN% Frontend not running.
)
call :findPidsOnPort %FRONTEND_PORT% FRONTEND_FINAL_PIDS
call :reportStatus "Frontend after shutdown" "%FRONTEND_FINAL_PIDS%" %FRONTEND_PORT%
echo.

rem === Final status ===
set "EXIT_CODE=0"
if defined BACKEND_FINAL_PIDS (
    echo %ERR% Unable to clear port %BACKEND_PORT%.
    set "EXIT_CODE=1"
) else (
    echo %OK% Backend port %BACKEND_PORT% is free.
)
if defined FRONTEND_FINAL_PIDS (
    echo %ERR% Unable to clear port %FRONTEND_PORT%.
    set "EXIT_CODE=1"
) else (
    echo %OK% Frontend port %FRONTEND_PORT% is free.
)

if "%EXIT_CODE%"=="0" (
    echo.
    echo %OK% All SynCVE services terminated successfully.
)
exit /b %EXIT_CODE%

:findPidsOnPort
setlocal EnableDelayedExpansion
set "PORT_VALUE=%~1"
set "PID_CAPTURE="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /c:":%PORT_VALUE% " ^| findstr /i "LISTENING"') do (
    if defined PID_CAPTURE (set "PID_CAPTURE=!PID_CAPTURE! %%P") else (set "PID_CAPTURE=%%P")
)
endlocal & set "%~2=%PID_CAPTURE%"
exit /b 0

:reportStatus
set "LABEL=%~1"
set "PID_SET=%~2"
set "PORT=%~3"
if not defined PID_SET (
    echo %INFO% %LABEL%: No process on port %PORT%.
) else (
    echo %INFO% %LABEL%: PIDs on port %PORT%: %PID_SET%
)
exit /b 0

:terminatePids
set "MODE=%~1"
set "PID_LIST=%~2"
if "%PID_LIST%"=="" exit /b 0
for %%P in (%PID_LIST%) do (
    if /i "%MODE%"=="FORCE" (
        taskkill /PID %%P /F >nul 2>&1
    ) else (
        taskkill /PID %%P >nul 2>&1
    )
)
exit /b 0
