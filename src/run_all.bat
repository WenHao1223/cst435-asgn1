@echo off
setlocal
cd /d "%~dp0"

rem Run the split pipeline into ONE timestamped folder so compare.exe can
rem find the averages written by the other three programs.
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set "STAMP=%%i"
if not defined STAMP (
    echo ERROR: could not create a timestamp.
    exit /b 1
)

set "RUN_DIR=results\%STAMP%"
if exist "%RUN_DIR%" (
    for /l %%n in (1,1,100) do (
        if not exist "results\%STAMP%_%%n" (
            set "RUN_DIR=results\%STAMP%_%%n"
            goto :folder_ready
        )
    )
)
:folder_ready
mkdir "%RUN_DIR%" 2>nul

echo Run folder: %RUN_DIR%
echo.

call .\sequential.exe "%RUN_DIR%"
if errorlevel 1 goto :failed

call .\openmp.exe "%RUN_DIR%"
if errorlevel 1 goto :failed

call .\cuda.exe "%RUN_DIR%"
if errorlevel 1 goto :failed

call .\compare.exe "%RUN_DIR%"
if errorlevel 1 goto :failed

echo.
echo Done. Results and charts are in %RUN_DIR%.
exit /b 0

:failed
echo.
echo Pipeline stopped: a program failed. See the error above.
exit /b 1
