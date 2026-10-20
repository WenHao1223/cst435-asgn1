@echo off
setlocal
cd /d "%~dp0"

rem Build every program when no argument is given:
rem   build.bat              -> benchmark.exe, sequential.exe, openmp.exe,
rem                             cuda.exe, compare.exe
rem Build a single source:
rem   build.bat cuda.cu    -> cuda.exe
set "SOURCES=benchmark.cpp sequential.cpp openmp.cpp cuda.cu compare.cpp"
if not "%~1"=="" set "SOURCES=%~1"

rem CUDA_PATH is set machine-wide by the toolkit installer, but an already-open
rem terminal may not have picked it up yet, so fall back to the registry.
if not defined CUDA_PATH (
    for /f "tokens=2,*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v CUDA_PATH 2^>nul') do set "CUDA_PATH=%%b"
)
if not defined CUDA_PATH (
    echo ERROR: CUDA_PATH is not set. Install the NVIDIA CUDA Toolkit first.
    exit /b 1
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: Visual Studio Build Tools not found ^(no vswhere.exe^).
    echo        nvcc needs the MSVC toolchain ^(cl.exe^) as its host compiler.
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
    echo ERROR: Could not locate a Visual Studio installation with the C++ toolset.
    exit /b 1
)

call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

rem Build for the local GPU (SASS) instead of generic PTX. Newer toolkits emit
rem PTX that older drivers reject with "unsupported toolchain"; targeting the
rem local architecture avoids that. Override with: set CUDA_ARCH=sm_89
if not defined CUDA_ARCH set "CUDA_ARCH=native"

set "FAILED=0"
for %%F in (%SOURCES%) do (
    if not exist "%%F" (
        echo ERROR: Source file "%%F" not found.
        set "FAILED=1"
    ) else (
        echo Building %%F ...
        "%CUDA_PATH%\bin\nvcc.exe" -x cu -O3 -std=c++17 -arch=%CUDA_ARCH% "%%F" -o "%%~nF.exe" -Xcompiler "/O2 /openmp"
        if errorlevel 1 set "FAILED=1"
    )
)

if "%FAILED%"=="1" (
    echo Build FAILED.
    exit /b 1
)

echo Build succeeded: %~dp0
endlocal
