@echo off
setlocal
cd /d "%~dp0"

rem Source file can be passed as the first argument (default: benchmark.cpp).
set "SOURCE=%~1"
if "%SOURCE%"=="" set "SOURCE=benchmark.cpp"
set "OUTPUT=%~n1"
if "%OUTPUT%"=="" set "OUTPUT=benchmark"

if not exist "%SOURCE%" (
    echo ERROR: Source file "%SOURCE%" not found.
    exit /b 1
)

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

"%CUDA_PATH%\bin\nvcc.exe" -x cu -O3 -std=c++17 -arch=%CUDA_ARCH% "%SOURCE%" -o "%OUTPUT%.exe" -Xcompiler "/O2 /openmp"
if errorlevel 1 (
    echo Build FAILED.
    exit /b 1
)

echo Build succeeded: %~dp0%OUTPUT%.exe
endlocal
