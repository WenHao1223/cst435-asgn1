@echo off
setlocal
cd /d "%~dp0"

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

"%CUDA_PATH%\bin\nvcc.exe" -x cu -O3 benchmark.cpp -o benchmark.exe -Xcompiler "/O2 /openmp"
if errorlevel 1 (
    echo Build FAILED.
    exit /b 1
)

echo Build succeeded: %~dp0benchmark.exe
endlocal
