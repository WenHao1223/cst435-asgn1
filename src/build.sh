#!/usr/bin/env bash
# Linux / macOS build script (Windows users: use build.bat).
#   ./build.sh              -> build everything that this machine supports
#   ./build.sh openmp.cpp   -> build one source
# CUDA programs (cuda.cu, benchmark.cpp) are skipped automatically when nvcc is missing.
# Override the compiler with:  CXX=g++-14 ./build.sh     GPU arch: CUDA_ARCH=sm_86 ./build.sh
set -u
cd "$(dirname "$0")"

SOURCES="${1:-sequential.cpp openmp.cpp compare.cpp cuda.cu benchmark.cpp}"
CUDA_ARCH="${CUDA_ARCH:-native}"
FAILED=0

# --- pick a C++ compiler that supports OpenMP ---------------------------------
if [ -z "${CXX:-}" ]; then
  for c in g++ g++-14 g++-13 g++-12 clang++; do
    command -v "$c" >/dev/null 2>&1 && CXX="$c" && break
  done
fi
[ -z "${CXX:-}" ] && { echo "ERROR: no C++ compiler found (install g++)."; exit 1; }

OMP_FLAG="-fopenmp"
if [ "$(uname)" = "Darwin" ]; then
  # Apple clang has no OpenMP. Use Homebrew GCC (brew install gcc) or libomp.
  if "$CXX" --version 2>&1 | grep -qi "apple"; then
    echo "NOTE: Apple clang has no -fopenmp. Run: brew install gcc, then: CXX=g++-14 ./build.sh"
    echo "      (or brew install libomp and the script will try -Xpreprocessor -fopenmp)."
    if [ -d "$(brew --prefix libomp 2>/dev/null)" ]; then
      P="$(brew --prefix libomp)"
      OMP_FLAG="-Xpreprocessor -fopenmp -I$P/include -L$P/lib -lomp"
    fi
  fi
fi

echo "C++ compiler: $CXX"
HAVE_NVCC=0
command -v nvcc >/dev/null 2>&1 && HAVE_NVCC=1

for f in $SOURCES; do
  [ -f "$f" ] || { echo "ERROR: $f not found."; FAILED=1; continue; }
  out="${f%.*}"
  case "$f" in
    cuda.cu|benchmark.cpp)
      if [ "$HAVE_NVCC" = 0 ]; then
        echo "Skipping $f (nvcc not found - no CUDA toolkit on this machine)."
        continue
      fi
      echo "Building $f (nvcc) ..."
      nvcc -x cu -O3 -std=c++17 -arch="$CUDA_ARCH" "$f" -o "$out" -Xcompiler "-O2,-fopenmp" || FAILED=1
      ;;
    *)
      echo "Building $f ..."
      # shellcheck disable=SC2086
      "$CXX" -O2 -std=c++17 $OMP_FLAG "$f" -o "$out" || FAILED=1
      ;;
  esac
done

[ "$FAILED" = 1 ] && { echo "Build FAILED."; exit 1; }
echo "Build succeeded."