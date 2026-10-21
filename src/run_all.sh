#!/usr/bin/env bash
# Linux / macOS pipeline (Windows users: use run_all.bat).
# Runs sequential -> openmp -> cuda (if built) -> compare into ONE timestamped folder.
set -u
cd "$(dirname "$0")"

RUN_DIR="results/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RUN_DIR"
echo "Run folder: $RUN_DIR"; echo

run() { # name
  if [ ! -x "./$1" ]; then echo "Skipping $1 (not built)."; return 0; fi
  "./$1" "$RUN_DIR" || { echo; echo "Pipeline stopped: $1 failed."; exit 1; }
}

run sequential
run openmp
run cuda
if [ -x ./cuda ]; then
  run compare
else
  echo "Skipping compare: compare needs cuda_averages.csv, and this machine has no CUDA build."
  echo "Send your sequential/openmp CSVs to the teammate with the GPU and run compare there."
fi

echo; echo "Done. Results and charts are in $RUN_DIR."