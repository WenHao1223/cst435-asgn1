# Sequential vs Parallel Processing in Matrix Multiplication: A Speedup and Efficiency Study using OpenMP and CUDA

CST435 Assignment 1. Benchmarks the same matrix multiplication on the sequential
CPU, the parallel CPU (OpenMP) and the GPU (CUDA), then reports execution time,
speedup and OpenMP parallel efficiency.

Works on **Windows** (`build.bat`, `run_all.bat`) and **Linux / macOS**
(`build.sh`, `run_all.sh`). Commands for each are shown side by side below.

## Requirements

| | Windows 10/11 | Linux | macOS |
| --- | --- | --- | --- |
| C++ compiler with OpenMP | Visual Studio Build Tools, "Desktop development with C++" workload | `g++` (`sudo apt install build-essential`) | Homebrew GCC: `brew install gcc` (Apple's clang has no OpenMP) |
| CUDA (for `cuda` and `benchmark`) | NVIDIA GPU + driver, CUDA Toolkit (`CUDA_PATH` set by the installer) | NVIDIA GPU + driver, CUDA Toolkit (`nvcc` on `PATH`) | **Not available** (no CUDA on macOS) |
| Python 3 with `matplotlib` (charts only) | `python` | `python3` | `python3` |

Without CUDA you can still build and run `sequential` and `openmp`. The Linux/macOS
scripts detect a missing `nvcc` and skip the CUDA programs automatically.

Install matplotlib with `python -m pip install matplotlib` (Windows) or
`python3 -m pip install matplotlib` (Linux/macOS).

## Files

| File | Purpose |
| --- | --- |
| `benchmark.cpp` | Standalone all-in-one benchmark. Runs all three implementations in one process (needs CUDA). |
| `common.h` | Shared sizes, seed, timing/validation helpers, run-folder and plot helpers. |
| `sequential.cpp` | Sequential CPU baseline. Writes raw readings, averages and its own graph. |
| `openmp.cpp` | OpenMP implementation with a thread sweep. Writes raw readings, averages (1-thread-baseline speedup/efficiency) and the strong-scaling graph. |
| `cuda.cu` | CUDA implementation. Writes raw GPU readings, averages and its own graph. |
| `compare.cpp` | Merges the three programs' averages into `averages.csv` and draws the comparison charts. |
| `plot_benchmark.py` | Chart generation (called by every program, or run by hand). |
| `build.bat` / `build.sh` | Builds every program, or a single source file (Windows / Linux + macOS). |
| `run_all.bat` / `run_all.sh` | Runs the split pipeline into one timestamped folder (Windows / Linux + macOS). |

Each split program writes its own raw readings, averages and graph. `compare`
does not re-run or re-plot the per-program results; it only combines them.

## Build

Open a terminal in this folder.

```bash
cd src
```

Then,

**Windows (cmd / PowerShell)**

```bat
build.bat
```

Produces `benchmark.exe  sequential.exe  openmp.exe  cuda.exe  compare.exe`.

**Linux / macOS (bash / zsh)**

```bash
chmod +x build.sh run_all.sh    # first time only
./build.sh
```

Produces `benchmark  sequential  openmp  cuda  compare` (no `.exe`). If `nvcc` is
not found, `cuda` and `benchmark` are skipped and the rest are built.

### Build one source only

| Windows | Linux / macOS |
| --- | --- |
| `build.bat cuda.cu` | `./build.sh cuda.cu` |

### Choose the GPU architecture

Both scripts compile with `-arch=native`, so the executable contains code for the
GPU in the machine. To target a specific architecture instead:

| Windows | Linux / macOS |
| --- | --- |
| `set CUDA_ARCH=sm_89` then `build.bat` | `CUDA_ARCH=sm_89 ./build.sh` |

### Choose the compiler (Linux / macOS)

The script picks `g++`, then `g++-14/13/12`, then `clang++`. Override it with:

```bash
CXX=g++-14 ./build.sh
```

**macOS:** with Apple's default compiler the script prints a hint. Run
`brew install gcc` and build with `CXX=g++-14 ./build.sh` (use the version number
Homebrew installed). Alternatively `brew install libomp` and the script will try
`-Xpreprocessor -fopenmp` with Apple clang.

## Run

### Option A - split pipeline (recommended)

**Windows**

```bat
run_all.bat
```

**Linux / macOS**

```bash
./run_all.sh
```

The script creates one timestamped folder, e.g. `results/2026-10-21_01-15-42/`,
then runs `sequential`, `openmp`, `cuda` and `compare` with that folder. Every run
gets a new folder, so earlier results are never overwritten.

`run_all.sh` skips `cuda` if it was not built. `compare` needs
`cuda_averages.csv`, so on a machine without a GPU it is skipped too (see
[No GPU on this machine](#no-gpu-on-this-machine)).

To run the programs:

**Windows**

```bat
.\sequential.exe
.\openmp.exe
.\cuda.exe
.\compare.exe
```

**Linux / macOS**

```bash
./sequential
./openmp
./cuda
./compare
```

`sequential`, `openmp` and `cuda` are independent: each writes its own files and
can run alone or in any order. If you omit the folder, each program creates its
own timestamped folder. `compare` then picks the newest averages file of each
kind under `results/` and writes the merged output to a new timestamped folder.
Pass a shared folder (or use `run_all`) when you want `compare` to use one
specific pipeline run.

### Option B - all-in-one benchmark (needs CUDA)

| Windows | Linux |
| --- | --- |
| `.\benchmark.exe` | `./benchmark` |

This runs sequential, OpenMP (including the thread sweep) and CUDA in one go and
writes everything to a timestamped folder, e.g. `results/2026-10-21_01-15-42/`.
It is independent of the split programs.

### No GPU on this machine

Only one teammate needs an NVIDIA GPU to produce the final comparison. On a
machine without CUDA, run `sequential` and `openmp` and send the resulting
`sequential_averages.csv` and `openmp_averages.csv` to the teammate with the GPU.
Put all three files in one folder and run `compare` on that folder:

| Windows | Linux |
| --- | --- |
| `.\compare.exe` | `./compare` |

Note that numbers from different machines are not directly comparable, so record
each machine's CPU, core count and GPU in the report.

## Validation

`sequential` runs two checks before benchmarking and names each one:

```
[PASS] 2x2 known result: [[1,2],[3,4]] x [[5,6],[7,8]] == [[19,22],[43,50]]
[PASS] 64x64 identity: A x I equals A
Validation PASSED: 2/2 checks passed
```

`openmp` and `cuda` compare every result against the built-in sequential
reference in `common.h` and print `yes`/`NO` in the `Matches_Sequential` column.
If a check fails, the program exits before writing results (sequential) or
reports `NO` for that size/thread count.

## Output

`run_all` writes everything into one timestamped folder. Each program owns its
files:

| Program | Files |
| --- | --- |
| `sequential` | `sequential_results.csv`, `sequential_averages.csv`, `sequential_execution_time.png` |
| `openmp` | `openmp_results.csv`, `openmp_averages.csv`, `openmp_strong_scaling.png` |
| `cuda` | `cuda_results.csv`, `cuda_averages.csv`, `cuda_execution_time.png` |
| `compare` | `averages.csv`, `execution_time_comparison.png`, `speedup_vs_sequential_cpu.png`, `gpu_speedup_over_parallel_cpu.png` |

CSV columns:

| File | Columns |
| --- | --- |
| `sequential_results.csv` | `Matrix_Size, Run, Sequential_CPU_ms` |
| `sequential_averages.csv` | `Matrix_Size, Sequential_CPU_ms` |
| `openmp_results.csv` | `Matrix_Size, Threads, Run, Parallel_CPU_ms` |
| `openmp_averages.csv` | `Matrix_Size, Threads, Parallel_CPU_ms, Speedup_vs_1_Thread, Parallel_Efficiency` |
| `cuda_results.csv` | `Matrix_Size, Run, Parallel_GPU_ms` |
| `cuda_averages.csv` | `Matrix_Size, Parallel_GPU_ms` |
| `averages.csv` | `Matrix_Size, Sequential_CPU_ms, Parallel_CPU_ms, Parallel_GPU_ms, Speedup_CPU_vs_Seq, Speedup_GPU_vs_CPU, Speedup_GPU_vs_Seq` |

`compare` uses the OpenMP averages at the highest thread count for
`Parallel_CPU_ms`, matching the all-in-one benchmark.

Charts:

- `sequential_execution_time.png` - sequential time vs matrix size (log scale).
- `cuda_execution_time.png` - the whole GPU operation (copies, kernel and copy back) vs matrix size.
- `openmp_strong_scaling.png` - OpenMP speedup and efficiency vs thread count, using the 1-thread OpenMP run as the baseline.
- `execution_time_comparison.png` - all three implementations vs matrix size.
- `speedup_vs_sequential_cpu.png` - OpenMP and CUDA speedup vs sequential CPU.
- `gpu_speedup_over_parallel_cpu.png` - CUDA speedup over the parallel CPU.

To redraw one program's charts for an existing folder (use `python` on Windows,
`python3` on Linux/macOS):

**Windows**

```bat
python plot_benchmark.py sequential results\2026-10-21_01-15-42
python plot_benchmark.py openmp     results\2026-10-21_01-15-42
python plot_benchmark.py cuda       results\2026-10-21_01-15-42
python plot_benchmark.py compare    results\2026-10-21_01-15-42
```

**Linux / macOS**

```bash
python3 plot_benchmark.py sequential results/2026-10-21_01-15-42
python3 plot_benchmark.py openmp     results/2026-10-21_01-15-42
python3 plot_benchmark.py cuda       results/2026-10-21_01-15-42
python3 plot_benchmark.py compare    results/2026-10-21_01-15-42
```

Running `plot_benchmark.py <folder>` without a mode still draws every chart,
which is what `benchmark` expects.

## How the metrics are calculated

- **Speedup** in `averages.csv` = sequential time / parallel time for the same
  matrix size.
- **OpenMP speedup** in `openmp_averages.csv` = 1-thread OpenMP time / N-thread
  OpenMP time. **Parallel efficiency** = that speedup / number of threads. Using
  the program's own 1-thread run keeps `openmp` independent of `sequential`;
  `compare` still measures CPU speedup against the sequential baseline.
- **GPU efficiency is intentionally not reported.** CUDA threads/cores are not
  directly comparable to CPU threads, so the GPU is discussed using execution
  time and speedup trends. `Parallel_GPU_ms` measures the whole GPU operation,
  the same way as `benchmark.cpp`: host-to-device copies, the kernel and the
  copy back.

## Configuration

Edit `common.h` to change the experiment:

```cpp
const int READINGS_PER_SIZE = 10;
const std::vector<int> MATRIX_SIZES = {128, 256, 512, 1024, 2048, 3072, 4096};
const unsigned SEED = 42;
```

The defaults are expensive: each size is repeated 10 times, the sequential
baseline at 4096 is slow, and the OpenMP thread sweep repeats every size for each
thread count. For a quick check, reduce `MATRIX_SIZES` and `READINGS_PER_SIZE`
before building.

`benchmark.cpp` keeps its own `matrix_sizes` list in `main`, so edit that file
too if you use the all-in-one program.

## Troubleshooting

- **`CUDA Error ... the provided PTX was compiled with an unsupported toolchain`**
  The executable was built without an architecture flag, or the driver is older
  than the toolkit. Rebuild with the build script (it uses `-arch=native`), or set
  `CUDA_ARCH` to the GPU's compute capability.
- **`ERROR: ... not found` from `compare`**
  At least one program has not written its averages yet. Run the missing
  program(s), or pass the shared run folder to `compare`. Without a GPU there is
  no `cuda_averages.csv`; see [No GPU on this machine](#no-gpu-on-this-machine).
- **Charts are not generated / `matplotlib` error**
  Install it with `python -m pip install matplotlib` (Windows) or
  `python3 -m pip install matplotlib` (Linux/macOS), then rerun the program or
  `plot_benchmark.py <mode> <folder>`.
- **`Permission denied` running `./build.sh` or `./run_all.sh`**
  Run `chmod +x build.sh run_all.sh` once.
- **`Skipping cuda.cu (nvcc not found ...)` on Linux**
  Install the CUDA Toolkit and make sure `nvcc` is on `PATH`
  (e.g. `export PATH=/usr/local/cuda/bin:$PATH`).
- **macOS: `-fopenmp` not supported / `omp.h` not found**
  Apple clang has no OpenMP. Use `brew install gcc` and `CXX=g++-14 ./build.sh`.
