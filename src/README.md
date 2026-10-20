# CST435 Assignment 1 - Matrix Multiplication Benchmark

Benchmarks the same matrix multiplication on the sequential CPU, the parallel
CPU (OpenMP) and the GPU (CUDA), then reports execution time, speedup and
OpenMP parallel efficiency.

## Requirements

- Windows 10/11 (the build script uses `cmd` and MSVC).
- NVIDIA GPU with a current driver.
- CUDA Toolkit, with `CUDA_PATH` set by the installer.
- Visual Studio Build Tools with the "Desktop development with C++" workload.
- Python 3 with `matplotlib` (only needed to draw the charts).

## Files

| File | Purpose |
| --- | --- |
| `0-benchmark.cpp` | Standalone all-in-one benchmark. Runs all three implementations in one process. |
| `common.h` | Shared sizes, seed, timing helpers and the sequential reference. |
| `2-sequential.cpp` | Sequential CPU baseline, writes `sequential_results.csv`. |
| `3-openmp.cpp` | OpenMP implementation with a thread sweep, writes `openmp_results.csv`. |
| `4-cuda.cu` | CUDA kernel, writes `cuda_results.csv` (kernel and end-to-end times). |
| `5-compare.cpp` | Merges the three CSVs, computes speedup/efficiency and draws the charts. |
| `plot_benchmark.py` | Chart generation (called by `5-compare` and `0-benchmark`). |
| `build.bat` | Builds every program, or a single source file. |

The split programs (`2`-`5`) are meant to be run in order. `0-benchmark` is the
self-contained alternative and does not need the others.

## Build

Open a terminal in this folder and run:

```bat
cd src
build.bat
```

This produces:

```
0-benchmark.exe  2-sequential.exe  3-openmp.exe  4-cuda.exe  5-compare.exe
```

To build only one source:

```bat
build.bat 4-cuda.cu
```

The script compiles with `-arch=native` so the executable contains code for the
GPU in the machine. To build for a specific architecture instead:

```bat
set CUDA_ARCH=sm_89
build.bat
```

## Run

### Option A - split pipeline (recommended)

```bat
cd src
.\2-sequential.exe
.\3-openmp.exe
.\4-cuda.exe
.\5-compare.exe
```

Run the programs in that order: `5-compare` reads the CSV files produced by the
first three and fails with a "Missing results/...csv" message if any are absent.

Each program prints a progress table and writes its CSV to `results/` in the
current directory. Run everything from `src` so the output lands in
`src/results/` and `5-compare` can find `plot_benchmark.py`.

### Option B - all-in-one benchmark

```bat
cd src
.\0-benchmark.exe
```

This runs sequential, OpenMP (including the thread sweep) and CUDA in one go and
writes everything to a timestamped folder, e.g. `results/2026-10-20_23-32-35/`.
Every run gets its own folder, so previous runs are never overwritten.

## Output

Split pipeline (`results/`):

| File | Contents |
| --- | --- |
| `sequential_results.csv` | `Matrix_Size, Run, Sequential_CPU_ms` |
| `openmp_results.csv` | `Matrix_Size, Threads, Run, Parallel_CPU_ms` |
| `cuda_results.csv` | `Matrix_Size, Run, Kernel_ms, E2E_ms` |
| `averages.csv` | Per size: sequential/OpenMP/GPU times, speedups, `GPU_Kernel_ms` |
| `openmp_scaling_averages.csv` | Per size and thread count: speedup and parallel efficiency |

Charts written to the same folder:

- `execution_time_comparison.png` - execution time vs matrix size (log scale).
- `speedup_vs_sequential_cpu.png` - OpenMP and CUDA speedup vs the sequential CPU.
- `gpu_speedup_over_parallel_cpu.png` - CUDA speedup over the parallel CPU.
- `openmp_strong_scaling.png` - OpenMP speedup and efficiency vs thread count.

`0-benchmark` writes the same charts plus its raw-reading CSVs
(`all_readings.csv`, `openmp_scaling_all_readings.csv`) into its timestamp folder.

To redraw the charts for an existing results folder:

```bat
python plot_benchmark.py results
python plot_benchmark.py results\2026-10-20_23-32-35
```

## How the metrics are calculated

- **Speedup** = sequential time / parallel time for the same matrix size.
- **OpenMP parallel efficiency** = speedup / number of threads. It is computed
  only for the OpenMP thread sweep (strong scaling).
- **GPU efficiency is intentionally not reported.** CUDA threads/cores are not
  directly comparable to CPU threads, so the GPU is discussed using execution
  time, kernel vs end-to-end time, and speedup trends.

## Configuration

Edit `common.h` to change the experiment:

```cpp
const int READINGS_PER_SIZE = 10;
const std::vector<int> MATRIX_SIZES = {128, 256, 512, 1024, 2048, 3072, 4096};
const unsigned SEED = 42;
```

`0-benchmark.cpp` keeps its own `matrix_sizes` list in `main`, so edit that file
too if you use the all-in-one program.

The defaults are expensive: 4096x4096 sequential runs dominate the runtime, and
every size is repeated 10 times (plus the OpenMP thread sweep). For a quick
check, reduce `MATRIX_SIZES` and `READINGS_PER_SIZE` before building.

## Troubleshooting

- **`CUDA Error ... the provided PTX was compiled with an unsupported toolchain`**
  The executable was built without an architecture flag, or the driver is older
  than the toolkit. Rebuild with `build.bat` (it uses `-arch=native`), or set
  `CUDA_ARCH` to the GPU's compute capability.
- **`Missing results/sequential_results.csv`** (or a similar message)
  Run `2-sequential.exe`, `3-openmp.exe` and `4-cuda.exe` before `5-compare.exe`.
- **Charts are not generated / `matplotlib` error**
  Install it with `python -m pip install matplotlib`, then rerun `5-compare` or
  `python plot_benchmark.py results`.
- **`warning C4849: OpenMP 'collapse' clause ignored`**
  MSVC supports OpenMP 2.0 and ignores `collapse(2)`. The parallel loop still
  runs correctly; the warning is harmless.
