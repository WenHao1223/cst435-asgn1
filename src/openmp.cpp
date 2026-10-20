// openmp.cpp - parallel CPU (OpenMP) with thread scaling
// Build: build.bat openmp.cpp
// Output: <run_dir>/openmp_results.csv   (Matrix_Size, Threads, Run, Parallel_CPU_ms)
//         <run_dir>/openmp_averages.csv  (Matrix_Size, Threads, Parallel_CPU_ms,
//                                         Speedup_vs_1_Thread, Parallel_Efficiency)
//         <run_dir>/openmp_strong_scaling.png
//
// Thread-level parallelism: the outer row loop is split across threads. Each
// thread writes disjoint rows of C, so no synchronisation is needed. The row
// loop is used on its own because MSVC's OpenMP 2.0 ignores collapse(2).
// Overheads to discuss: thread creation/scheduling, memory-bandwidth limits,
// cache effects (B is read column-wise), and fewer cores than threads.
//
// Speedup and efficiency use this program's own 1-thread run as the
// strong-scaling baseline, so openmp.exe does not need sequential.exe.
#include "common.h"
#include <omp.h>
#include <fstream>
#include <iomanip>

void openmpMatrixMultiply(const std::vector<float> &A, const std::vector<float> &B,
                          std::vector<float> &C, int size)
{
#pragma omp parallel for
    for (int row = 0; row < size; ++row)
        for (int col = 0; col < size; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k)
                sum += A[row * size + k] * B[k * size + col];
            C[row * size + col] = sum;
        }
}

std::vector<int> openmpThreadCounts()
{
    int max_threads = omp_get_max_threads();
    std::vector<int> counts;
    for (int t = 1; t < max_threads; t *= 2)
        counts.push_back(t);
    counts.push_back(max_threads);
    return counts;
}

int main(int argc, char *argv[])
{
    std::string run_dir = resolveRunDirectory(argc, argv);
    std::cout << "Run directory: " << run_dir << std::endl;

    std::ofstream raw(run_dir + "/openmp_results.csv");
    std::ofstream averages(run_dir + "/openmp_averages.csv");
    if (!raw || !averages)
    {
        std::cerr << "ERROR: could not open the output CSV files in " << run_dir << std::endl;
        return 1;
    }
    raw << "Matrix_Size,Threads,Run,Parallel_CPU_ms" << std::endl;
    averages << "Matrix_Size,Threads,Parallel_CPU_ms,Speedup_vs_1_Thread,Parallel_Efficiency" << std::endl;

    std::vector<int> thread_counts = openmpThreadCounts();
    std::cout << "Thread counts:";
    for (int t : thread_counts)
        std::cout << " " << t;
    std::cout << std::endl;
    std::cout << "Matrix_Size,Threads,Average_ms,Speedup,Efficiency,Matches_Sequential" << std::endl;

    for (int N : MATRIX_SIZES)
    {
        std::vector<float> A(N * N), B(N * N), C(N * N, 0.0f), ref(N * N, 0.0f);
        initializeMatrix(A, N, SEED);
        initializeMatrix(B, N, SEED + 1);
        sequentialMatrixMultiply(A, B, ref, N); // known-correct result

        struct ThreadResult
        {
            int threads;
            double average_ms;
            bool correct;
        };
        std::vector<ThreadResult> results;

        for (int threads : thread_counts)
        {
            omp_set_num_threads(threads);

            // Correctness check for this thread count
            openmpMatrixMultiply(A, B, C, N);
            bool correct = maxRelDiff(C, ref) < 1e-4;

            double total = 0.0;
            for (int run = 1; run <= READINGS_PER_SIZE; ++run)
            {
                double ms = timeMs([&]()
                                   { openmpMatrixMultiply(A, B, C, N); });
                total += ms;
                raw << N << "," << threads << "," << run << ","
                    << std::fixed << std::setprecision(4) << ms << std::endl;
            }

            results.push_back({threads, total / READINGS_PER_SIZE, correct});
        }

        // Strong-scaling baseline: this program's own 1-thread run.
        double baseline_ms = 0.0;
        for (const ThreadResult &result : results)
        {
            if (result.threads == 1)
                baseline_ms = result.average_ms;
        }
        if (baseline_ms <= 0.0)
        {
            std::cerr << "ERROR: the thread sweep must include 1 thread." << std::endl;
            return 1;
        }

        for (const ThreadResult &result : results)
        {
            double speedup = baseline_ms / result.average_ms;
            double efficiency = speedup / result.threads;

            averages << N << "," << result.threads << ","
                     << std::fixed << std::setprecision(4) << result.average_ms << ","
                     << std::setprecision(2) << speedup << ","
                     << std::setprecision(4) << efficiency << std::endl;

            std::cout << N << "x" << N << "," << result.threads << ","
                      << std::fixed << std::setprecision(4) << result.average_ms << ","
                      << std::setprecision(2) << speedup << "x,"
                      << std::setprecision(4) << efficiency << ","
                      << (result.correct ? "yes" : "NO") << std::endl;
        }
    }

    std::cout << "Wrote " << run_dir << "/openmp_results.csv" << std::endl;
    std::cout << "Wrote " << run_dir << "/openmp_averages.csv" << std::endl;

    if (!runPlotScript(argv[0], "openmp \"" + run_dir + "\""))
        std::cerr << "Warning: chart generation failed (is matplotlib installed?)." << std::endl;

    return 0;
}
