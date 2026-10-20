// openmp.cpp - parallel CPU (OpenMP) with thread scaling
// Build: g++ -O2 -fopenmp -std=c++17 openmp.cpp -o openmp
// Output: results/openmp_results.csv  (Matrix_Size,Threads,Run,Parallel_CPU_ms)
//
// Thread-level parallelism: collapse(2) merges the row and col loops into one
// N*N iteration space that OpenMP splits across threads. Each (row,col) writes a
// different C element, so no synchronisation is needed.
// Overheads to discuss: thread creation/scheduling, memory-bandwidth limits,
// cache effects (B is read column-wise), and fewer cores than threads.
#include "common.h"
#include <omp.h>
#include <fstream>
#include <iomanip>

void openmpMatrixMultiply(const std::vector<float> &A, const std::vector<float> &B,
                          std::vector<float> &C, int size)
{
#pragma omp parallel for collapse(2)
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

int main()
{
    ensureResultsDir();
    std::ofstream file(RESULTS_DIR + "/openmp_results.csv");
    file << "Matrix_Size,Threads,Run,Parallel_CPU_ms" << std::endl;

    std::vector<int> thread_counts = openmpThreadCounts();
    std::cout << "Thread counts:";
    for (int t : thread_counts)
        std::cout << " " << t;
    std::cout << std::endl;
    std::cout << "Matrix_Size,Threads,Average_ms,Correct" << std::endl;

    for (int N : MATRIX_SIZES)
    {
        std::vector<float> A(N * N), B(N * N), C(N * N, 0.0f), ref(N * N, 0.0f);
        initializeMatrix(A, N, SEED);
        initializeMatrix(B, N, SEED + 1);
        sequentialMatrixMultiply(A, B, ref, N); // known-correct result

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
                file << N << "," << threads << "," << run << ","
                     << std::fixed << std::setprecision(4) << ms << std::endl;
            }
            std::cout << N << "x" << N << "," << threads << ","
                      << std::fixed << std::setprecision(4) << total / READINGS_PER_SIZE
                      << "," << (correct ? "yes" : "NO") << std::endl;
        }
    }
    std::cout << "Wrote " << RESULTS_DIR << "/openmp_results.csv" << std::endl;
    return 0;
}