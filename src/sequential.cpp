// sequential.cpp - sequential CPU baseline
// Build: build.bat sequential.cpp
// Output: <run_dir>/sequential_results.csv   (Matrix_Size, Run, Sequential_CPU_ms)
//         <run_dir>/sequential_averages.csv  (Matrix_Size, Sequential_CPU_ms)
//         <run_dir>/sequential_execution_time.png
//
// Pseudocode:
//   for row in 0..N-1
//     for col in 0..N-1
//       sum = 0
//       for k in 0..N-1: sum += A[row][k] * B[k][col]
//       C[row][col] = sum
// Complexity: O(N^3) multiply-adds, O(N^2) memory.
#include "common.h"
#include <fstream>
#include <iomanip>

bool validate()
{
    int passed = 0;
    const int total = 2;
    bool ok = true;

    // Test 1: known 2x2 result  [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
    std::vector<float> A = {1, 2, 3, 4}, B = {5, 6, 7, 8}, C(4, 0.0f);
    sequentialMatrixMultiply(A, B, C, 2);
    std::vector<float> expected = {19, 22, 43, 50};
    if (C != expected)
    {
        std::cerr << "[FAIL] 2x2 known result: [[1,2],[3,4]] x [[5,6],[7,8]]" << std::endl;
        std::cerr << "       expected [19,22,43,50], got [";
        for (size_t i = 0; i < C.size(); ++i)
            std::cerr << (i ? "," : "") << C[i];
        std::cerr << "]" << std::endl;
        ok = false;
    }
    else
    {
        std::cout << "[PASS] 2x2 known result: [[1,2],[3,4]] x [[5,6],[7,8]] == [[19,22],[43,50]]" << std::endl;
        ++passed;
    }

    // Test 2: A x Identity must equal A
    const int n = 64;
    std::vector<float> M(n * n), I(n * n, 0.0f), R(n * n, 0.0f);
    initializeMatrix(M, n, SEED);
    for (int i = 0; i < n; ++i)
        I[i * n + i] = 1.0f;
    sequentialMatrixMultiply(M, I, R, n);
    double diff = maxRelDiff(R, M);
    if (diff > 1e-6)
    {
        std::cerr << "[FAIL] 64x64 identity: A x I equals A (max relative diff "
                  << std::scientific << diff << ")" << std::endl;
        ok = false;
    }
    else
    {
        std::cout << "[PASS] 64x64 identity: A x I equals A" << std::endl;
        ++passed;
    }

    std::cout << "Validation " << (ok ? "PASSED" : "FAILED")
              << ": " << passed << "/" << total << " checks passed" << std::endl;
    return ok;
}

int main(int argc, char *argv[])
{
    if (!validate())
        return 1;

    std::string run_dir = resolveRunDirectory(argc, argv);
    std::cout << "Run directory: " << run_dir << std::endl;

    std::ofstream raw(run_dir + "/sequential_results.csv");
    std::ofstream averages(run_dir + "/sequential_averages.csv");
    if (!raw || !averages)
    {
        std::cerr << "ERROR: could not open the output CSV files in " << run_dir << std::endl;
        return 1;
    }

    raw << "Matrix_Size,Run,Sequential_CPU_ms" << std::endl;
    averages << "Matrix_Size,Sequential_CPU_ms" << std::endl;

    std::cout << "Matrix_Size,Average_ms" << std::endl;
    for (int N : MATRIX_SIZES)
    {
        std::vector<float> A(N * N), B(N * N), C(N * N, 0.0f);
        initializeMatrix(A, N, SEED);
        initializeMatrix(B, N, SEED + 1);

        double total = 0.0;
        for (int run = 1; run <= READINGS_PER_SIZE; ++run)
        {
            double ms = timeMs([&]()
                               { sequentialMatrixMultiply(A, B, C, N); });
            total += ms;
            raw << N << "," << run << "," << std::fixed << std::setprecision(4) << ms << std::endl;
        }

        double average_ms = total / READINGS_PER_SIZE;
        averages << N << "," << std::fixed << std::setprecision(4) << average_ms << std::endl;
        std::cout << N << "x" << N << "," << std::fixed << std::setprecision(4)
                  << average_ms << std::endl;
    }

    std::cout << "Wrote " << run_dir << "/sequential_results.csv" << std::endl;
    std::cout << "Wrote " << run_dir << "/sequential_averages.csv" << std::endl;

    if (!runPlotScript(argv[0], "sequential \"" + run_dir + "\""))
        std::cerr << "Warning: chart generation failed (is matplotlib installed?)." << std::endl;

    return 0;
}
