// sequential.cpp - sequential CPU baseline
// Build: g++ -O2 -std=c++17 sequential.cpp -o sequential
// Output: results/sequential_results.csv  (Matrix_Size,Run,Sequential_CPU_ms)
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
    bool ok = true;

    // Test 1: known 2x2 result  [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
    std::vector<float> A = {1, 2, 3, 4}, B = {5, 6, 7, 8}, C(4, 0.0f);
    sequentialMatrixMultiply(A, B, C, 2);
    std::vector<float> expected = {19, 22, 43, 50};
    if (C != expected)
    {
        std::cerr << "FAIL: 2x2 known-result test\n";
        ok = false;
    }

    // Test 2: A x Identity must equal A
    const int n = 64;
    std::vector<float> M(n * n), I(n * n, 0.0f), R(n * n, 0.0f);
    initializeMatrix(M, n, SEED);
    for (int i = 0; i < n; ++i)
        I[i * n + i] = 1.0f;
    sequentialMatrixMultiply(M, I, R, n);
    if (maxRelDiff(R, M) > 1e-6)
    {
        std::cerr << "FAIL: identity test\n";
        ok = false;
    }

    std::cout << (ok ? "Validation PASSED" : "Validation FAILED") << std::endl;
    return ok;
}

int main()
{
    if (!validate())
        return 1;

    ensureResultsDir();
    std::ofstream file(RESULTS_DIR + "/sequential_results.csv");
    file << "Matrix_Size,Run,Sequential_CPU_ms" << std::endl;

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
            file << N << "," << run << "," << std::fixed << std::setprecision(4) << ms << std::endl;
        }
        std::cout << N << "x" << N << "," << std::fixed << std::setprecision(4)
                  << total / READINGS_PER_SIZE << std::endl;
    }
    std::cout << "Wrote " << RESULTS_DIR << "/sequential_results.csv" << std::endl;
    return 0;
}