// cuda.cu - parallel GPU (CUDA) matrix multiplication
// Build: build.bat 4-cuda.cu
// Output: results/cuda_results.csv  (Matrix_Size,Run,Kernel_ms,E2E_ms)
//
// One CUDA thread computes one C[row][col] element and walks the full k range.
// Kernel_ms measures only the kernel launch (CUDA events); E2E_ms includes the
// host-to-device copies, the kernel and the copy back.
#include "common.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>

// =================================================================
// CUDA Error Checking Wrapper
// =================================================================
#define CUDA_CHECK(err)                                                       \
    do                                                                        \
    {                                                                         \
        cudaError_t err_ = (err);                                             \
        if (err_ != cudaSuccess)                                              \
        {                                                                     \
            std::cerr << "CUDA Error in " << __FILE__ << " line " << __LINE__ \
                      << ": " << cudaGetErrorString(err_) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void matrixMulGpu(const float *A, const float *B, float *C, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size)
    {
        float sum = 0.0f;
        for (int k = 0; k < size; ++k)
        {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

int main()
{
    ensureResultsDir();
    std::ofstream file(RESULTS_DIR + "/cuda_results.csv");
    if (!file)
    {
        std::cerr << "ERROR: could not open " << RESULTS_DIR << "/cuda_results.csv" << std::endl;
        return 1;
    }
    file << "Matrix_Size,Run,Kernel_ms,E2E_ms" << std::endl;

    std::cout << "Matrix_Size,Kernel_ms,E2E_ms,Correct" << std::endl;

    const dim3 threads_per_block(16, 16);

    for (int N : MATRIX_SIZES)
    {
        size_t matrix_bytes = static_cast<size_t>(N) * N * sizeof(float);

        std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, 0.0f), h_ref(N * N, 0.0f);
        initializeMatrix(h_A, N, SEED);
        initializeMatrix(h_B, N, SEED + 1);
        sequentialMatrixMultiply(h_A, h_B, h_ref, N); // known-correct result

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, matrix_bytes));

        dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                        (N + threads_per_block.y - 1) / threads_per_block.y);

        cudaEvent_t kernel_start, kernel_stop;
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));

        double kernel_total = 0.0;
        double e2e_total = 0.0;
        bool correct = true;

        for (int run = 1; run <= READINGS_PER_SIZE; ++run)
        {
            double e2e_ms = timeMs([&]()
                                   {
                CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaEventRecord(kernel_start));
                matrixMulGpu<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(kernel_stop));
                CUDA_CHECK(cudaEventSynchronize(kernel_stop));
                CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost)); });

            float kernel_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));

            // The kernel is deterministic, so checking the first run is enough.
            if (run == 1 && maxRelDiff(h_C, h_ref) > 1e-4)
            {
                correct = false;
            }

            kernel_total += kernel_ms;
            e2e_total += e2e_ms;

            file << N << "," << run << ","
                 << std::fixed << std::setprecision(4) << kernel_ms << ","
                 << e2e_ms << std::endl;
        }

        CUDA_CHECK(cudaEventDestroy(kernel_start));
        CUDA_CHECK(cudaEventDestroy(kernel_stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        std::cout << N << "x" << N << ","
                  << std::fixed << std::setprecision(4)
                  << kernel_total / READINGS_PER_SIZE << ","
                  << e2e_total / READINGS_PER_SIZE << ","
                  << (correct ? "yes" : "NO") << std::endl;
    }

    std::cout << "Wrote " << RESULTS_DIR << "/cuda_results.csv" << std::endl;
    return 0;
}
