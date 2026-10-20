// cuda.cu - parallel GPU (CUDA) matrix multiplication
// Build: build.bat cuda.cu
// Output: <run_dir>/cuda_results.csv   (Matrix_Size, Run, Parallel_GPU_ms)
//         <run_dir>/cuda_averages.csv  (Matrix_Size, Parallel_GPU_ms)
//         <run_dir>/cuda_execution_time.png
//
// One CUDA thread computes one C[row][col] element and walks the full k range.
// Parallel_GPU_ms is measured the same way as benchmark.cpp: host-to-device
// copies, the kernel and the copy back, i.e. the whole GPU operation.
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

int main(int argc, char *argv[])
{
    std::string run_dir = resolveRunDirectory(argc, argv);
    std::cout << "Run directory: " << run_dir << std::endl;

    std::ofstream raw(run_dir + "/cuda_results.csv");
    std::ofstream averages(run_dir + "/cuda_averages.csv");
    if (!raw || !averages)
    {
        std::cerr << "ERROR: could not open the output CSV files in " << run_dir << std::endl;
        return 1;
    }
    raw << "Matrix_Size,Run,Parallel_GPU_ms" << std::endl;
    averages << "Matrix_Size,Parallel_GPU_ms" << std::endl;

    std::cout << "Matrix_Size,Average_ms,Matches_Sequential" << std::endl;

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

        double total = 0.0;
        bool correct = true;

        for (int run = 1; run <= READINGS_PER_SIZE; ++run)
        {
            double gpu_ms = timeMs([&]()
                                   {
                CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice));
                matrixMulGpu<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost)); });

            // The kernel is deterministic, so checking the first run is enough.
            if (run == 1 && maxRelDiff(h_C, h_ref) > 1e-4)
            {
                correct = false;
            }

            total += gpu_ms;
            raw << N << "," << run << ","
                << std::fixed << std::setprecision(4) << gpu_ms << std::endl;
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        double average_ms = total / READINGS_PER_SIZE;

        averages << N << "," << std::fixed << std::setprecision(4)
                 << average_ms << std::endl;

        std::cout << N << "x" << N << ","
                  << std::fixed << std::setprecision(4) << average_ms << ","
                  << (correct ? "yes" : "NO") << std::endl;
    }

    std::cout << "Wrote " << run_dir << "/cuda_results.csv" << std::endl;
    std::cout << "Wrote " << run_dir << "/cuda_averages.csv" << std::endl;

    if (!runPlotScript(argv[0], "cuda \"" + run_dir + "\""))
        std::cerr << "Warning: chart generation failed (is matplotlib installed?)." << std::endl;

    return 0;
}
