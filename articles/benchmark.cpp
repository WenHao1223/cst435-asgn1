#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

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

// =================================================================
// Matrix Multiplication Implementations
// =================================================================

void initializeMatrix(std::vector<float> &matrix, int size)
{
    for (int i = 0; i < size * size; ++i)
    {
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void sequentialMatrixMultiply(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int size)
{
    for (int row = 0; row < size; ++row)
    {
        for (int col = 0; col < size; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k)
            {
                sum += A[row * size + k] * B[k * size + col];
            }
            C[row * size + col] = sum;
        }
    }
}

void openmpMatrixMultiply(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int size)
{
#pragma omp parallel for collapse(2)
    for (int row = 0; row < size; ++row)
    {
        for (int col = 0; col < size; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k)
            {
                sum += A[row * size + k] * B[k * size + col];
            }
            C[row * size + col] = sum;
        }
    }
}

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

// =================================================================
// Main Benchmark Runner
// =================================================================
int main()
{
    srand(static_cast<unsigned>(time(0)));

    std::vector<int> matrix_sizes = {128, 256, 512, 1024, 2048, 3072, 4096};

    // Print the header for our CSV data table
    std::cout << "Matrix_Size,Sequential_CPU_ms,Parallel_CPU_ms,Parallel_GPU_ms,"
              << "Speedup_CPU_vs_Seq,Speedup_GPU_vs_CPU,Speedup_GPU_vs_Seq" << std::endl;

    for (int N : matrix_sizes)
    {
        // --- Host Memory Allocation & Initialization ---
        std::vector<float> h_A(N * N);
        std::vector<float> h_B(N * N);
        std::vector<float> h_C(N * N, 0.0f); // Re-use one C matrix for all results

        initializeMatrix(h_A, N);
        initializeMatrix(h_B, N);

        // --- 1. Sequential CPU Benchmark ---
        auto start_seq = std::chrono::high_resolution_clock::now();
        sequentialMatrixMultiply(h_A, h_B, h_C, N);
        auto end_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_seq = end_seq - start_seq;

        // --- 2. Parallel CPU (OpenMP) Benchmark ---
        auto start_omp = std::chrono::high_resolution_clock::now();
        openmpMatrixMultiply(h_A, h_B, h_C, N);
        auto end_omp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_omp = end_omp - start_omp;

        // ---3. Parallel GPU (CUDA) Benchmark ---
        float *d_A, *d_B, *d_C;
        size_t matrix_bytes = N * N * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, matrix_bytes));

        auto start_gpu = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice));
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrixMulGpu<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost));
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_gpu = end_gpu - start_gpu;

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        // --- 4. Calculate Speedups ---
        double speedup_cpu_vs_seq = duration_seq.count() / duration_omp.count();
        double speedup_gpu_vs_cpu = duration_omp.count() / duration_gpu.count();
        double speedup_gpu_vs_seq = duration_seq.count() / duration_gpu.count();

        // --- 5. Report Results for this Size ---
        std::cout << N << "x" << N << ","
                  << std::fixed << std::setprecision(4) << duration_seq.count() << ","
                  << std::fixed << std::setprecision(4) << duration_omp.count() << ","
                  << std::fixed << std::setprecision(4) << duration_gpu.count() << ","
                  << std::fixed << std::setprecision(2) << speedup_cpu_vs_seq << "x,"
                  << std::fixed << std::setprecision(2) << speedup_gpu_vs_cpu << "x,"
                  << std::fixed << std::setprecision(2) << speedup_gpu_vs_seq << "x" << std::endl;
    }

    std::cout << "\nBenchmark complete." << std::endl;

    return 0;
}