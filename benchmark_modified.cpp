#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <filesystem>
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
// Benchmark Data & Output Helpers
// =================================================================
const int READINGS_PER_SIZE = 3;
const std::string OUTPUT_DIR = "results";
const std::string ALL_READINGS_CSV = OUTPUT_DIR + "/all_readings.csv";
const std::string AVERAGES_CSV = OUTPUT_DIR + "/averages.csv";

struct Reading
{
    int matrixSize;
    int run; // 1-based repetition index
    double sequentialMs;
    double openmpMs;
    double cudaMs;
};

struct AverageReading
{
    int matrixSize;
    double sequentialMs;
    double openmpMs;
    double cudaMs;
};

template <typename Func>
double timeMs(Func &&fn)
{
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void writeAllReadingsCsv(const std::vector<Reading> &readings)
{
    std::ofstream file(ALL_READINGS_CSV);
    if (!file)
    {
        std::cerr << "Error: could not open " << ALL_READINGS_CSV << " for writing." << std::endl;
        exit(EXIT_FAILURE);
    }

    file << "Matrix_Size,Run,Sequential_CPU_ms,Parallel_CPU_ms,Parallel_GPU_ms" << std::endl;
    for (const Reading &reading : readings)
    {
        file << reading.matrixSize << ","
             << reading.run << ","
             << std::fixed << std::setprecision(4) << reading.sequentialMs << ","
             << reading.openmpMs << ","
             << reading.cudaMs << std::endl;
    }
}

std::vector<AverageReading> averageReadings(const std::vector<Reading> &readings)
{
    std::vector<AverageReading> averages;
    for (size_t i = 0; i + READINGS_PER_SIZE <= readings.size(); i += READINGS_PER_SIZE)
    {
        AverageReading average{readings[i].matrixSize, 0.0, 0.0, 0.0};
        for (int run = 0; run < READINGS_PER_SIZE; ++run)
        {
            average.sequentialMs += readings[i + run].sequentialMs;
            average.openmpMs += readings[i + run].openmpMs;
            average.cudaMs += readings[i + run].cudaMs;
        }
        average.sequentialMs /= READINGS_PER_SIZE;
        average.openmpMs /= READINGS_PER_SIZE;
        average.cudaMs /= READINGS_PER_SIZE;
        averages.push_back(average);
    }
    return averages;
}

void writeAveragesCsv(const std::vector<AverageReading> &averages)
{
    std::ofstream file(AVERAGES_CSV);
    if (!file)
    {
        std::cerr << "Error: could not open " << AVERAGES_CSV << " for writing." << std::endl;
        exit(EXIT_FAILURE);
    }

    file << "Matrix_Size,Sequential_CPU_ms,Parallel_CPU_ms,Parallel_GPU_ms,"
         << "Speedup_CPU_vs_Seq,Speedup_GPU_vs_CPU,Speedup_GPU_vs_Seq" << std::endl;
    for (const AverageReading &average : averages)
    {
        double speedup_cpu_vs_seq = average.sequentialMs / average.openmpMs;
        double speedup_gpu_vs_cpu = average.openmpMs / average.cudaMs;
        double speedup_gpu_vs_seq = average.sequentialMs / average.cudaMs;

        file << average.matrixSize << ","
             << std::fixed << std::setprecision(4) << average.sequentialMs << ","
             << average.openmpMs << ","
             << average.cudaMs << ","
             << std::setprecision(2) << speedup_cpu_vs_seq << ","
             << speedup_gpu_vs_cpu << ","
             << speedup_gpu_vs_seq << std::endl;
    }
}

// =================================================================
// Main Benchmark Runner
// =================================================================
int main()
{
    srand(static_cast<unsigned>(time(0)));

    std::vector<int> matrix_sizes = {128, 256, 512, 1024, 2048, 3072, 4096};
    std::vector<Reading> readings;

    std::filesystem::create_directories(OUTPUT_DIR);

    // Print the header for our CSV data table
    std::cout << "Matrix_Size,Run,Sequential_CPU_ms,Parallel_CPU_ms,Parallel_GPU_ms,"
              << "Speedup_CPU_vs_Seq,Speedup_GPU_vs_CPU,Speedup_GPU_vs_Seq" << std::endl;

    for (int N : matrix_sizes)
    {
        // --- Host Memory Allocation & Initialization ---
        std::vector<float> h_A(N * N);
        std::vector<float> h_B(N * N);
        std::vector<float> h_C(N * N, 0.0f); // Re-use one C matrix for all results

        initializeMatrix(h_A, N);
        initializeMatrix(h_B, N);

        // --- Device Memory Allocation ---
        float *d_A, *d_B, *d_C;
        size_t matrix_bytes = N * N * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, matrix_bytes));

        for (int run = 1; run <= READINGS_PER_SIZE; ++run)
        {
            // --- 1. Sequential CPU Benchmark ---
            double duration_seq = timeMs([&]()
                                         { sequentialMatrixMultiply(h_A, h_B, h_C, N); });

            // --- 2. Parallel CPU (OpenMP) Benchmark ---
            double duration_omp = timeMs([&]()
                                         { openmpMatrixMultiply(h_A, h_B, h_C, N); });

            // --- 3. Parallel GPU (CUDA) Benchmark ---
            double duration_gpu = timeMs([&]()
                                         {
                CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice));
                dim3 threadsPerBlock(16, 16);
                dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
                matrixMulGpu<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost)); });

            readings.push_back({N, run, duration_seq, duration_omp, duration_gpu});

            // --- 4. Calculate Speedups ---
            double speedup_cpu_vs_seq = duration_seq / duration_omp;
            double speedup_gpu_vs_cpu = duration_omp / duration_gpu;
            double speedup_gpu_vs_seq = duration_seq / duration_gpu;

            // --- 5. Report Results for this Reading ---
            std::cout << N << "x" << N << "," << run << ","
                      << std::fixed << std::setprecision(4) << duration_seq << ","
                      << duration_omp << ","
                      << duration_gpu << ","
                      << std::setprecision(2) << speedup_cpu_vs_seq << "x,"
                      << speedup_gpu_vs_cpu << "x,"
                      << speedup_gpu_vs_seq << "x" << std::endl;
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    writeAllReadingsCsv(readings);

    std::vector<AverageReading> averages = averageReadings(readings);
    writeAveragesCsv(averages);

    std::cout << "\nAverage of " << READINGS_PER_SIZE << " readings per matrix size:" << std::endl;
    std::cout << "Matrix_Size,Sequential_CPU_ms,Parallel_CPU_ms,Parallel_GPU_ms,"
              << "Speedup_CPU_vs_Seq,Speedup_GPU_vs_CPU,Speedup_GPU_vs_Seq" << std::endl;
    for (const AverageReading &average : averages)
    {
        std::cout << average.matrixSize << "x" << average.matrixSize << ","
                  << std::fixed << std::setprecision(4) << average.sequentialMs << ","
                  << average.openmpMs << ","
                  << average.cudaMs << ","
                  << std::setprecision(2) << average.sequentialMs / average.openmpMs << "x,"
                  << average.openmpMs / average.cudaMs << "x,"
                  << average.sequentialMs / average.cudaMs << "x" << std::endl;
    }

    std::cout << "\nRaw readings written to " << ALL_READINGS_CSV << std::endl;
    std::cout << "Average readings written to " << AVERAGES_CSV << std::endl;
    std::cout << "Benchmark complete." << std::endl;

    return 0;
}
