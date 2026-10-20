#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <algorithm>
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
const std::string OUTPUT_ROOT = "results";
const std::string ALL_READINGS_FILENAME = "all_readings.csv";
const std::string AVERAGES_FILENAME = "averages.csv";
const std::string SCALING_ALL_READINGS_FILENAME = "openmp_scaling_all_readings.csv";
const std::string SCALING_AVERAGES_FILENAME = "openmp_scaling_averages.csv";
const std::string PLOT_SCRIPT = "plot_benchmark.py";
const std::string PYTHON_COMMAND = "python";

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

struct ScalingReading
{
    int matrixSize;
    int threads;
    int run; // 1-based repetition index
    double openmpMs;
};

struct ScalingAverage
{
    int matrixSize;
    int threads;
    double sequentialMs;
    double openmpMs;
    double speedup;
    double efficiency;
};

std::vector<int> openmpThreadCounts()
{
    int max_threads = omp_get_max_threads();
    std::vector<int> thread_counts;
    for (int threads = 1; threads < max_threads; threads *= 2)
    {
        thread_counts.push_back(threads);
    }
    thread_counts.push_back(max_threads);
    return thread_counts;
}

template <typename Func>
double timeMs(Func &&fn)
{
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string createRunDirectory()
{
    std::time_t now = std::time(nullptr);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &now);
#else
    localtime_r(&now, &local_time);
#endif

    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", &local_time);

    std::filesystem::path run_dir = std::filesystem::path(OUTPUT_ROOT) / timestamp;
    for (int suffix = 1; std::filesystem::exists(run_dir); ++suffix)
    {
        run_dir = std::filesystem::path(OUTPUT_ROOT) / (std::string(timestamp) + "_" + std::to_string(suffix));
    }

    std::filesystem::create_directories(run_dir);
    return run_dir.string();
}

void writeAllReadingsCsv(const std::vector<Reading> &readings, const std::string &output_dir)
{
    std::filesystem::path csv_path = std::filesystem::path(output_dir) / ALL_READINGS_FILENAME;
    std::ofstream file(csv_path);
    if (!file)
    {
        std::cerr << "Error: could not open " << csv_path.string() << " for writing." << std::endl;
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

void writeAveragesCsv(const std::vector<AverageReading> &averages, const std::string &output_dir)
{
    std::filesystem::path csv_path = std::filesystem::path(output_dir) / AVERAGES_FILENAME;
    std::ofstream file(csv_path);
    if (!file)
    {
        std::cerr << "Error: could not open " << csv_path.string() << " for writing." << std::endl;
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

double sequentialMsForSize(const std::vector<AverageReading> &averages, int matrix_size)
{
    auto match = std::find_if(averages.begin(), averages.end(),
                              [matrix_size](const AverageReading &average)
                              { return average.matrixSize == matrix_size; });
    return match != averages.end() ? match->sequentialMs : 0.0;
}

std::vector<ScalingAverage> averageScalingReadings(const std::vector<ScalingReading> &readings,
                                                   const std::vector<AverageReading> &averages)
{
    std::vector<ScalingAverage> scaling_averages;
    for (size_t i = 0; i + READINGS_PER_SIZE <= readings.size(); i += READINGS_PER_SIZE)
    {
        const ScalingReading &first = readings[i];
        double openmp_ms = 0.0;
        for (int run = 0; run < READINGS_PER_SIZE; ++run)
        {
            openmp_ms += readings[i + run].openmpMs;
        }
        openmp_ms /= READINGS_PER_SIZE;

        double sequential_ms = sequentialMsForSize(averages, first.matrixSize);
        double speedup = sequential_ms / openmp_ms;

        scaling_averages.push_back(
            {first.matrixSize, first.threads, sequential_ms, openmp_ms,
             speedup, speedup / first.threads});
    }
    return scaling_averages;
}

void writeScalingReadingsCsv(const std::vector<ScalingReading> &readings,
                             const std::vector<AverageReading> &averages,
                             const std::string &output_dir)
{
    std::filesystem::path csv_path = std::filesystem::path(output_dir) / SCALING_ALL_READINGS_FILENAME;
    std::ofstream file(csv_path);
    if (!file)
    {
        std::cerr << "Error: could not open " << csv_path.string() << " for writing." << std::endl;
        exit(EXIT_FAILURE);
    }

    file << "Matrix_Size,Threads,Run,Sequential_CPU_ms,Parallel_CPU_ms,"
         << "Speedup_CPU_vs_Seq,Parallel_Efficiency" << std::endl;
    for (const ScalingReading &reading : readings)
    {
        double sequential_ms = sequentialMsForSize(averages, reading.matrixSize);
        double speedup = sequential_ms / reading.openmpMs;

        file << reading.matrixSize << ","
             << reading.threads << ","
             << reading.run << ","
             << std::fixed << std::setprecision(4) << sequential_ms << ","
             << reading.openmpMs << ","
             << std::setprecision(2) << speedup << ","
             << std::setprecision(4) << speedup / reading.threads << std::endl;
    }
}

void writeScalingAveragesCsv(const std::vector<ScalingAverage> &scaling_averages, const std::string &output_dir)
{
    std::filesystem::path csv_path = std::filesystem::path(output_dir) / SCALING_AVERAGES_FILENAME;
    std::ofstream file(csv_path);
    if (!file)
    {
        std::cerr << "Error: could not open " << csv_path.string() << " for writing." << std::endl;
        exit(EXIT_FAILURE);
    }

    file << "Matrix_Size,Threads,Sequential_CPU_ms,Parallel_CPU_ms,"
         << "Speedup_CPU_vs_Seq,Parallel_Efficiency" << std::endl;
    for (const ScalingAverage &average : scaling_averages)
    {
        file << average.matrixSize << ","
             << average.threads << ","
             << std::fixed << std::setprecision(4) << average.sequentialMs << ","
             << average.openmpMs << ","
             << std::setprecision(2) << average.speedup << ","
             << std::setprecision(4) << average.efficiency << std::endl;
    }
}

bool runPlotScript(const char *executable, const std::string &output_dir)
{
    std::vector<std::string> candidates = {PLOT_SCRIPT};

    std::filesystem::path executable_path(executable != nullptr ? executable : "");
    if (executable_path.has_parent_path())
    {
        candidates.push_back((executable_path.parent_path() / PLOT_SCRIPT).string());
    }

    for (const std::string &script : candidates)
    {
        if (!std::filesystem::exists(script))
        {
            continue;
        }

        std::string command = PYTHON_COMMAND + " \"" + script + "\" \"" + output_dir + "\"";
        if (std::system(command.c_str()) != 0)
        {
            std::cerr << "Warning: graph export failed "
                      << "(is Python with matplotlib installed?)." << std::endl;
            return false;
        }
        std::cout << "Graphs written to " << output_dir << std::endl;
        return true;
    }

    std::cerr << "Warning: " << PLOT_SCRIPT << " not found; skipping graph export." << std::endl;
    return false;
}

// =================================================================
// Main Benchmark Runner
// =================================================================
int main(int argc, char *argv[])
{
    srand(static_cast<unsigned>(time(0)));

    std::vector<int> matrix_sizes = {
        128,
        256,
        512,
        1024,
        // 2048,
        // 3072,
        // 4096
    };
    std::vector<Reading> readings;
    std::vector<ScalingReading> scaling_readings;

    std::vector<int> thread_counts = openmpThreadCounts();
    int max_threads = omp_get_max_threads();

    std::string output_dir = createRunDirectory();
    std::cout << "Results directory: " << output_dir << std::endl;
    std::cout << "OpenMP strong scaling thread counts:";
    for (int threads : thread_counts)
    {
        std::cout << " " << threads;
    }
    std::cout << std::endl;

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

        // --- 6. OpenMP Strong-Scaling Benchmark (fixed size, varying threads) ---
        for (int threads : thread_counts)
        {
            omp_set_num_threads(threads);
            for (int run = 1; run <= READINGS_PER_SIZE; ++run)
            {
                double duration_omp = timeMs([&]()
                                             { openmpMatrixMultiply(h_A, h_B, h_C, N); });
                scaling_readings.push_back({N, threads, run, duration_omp});
            }
        }
        omp_set_num_threads(max_threads);
    }

    writeAllReadingsCsv(readings, output_dir);

    std::vector<AverageReading> averages = averageReadings(readings);
    writeAveragesCsv(averages, output_dir);

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

    std::vector<ScalingAverage> scaling_averages = averageScalingReadings(scaling_readings, averages);
    writeScalingReadingsCsv(scaling_readings, averages, output_dir);
    writeScalingAveragesCsv(scaling_averages, output_dir);

    std::cout << "\nOpenMP strong scaling (" << READINGS_PER_SIZE
              << " readings per thread count, efficiency = speedup / threads):" << std::endl;
    std::cout << "Matrix_Size,Threads,Sequential_CPU_ms,Parallel_CPU_ms,"
              << "Speedup_CPU_vs_Seq,Parallel_Efficiency" << std::endl;
    for (const ScalingAverage &average : scaling_averages)
    {
        std::cout << average.matrixSize << "x" << average.matrixSize << ","
                  << average.threads << ","
                  << std::fixed << std::setprecision(4) << average.sequentialMs << ","
                  << average.openmpMs << ","
                  << std::setprecision(2) << average.speedup << "x,"
                  << std::setprecision(4) << average.efficiency << std::endl;
    }

    std::cout << "\nRaw readings written to "
              << (std::filesystem::path(output_dir) / ALL_READINGS_FILENAME).string() << std::endl;
    std::cout << "Average readings written to "
              << (std::filesystem::path(output_dir) / AVERAGES_FILENAME).string() << std::endl;
    std::cout << "OpenMP scaling readings written to "
              << (std::filesystem::path(output_dir) / SCALING_ALL_READINGS_FILENAME).string() << std::endl;
    std::cout << "OpenMP scaling averages written to "
              << (std::filesystem::path(output_dir) / SCALING_AVERAGES_FILENAME).string() << std::endl;

    runPlotScript(argc > 0 ? argv[0] : nullptr, output_dir);

    std::cout << "Benchmark complete." << std::endl;

    return 0;
}
