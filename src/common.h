// common.h - shared helpers used by every program
// SAME sizes, seed and run count so the data is consistent
#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <ctime>
#include <cstdlib>

const int READINGS_PER_SIZE = 10;
const std::vector<int> MATRIX_SIZES = {
    128,
    256,
    512,
    1024,
    2048,
    3072,
    4096,
};
const std::string RESULTS_DIR = "results";
const unsigned SEED = 42;

// Same seed -> same matrices in every program.
inline void initializeMatrix(std::vector<float> &m, int size, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < size * size; ++i)
        m[i] = dist(gen);
}

template <typename Func>
double timeMs(Func &&fn)
{
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline void ensureResultsDir()
{
    std::filesystem::create_directories(RESULTS_DIR);
}

// Creates results/<YYYY-MM-DD_HH-MM-SS>/ so every run keeps its own output.
inline std::string createRunDirectory()
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

    std::filesystem::path run_dir = std::filesystem::path(RESULTS_DIR) / timestamp;
    for (int suffix = 1; std::filesystem::exists(run_dir); ++suffix)
    {
        run_dir = std::filesystem::path(RESULTS_DIR) /
                  (std::string(timestamp) + "_" + std::to_string(suffix));
    }

    std::filesystem::create_directories(run_dir);
    return run_dir.string();
}

// Uses the folder passed on the command line, otherwise creates a new
// timestamped one. run_all.bat passes the same folder to every program.
inline std::string resolveRunDirectory(int argc, char *argv[])
{
    if (argc > 1)
    {
        std::filesystem::path run_dir(argv[1]);
        std::filesystem::create_directories(run_dir);
        return run_dir.string();
    }
    return createRunDirectory();
}

// Windows usually ships "python"; Linux/macOS usually only "python3".
#ifdef _WIN32
const char *const PYTHON_CMD = "python";
#else
const char *const PYTHON_CMD = "python3";
#endif

// Runs plot_benchmark.py from the current directory or next to the executable.
inline bool runPlotScript(const char *executable, const std::string &arguments)
{
    std::vector<std::string> candidates = {"plot_benchmark.py"};

    std::filesystem::path executable_path(executable != nullptr ? executable : "");
    if (executable_path.has_parent_path())
    {
        candidates.push_back((executable_path.parent_path() / "plot_benchmark.py").string());
    }

    for (const std::string &script : candidates)
    {
        if (!std::filesystem::exists(script))
        {
            continue;
        }

        std::string command = std::string(PYTHON_CMD) + " \"" + script + "\" " + arguments;
        return std::system(command.c_str()) == 0;
    }

    std::cerr << "Warning: plot_benchmark.py not found; skipping charts." << std::endl;
    return false;
}

// Reference algorithm (owned by Member 2). Members 3 and 4 use it to validate their output.
inline void sequentialMatrixMultiply(const std::vector<float> &A, const std::vector<float> &B,
                                     std::vector<float> &C, int size)
{
    for (int row = 0; row < size; ++row)
        for (int col = 0; col < size; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k)
                sum += A[row * size + k] * B[k * size + col];
            C[row * size + col] = sum;
        }
}

// Largest relative difference between two result matrices.
inline double maxRelDiff(const std::vector<float> &X, const std::vector<float> &Y)
{
    double worst = 0.0;
    for (size_t i = 0; i < X.size(); ++i)
    {
        double denom = std::max(1e-6, std::fabs((double)Y[i]));
        worst = std::max(worst, std::fabs((double)X[i] - (double)Y[i]) / denom);
    }
    return worst;
}