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