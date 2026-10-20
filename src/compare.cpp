// compare.cpp - merge the per-program averages and draw comparison charts
// Build: build.bat compare.cpp
// Run AFTER sequential, openmp and cuda have written their averages into the
// same run folder.
// Output: <run_dir>/averages.csv
//         <run_dir>/execution_time_comparison.png
//         <run_dir>/speedup_vs_sequential_cpu.png
//         <run_dir>/gpu_speedup_over_parallel_cpu.png
//
// This program only combines the three programs' results: each implementation
// writes its own raw readings, averages and graph.
#include "common.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>

// Reads a CSV (skipping the header) into rows of doubles.
std::vector<std::vector<double>> readCsv(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
    {
        std::cerr << "ERROR: " << path << " not found." << std::endl;
        std::cerr << "       Run the matching program with the same run folder first." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ','))
            row.push_back(std::stod(cell));
        rows.push_back(row);
    }
    return rows;
}

int main(int argc, char *argv[])
{
    std::string run_dir = resolveRunDirectory(argc, argv);
    std::cout << "Run directory: " << run_dir << std::endl;

    // sequential averages: [size, ms]
    std::map<int, double> sequential_ms;
    for (auto &row : readCsv(run_dir + "/sequential_averages.csv"))
        sequential_ms[(int)row[0]] = row[1];

    // openmp averages: [size, threads, ms, speedup, efficiency]
    std::map<std::pair<int, int>, double> openmp_ms;
    int max_threads = 1;
    for (auto &row : readCsv(run_dir + "/openmp_averages.csv"))
    {
        openmp_ms[{(int)row[0], (int)row[1]}] = row[2];
        max_threads = std::max(max_threads, (int)row[1]);
    }

    // cuda averages: [size, kernel_ms, e2e_ms]
    std::map<int, double> gpu_kernel_ms, gpu_e2e_ms;
    for (auto &row : readCsv(run_dir + "/cuda_averages.csv"))
    {
        gpu_kernel_ms[(int)row[0]] = row[1];
        gpu_e2e_ms[(int)row[0]] = row[2];
    }

    std::ofstream out(run_dir + "/averages.csv");
    if (!out)
    {
        std::cerr << "ERROR: could not open " << run_dir << "/averages.csv" << std::endl;
        return 1;
    }

    out << "Matrix_Size,Sequential_CPU_ms,Parallel_CPU_ms,Parallel_GPU_ms,"
        << "Speedup_CPU_vs_Seq,Speedup_GPU_vs_CPU,Speedup_GPU_vs_Seq,GPU_Kernel_ms" << std::endl;
    std::cout << "Matrix_Size,Seq_ms,OpenMP_ms,GPU_E2E_ms,GPU_Kernel_ms,CPU/Seq,GPU/CPU,GPU/Seq" << std::endl;

    for (auto &[size, seq_ms] : sequential_ms)
    {
        double omp_ms = openmp_ms[{size, max_threads}];
        double gpu_ms = gpu_e2e_ms[size];
        double kernel_ms = gpu_kernel_ms[size];

        out << size << "," << std::fixed << std::setprecision(4)
            << seq_ms << "," << omp_ms << "," << gpu_ms << ","
            << std::setprecision(2) << seq_ms / omp_ms << ","
            << omp_ms / gpu_ms << ","
            << seq_ms / gpu_ms << ","
            << std::setprecision(4) << kernel_ms << std::endl;

        std::cout << size << "x" << size << "," << std::fixed << std::setprecision(4)
                  << seq_ms << "," << omp_ms << "," << gpu_ms << "," << kernel_ms << ","
                  << std::setprecision(2) << seq_ms / omp_ms << "x,"
                  << omp_ms / gpu_ms << "x,"
                  << seq_ms / gpu_ms << "x" << std::endl;
    }

    std::cout << "Wrote " << run_dir << "/averages.csv" << std::endl;
    std::cout << "Using OpenMP results at " << max_threads << " threads." << std::endl;

    if (!runPlotScript(argv[0], "compare \"" + run_dir + "\""))
        std::cerr << "Warning: chart generation failed (is matplotlib installed?)." << std::endl;

    return 0;
}
