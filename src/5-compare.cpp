// compare.cpp - merge results from Members 2-4, compute speedup/efficiency
// Build: g++ -O2 -std=c++17 compare.cpp -o compare
// Run AFTER sequential, openmp and cuda_bench have produced their CSVs in results/.
// Output (same names/columns as the original benchmark so plot_benchmark.py still works):
//   results/averages.csv                  - per size: seq, OpenMP(max threads), GPU end-to-end, speedups
//   results/openmp_scaling_averages.csv   - per size & thread count: speedup + efficiency
//   (GPU_Kernel_ms is appended as an extra last column of averages.csv)
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <filesystem>

const std::string DIR = "results";

// Reads a CSV (skipping the header) into rows of doubles.
std::vector<std::vector<double>> readCsv(const std::string &name)
{
    std::ifstream in(DIR + "/" + name);
    if (!in)
    {
        std::cerr << "Missing " << DIR << "/" << name << " - run that member's program first.\n";
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

int main()
{
    // sequential: [size, run, ms]
    std::map<int, std::vector<double>> seq, gpuKernel, gpuE2E;
    for (auto &r : readCsv("sequential_results.csv"))
        seq[(int)r[0]].push_back(r[2]);
    // cuda: [size, run, kernel_ms, e2e_ms]
    for (auto &r : readCsv("cuda_results.csv"))
    {
        gpuKernel[(int)r[0]].push_back(r[2]);
        gpuE2E[(int)r[0]].push_back(r[3]);
    }
    // openmp: [size, threads, run, ms]
    std::map<std::pair<int, int>, std::vector<double>> omp;
    int maxThreads = 1;
    for (auto &r : readCsv("openmp_results.csv"))
    {
        omp[{(int)r[0], (int)r[1]}].push_back(r[3]);
        maxThreads = std::max(maxThreads, (int)r[1]);
    }

    auto avg = [](const std::vector<double> &v)
    {
        double s = 0;
        for (double x : v)
            s += x;
        return v.empty() ? 0.0 : s / v.size();
    };

    // ---- averages.csv ----
    std::ofstream out(DIR + "/averages.csv");
    out << "Matrix_Size,Sequential_CPU_ms,Parallel_CPU_ms,Parallel_GPU_ms,"
        << "Speedup_CPU_vs_Seq,Speedup_GPU_vs_CPU,Speedup_GPU_vs_Seq,GPU_Kernel_ms" << std::endl;
    std::cout << "Matrix_Size,Seq_ms,OpenMP_ms,GPU_E2E_ms,GPU_Kernel_ms,CPU/Seq,GPU/CPU,GPU/Seq" << std::endl;
    for (auto &[size, runs] : seq)
    {
        double s = avg(runs);
        double o = avg(omp[{size, maxThreads}]);
        double g = avg(gpuE2E[size]);
        double k = avg(gpuKernel[size]);
        out << size << "," << std::fixed << std::setprecision(4) << s << "," << o << "," << g << ","
            << std::setprecision(2) << s / o << "," << o / g << "," << s / g << ","
            << std::setprecision(4) << k << std::endl;
        std::cout << size << "x" << size << "," << std::fixed << std::setprecision(4) << s << ","
                  << o << "," << g << "," << k << "," << std::setprecision(2)
                  << s / o << "x," << o / g << "x," << s / g << "x" << std::endl;
    }

    // ---- openmp_scaling_averages.csv ----
    std::ofstream sc(DIR + "/openmp_scaling_averages.csv");
    sc << "Matrix_Size,Threads,Sequential_CPU_ms,Parallel_CPU_ms,"
       << "Speedup_CPU_vs_Seq,Parallel_Efficiency" << std::endl;
    std::cout << "\nMatrix_Size,Threads,Seq_ms,OpenMP_ms,Speedup,Efficiency" << std::endl;
    for (auto &[key, runs] : omp)
    {
        int size = key.first, threads = key.second;
        double s = avg(seq[size]), o = avg(runs);
        double speedup = s / o, eff = speedup / threads;
        sc << size << "," << threads << "," << std::fixed << std::setprecision(4) << s << "," << o << ","
           << std::setprecision(2) << speedup << "," << std::setprecision(4) << eff << std::endl;
        std::cout << size << "x" << size << "," << threads << "," << std::fixed << std::setprecision(4)
                  << s << "," << o << "," << std::setprecision(2) << speedup << "x,"
                  << std::setprecision(4) << eff << std::endl;
    }

    // ---- charts (reuses the existing plot script if present) ----
    if (std::filesystem::exists("plot_benchmark.py"))
    {
        std::string cmd = "python plot_benchmark.py \"" + DIR + "\"";
        if (std::system(cmd.c_str()) != 0)
            std::cerr << "Warning: plot failed (is matplotlib installed?)." << std::endl;
    }
    else
        std::cerr << "plot_benchmark.py not found; skipping charts." << std::endl;

    std::cout << "Wrote " << DIR << "/averages.csv and " << DIR << "/openmp_scaling_averages.csv" << std::endl;
    return 0;
}