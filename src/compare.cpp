// compare.cpp - merge the per-program averages and draw comparison charts
// Build: build.bat compare.cpp
// Run after sequential, openmp and cuda have produced their averages files.
// With a folder argument, the three files are read from that folder (this is
// what run_all.bat does). Without an argument, the newest file of each kind
// under results/ is used, so the three programs can be run independently.
// Output: a new timestamped folder (or the folder argument) containing
//         averages.csv and the three comparison charts.
#include "common.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>

// Reads a CSV (skipping the header) into rows of doubles.
std::vector<std::vector<double>> readCsv(const std::filesystem::path &path)
{
    std::ifstream in(path);
    if (!in)
    {
        std::cerr << "ERROR: " << path.string() << " not found." << std::endl;
        return {};
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

// Finds the most recently written file with this name anywhere under results/.
std::filesystem::path newestResultsFile(const std::string &filename)
{
    std::filesystem::path newest;
    std::filesystem::file_time_type newest_time{};

    if (!std::filesystem::exists(RESULTS_DIR))
        return newest;

    for (const auto &entry : std::filesystem::recursive_directory_iterator(RESULTS_DIR))
    {
        if (!entry.is_regular_file() || entry.path().filename().string() != filename)
            continue;

        auto write_time = entry.last_write_time();
        if (newest.empty() || write_time > newest_time)
        {
            newest = entry.path();
            newest_time = write_time;
        }
    }
    return newest;
}

int main(int argc, char *argv[])
{
    const bool use_folder = argc > 1;

    // Resolve the three inputs first, so a missing file does not leave an
    // empty timestamped folder behind.
    std::filesystem::path sequential_path =
        use_folder ? std::filesystem::path(argv[1]) / "sequential_averages.csv"
                   : newestResultsFile("sequential_averages.csv");
    std::filesystem::path openmp_path =
        use_folder ? std::filesystem::path(argv[1]) / "openmp_averages.csv"
                   : newestResultsFile("openmp_averages.csv");
    std::filesystem::path cuda_path =
        use_folder ? std::filesystem::path(argv[1]) / "cuda_averages.csv"
                   : newestResultsFile("cuda_averages.csv");

    std::vector<std::string> missing;
    if (!std::filesystem::exists(sequential_path))
        missing.push_back("sequential_averages.csv (run sequential.exe)");
    if (!std::filesystem::exists(openmp_path))
        missing.push_back("openmp_averages.csv (run openmp.exe)");
    if (!std::filesystem::exists(cuda_path))
        missing.push_back("cuda_averages.csv (run cuda.exe)");

    if (!missing.empty())
    {
        std::cerr << "ERROR: could not find these results:" << std::endl;
        for (const std::string &name : missing)
            std::cerr << "  - " << name << std::endl;
        std::cerr << "Run the missing program(s) first, or pass one shared run folder:" << std::endl;
        std::cerr << "  compare.exe results\\<timestamp>" << std::endl;
        return 1;
    }

    // Without a shared folder, write the merged results to a new timestamped
    // folder and report which per-program files were used.
    std::string run_dir = resolveRunDirectory(argc, argv);
    std::cout << "Output directory: " << run_dir << std::endl;
    if (!use_folder)
    {
        std::cout << "Using sequential averages: " << sequential_path.string() << std::endl;
        std::cout << "Using OpenMP averages:     " << openmp_path.string() << std::endl;
        std::cout << "Using CUDA averages:       " << cuda_path.string() << std::endl;
    }

    // sequential averages: [size, ms]
    std::map<int, double> sequential_ms;
    for (auto &row : readCsv(sequential_path))
        sequential_ms[(int)row[0]] = row[1];

    // openmp averages: [size, threads, ms, speedup, efficiency]
    std::map<std::pair<int, int>, double> openmp_ms;
    int max_threads = 1;
    for (auto &row : readCsv(openmp_path))
    {
        openmp_ms[{(int)row[0], (int)row[1]}] = row[2];
        max_threads = std::max(max_threads, (int)row[1]);
    }

    // cuda averages: [size, ms]
    std::map<int, double> gpu_ms;
    for (auto &row : readCsv(cuda_path))
        gpu_ms[(int)row[0]] = row[1];

    std::ofstream out(std::filesystem::path(run_dir) / "averages.csv");
    if (!out)
    {
        std::cerr << "ERROR: could not open " << run_dir << "/averages.csv" << std::endl;
        return 1;
    }

    out << "Matrix_Size,Sequential_CPU_ms,Parallel_CPU_ms,Parallel_GPU_ms,"
        << "Speedup_CPU_vs_Seq,Speedup_GPU_vs_CPU,Speedup_GPU_vs_Seq" << std::endl;
    std::cout << "Matrix_Size,Seq_ms,OpenMP_ms,GPU_ms,CPU/Seq,GPU/CPU,GPU/Seq" << std::endl;

    for (auto &[size, seq_ms] : sequential_ms)
    {
        double omp_ms = openmp_ms[{size, max_threads}];
        double gpu_time_ms = gpu_ms[size];

        out << size << "," << std::fixed << std::setprecision(4)
            << seq_ms << "," << omp_ms << "," << gpu_time_ms << ","
            << std::setprecision(2) << seq_ms / omp_ms << ","
            << omp_ms / gpu_time_ms << ","
            << seq_ms / gpu_time_ms << std::endl;

        std::cout << size << "x" << size << "," << std::fixed << std::setprecision(4)
                  << seq_ms << "," << omp_ms << "," << gpu_time_ms << ","
                  << std::setprecision(2) << seq_ms / omp_ms << "x,"
                  << omp_ms / gpu_time_ms << "x,"
                  << seq_ms / gpu_time_ms << "x" << std::endl;
    }

    std::cout << "Wrote " << run_dir << "/averages.csv" << std::endl;
    std::cout << "Using OpenMP results at " << max_threads << " threads." << std::endl;

    if (!runPlotScript(argv[0], "compare \"" + run_dir + "\""))
        std::cerr << "Warning: chart generation failed (is matplotlib installed?)." << std::endl;

    return 0;
}
