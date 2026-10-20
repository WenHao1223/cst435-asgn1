"""Render comparison charts from the benchmark CSV files.

Usage: python plot_benchmark.py <run_dir>
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

TIME_SERIES = [
    ("Sequential_CPU_ms", "Sequential (CPU)", "#FF7F0E", "o", "-"),
    ("Parallel_CPU_ms", "Parallel (OpenMP)", "#1E90FF", "s", "--"),
    ("Parallel_GPU_ms", "Parallel (CUDA)", "#00C853", "^", "-."),
]

SPEEDUP_VS_SEQUENTIAL_SERIES = [
    ("Speedup_CPU_vs_Seq", "Parallel CPU (OpenMP) Speedup", "#1E90FF", "s", "-"),
    ("Speedup_GPU_vs_Seq", "Parallel GPU (CUDA) Speedup", "#00C853", "^", "-"),
]

GPU_VS_CPU_SERIES = [
    ("Speedup_GPU_vs_CPU", "Parallel GPU (CUDA) Speedup", "#FF7F0E", "D", "-"),
]


def read_csv(path):
    """Read a CSV file and return a list of dictionaries, one per row."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def column(rows, name):
    """Return a list of values from a specific column in the CSV rows."""
    return [float(row[name]) for row in rows]


def matrix_sizes(rows):
    """Return a list of matrix sizes from the CSV rows."""
    return [int(row["Matrix_Size"]) for row in rows]


def configure_axes(ax, sizes, title, ylabel):
    """Configure the axes of a matplotlib plot."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{size}x{size}" for size in sizes])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Matrix size (N x N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)


def save_figure(figure, output_dir, filename):
    """Save a matplotlib figure to a file."""
    path = os.path.join(output_dir, filename)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_execution_time(output_dir, averages, readings):
    """Plot execution time for each implementation."""
    sizes = matrix_sizes(averages)
    figure, ax = plt.subplots(figsize=(9, 5.5))

    for name, label, color, marker, linestyle in TIME_SERIES:
        ax.plot(
            sizes,
            column(averages, name),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
        if readings:
            # Show every individual reading as a faint dot for spread visibility.
            for size in sizes:
                values = [
                    float(row[name])
                    for row in readings
                    if int(row["Matrix_Size"]) == size
                ]
                ax.scatter(
                    [size] * len(values),
                    values,
                    color=color,
                    alpha=0.25,
                    s=16,
                    zorder=1,
                )

    configure_axes(
        ax,
        sizes,
        "Execution Time vs. Matrix Size",
        "Execution time (ms, log scale)",
    )
    ax.set_yscale("log")
    ax.legend()
    return save_figure(figure, output_dir, "execution_time_comparison.png")


def plot_speedup(output_dir, averages, title, filename, series):
    """Plot speedup for each implementation."""
    sizes = matrix_sizes(averages)
    figure, ax = plt.subplots(figsize=(9, 5.5))

    for name, label, color, marker, linestyle in series:
        ax.plot(
            sizes,
            column(averages, name),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, label="No speedup")
    configure_axes(ax, sizes, title, "Speedup (x)")
    ax.set_ylim(bottom=0)
    ax.legend()
    return save_figure(figure, output_dir, filename)


def plot_speedup_bars(output_dir, averages, title, filename, series):
    """Plot speedup per matrix size as grouped bars."""
    sizes = matrix_sizes(averages)
    positions = list(range(len(sizes)))
    bar_width = 0.8 / len(series)
    figure, ax = plt.subplots(figsize=(9, 5.5))

    for index, (name, label, color, _, _) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * bar_width
        bars = ax.bar(
            [position + offset for position in positions],
            column(averages, name),
            width=bar_width,
            color=color,
            label=label,
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{size}x{size}" for size in sizes])
    ax.set_xlabel("Matrix size (N x N)")
    ax.set_ylabel("Speedup (x)")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend()
    return save_figure(figure, output_dir, filename)


def main():
    """Main function to read CSV files and generate plots."""
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    averages_path = os.path.join(results_dir, "averages.csv")
    if not os.path.isfile(averages_path):
        sys.exit(f"error: {averages_path} not found")

    readings_path = os.path.join(results_dir, "all_readings.csv")
    readings = read_csv(readings_path) if os.path.isfile(readings_path) else []
    averages = read_csv(averages_path)

    paths = [
        plot_execution_time(results_dir, averages, readings),
        plot_speedup_bars(
            results_dir,
            averages,
            "Speedup Relative to Sequential CPU",
            "speedup_vs_sequential_cpu.png",
            SPEEDUP_VS_SEQUENTIAL_SERIES,
        ),
        plot_speedup(
            results_dir,
            averages,
            "GPU Speedup Over Parallel CPU",
            "gpu_speedup_over_parallel_cpu.png",
            GPU_VS_CPU_SERIES,
        ),
    ]

    for path in paths:
        print(f"graph written to {path}")


if __name__ == "__main__":
    main()
