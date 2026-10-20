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
    ("Sequential_CPU_ms", "Sequential (CPU)", "#000000", "o", "-"),
    ("Parallel_CPU_ms", "Parallel (OpenMP)", "#FF7F0E", "s", "--"),
    ("Parallel_GPU_ms", "Parallel (CUDA)", "#1E90FF", "^", "-."),
]

SPEEDUP_VS_SEQUENTIAL_SERIES = [
    ("Speedup_CPU_vs_Seq", "Parallel CPU (OpenMP) Speedup", "#E53935", "s", "-"),
    ("Speedup_GPU_vs_Seq", "Parallel GPU (CUDA) Speedup", "#1E90FF", "^", "-"),
]

GPU_VS_CPU_SERIES = [
    ("Speedup_GPU_vs_CPU", "Parallel GPU (CUDA) Speedup", "#1E90FF", "D", "-"),
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


def plot_openmp_scaling(output_dir, scaling_averages):
    """Plot OpenMP speedup and parallel efficiency against thread count."""
    sizes = sorted({int(row["Matrix_Size"]) for row in scaling_averages})
    thread_counts = sorted({int(row["Threads"]) for row in scaling_averages})
    colors = [
        plt.cm.viridis(0.15 + 0.7 * index / max(1, len(sizes) - 1))
        for index in range(len(sizes))
    ]

    figure, (speedup_ax, efficiency_ax) = plt.subplots(1, 2, figsize=(13, 5.5))

    for index, size in enumerate(sizes):
        rows = sorted(
            (row for row in scaling_averages if int(row["Matrix_Size"]) == size),
            key=lambda row: int(row["Threads"]),
        )
        threads = [int(row["Threads"]) for row in rows]
        color = colors[index]
        label = f"{size}x{size}"

        speedup_ax.plot(
            threads,
            [float(row["Speedup_CPU_vs_Seq"]) for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        efficiency_ax.plot(
            threads,
            [float(row["Parallel_Efficiency"]) for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

    speedup_ax.plot(
        thread_counts,
        thread_counts,
        linestyle=":",
        color="black",
        label="Ideal speedup",
    )

    panels = [
        (speedup_ax, "Speedup vs. Threads", "Speedup (x)"),
        (efficiency_ax, "Parallel Efficiency vs. Threads", "Parallel efficiency"),
    ]
    for ax, title, ylabel in panels:
        ax.set_xscale("log", base=2)
        ax.set_xticks(thread_counts)
        ax.set_xticklabels([str(threads) for threads in thread_counts])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    speedup_ax.legend()
    figure.suptitle("OpenMP Strong Scaling")
    return save_figure(figure, output_dir, "openmp_strong_scaling.png")


def main():
    """Main function to read CSV files and generate plots."""
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    averages_path = os.path.join(results_dir, "averages.csv")
    if not os.path.isfile(averages_path):
        sys.exit(f"error: {averages_path} not found")

    readings_path = os.path.join(results_dir, "all_readings.csv")
    readings = read_csv(readings_path) if os.path.isfile(readings_path) else []
    averages = read_csv(averages_path)

    scaling_path = os.path.join(results_dir, "openmp_scaling_averages.csv")
    scaling_averages = read_csv(scaling_path) if os.path.isfile(scaling_path) else []

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

    if scaling_averages:
        paths.append(plot_openmp_scaling(results_dir, scaling_averages))

    for path in paths:
        print(f"graph written to {path}")


if __name__ == "__main__":
    main()
