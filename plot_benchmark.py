"""Render comparison charts from the benchmark CSV files.

Usage: python plot_benchmark.py [results_dir]
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

TIME_SERIES = [
    ("Sequential_CPU_ms", "Sequential (CPU)", "#4d4d4d", "o"),
    ("Parallel_CPU_ms", "Parallel (OpenMP)", "#1f77b4", "s"),
    ("Parallel_GPU_ms", "Parallel (CUDA)", "#2ca02c", "^"),
]

SPEEDUP_SERIES = [
    ("Speedup_CPU_vs_Seq", "OpenMP vs Sequential", "#1f77b4", "s"),
    ("Speedup_GPU_vs_CPU", "CUDA vs OpenMP", "#ff7f0e", "D"),
    ("Speedup_GPU_vs_Seq", "CUDA vs Sequential", "#2ca02c", "^"),
]


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def column(rows, name):
    return [float(row[name]) for row in rows]


def matrix_sizes(rows):
    return [int(row["Matrix_Size"]) for row in rows]


def configure_axes(ax, sizes, title, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Matrix size (N x N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="--", alpha=0.4)


def save_figure(figure, output_dir, filename):
    path = os.path.join(output_dir, filename)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_execution_time(output_dir, averages, readings):
    sizes = matrix_sizes(averages)
    figure, ax = plt.subplots(figsize=(9, 5.5))

    for name, label, color, marker in TIME_SERIES:
        ax.plot(
            sizes,
            column(averages, name),
            color=color,
            marker=marker,
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
        "Execution Time Comparison",
        "Execution time (ms, log scale)",
    )
    ax.set_yscale("log")
    ax.legend()
    return save_figure(figure, output_dir, "execution_time_comparison.png")


def plot_speedups(output_dir, averages):
    sizes = matrix_sizes(averages)
    figure, ax = plt.subplots(figsize=(9, 5.5))

    for name, label, color, marker in SPEEDUP_SERIES:
        ax.plot(
            sizes,
            column(averages, name),
            color=color,
            marker=marker,
            linewidth=2,
            label=label,
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, label="No speedup")
    configure_axes(ax, sizes, "Speedup Comparison", "Speedup (x)")
    ax.legend()
    return save_figure(figure, output_dir, "speedup_comparison.png")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    averages_path = os.path.join(results_dir, "averages.csv")
    if not os.path.isfile(averages_path):
        sys.exit(f"error: {averages_path} not found")

    readings_path = os.path.join(results_dir, "all_readings.csv")
    readings = read_csv(readings_path) if os.path.isfile(readings_path) else []
    averages = read_csv(averages_path)

    paths = [
        plot_execution_time(results_dir, averages, readings),
        plot_speedups(results_dir, averages),
    ]

    for path in paths:
        print(f"graph written to {path}")


if __name__ == "__main__":
    main()
