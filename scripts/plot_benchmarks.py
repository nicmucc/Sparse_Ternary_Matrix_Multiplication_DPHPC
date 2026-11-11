"""Plot `make sweep` results: runtime vs sparsity, runtime vs size, speedup over BLAS."""
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# Color identifies the algorithm family, line style/marker the variant.
STYLE = {
    "BLAS":         ("#0b0b0b", "-",  "o"),
    "Naive":        ("#8a8984", "-",  "o"),
    "RSR":          ("#2a78d6", "-",  "o"),
    "RSR++":        ("#2a78d6", "--", "s"),
    "RSR binary split v4.2": ("#6051a7", "-", "D"),
    "RSR ternary v3.1":      ("#a33c75", "-", "o"),
    "RSR ternary v3.3":      ("#a33c75", ":", "^"),
    "SSR basic":    ("#eb6834", "-",  "o"),
    "SSR unrolled": ("#eb6834", "--", "s"),
    "SSR tree":     ("#eb6834", ":",  "^"),
    "TCSC basic":   ("#1baf7a", "-",  "o"),
    "TCSC merged":  ("#1baf7a", "--", "s"),
    "Eigen CSR":    ("#eda100", "-",  "o"),
    "Eigen CSC":    ("#eda100", "--", "s"),
}


def read(rows, x_field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["implementation"], float(row[x_field]))].append(row)

    series = defaultdict(list)
    for (name, x), datasets in grouped.items():
        centre = median(float(row["median_us"]) for row in datasets)
        samples = [float(value) for row in datasets
                   for value in row.get("sample_us", "").split(";") if value]
        series[name].append((x, centre, samples))
    return {name: sorted(points) for name, points in series.items()}


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e")


def draw(ax, series):
    for name, points in series.items():
        color, line, marker = STYLE.get(name, ("#52514e", "-", "o"))
        xs = [x for x, _, _ in points]
        ys = [median for _, median, _ in points]

        # Pool batch averages across seeds; curves use medians of dataset medians
        spreads = [stdev(samples) if len(samples) > 1 else 0
                   for _, _, samples in points]
        if any(spreads):
            lower = [max(0, y - spread) for y, spread in zip(ys, spreads)]
            upper = [y + spread for y, spread in zip(ys, spreads)]
            ax.fill_between(xs, lower, upper, color=color, alpha=0.12, linewidth=0)

        width = 2.6 if name == "BLAS" else 2
        ax.plot(xs, ys, color=color, linestyle=line, marker=marker, markersize=6,
                linewidth=width, markeredgecolor="white", markeredgewidth=1, label=name)


def legend(ax):
    title = r"Bands: $\pm 1$ pooled sample SD" if ax.collections else None
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=10, title=title)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def percent(x, _):
    label = f"{x:.0%}"
    return label.replace("%", r"\%") if plt.rcParams["text.usetex"] else label


def main():
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "build/results")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "out/plots")
    out.mkdir(parents=True, exist_ok=True)
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.size": 11,
        "font.family": "serif" if use_tex else "DejaVu Sans",
        "font.serif": ["Computer Modern Roman"],
    })

    with (results / "benchmarks.csv").open() as stream:
        rows = list(csv.DictReader(stream))

    # Identify the generated sweep families from the measured configurations
    shapes = defaultdict(set)
    size_groups = defaultdict(set)
    for row in rows:
        shapes[(row["m"], row["k"], row["n"])].add(row["target_sparsity"])
        if row["k"] == row["n"]:
            size_groups[(row["m"], row["target_sparsity"])].add(row["k"])
    sparsity_shape = max(shapes, key=lambda shape: len(shapes[shape]))
    size_group = max(size_groups, key=lambda group: len(size_groups[group]))

    sparsity_rows = [row for row in rows
                     if (row["m"], row["k"], row["n"]) == sparsity_shape]
    size_rows = [row for row in rows
                 if row["k"] == row["n"]
                 and (row["m"], row["target_sparsity"]) == size_group]
    shape = "M={}, K={}, N={}".format(*sparsity_shape)
    by_sparsity = read(sparsity_rows, "target_sparsity")

    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax, by_sparsity)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(percent))
    style_axes(ax, f"Runtime vs sparsity ({shape})", "Fraction of zero weights",
               r"Median time per call ($\mu$s, log scale, lower is better)")
    legend(ax)
    save(fig, out, "runtime_vs_sparsity")

    by_size = read(size_rows, "k")
    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax, by_size)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    sizes = sorted({x for points in by_size.values() for x, _, _ in points})
    ax.set_xticks(sizes, [str(int(s)) for s in sizes])
    ax.xaxis.set_minor_formatter(NullFormatter())
    zeros = percent(float(size_group[1]), None)
    style_axes(ax, f"Runtime vs size (M={size_group[0]}, {zeros} zeros)",
               "K = N", r"Median time per call ($\mu$s, log scale, lower is better)")
    legend(ax)
    save(fig, out, "runtime_vs_size")

    blas = {row["dataset"]: row for row in sparsity_rows if row["implementation"] == "BLAS"}
    speedup_rows = []
    for row in sparsity_rows:
        if row["implementation"] == "BLAS":
            continue

        # Pair each implementation with BLAS on the same saved dataset
        baseline = blas[row["dataset"]]
        samples = [float(value) for value in row.get("sample_us", "").split(";") if value]
        blas_samples = [float(value) for value in baseline.get("sample_us", "").split(";") if value]
        speedup_rows.append({
            **row,
            "median_us": float(baseline["median_us"]) / float(row["median_us"]),
            "sample_us": ";".join(str(b / t) for b, t in zip(blas_samples, samples)),
        })

    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax, read(speedup_rows, "target_sparsity"))
    ax.axhline(1, color="#0b0b0b", linewidth=1.2, linestyle="-", zorder=0)
    ax.text(ax.get_xlim()[0], 1, " BLAS", va="bottom", ha="left", color="#0b0b0b", fontsize=10)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: rf"{y:g}$\times$"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(percent))
    style_axes(ax, f"Speedup over BLAS vs sparsity ({shape})", "Fraction of zero weights",
               r"Speedup over BLAS (above $1\times$ is faster)")
    legend(ax)
    save(fig, out, "speedup_vs_sparsity")
    print(f"Plots written to {out}/")


if __name__ == "__main__":
    main()
