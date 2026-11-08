"""Plot `make sweep` results: runtime vs sparsity, runtime vs size, speedup over BLAS."""
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import stdev

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
    "SSR basic":    ("#eb6834", "-",  "o"),
    "SSR unrolled": ("#eb6834", "--", "s"),
    "SSR tree":     ("#eb6834", ":",  "^"),
    "TCSC basic":   ("#1baf7a", "-",  "o"),
    "TCSC merged":  ("#1baf7a", "--", "s"),
    "Eigen CSR":    ("#eda100", "-",  "o"),
    "Eigen CSC":    ("#eda100", "--", "s"),
}


def read(path, x_field):
    series = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            samples = [float(value) for value in row.get("sample_us", "").split(";") if value]
            series[row["implementation"]].append(
                (float(row[x_field]), float(row["median_us"]), samples))
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

        # Samples are batch averages; preserve medians as the curve centres
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
    title = r"Bands: $\pm 1$ sample SD" if ax.collections else None
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

    with (results / "sparsity.csv").open() as stream:
        first = next(csv.DictReader(stream))
    shape = f"M={first['m']}, K={first['k']}, N={first['n']}"
    by_sparsity = read(results / "sparsity.csv", "target_sparsity")

    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax, by_sparsity)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(percent))
    style_axes(ax, f"Runtime vs sparsity ({shape})", "Fraction of zero weights",
               r"Median time per call ($\mu$s, log scale, lower is better)")
    legend(ax)
    save(fig, out, "runtime_vs_sparsity")

    with (results / "size.csv").open() as stream:
        first = next(csv.DictReader(stream))
    by_size = read(results / "size.csv", "k")
    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax, by_size)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    sizes = sorted({x for points in by_size.values() for x, _, _ in points})
    ax.set_xticks(sizes, [str(int(s)) for s in sizes])
    ax.xaxis.set_minor_formatter(NullFormatter())
    zeros = percent(float(first["target_sparsity"]), None)
    style_axes(ax, f"Runtime vs size (M={first['m']}, {zeros} zeros)",
               "K = N", r"Median time per call ($\mu$s, log scale, lower is better)")
    legend(ax)
    save(fig, out, "runtime_vs_size")

    blas = {x: (median, samples) for x, median, samples in by_sparsity["BLAS"]}
    speedups = {}
    for name, points in by_sparsity.items():
        if name == "BLAS":
            continue

        # Match repetitions so the spread includes noise in both implementations
        speedups[name] = [
            (x, blas[x][0] / median,
             [b / t for b, t in zip(blas[x][1], samples)])
            for x, median, samples in points
        ]

    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax, speedups)
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
