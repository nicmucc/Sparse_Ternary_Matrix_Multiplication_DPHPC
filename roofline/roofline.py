"""Measure all kernels into kernels.csv, or replot the accumulated measurements."""
import argparse
import csv
import io
import os
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "build/results/roofline/release/kernels.csv"
OUTPUT = ROOT / "out/plots/roofline/release/roofline.png"
FIELDS = ["kernel", "M", "K", "N", "sparsity", "threads", "time_ms",
          "flops", "bytes", "ssr_width", "backend", "region", "machine", "build_profile", "seed"]

# Hardware ceilings are reference bounds, not sustained measurements.
HARDWARE = {
    "ultra155h": {"label": "Intel Core Ultra 7 155H", "freq": 4.80,
                 "simd_bits": 256, "cores": 1,
                 "memory": {"ddr5-5600": 89.6, "lpddr5x-7467": 119.472}},
    "xeon": {"label": "Intel Xeon Gold 6240", "freq": 2.60,
             "simd_bits": 512, "cores": 18,
             "memory": {"ddr4-2933": 140.784}},
}


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return result


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames != FIELDS:
            raise ValueError("Incompatible CSV format; use make clean-roofline to start again")
        for row in rows:
            for key in ("M", "K", "N", "threads", "ssr_width"):
                if int(row[key]) < 1:
                    raise ValueError(f"Invalid {key} in CSV")
            for key in ("time_ms", "bytes"):
                positive(row[key])
            if not math.isfinite(float(row["flops"])) or float(row["flops"]) < 0:
                raise ValueError("Invalid FLOP count in CSV")
            if not 0 <= float(row["sparsity"]) <= 1:
                raise ValueError("Invalid sparsity in CSV")
    return rows


def measure(args):
    previous = read_rows(args.points)
    if any(row["machine"] != args.machine for row in previous):
        raise ValueError("Use a separate --points file for each machine")
    if not args.benchmark.is_file():
        raise ValueError("Benchmark executable missing: run make roofline")
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads),
           "OPENBLAS_NUM_THREADS": str(args.threads)}
    command = [str(args.benchmark.resolve()), "--dataset", str(args.dataset.resolve()),
               "--samples", str(args.samples), "--min-ms", str(args.min_ms),
               "--width", str(args.width)]
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE,
                            env=env, cwd=ROOT)
    new_rows = [from_benchmark(row, args) for row in csv.DictReader(io.StringIO(result.stdout))]
    if not new_rows:
        raise ValueError("Benchmark produced no measurements")
    # Replace atomically after a successful complete run, preserving previous rows.
    args.points.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.points.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(previous + new_rows)
        temporary.replace(args.points)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Added {len(new_rows)} measurements to {args.points}")


def from_benchmark(row, args):
    return dict(kernel=row["implementation"], M=row["m"], K=row["k"], N=row["n"],
                sparsity=row["sparsity"], threads=str(args.threads),
                time_ms=float(row["median_us"]) / 1000, flops=row["model_flops"],
                bytes=row["model_bytes"], ssr_width=row["width"], backend=row["backend"],
                region=row["scope"], machine=args.machine, build_profile=args.build_profile,
                seed=row["seed"])


def configuration(row):
    return (int(row["M"]), int(row["K"]), int(row["N"]), float(row["sparsity"]),
            int(row["threads"]), int(row["ssr_width"]), row["build_profile"], row["region"])


def plot(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = read_rows(args.points)
    if not rows:
        raise ValueError("No measurements found; run make roofline first")
    if any(row["machine"] != args.machine for row in rows):
        raise ValueError("CSV contains a different machine; use a separate --points file")
    hw = HARDWARE[args.machine]
    if args.machine == "ultra155h" and args.profile == "socket":
        raise ValueError("Ultra 155H has heterogeneous cores: use the core reference")
    cores = 1 if args.profile == "core" else hw["cores"]
    if any(int(row["threads"]) > cores for row in rows):
        raise ValueError("CSV thread budget exceeds the selected hardware profile")
    frequency = args.frequency_ghz or hw["freq"]
    peak = args.peak_gflops or cores * frequency * (hw["simd_bits"] / 32) * 2 * 2
    memory = args.memory or next(iter(hw["memory"]))
    if memory not in hw["memory"]:
        raise ValueError("Memory type is incompatible with the selected machine")
    bw = args.mem_bw or hw["memory"][memory]
    ridge = peak / bw
    points = [(row, float(row["flops"]) / float(row["bytes"]),
               float(row["flops"]) / float(row["time_ms"]) / 1e6)
              for row in rows if float(row["flops"]) > 0]
    if not points:
        raise ValueError("Zero-FLOP measurements cannot be displayed on logarithmic axes")

    xmin = min([1e-2, ridge / 4] + [i / 4 for _, i, _ in points])
    xmax = max([1e3, ridge * 4] + [i * 4 for _, i, _ in points])
    ymin = min([1.] + [p / 2 for _, _, p in points])
    ymax = max([1e4, peak * 2] + [p * 2 for _, _, p in points])
    plt.rcParams.update({"text.usetex": False, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#d0d0d0")
    ax.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(ymin, ymax))
    intensity = np.logspace(np.log10(xmin), np.log10(ridge), 350)
    ax.plot(intensity, bw * intensity, color="black", lw=1.5)
    intensity = np.logspace(np.log10(ridge), np.log10(xmax), 200)
    ax.plot(intensity, np.full_like(intensity, peak), color="black", lw=1.5)
    ax.set_xlabel("Modelled arithmetic intensity [FLOP/byte]", fontsize=14)
    ax.grid(True, which="both", ls="-", lw=0.8, color="white", alpha=0.8)
    ax.tick_params(axis="both", which="major", labelsize=12, length=6, width=1.5)
    plt.subplots_adjust(top=0.82)
    fig.canvas.draw()
    bbox = ax.get_position()
    scope = "P-core reference" if args.machine == "ultra155h" else (
        "1 core" if args.profile == "core" else "1 socket")
    fig.text(bbox.x0, 0.89, f"Modelled Roofline on {hw['label']} ({scope}) — FP32",
             ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(bbox.x0, 0.854, "Performance [GFLOP/s]",
             ha="left", va="top", fontsize=12)
    x = min(0.8, ridge / 10)
    x = min(max(x, xmin * 3, ymin * 3 / bw), ridge / 1.8)
    y = bw * x
    p1, p2 = ax.transData.transform((x, y)), ax.transData.transform((x * 2, y * 2))
    angle = np.degrees(np.arctan2(*(p2 - p1)[::-1]))
    memory_label = "Memory roof" if args.mem_bw else "Memory upper bound"
    ax.text(x, y * 0.72, f"{memory_label} ({bw:.1f} GB/s)",
            rotation=angle, rotation_mode="anchor", ha="center", va="center",
            fontsize=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    kind = "measured" if args.peak_gflops else "theoretical"
    ax.text(ridge * 1.3, peak * 1.08, f"Compute roof ({peak:.1f} GFLOP/s, {kind})",
            ha="left", va="bottom", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))

    names = list(dict.fromkeys(row["kernel"] for row, _, _ in points))
    configs = list(dict.fromkeys(configuration(row) for row, _, _ in points))
    marker_shapes = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
    def marker(index):
        return marker_shapes[index] if index < len(marker_shapes) else f"$ {index + 1} $"
    colors = {name: plt.get_cmap("tab20")((i % 10) * 2 + i // 10) for i, name in enumerate(names)}
    for row, intensity, performance in points:
        index = configs.index(configuration(row))
        ax.scatter(intensity, performance, s=65, color=colors[row["kernel"]],
                   marker=marker(index), edgecolors="black", linewidths=0.5, zorder=5)
    handles = [Line2D([], [], color=colors[name], marker="o", linestyle="",
                      markeredgecolor="black", label=name) for name in names]
    for index, (m, k, n, sparsity, threads, width, build_profile, region) in enumerate(configs):
        label = f"{m}×{k}×{n}, {sparsity:.3%} zeros, {threads}T"
        if len({c[5:] for c in configs}) > 1:
            label += f", L={width}, {build_profile}, {region}"
        handles.append(Line2D([], [], color="black", marker=marker(index), linestyle="",
                              markerfacecolor="white", label=label))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Model: source-level FLOPs / representation footprint; DRAM traffic is not measured.")
    print(f"Plotted {len(points)} measurements: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help="append a new run, then plot")
    parser.add_argument("--clean", action="store_true", help="remove the default CSV and plot")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/benchmarks/m16_k1024_n1024_s0.95_seed42")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--min-ms", type=positive, default=10)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--build-profile", default="release")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "build/release/benchmark")
    parser.add_argument("--points", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--machine", choices=HARDWARE, default="ultra155h")
    parser.add_argument("--profile", choices=("core", "socket"), default="core")
    parser.add_argument("--memory", choices=("ddr5-5600", "lpddr5x-7467", "ddr4-2933"))
    parser.add_argument("--mem-bw", type=positive)
    parser.add_argument("--frequency-ghz", type=positive)
    parser.add_argument("--peak-gflops", type=positive)
    args = parser.parse_args()
    if args.clean:
        args.points.unlink(missing_ok=True)
        args.output.unlink(missing_ok=True)
        return
    if args.samples < 3 or args.threads < 1 or not 1 <= args.width <= 8:
        parser.error("samples must be >=3, threads positive, width in [1,8]")
    hw = HARDWARE[args.machine]
    if args.machine == "ultra155h" and args.profile == "socket":
        parser.error("Ultra 155H has heterogeneous cores: use the core reference")
    if args.memory and args.memory not in hw["memory"]:
        parser.error("Memory type is incompatible with the selected machine")
    if args.measure and args.threads > (1 if args.profile == "core" else hw["cores"]):
        parser.error("Thread budget exceeds the selected hardware profile")
    if args.measure and args.profile == "core" and args.threads != 1:
        parser.error("the core profile requires one thread")
    try:
        if args.measure:
            measure(args)
        plot(args)
    except (OSError, ValueError, argparse.ArgumentTypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
