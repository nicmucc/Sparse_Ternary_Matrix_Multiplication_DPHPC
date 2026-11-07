"""Plot benchmark CSV results. Requires matplotlib and NumPy."""
import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with path.open() as stream:
        rows = [{k.strip(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(stream)]
    if not rows or "target_sparsity" not in rows[0]:
        raise ValueError("CSV must contain target_sparsity; rerun the benchmark")
    integer = ["m", "k", "n", "width", "seed", "pattern_rows", "inputs",
               "omp_threads", "blas_threads"]
    real = ["sparsity", "median_us", "speedup_blas", "preprocess_us", "weight_bytes"]
    for row in rows:
        for key in integer:
            row[key] = int(row[key]) if row[key] != "unset" else -1
        for key in real:
            row[key] = float(row[key])
        row["target"] = float(row["target_sparsity"]) if row["target_sparsity"] else row["sparsity"]
        row["shape"] = (row["m"], row["k"], row["n"])
    if len({(r["omp_threads"], r["blas_threads"], r["scope"]) for r in rows}) != 1:
        raise ValueError("Plot one thread configuration and measurement scope at a time")
    return rows


def configuration_effects(rows, names):
    effects, labels = [], []
    for varying, label in [("pattern_rows", "Prototype rows"),
                           ("inputs", "Input count"), ("width", "Block width")]:
        levels = sorted({row[varying] for row in rows})
        if len(levels) < 2:
            continue
        a, b = levels[0], levels[-1]
        groups = defaultdict(dict)
        fields = ["implementation", "shape", "target", "seed", "width",
                  "pattern_rows", "inputs", "omp_threads", "blas_threads", "scope"]
        fields.remove(varying)
        for row in rows:
            if row[varying] in (a, b):
                groups[tuple(row[k] for k in fields)][row[varying]] = row["median_us"]
        paired = defaultdict(list)
        for key, values in groups.items():
            if a in values and b in values:
                paired[key[0]].append(values[a] / values[b])
        effects.append([math.exp(statistics.mean(math.log(x) for x in paired[name]))
                        if paired[name] else math.nan for name in names])
        labels.append(f"{label} {b} vs {a}")
    return effects, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--inputs", type=int, default=1)
    parser.add_argument("--pattern-rows", type=int, default=0)
    args = parser.parse_args()
    rows = read_rows(args.csv)
    out = args.out or args.csv.parent / "plots"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm

    run = args.csv.parent / "run.json"
    metadata = json.loads(run.read_text()) if run.exists() else {}
    targets = sorted({row["target"] for row in rows})
    names = list(dict.fromkeys(r["implementation"] for r in rows))
    shapes = sorted({r["shape"] for r in rows})
    columns = [(shape, target) for shape in shapes for target in targets]
    labels = [f"{m} × {k} × {n}\n{target:.0%} zeros"
              for (m, k, n), target in columns]
    selected = [r for r in rows if r["width"] == args.width
                and r["inputs"] == args.inputs and r["pattern_rows"] == args.pattern_rows]
    if not selected:
        raise ValueError("No rows match the selected width, input count, and pattern count")
    geomean = lambda xs: math.exp(statistics.mean(math.log(x) for x in xs))
    def matrix(field, subset=selected, aggregate=statistics.median):
        groups = defaultdict(list)
        for row in subset:
            groups[(row["implementation"], row["shape"], row["target"])].append(row[field])
        return np.array([[aggregate(groups[(name, shape, target)])
                          if groups[(name, shape, target)] else np.nan
                          for shape, target in columns] for name in names])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    pdf = PdfPages(out / "benchmark-plots.pdf")
    def save(fig, name):
        fig.savefig(out / (name+".png"), dpi=170, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    context = (f"Width {args.width} · {args.inputs} input(s) · "
               + ("independent weight rows" if args.pattern_rows == 0
                  else f"{args.pattern_rows} repeating prototype rows"))
    machine = metadata.get("cpu", metadata.get("machine", ""))
    footer = ("Columns: M × K × N and target zero fraction; actual sparsity varies by seed. "
              "Times include API-internal allocations.\n"+str(machine))

    def ratio_heatmap(ax, values, xlabels, title):
        finite = values[np.isfinite(values) & (values > 0)]
        span = max(1.0, float(np.max(np.abs(np.log2(finite))))) if finite.size else 1
        transformed = np.ma.masked_invalid(np.log2(values))
        ax.imshow(transformed, cmap="RdBu", norm=TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span),
                  aspect="auto")
        ax.set_xticks(range(len(xlabels)), xlabels)
        ax.set_yticks(range(len(names)), names)
        ax.tick_params(length=0, pad=9)
        ax.set_title(title, loc="left", fontsize=12, pad=16)
        for i in range(len(names)):
            for j in range(len(xlabels)):
                value = values[i, j]
                if np.isfinite(value):
                    color = "white" if abs(math.log2(value))/span > 0.62 else "#17212b"
                    ax.text(j, i, f"{value:.2f}×", ha="center", va="center", color=color)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    speedups = matrix("speedup_blas", aggregate=geomean)
    ratio_heatmap(ax, speedups, labels, "Speedup over BLAS: above 1× is faster")
    fig.suptitle("Which implementations beat BLAS?", x=0.12, ha="left", fontsize=19, weight="bold")
    fig.text(0.12, 0.90, context+" · geometric mean across seeds", fontsize=11)
    fig.text(0.12, 0.015, "Blue = faster · red = slower · white = equal\n"+footer, fontsize=10)
    fig.subplots_adjust(top=0.83, bottom=0.17, left=0.16, right=0.97)
    save(fig, "01-speedup")

    effects, effect_labels = configuration_effects(rows, names)
    if effects:
        fig, ax = plt.subplots(figsize=(11, 7.5))
        ratio_heatmap(ax, np.array(effects).T, effect_labels,
                      "Above 1× means the changed configuration is faster")
        fig.suptitle("What changes performance?", x=0.12, ha="left", fontsize=19, weight="bold")
        fig.text(0.12, 0.90, "Geometric mean of matched pairs across all other measured settings", fontsize=11)
        fig.text(0.12, 0.015, "Blue = improvement · red = slowdown. Pattern repetition can also change actual sparsity.\nMatched shapes, target sparsity, seeds, and threads. Times include API-internal allocations.\n"+str(machine), fontsize=10)
        fig.subplots_adjust(top=0.82, bottom=0.18, left=0.18, right=0.97)
        save(fig, "02-configuration-effects")

    latency = matrix("median_us")
    fig, axes = plt.subplots(1, len(shapes), figsize=(5*len(shapes), 7), squeeze=False)
    colors = ["#2474ad", "#dd7f21", "#538f4e"]
    for s, shape in enumerate(shapes):
        ax = axes[0, s]
        for t, target in enumerate(targets):
            col = columns.index((shape, target))
            offset = (t-(len(targets)-1)/2)*0.23
            ax.scatter(latency[:, col], np.arange(len(names))+offset, s=55,
                       color=colors[t%len(colors)], label=f"{target:.0%} zeros", zorder=3)
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Latency (µs, log scale; lower is better)")
        ax.set_title(" × ".join(map(str, shape)), pad=16)
        ax.grid(axis="x", alpha=0.2)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.55, 0.065),
               ncol=len(targets), frameon=False)
    fig.suptitle("How long does inference take?", x=0.10, ha="left", fontsize=19, weight="bold")
    fig.text(0.10, 0.90, context+" · median across seeds; each panel has its own x scale", fontsize=11)
    fig.text(0.10, 0.015, footer, fontsize=10)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.98, wspace=0.5)
    save(fig, "03-latency")

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    for ax, field, divisor, title, unit in [
        (axes[0], "preprocess_us", 1000, "One-time weight preparation", "ms"),
        (axes[1], "weight_bytes", 1024, "Stored weights and metadata", "KiB")]:
        values = matrix(field)/divisor
        positive = values[np.isfinite(values) & (values>0)]
        floor = float(positive.min()) / 2 if positive.size else 1.0
        transformed = np.ma.masked_invalid(np.log10(np.maximum(values, floor)))
        ax.imshow(transformed, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(len(names)), names)
        ax.set_title(title+" ("+unit+")", loc="left", pad=18)
        ax.tick_params(length=0, pad=8)
        for i in range(len(names)):
            for j in range(len(labels)):
                value = values[i,j]
                if np.isfinite(value):
                    v = math.log10(max(value, floor))
                    midpoint = (float(transformed.min())+float(transformed.max()))/2
                    text = f"{value:.3g}"
                    ax.text(j,i,text,ha="center",va="center",
                            color="white" if v>midpoint else "#17212b", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("What does preprocessing cost?", x=0.09, ha="left", fontsize=19, weight="bold")
    fig.text(0.09, 0.91, context+" · median across seeds · lower is better in both panels", fontsize=11)
    fig.text(0.09, 0.015, "Storage excludes inputs, outputs, scratch and allocator overhead. Preparation is a single measurement per run.\n"+footer, fontsize=10)
    fig.subplots_adjust(top=0.83, bottom=0.23, left=0.09, right=0.99, wspace=0.30)
    save(fig, "04-preprocessing-storage")
    pdf.close()
    summary = []
    times = matrix("median_us")
    for j, ((m,k,n), target) in enumerate(columns):
        if not np.isfinite(times[:,j]).any():
            continue
        best = int(np.nanargmin(times[:,j]))
        summary.append(f"{m}x{k}x{n}, {target:.0%} zeros: {names[best]} "
                       f"({times[best,j]:.3g} us; {speedups[best,j]:.2f}x BLAS)")
    (out/"summary.txt").write_text(context+"\n"+"\n".join(summary)+"\n")
    print("\n".join(summary))
    print(f"Plots: {out}")


if __name__ == "__main__":
    main()
