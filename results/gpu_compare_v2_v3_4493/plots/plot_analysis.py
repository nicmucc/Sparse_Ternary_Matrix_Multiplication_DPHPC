#!/usr/bin/env python3
"""Reproduce focused plots and interpretation for the recorded job 4493 smoke run."""
import csv
import hashlib
import html
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
sys.path.insert(0, str(RUN / "source" / "bench"))
from plot_gpu_compare import load_results

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np


def main():
    env, versions, workloads, stats, raw = load_results(RUN)
    assert versions == ("v2", "v3") and env["suite"] == "smoke"
    assert env["samples"] == 5 and len(workloads) == 6 and len(raw) == 192
    assert "RTX 5060 Ti" in env["gpu"]
    for relative, expected in env["sources"].items():
        assert hashlib.sha256((RUN / "source" / relative).read_bytes()).hexdigest() == expected, relative
    correctness = json.loads((RUN / "correctness.json").read_text())
    assert len(correctness) == 4 and all(c["status"] == "PASS" for c in correctness)
    for version in versions:
        assert "195 cases passed, 0 failed" in (RUN / f"correctness_{version}_test_correctness.log").read_text()
        assert "108 cases" in (RUN / f"correctness_{version}_test_gpu_correctness.log").read_text()
    sanitizer_logs = sorted(RUN.glob("sanitize_*.log"))
    assert len(sanitizer_logs) == 12
    for path in sanitizer_logs:
        text = path.read_text()
        expected = "0 hazards displayed (0 errors, 0 warnings)" if "racecheck" in path.name else "ERROR SUMMARY: 0 errors"
        assert expected in text, path
    log = (RUN.parent / "spr-gpu-4493.out").read_text()
    assert len(re.findall(r"^\[\d+/6\] v[23]_.*: PASS$", log, re.MULTILINE)) == 12
    assert "Completed:" in log and "failed version/configurations: 0" in log
    Ms, ks = [1, 33, 257], [4, 8]
    def stat(version, m, k, metric):
        return np.array(stats[(version, m, 257, 257, k, .5, metric)]) * 1000
    def ratio(m, k, metric):
        return stat("v2", m, k, metric)[1] / stat("v3", m, k, metric)[1]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for ax, metric, title in zip(axes, ("gpu_event", "end_to_end_wall"), ("GPU kernel", "End-to-end: bind + run + finish")):
        values = np.array([[ratio(m, k, metric) for m in Ms] for k in ks])
        image = ax.imshow(np.log2(values), cmap="RdBu", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
        for y in range(len(ks)):
            for x in range(len(Ms)):
                ax.text(x, y, f"{values[y, x]:.2f}×", ha="center", va="center", fontsize=17,
                        color="white" if abs(math.log2(values[y, x])) > .6 else "#152238", fontweight="bold")
        ax.set_xticks(range(3), [str(m) for m in Ms])
        ax.set_yticks(range(2), [f"k = {k}" for k in ks])
        ax.set_xlabel("Input rows M")
        ax.set_title(title, fontsize=13, pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(image, ax=axes, shrink=.85, pad=.03, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["0.5×", "1×", "2×"])
    cb.set_label("v2 / v3: blue favors v3; red favors v2")
    fig.suptitle("V3 helps small batches; the benefit fades at M = 257", fontsize=16)
    fig.supxlabel("RTX 5060 Ti · K = N = 257 · 50% zeros · ratio of medians from 5 samples", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"01_speedups.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    colors = {"v2": "#2563eb", "v3": "#d75a18"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7), layout="constrained", sharex=True, sharey="row")
    positions = np.arange(3)
    for row, metric in enumerate(("gpu_event", "end_to_end_wall")):
        for col, k in enumerate(ks):
            ax = axes[row, col]
            for vi, version in enumerate(versions):
                values = np.array([stat(version, m, k, metric) for m in Ms])
                x = positions + (-.055 if vi == 0 else .055)
                ax.errorbar(x, values[:, 1], yerr=[values[:, 1] - values[:, 0], values[:, 2] - values[:, 1]],
                            marker="o" if vi == 0 else "s", capsize=5, linewidth=1.8,
                            color=colors[version], label=version)
                for i, m in enumerate(Ms):
                    samples = [float(r["ms"]) * 1000 for r in raw if r["version"] == version and int(r["M"]) == m
                               and int(r["k"]) == k and r["metric"] == metric]
                    ax.scatter(x[i] + np.linspace(-.03, .03, len(samples)), samples,
                               color=colors[version], alpha=.35, s=18, zorder=3)
            ax.set_yscale("log", base=2)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True, which="both", axis="y", alpha=.2)
            ax.set_xticks(positions, [str(m) for m in Ms])
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_title(("GPU kernel" if row == 0 else "End-to-end") + f" · k = {k}")
            ax.legend(loc="upper left")
            ax.set_xlabel("Input rows M (equally spaced)")
            if col == 0:
                ax.set_ylabel("Latency (µs, log scale)")
    for row, metric in enumerate(("gpu_event", "end_to_end_wall")):
        values = [float(r["ms"]) * 1000 for r in raw if r["metric"] == metric]
        axes[row, 0].set_ylim(min(values) * .8, max(values) * 1.25)
    fig.suptitle("Absolute latency: k = 4 is fastest for both versions in this run", fontsize=16)
    fig.supxlabel("RTX 5060 Ti · K = N = 257 · 50% zeros · markers: medians; whiskers: IQR; faint dots: all 5 samples", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"02_latency_variability.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Compare the fastest tested width, rather than selecting by largest speedup.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    best_rows = []
    for ax, metric, title in zip(axes, ("gpu_event", "end_to_end_wall"),
                                ("GPU kernel", "End-to-end: bind + run + finish")):
        best = {}
        for version in versions:
            best[version] = [min((stat(version, m, k, metric)[1], k) for k in ks) for m in Ms]
        for vi, version in enumerate(versions):
            values = np.array([stat(version, m, selected_k, metric)
                               for m, (_, selected_k) in zip(Ms, best[version])])
            bars = ax.bar(positions + (vi - .5) * .36, values[:, 1], width=.34,
                          color=colors[version], label=version,
                          yerr=[values[:, 1] - values[:, 0], values[:, 2] - values[:, 1]],
                          capsize=4, zorder=3)
            for bar, m, (median, selected_k), q3 in zip(bars, Ms, best[version], values[:, 2]):
                ax.text(bar.get_x() + bar.get_width()/2, q3 + max(values[:, 1])*.025,
                        f"{median:.1f}", ha="center", fontsize=9)
                best_rows.append([version, m, metric, selected_k, median])
        ax.set_ylim(0, max(value[0] for version in versions for value in best[version]) * 1.27)
        for i, m in enumerate(Ms):
            speedup = best["v2"][i][0] / best["v3"][i][0]
            ax.text(i, .97, f"{speedup:.2f}×", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontweight="bold",
                    color="#15803d" if speedup > 1.01 else "#b91c1c" if speedup < .99 else "#444444")
        ax.set_xticks(positions, [str(m) for m in Ms])
        ax.set_xlabel("Input rows M")
        ax.set_ylabel("Latency (µs)")
        ax.set_title(title, pad=12)
        ax.grid(True, axis="y", alpha=.2, zorder=0)
        ax.legend(loc="upper left", bbox_to_anchor=(.01, .88), frameon=False)
    fig.suptitle("Fastest tested width: k = 4 for both versions at every M", fontsize=15)
    fig.supxlabel("RTX 5060 Ti · K = N = 257 · 50% zeros · top labels: v2/v3 speedup · bars: medians · whiskers: IQR\n"
                  "Selected from k={4,8} using the same 5 samples; selection has no held-out evaluation", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"03_best_tested_k.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (HERE / "best_tested_k_us.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "M", "metric", "best_tested_k", "median_us"])
        writer.writerows(best_rows)

    rows = []
    for m in Ms:
        for k in ks:
            entry = dict(M=m, k=k)
            for metric in ("gpu_event", "run_wall", "end_to_end_wall"):
                for version in versions:
                    entry[f"{version}_{metric}_us"] = float(stat(version, m, k, metric)[1])
                entry[f"speedup_{metric}"] = float(ratio(m, k, metric))
            rows.append(entry)
    with (HERE / "comparison_us.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = ["# V3 versus v2: interpretation of job 4493", "",
        "This is the **smoke run**, not the full campaign: RTX 5060 Ti, CUDA compiler 12.9, sm_120, K=N=257, 50% zero weights, seed 42, M={1,33,257}, k={4,8}, five warm-ups and five samples per inference metric.", "",
        "V2 launches 128 input rows per CUDA block with one thread per SPR task. V3 launches 32 input rows by eight helpers (256 threads per block), with independent segment sums and cooperative output reductions. This is one fixed configuration, not a helper-count tuning sweep.", "",
        "## Validation", "",
        "Both versions passed 195 common correctness cases and 108 GPU edge cases (303 per version). Edge cases include two input bindings and three runs per binding. All 12 sanitizer invocations reported zero errors; racecheck also reported zero warnings. All 12 benchmark executions passed reference validation. The largest smoke workloads use sampled reference checks; all outputs are checked for finiteness. These results support correctness on the tested cases, not a proof for every shape.", "",
        "Recomputed medians and quartiles match the recorded timing summary. All source-snapshot hashes match environment.json. There are 192 raw records: 12 preparation observations and 180 inference samples.", "",
        "## Measured results", "",
        "Speedup = median(v2)/median(v3); values above 1 favor v3.", "",
        "| M | k | GPU v2 (µs) | GPU v3 (µs) | GPU speedup | End-to-end v2 (µs) | End-to-end v3 (µs) | End-to-end speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['M']} | {r['k']} | {r['v2_gpu_event_us']:.2f} | {r['v3_gpu_event_us']:.2f} | {r['speedup_gpu_event']:.3f}× | {r['v2_end_to_end_wall_us']:.2f} | {r['v3_end_to_end_wall_us']:.2f} | {r['speedup_end_to_end_wall']:.3f}× |")
    report += ["", "![Matched-width speedups](01_speedups.png)", "",
        "## Interpretation", "",
        "1. **Small-batch kernel improvements are clear within this run.** At M=1 and M=33, GPU-event speedups range from 1.50× to 1.92×. The kernel IQRs do not overlap in these four cases. The plausible explanation is more parallelism within a task when input-row parallelism is scarce; the data do not isolate reduction speed from changed launch geometry, metadata access or other effects.", "",
        "2. **The improvement does not extend to every larger-batch case.** At M=257,k=4, GPU latency rises from 49.664 to 61.760 µs: v3 takes 24.4% longer. IQRs are separated. At k=8, 124.800 versus 127.072 µs is close to parity and the IQRs overlap. Synchronized run() slightly favors v3 there, so the roughly 2% GPU-event difference is not a robust basis for choosing a version.", "",
        "3. **End-to-end time matters.** At M=1,k=4, a 1.57× GPU gain becomes 1.03× end-to-end: the median improves by only 1.612 µs. At M=33, end-to-end gains are 1.22× (k=4) and 1.35× (k=8). At M=257,k=4, total time is effectively tied. At M=257,k=8, v3 takes 12.9% longer end-to-end despite near-equal kernel time. Allocation, transposition and transfer costs are included, but not separately measured; this regression cannot be attributed specifically to the reduction or transpose.", "",
        "4. **Compare absolute latency as well as matched-k speedup.** k=4 has the lowest measured kernel and end-to-end median for both versions at every M here. The largest matched-k speedup (1.92× at M=1,k=8) is not the fastest configuration: v3 k=8 takes 35.808 µs, versus 17.312 µs for v3 k=4 and 27.264 µs for v2 k=4. Comparing the fastest tested k for each version yields GPU speedups of 1.57×, 1.53× and 0.80× at M=1,33,257; end-to-end speedups are 1.03×,1.22× and 1.00×.", "",
        "5. **Likely scaling tradeoff, not a proven bottleneck.** V3 trades row parallelism for helper parallelism, adds shared-memory traffic and barriers at every tree level, and leaves merges sequential. Once v2 has enough independent input rows, these costs can outweigh the cooperation benefit. Kernel timings alone do not prove occupancy, memory-bandwidth or barrier-stall causes; profiling is needed to separate them.", "",
        "![Absolute latency and variability](02_latency_variability.png)", "",
        "![Fastest tested width comparison](03_best_tested_k.png)", "",
        "## Next experiment", "",
        "Run `sbatch --cpus-per-task=3 bench/gpu_compare.sbatch full` in the existing cluster project. The full campaign adds K=N=1024/4096, M=8/31/32/128/512, k=16 and 0%/90% zeros, with 30 samples. Keep the current v3 launch fixed first. Then compare 2/4/8/16 helpers on promising workloads. Do not infer an optimal helper count or a dispatch threshold from this one fixed eight-helper smoke run.", "",
        "Five samples, one seed and one job give limited evidence for stability, especially for small differences. Inputs repeat, caches are not flushed, and clocks are not locked. IQRs show within-run sample spread, not confidence intervals across jobs.", ""]
    (HERE / "interpretation.md").write_text("\n".join(report))
    sections = ["<!doctype html><meta charset='utf-8'><title>V3 versus v2: job 4493</title>",
        "<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#172033}img{width:100%}table{border-collapse:collapse}td,th{padding:6px 10px;border:1px solid #ddd}</style>",
        "<h1>V3 versus v2: RTX 5060 Ti smoke run</h1>",
        "<p>K=N=257 · 50% zeros · five samples · all recorded correctness/sanitizer checks passed.</p>",
        "<p>V3 accelerates small-batch GPU kernels, but the larger-batch and end-to-end results are mixed. See <a href='interpretation.md'>full interpretation</a> and <a href='comparison_us.csv'>numerical comparison</a>.</p>"]
    for stem, caption in (("01_speedups", "Matched-k speedup: above 1 favors v3"),
                          ("02_latency_variability", "Absolute latency and sample variation"),
                          ("03_best_tested_k", "Fastest tested k for each version")):
        sections += [f"<h2>{caption}</h2><img src='{stem}.png' alt='{html.escape(caption)}'>",
                     f"<p><a href='{stem}.pdf'>PDF</a> · <a href='{stem}.png'>PNG</a></p>"]
    (HERE / "report.html").write_text("\n".join(sections))
    inputs = [RUN / name for name in ("raw_samples.csv", "timing_summary.csv", "environment.json",
                                      "correctness.json", "failures.json")]
    inputs += sanitizer_logs + sorted(RUN.glob("correctness_*.log"))
    (HERE / "analysis_validation.json").write_text(json.dumps({
        "job_id": 4493, "versions": versions, "workloads_per_version": len(workloads),
        "samples_per_metric": env["samples"], "correctness_cases_per_version": 303,
        "zero_error_sanitizer_logs": len(sanitizer_logs), "raw_rows": len(raw),
        "source_hashes_verified": len(env["sources"]),
        "max_recorded_initial_error": max(float(r["max_abs_error"]) for r in raw),
        "input_hashes": {str(p.relative_to(RUN)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    }, indent=2))
    print("Validated: 303 correctness cases per version, 12 zero-error sanitizer logs, 12 PASS benchmark executions, 192 raw records, source snapshot hashes.")
    print(f"Wrote focused plots and interpretation to {HERE}")


if __name__ == "__main__":
    main()
