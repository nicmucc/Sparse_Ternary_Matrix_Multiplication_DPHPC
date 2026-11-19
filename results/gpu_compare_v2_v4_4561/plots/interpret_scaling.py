#!/usr/bin/env python3
"""Validate job 4561 and explain v2/v4 batch scaling; retain previous v3 as context."""
import csv
import hashlib
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PREVIOUS = RUN.parent / "gpu_compare_v2_v3_4548"
sys.path.insert(0, str(RUN / "source" / "bench"))
from plot_gpu_compare import load_results

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    env, versions, workloads, stats, raw = load_results(RUN)
    assert versions == ("v2", "v4") and env["samples"] == 30
    assert env["suite"] == "smoke" and len(workloads) == 6 and len(raw) == 1092
    for relative, digest in env["sources"].items():
        assert hashlib.sha256((RUN / "source" / relative).read_bytes()).hexdigest() == digest, relative
    checks = json.loads((RUN / "correctness.json").read_text())
    assert len(checks) == 4 and all(c["status"] == "PASS" for c in checks)
    for version in versions:
        assert "195 cases passed, 0 failed" in (RUN / f"correctness_{version}_test_correctness.log").read_text()
        assert "108 cases" in (RUN / f"correctness_{version}_test_gpu_correctness.log").read_text()
    sanitizers = sorted(RUN.glob("sanitize_*.log"))
    assert len(sanitizers) == 12
    for p in sanitizers:
        expected = "0 hazards displayed (0 errors, 0 warnings)" if "racecheck" in p.name else "ERROR SUMMARY: 0 errors"
        assert expected in p.read_text(), p
    logs = list(RUN.glob("v[24]_M*.log"))
    assert len(logs) == 12 and all("PASS, final max error" in p.read_text() for p in logs)
    old_env, old_versions, old_workloads, old_stats, _ = load_results(PREVIOUS)
    assert old_versions == ("v2", "v3") and workloads == old_workloads
    for relative in ("gpu/spr/spr_v2.cu", "bench/bench_gpu_compare.cu"):
        assert env["sources"][relative] == old_env["sources"][relative]
    v4_code = (RUN / "source/gpu/spr/spr_v4.cu").read_text()
    assert v4_code.count("__syncthreads()") == 1 and "dim3 threads(32, 8)" in v4_code
    assert "partial[" not in v4_code
    Ms, ks = [1, 33, 257], [4, 8]
    metrics = ("gpu_event", "run_wall", "end_to_end_wall")
    def stat(data, version, m, k, metric):
        return np.array(data[(version, m, 257, 257, k, .5, metric)]) * 1000
    rows = []
    for m in Ms:
        for k in ks:
            r = dict(M=m, k=k)
            for metric in metrics:
                v2 = stat(stats, "v2", m, k, metric)[1]
                v4 = stat(stats, "v4", m, k, metric)[1]
                r[f"v2_{metric}_us"] = v2
                r[f"v4_{metric}_us"] = v4
                r[f"speedup_{metric}"] = v2 / v4
                r[f"v4_{metric}_time_change_pct"] = 100 * (v4 / v2 - 1)
            r["previous_v3_gpu_us"] = stat(old_stats, "v3", m, k, "gpu_event")[1]
            r["v4_vs_previous_v3_gpu_time_change_pct"] = 100 * (r["v4_gpu_event_us"] / r["previous_v3_gpu_us"] - 1)
            r["v2_gpu_cross_job_change_pct"] = 100 * (r["v2_gpu_event_us"] / stat(old_stats, "v2", m, k, "gpu_event")[1] - 1)
            r["v2_e2e_cross_job_change_pct"] = 100 * (r["v2_end_to_end_wall_us"] / stat(old_stats, "v2", m, k, "end_to_end_wall")[1] - 1)
            rows.append(r)
    # The reported fastest tested width must actually be supported by the data.
    for version in versions:
        for m in Ms:
            for metric in metrics:
                assert min(ks, key=lambda k: stat(stats, version, m, k, metric)[1]) == 4
    with (HERE / "comparison_us.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    positions = np.arange(len(Ms))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for ax, metric, title in zip(axes, ("gpu_event", "end_to_end_wall"), ("GPU kernel", "End-to-end: bind + run + finish")):
        for k, color in zip(ks, ("#2563eb", "#d75a18")):
            values = [next(r[f"speedup_{metric}"] for r in rows if r["M"] == m and r["k"] == k) for m in Ms]
            ax.plot(positions, values, "o-", color=color, linewidth=2, label=f"k={k}")
            for x, y in zip(positions, values):
                ax.annotate(f"{y:.2f}×", (x, y), xytext=(0, 9 if k == 8 else -17),
                            textcoords="offset points", ha="center", color=color)
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_ylim(.75, 2.45 if metric == "gpu_event" else 1.85)
        ax.set_xticks(positions, [str(m) for m in Ms])
        ax.set_xlabel("Input rows M (equally spaced)")
        ax.set_ylabel("Speedup v2 / v4; above 1 favors v4")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=.2)
        ax.legend(loc="upper right")
    fig.suptitle("V4's kernel advantage largely disappears at M=257", fontsize=16)
    fig.supxlabel("RTX 5060 Ti · K=N=257 · 50% zeros · ratios of medians from 30 samples; lines connect measured categories", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"01_speedup_vs_M.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained", sharey=True)
    for ax, k in zip(axes, ks):
        for version, color, marker in (("v2", "#2563eb", "o"), ("v4", "#d75a18", "s")):
            values = np.array([stat(stats, version, m, k, "gpu_event") / m for m in Ms])
            ax.plot(positions, values[:, 1], marker + "-", color=color, linewidth=2, label=version)
            ax.fill_between(positions, values[:, 0], values[:, 2], color=color, alpha=.15)
            for x, y in zip(positions, values[:, 1]):
                ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8 if version == "v2" else -16),
                            textcoords="offset points", ha="center", color=color, fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(.12, 110)
        ax.set_xticks(positions, [str(m) for m in Ms])
        ax.set_xlabel("Input rows M (equally spaced)")
        ax.set_title(f"k={k}")
        ax.grid(True, axis="y", which="both", alpha=.2)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Amortized GPU time per row (µs, log scale)")
    fig.suptitle("Both versions gain throughput with batching; v2 catches up", fontsize=15)
    fig.supxlabel("Kernel batch time divided by M; this is not isolated single-row request latency\nMarkers: medians; bands: sample IQR divided by M, not confidence intervals", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"02_amortized_gpu_time.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    report = ["# V2 versus v4: interpretation of job 4561", "",
        "The kernel speedup does diminish at the largest tested M, although it is not monotonically decreasing: M=33 is similar to or better than M=1. These are smoke results on an RTX 5060 Ti: K=N=257, 50% zeros, seed 42, M={1,33,257}, k={4,8}, five warm-ups and 30 samples per inference metric.", "",
        "## Validation", "",
        "Both versions passed 195 common correctness cases and 108 edge cases each. All 12 sanitizer invocations reported zero errors, and racecheck reported zero warnings; all 12 benchmark executions passed their reference checks. Source-snapshot hashes were verified, and all medians/quartiles were recomputed from 1092 raw records. M=257 benchmark cases use sampled FP64 reference checks plus a full finite-output scan; the dedicated correctness suites check every output on their test cases.", "",
        "## Same-job results", "",
        "Speedup = median(v2)/median(v4), above 1 favors v4.", "",
        "| M | k | GPU v2 (µs) | GPU v4 (µs) | GPU speedup | End-to-end speedup |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['M']} | {r['k']} | {r['v2_gpu_event_us']:.3f} | {r['v4_gpu_event_us']:.3f} | {r['speedup_gpu_event']:.3f}× | {r['speedup_end_to_end_wall']:.3f}× |")
    report += ["", "![Speedup versus M](01_speedup_vs_M.png)", "",
        "At M=1 and M=33, v4's kernel is 1.72–2.22× faster than v2. At M=257,k=4, v4 takes 53.376 µs versus 50.976 µs: 4.7% longer. At k=8 it takes 121.152 versus 125.424 µs: 3.4% less time. Thus the larger-batch outcome is close to parity, with a small loss at k=4 and a small gain at k=8. Synchronized run() shows the same direction in both cases.", "",
        "k=4 remains the fastest tested k for both versions at every M, for both GPU and end-to-end timings. A larger matched-k speedup at k=8 does not imply that k=8 has the lowest latency.", "",
        "## Why increasing M changes the tradeoff", "",
        "V2 gives one thread to each (input row, SPR block) task, with 128 rows per CUDA block. V4 gives eight helpers to each task and tiles 32 input rows per CUDA block. Helpers compute independent segment sums, synchronize once, then helper 0 evaluates the whole tree.", "",
        "At small M, v2 has few active input-row lanes. V4 creates additional parallel work across segments, which can expose more concurrent work and memory accesses. At larger M, v2 can already use independent rows to populate warps and hide latency. V4 still reserves eight helpers per row, but seven do no tree evaluation after the segment phase. The resource and scheduling tradeoff is therefore less favorable. This is an explanation consistent with the mapping and measured trend, not a measured attribution to occupancy, bandwidth or a specific stall source.", "",
        "For example, at M=1, v2's (128,1) block assigns one thread to a valid row, while v4's (32,8) block assigns eight. At M=257, v2 launches three 128-thread blocks per SPR block, while v4 launches nine 256-thread blocks. Both ultimately evaluate 257 row trees per SPR block, but v4 has many more helper threads associated with that work.", "",
        "![Amortized GPU time](02_amortized_gpu_time.png)", "",
        "Both kernels improve in amortized time per row as M grows. For k=4, v2 goes from 26.864 µs/row at M=1 to approximately 0.198 at M=257; v4 goes from 15.616 to approximately 0.208. V4 is not becoming inefficient simply because it processes more rows; v2 catches up by using its row-level parallelism. Batch time divided by M describes throughput, not individual request latency.", "",
        "## End-to-end latency is a separate result", "",
        "End-to-end medians favor v4 in all six cases, by 1.10–1.62×. This trend is not monotonic in M: for k=4 the ratios are 1.16×, 1.10× and 1.15×; for k=8 they are 1.47×, 1.62× and 1.14×.", "",
        "At M=257,k=4, v4's end-to-end time is 457.880 µs versus v2's 527.875 µs, despite v4's slower kernel. The total-latency advantage comes from outside the kernel timing boundary; allocations, transposition and transfers are included but not measured individually. Both implementations have the same general bind/finish behavior, so the difference should not be described as a demonstrated transpose optimization.", "",
        "The unchanged v2 end-to-end baseline also shifts between jobs: relative to job 4548 it is 5.7% slower at M=257,k=4 and 11.4% slower at M=257,k=8. These variations reinforce the need to distinguish measured total latency from an isolated algorithm improvement.", "",
        "## Context from the preceding v3 experiment", "",
        "The preceding v3 job 4548 used the same input shapes, seed, v2 source and benchmark harness, but only five samples and a separate allocation. Cross-job v3/v4 comparisons are descriptive, not a controlled isolation of the code change.", "",
        "| M | k | Prior v3 GPU (µs) | Current v4 GPU (µs) | v4 time change |",
        "|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['M']} | {r['k']} | {r['previous_v3_gpu_us']:.3f} | {r['v4_gpu_event_us']:.3f} | {r['v4_vs_previous_v3_gpu_time_change_pct']:+.1f}% |")
    report += ["",
        "The strongest difference is M=33,k=8: 53.408 µs for prior v3 versus 37.248 µs for v4, a 30.3% reduction in GPU time. At M=257,k=4 the v2/v3 kernel gap had been 18.5%; it is now 4.7% for v2/v4. This makes segment-only cooperation a useful next candidate, while M=1,k=8 shows a small cross-job regression versus v3 and prevents claiming universal dominance.", "",
        "## Practical conclusion", "",
        "The evidence supports experimenting with a workload-dependent helper count rather than increasing helpers for every M. Keep the current v4 as a small-batch candidate; for larger M compare helper counts 1,2,4,8 with otherwise identical metadata and mapping. Then test the full K,N,k,sparsity sweep before selecting a dispatch rule. Larger K can increase segment-sum work, larger N supplies more independent SPR blocks, and larger k changes the tree size. A crossover cannot be inferred from M alone or located precisely between just 33 and 257.", "",
        "The 30 samples provide more within-job observations than previous smoke runs, but this remains one matrix shape, one sparsity, one seed and one job. Sample IQRs are not cross-job confidence intervals, and no hardware-counter profile was collected.", ""]
    (HERE / "interpretation.md").write_text("\n".join(report))
    document = ["<!doctype html><meta charset='utf-8'><title>V2 versus v4 batch scaling</title>",
        "<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#172033}img{width:100%}</style>",
        "<h1>V2 versus v4: batch scaling</h1>",
        "<p>Job 4561 · RTX 5060 Ti · K=N=257 · 50% zeros · 30 samples. V4's kernel advantage largely disappears at M=257; end-to-end latency includes additional effects.</p>",
        "<p><a href='interpretation.md'>Full interpretation</a> · <a href='comparison_us.csv'>Numerical comparison</a></p>"]
    for stem, title in (("01_speedup_vs_M", "Kernel and end-to-end speedup versus M"),
                        ("02_amortized_gpu_time", "Amortized GPU time per row"),
                        ("K257_N257_z0.5", "Absolute latency and sample IQR")):
        document += [f"<h2>{html.escape(title)}</h2><img src='{stem}.png' alt='{html.escape(title)}'>",
                     f"<p><a href='{stem}.pdf'>PDF</a></p>"]
    (HERE / "report.html").write_text("\n".join(document))
    inputs = [RUN / n for n in ("environment.json", "raw_samples.csv", "timing_summary.csv", "correctness.json", "failures.json")]
    inputs += sanitizers + sorted(RUN.glob("correctness_*.log"))
    (HERE / "interpretation_validation.json").write_text(json.dumps({
        "job": 4561, "samples": 30, "raw_records": len(raw), "workloads_per_version": 6,
        "correctness_cases_per_version": 303, "zero_error_sanitizer_checks": 12,
        "source_hashes_verified": len(env["sources"]),
        "input_hashes": {str(p.relative_to(RUN)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    }, indent=2))
    print("Validated source hashes, raw timing statistics, correctness/sanitizer logs and prior v3 context.")
    print(HERE / "interpretation.md")


if __name__ == "__main__":
    main()
