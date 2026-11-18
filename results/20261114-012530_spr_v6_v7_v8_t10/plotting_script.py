#!/usr/bin/env python3
"""Benchmark SPR V6/V7/V8 at 10 requested threads, without correctness checks.

Run with /usr/bin/python3 bench/compare_v6_v7_v8.py.
Replot existing measurements with --plots-only results/RUN.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THREADS = 10
KERNELS = [f"spr.v{v}.b32" for v in [6, 7, 8]]
MS = [1, 32, 64, 256, 512, 1024]
SHAPES = [(1024, 4096), (2048, 2048), (4096, 1024), (2048, 8192)]
SPARSITIES = [0.3, 0.5, 0.7, 0.9]
KS = [2, 3, 4]
CLUSTER_MS = [64, 256, 512, 1024]
CLUSTER_SHAPES = [(1024, 4096), (2048, 8192)]


def geomean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_command(command, log, env):
    with log.open("a") as f:
        f.write("\n$ " + " ".join(map(str, command)) + "\n")
        f.flush()
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)


def summarize(out):
    manifest = json.loads((out / "manifest.json").read_text())
    rows = read_csv(out / "raw.csv")
    expected = len(manifest["jobs"]) * len(SPARSITIES) * len(KS) * len(KERNELS)
    assert len(rows) == expected, (len(rows), expected)
    assert all(r["threads"] == str(THREADS) and r["ok"] == "1" and
               float(r["max_err"]) == -1 and int(r["rounds"]) == manifest["rounds"] and
               math.isfinite(float(r["median_ms"])) and float(r["median_ms"]) > 0 for r in rows)
    for r in rows:
        for field in ["M", "K", "N", "bytes"]:
            r[field] = int(r[field])
        for field in ["sparsity", "median_ms", "adds_per_row"]:
            r[field] = float(r[field])
        r["pattern"] = r["tag"].removeprefix("v678_")

    def workload(r):
        return r["pattern"], r["M"], r["K"], r["N"], r["sparsity"]

    lookup = {(workload(r), r["param"], r["kernel"]): r for r in rows}
    assert len(lookup) == expected, "duplicate timing configurations"
    paired, tuned = [], []
    for w in sorted({workload(r) for r in rows}):
        for k in KS:
            a, b, c = [lookup[(w, f"k={k}", kernel)] for kernel in KERNELS]
            assert a["adds_per_row"] == b["adds_per_row"] == c["adds_per_row"]
            assert a["bytes"] == b["bytes"]
            # V8 adds one four-int Task descriptor per rectangle to the same format.
            assert c["bytes"] > a["bytes"] and (c["bytes"] - a["bytes"]) % 16 == 0
            paired.append(dict(pattern=w[0], M=w[1], K=w[2], N=w[3], sparsity=w[4], k=k,
                               v6_ms=a["median_ms"], v7_ms=b["median_ms"], v8_ms=c["median_ms"],
                               v8_vs_v6=a["median_ms"] / c["median_ms"],
                               v8_vs_v7=b["median_ms"] / c["median_ms"],
                               v7_vs_v6=a["median_ms"] / b["median_ms"],
                               v8_tasks=(c["bytes"] - a["bytes"]) // 16))
        best = [min((lookup[(w, f"k={k}", kernel)] for k in KS),
                    key=lambda r: r["median_ms"]) for kernel in KERNELS]
        tuned.append(dict(pattern=w[0], M=w[1], K=w[2], N=w[3], sparsity=w[4],
                          **{f"v{v}_k": r["param"] for v, r in zip([6, 7, 8], best)},
                          **{f"v{v}_ms": r["median_ms"] for v, r in zip([6, 7, 8], best)},
                          v8_vs_v6=best[0]["median_ms"] / best[2]["median_ms"],
                          v8_vs_v7=best[1]["median_ms"] / best[2]["median_ms"]))
    write_csv(out / "all_k_comparison.csv", paired)
    primary = [r for r in paired if r["k"] == 3]
    write_csv(out / "comparison.csv", primary)
    write_csv(out / "tuned_comparison.csv", tuned)
    regular = [r for r in primary if r["pattern"] == "iid" and r["N"] != 32]
    narrow = [r for r in primary if r["pattern"] == "iid" and r["N"] == 32]
    clustered = [r for r in primary if r["pattern"] == "column_clustered"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    indices = list(range(len(MS)))

    def mean(data, field):
        return geomean([r[field] for r in data])

    def finish(fig, axs, title, filename, legend_cols=3):
        h, labels = axs.flat[0].get_legend_handles_labels()
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0.10, 1, 0.94))
        if labels:
            fig.legend(h, labels, loc="lower center", ncol=legend_cols, frameon=False)
        fig.savefig(out / filename, dpi=170, bbox_inches="tight")
        plt.close(fig)

    def m_axis(ax, ratio=False):
        ax.set_xticks(indices, labels=MS)
        ax.set_xlabel("M (input samples)")
        ax.grid(alpha=0.25)
        if ratio:
            ax.axhline(1, color="black", linestyle="--", linewidth=0.9)
            ax.set_ylabel("V8 speedup (>1 is faster)")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False)
    for ax, baseline in zip(axs.flat, [6, 7]):
        for s in SPARSITIES:
            ax.plot(indices, [mean([r for r in regular if r["M"] == M and r["sparsity"] == s],
                                   f"v8_vs_v{baseline}") for M in MS], "o-", label=f"{s:.0%} sparse")
        ax.set_title(f"V8 / V{baseline}" + (" (V7 serial at M≤32)" if baseline == 7 else ""))
        m_axis(ax, ratio=True)
    finish(fig, axs, "Rectangle scheduling: V8 vs V6/V7 — 10 requested threads, B=32, k=3\n"
           "Geometric means over four regular K×N shapes", "speedup.png", 4)

    def runtime_figure(data, title, filename, microseconds=False):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for ax, s in zip(axs.flat, SPARSITIES):
            for v in [6, 7, 8]:
                times = [mean([r for r in data if r["M"] == M and r["sparsity"] == s], f"v{v}_ms") for M in MS]
                ax.plot(indices, [t * (1000 if microseconds else 1) for t in times], "o-", label=f"V{v}")
            ax.set(title=f"{s:.0%} sparse", yscale="log",
                   ylabel="inference time [µs]" if microseconds else "inference time [ms]")
            m_axis(ax)
        finish(fig, axs, title, filename)

    runtime_figure(regular, "Inference latency — 10 requested threads, B=32, k=3\n"
                   "Geometric means over four regular K×N shapes; lower is better", "runtime.png")
    runtime_figure(narrow, "Narrow output: K=1024, N=32 — 10 requested threads, k=3\n"
                   "Absolute latency; lower is better", "narrow_runtime.png", True)

    for M in MS:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for ax, (K, N) in zip(axs.flat, SHAPES):
            rs = [r for r in regular if (r["M"], r["K"], r["N"]) == (M, K, N)]
            for v in [6, 7, 8]:
                ax.plot([s * 100 for s in SPARSITIES],
                        [next(r[f"v{v}_ms"] for r in rs if r["sparsity"] == s) for s in SPARSITIES],
                        "o-", label=f"V{v}")
            ax.set(title=f"K={K}, N={N}", xlabel="weight sparsity [%]", ylabel="inference time [ms]")
            ax.grid(alpha=0.25)
        finish(fig, axs, f"M={M}: runtime by sparsity — 10 requested threads, B=32, k=3", f"runtime_M{M}.png")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (K, N) in zip(axs.flat, SHAPES):
        for s in SPARSITIES:
            ax.plot(indices, [next(r["v8_vs_v6"] for r in regular
                                   if (r["M"], r["K"], r["N"], r["sparsity"]) == (M, K, N, s)) for M in MS],
                    "o-", label=f"{s:.0%} sparse")
        ax.set_title(f"K={K}, N={N}")
        m_axis(ax, ratio=True)
    finish(fig, axs, "V8 vs V6 by matrix shape — 10 requested threads, B=32, k=3", "matrix_speedup.png", 4)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for row, (K, N) in zip(axs, CLUSTER_SHAPES):
        for ax, baseline in zip(row, [6, 7]):
            for pattern in ["iid", "column_clustered"]:
                data = [r for r in primary if (r["pattern"], r["K"], r["N"]) == (pattern, K, N)]
                ax.plot(range(len(CLUSTER_MS)),
                        [mean([r for r in data if r["M"] == M], f"v8_vs_v{baseline}") for M in CLUSTER_MS],
                        "o-", label="random zeros" if pattern == "iid" else "clustered zero columns")
            ax.set(title=f"K={K}, N={N}: V8 / V{baseline}", xlabel="M (input samples)",
                   ylabel="V8 speedup (>1 is faster)")
            ax.set_xticks(range(len(CLUSTER_MS)), labels=CLUSTER_MS)
            ax.axhline(1, color="black", linestyle="--", linewidth=0.9)
            ax.grid(alpha=0.25)
    finish(fig, axs, "Effect of sparse structure — 10 requested threads, k=3\n"
           "Geometric means over four sparsity levels", "pattern_speedup.png", 2)

    peaks = json.loads((out / "peaks.json").read_text())
    assert peaks["threads"] == THREADS
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    xs = [10 ** (-2 + i / 40) for i in range(201)]
    for ax, (K, N) in zip(axs.flat, SHAPES):
        for level, style in [("dram", "-"), ("l2", "--"), ("l1", ":")]:
            bw = peaks[f"bw_{level}_gbs"]
            ax.plot(xs, [min(peaks["peak_simd_gflops"], x * bw) for x in xs],
                    color="gray", linestyle=style, alpha=0.65, linewidth=1,
                    label=f"{level.upper()} roof ({bw:.0f} GB/s)")
        for v, marker in zip([6, 7, 8], ["o", "s", "^"]):
            rs = [lookup[(("iid", 512, K, N, s), "k=3", f"spr.v{v}.b32")] for s in SPARSITIES]
            work = [r["M"] * r["adds_per_row"] for r in rs]
            ai = [w / (r["bytes"] + 4 * r["M"] * (r["K"] + r["N"])) for r, w in zip(rs, work)]
            perf = [w / r["median_ms"] / 1e6 for r, w in zip(rs, work)]
            ax.scatter(ai, perf, s=50, marker=marker, label=f"V{v}", zorder=3)
        ax.set(title=f"M=512, K={K}, N={N}", xscale="log", yscale="log", xlim=(0.02, 1000),
               ylim=(0.5, peaks["peak_simd_gflops"] * 2), xlabel="additions / compulsory byte", ylabel="GFLOP/s (fp32 additions)")
        ax.grid(alpha=0.25)
    finish(fig, axs, "SPR roofline — fresh local M4 Pro measurements, 10 threads, B=32, k=3", "roofline.png", 3)

    summary = []
    for M in MS:
        rs = [r for r in regular if r["M"] == M]
        summary.append(dict(M=M, **{f"v{v}_ms": mean(rs, f"v{v}_ms") for v in [6, 7, 8]},
                            v8_vs_v6=mean(rs, "v8_vs_v6"), v8_vs_v7=mean(rs, "v8_vs_v7")))
    write_csv(out / "summary.csv", summary)
    lines = ["# SPR V6 / V7 / V8: 10-thread comparison", "", manifest["cpu"].replace("\n", "  \n"),
             "", "V6 uses static collapsed tile × group scheduling; V7 assigns whole M tiles; V8 dynamically schedules rectangles sized by estimated sparse-instruction work.",
             "All three kernels are B=32. The primary results hold k=3 fixed across all versions, matrices and sparsities.",
             "The secondary tuned comparison independently selects the fastest k in {2,3,4} for each version and workload; it does not isolate scheduling as strictly.",
             "", "## Measurement method", "",
             f"- M: {MS}; regular K×N: {SHAPES}; sparsities: 30%, 50%, 70%, 90%.",
             "- Additional K=1024, N=32 cases are reported separately from regular-matrix averages.",
             f"- Clustered diagnostic: M={CLUSTER_MS}, K×N={CLUSTER_SHAPES}; floor(s*N) leading columns are zero, remaining columns dense random ±1.",
             "- Identical seed-42 inputs and weights for each matching workload across versions and k values.",
             f"- {manifest['rounds']} interleaved timed rounds, rotating configuration order, after 5 warmups. Medians are reported.",
             "- Exactly 10 requested OpenMP threads; OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0, OMP_DYNAMIC=FALSE. macOS controls core placement.",
             "- V7 runs serially at M=1 and M=32 because there is only one sample tile. Other cases can also use fewer than 10 useful workers.",
             "- Preparation, input conversion and bind() are outside inference timing. V8 builds its reusable task plan in bind(). Thread-private scratch allocation remains inside inference for all versions.",
             "- Correctness checks were explicitly skipped: no correctness suite, benchmark --verify none, --ref-rows 0. max_err=-1 means unchecked; ok=1 is not new correctness evidence.",
             "- Fresh 10-thread machine ceilings are saved in peaks.json. No historical machine bandwidths are used.",
             "", "## Primary results: fixed k=3", "",
             "Geometric means over the four regular K×N shapes and four sparsity levels at each M. Speedup = baseline runtime / V8 runtime; >1 means V8 is faster.", "",
             "| M | V6 [ms] | V7 [ms] | V8 [ms] | V8 / V6 | V8 / V7 |", "|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['M']} | {r['v6_ms']:.4f} | {r['v7_ms']:.4f} | {r['v8_ms']:.4f} | {r['v8_vs_v6']:.3f}× | {r['v8_vs_v7']:.3f}× |")
    lines += ["", "## Dependence on sparsity", "", "| M | Sparsity | V8 / V6 | V8 / V7 |", "|---|---|---|---|"]
    for M in MS:
        for s in SPARSITIES:
            rs = [r for r in regular if r["M"] == M and r["sparsity"] == s]
            lines.append(f"| {M} | {s:.0%} | {mean(rs, 'v8_vs_v6'):.3f}× | {mean(rs, 'v8_vs_v7'):.3f}× |")
    lines += ["", "## Pattern and tuning diagnostics", "", "| Dataset | Points | V8 / V6 | V8 / V7 |", "|---|---|---|---|"]
    tuned_regular = [r for r in tuned if r["pattern"] == "iid" and r["N"] != 32]
    for name, data in [("Random zeros, regular shapes, fixed k=3", regular),
                       ("Clustered zero columns, fixed k=3", clustered),
                       ("Narrow N=32, fixed k=3", narrow),
                       ("Random zeros, regular shapes, independently tuned k", tuned_regular)]:
        lines.append(f"| {name} | {len(data)} | {mean(data, 'v8_vs_v6'):.3f}× | {mean(data, 'v8_vs_v7'):.3f}× |")
    lines += ["", "Clustered and narrow results are deliberately excluded from the main regular-matrix averages.",
              "The clustered pattern is synthetic; it demonstrates sensitivity to unequal group work, rather than predicting all structured-sparsity workloads.",
              "Task counts in comparison.csv are inferred from V8's extra metadata bytes: four 32-bit ints per Task. The underlying instruction streams perform the same addition count at matching k.",
              "Near-unity differences can reflect measurement noise. This is one timing campaign with one deterministic weight sample per workload; no confidence intervals are claimed.",
              "Roofline intensity uses compulsory instruction-format + X + Y bytes, including V8 task descriptors. It is not measured cache traffic or a direct measure of cache reuse.",
              "", "## Plots and data", "",
              "- [V8 speedup vs V6 and V7](speedup.png)", "- [Absolute runtime](runtime.png)",
              "- [V8 vs V6 by matrix shape](matrix_speedup.png)", "- [Random vs clustered zeros](pattern_speedup.png)",
              "- [Narrow-output absolute runtime](narrow_runtime.png)", "- [Roofline](roofline.png)",
              "- Per-size runtime curves: " + ", ".join(f"[M={M}](runtime_M{M}.png)" for M in MS),
              "- comparison.csv: fixed k=3; all_k_comparison.csv: all matching k; tuned_comparison.csv: independently fastest k; raw.csv: all timing configurations.",
              "", "![V8 speedup](speedup.png)", "", "![Runtime](runtime.png)"]
    if (out / "interpretation.md").exists():
        lines.insert(2, "[Interpretation and matched-pattern comparisons](interpretation.md)\n")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    validation = dict(timing_configurations=len(rows), fixed_k_workloads=len(primary),
                      correctness_checks_run=False, unchecked_configurations=len(rows), threads=THREADS,
                      source_matches_current={v: (out / f"source/cpu/spr/spr_v{v}.cpp").read_bytes() ==
                                              (ROOT / f"cpu/spr/spr_v{v}.cpp").read_bytes() for v in [6, 7, 8]})
    (out / "timing_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print("\n".join(lines[lines.index("## Primary results: fixed k=3"):lines.index("## Dependence on sparsity")]), flush=True)
    print(f"Results: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--plots-only", type=Path)
    args = parser.parse_args()
    if args.plots_only:
        summarize(args.plots_only.resolve())
        return
    assert args.rounds > 0
    stamp = datetime.datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / f"{stamp}_spr_v6_v7_v8_t10"
    out.mkdir(parents=True, exist_ok=False)
    jobs = [dict(pattern="iid", M=M, K=K, N=N) for M in MS for K, N in SHAPES + [(1024, 32)]]
    jobs += [dict(pattern="column_clustered", M=M, K=K, N=N)
             for M in CLUSTER_MS for K, N in CLUSTER_SHAPES]
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0", OMP_DYNAMIC="FALSE")
    removed_overrides = {k: env.pop(k) for k in ["SPR_L1_BYTES", "SPR_L2_BYTES", "SPR_LLC_BYTES"] if k in env}
    cpu = subprocess.check_output(["sysctl", "machdep.cpu.brand_string", "hw.physicalcpu",
                                   "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu",
                                   "hw.cachelinesize", "hw.perflevel0.l1dcachesize", "hw.perflevel0.l2cachesize",
                                   "hw.perflevel0.cpusperl2", "hw.perflevel1.l1dcachesize", "hw.perflevel1.l2cachesize"], text=True)
    assert "Apple M4 Pro" in cpu
    manifest = dict(cpu=cpu, threads=THREADS, kernels=KERNELS, jobs=jobs, ks=KS, primary_k=3,
                    sparsities=SPARSITIES, rounds=args.rounds, warmup=5, prep_rounds=1, seed=42,
                    verification="none", ref_rows=0, correctness_checks_run=False,
                    removed_cache_overrides=removed_overrides, command=sys.argv,
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sources = [*(ROOT / "cpu/spr").glob("spr_v[678].cpp"), *(ROOT / "include/spr").glob("*.hpp"),
               ROOT / "cpu/core.cpp", ROOT / "bench/bench_cpu.cpp", ROOT / "bench/peaks.cpp",
               ROOT / "CMakeLists.txt", ROOT / "build/compile_commands.json", Path(__file__).resolve()]
    for source in sources:
        dest = out / "source" / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    (out / "source.diff").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
    hashes = {name: hashlib.sha256((ROOT / "build" / name).read_bytes()).hexdigest() for name in ["bench_cpu", "peaks"]}
    (out / "binary_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Results: {out}\nMeasuring fresh 10-thread M4 Pro roofline ceilings; no correctness checks.", flush=True)
    peaks = subprocess.check_output([str(ROOT / "build/peaks"), "--threads", str(THREADS)], env=env, text=True)
    assert json.loads(peaks)["threads"] == THREADS
    (out / "peaks.json").write_text(peaks)
    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {job['pattern']} M={job['M']}, K={job['K']}, N={job['N']}", flush=True)
        run_command([ROOT / "build/bench_cpu", "--M", job["M"], "--K", job["K"], "--N", job["N"],
                     "--s", ",".join(map(str, SPARSITIES)), "--sparsity-pattern", job["pattern"],
                     "--k", ",".join(map(str, KS)), "--threads", THREADS,
                     "--kernels", ",".join("=" + k for k in KERNELS), "--rounds", args.rounds,
                     "--warmup", 5, "--prep-rounds", 1, "--verify", "none", "--ref-rows", 0,
                     "--seed", 42, "--tag", "v678_" + job["pattern"], "--out", out / "raw.csv"],
                    out / "benchmark.log", env)
        recent = read_csv(out / "raw.csv")[-36:]
        assert len(recent) == 36 and all(float(r["max_err"]) == -1 and float(r["median_ms"]) > 0 for r in recent)
    summarize(out)


if __name__ == "__main__":
    main()
