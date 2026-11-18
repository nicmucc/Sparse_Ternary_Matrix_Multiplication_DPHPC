#!/usr/bin/env python3
"""Compare V6 tile x group parallelism with V7 whole-tile parallelism on the local M4 Pro.

Build with CMake, then run /usr/bin/python3 bench/compare_v6_v7.py.
Replot with --plots-only results/RUN. Requires Matplotlib.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
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

KERNELS = ["spr.v6.b32", "spr.v7.b32"]
SHAPES = [(1024, 4096), (2048, 2048), (4096, 1024), (2048, 8192), (1024, 32)]
MS = [32, 64, 256, 512, 1024]
PATTERN_MS = [256, 512]
ROOF_M = 512


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def command(cmd, log, env):
    with log.open("a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def summarize(out):
    manifest = json.loads((out / "manifest.json").read_text())
    threads, jobs = manifest["threads"], manifest["jobs"]
    rows = [r for t in threads for r in read_csv(out / f"t{t:02}/raw.csv")]
    expected = len(threads) * len(jobs) * len(manifest["sparsities"]) * len(manifest["ks"]) * 2
    assert len(rows) == expected, (len(rows), expected)
    assert all(r["ok"] == "1" and int(r["rounds"]) == manifest["rounds"] and
               0 <= float(r["max_err"]) <= 2e-3 * math.sqrt(int(r["K"])) for r in rows)
    write_csv(out / "raw.csv", rows)
    for r in rows:
        for key in ("threads", "M", "K", "N"):
            r[key] = int(r[key])
        for key in ("sparsity", "median_ms", "prep_ms", "bytes", "adds_per_row"):
            r[key] = float(r[key])
        r["pattern"] = r["tag"].removeprefix("v6v7_")
    workload = lambda r: (r["pattern"], r["M"], r["K"], r["N"], r["sparsity"])
    lookup = {(r["threads"], r["kernel"], workload(r), r["param"]): r for r in rows}
    assert len(lookup) == expected, "duplicate benchmark rows"
    best = {}
    for r in rows:
        key = (r["threads"], r["kernel"], workload(r))
        if key not in best or r["median_ms"] < best[key]["median_ms"]:
            best[key] = r
    workloads = sorted({workload(r) for r in rows})
    paired, tuned = [], []
    peaks = {t: json.loads((out / f"t{t:02}/peaks.json").read_text()) for t in threads}
    for w in workloads:
        # Select one common k from V6/1T, then hold it fixed for both versions
        # and all thread counts to isolate the scheduling change.
        param = best[(1, KERNELS[0], w)]["param"]
        for t in threads:
            a, b = [lookup[(t, k, w, param)] for k in KERNELS]
            assert a["bytes"] == b["bytes"] and a["adds_per_row"] == b["adds_per_row"], "comparison must isolate scheduling"
            a0, b0 = [lookup[(1, k, w, param)] for k in KERNELS]
            work = b["M"] * b["adds_per_row"]
            ai = work / (b["bytes"] + 4 * b["M"] * (b["K"] + b["N"]))
            paired.append(dict(pattern=w[0], M=w[1], K=w[2], N=w[3], sparsity=w[4], threads=t,
                               param=param, v6_ms=a["median_ms"], v7_ms=b["median_ms"],
                               v7_vs_v6=a["median_ms"] / b["median_ms"],
                               v6_scaling=a0["median_ms"] / a["median_ms"],
                               v7_scaling=b0["median_ms"] / b["median_ms"],
                               intensity=ai, v6_gflops=work / a["median_ms"] / 1e6,
                               v7_gflops=work / b["median_ms"] / 1e6))
            aa, bb = [best[(t, k, w)] for k in KERNELS]
            tuned.append(dict(pattern=w[0], M=w[1], K=w[2], N=w[3], sparsity=w[4], threads=t,
                              v6_param=aa["param"], v7_param=bb["param"], v6_ms=aa["median_ms"],
                              v7_ms=bb["median_ms"], v7_vs_v6=aa["median_ms"] / bb["median_ms"]))
    write_csv(out / "comparison.csv", paired)
    write_csv(out / "tuned_comparison.csv", tuned)
    regular = [r for r in paired if r["pattern"] == "iid" and r["N"] != 32]
    sps = manifest["sparsities"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def panels(n):
        return plt.subplots(math.ceil(n / 2), 2, figsize=(12, 3.9 * math.ceil(n / 2)), squeeze=False)

    def finish(fig, axs, n, title, filename):
        for ax in list(axs.flat)[n:]:
            ax.axis("off")
        h, labels = axs.flat[0].get_legend_handles_labels()
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.06 + 0.025 * math.ceil(len(labels) / 4), 1, 0.94))
        fig.legend(h, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
        fig.savefig(out / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def axes_style(ax):
        ax.set_xticks(threads)
        ax.set_xlabel("requested OpenMP threads")
        ax.axvline(10, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.25)

    fig, axs = panels(len(MS))
    ratio_limits = (min(0.95, min(r["v7_vs_v6"] for r in regular) * 0.98),
                    max(1.05, max(r["v7_vs_v6"] for r in regular) * 1.03))
    for ax, M in zip(axs.flat, MS):
        for s in sps:
            ax.plot(threads, [geomean([r["v7_vs_v6"] for r in regular if r["M"] == M and r["threads"] == t and r["sparsity"] == s]) for t in threads], "o-", label=f"{s:.0%} sparse")
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set(title=f"M={M}; geomean over 4 regular matrix shapes", ylabel="V7 speedup vs V6 (>1 is faster)")
        ax.set_ylim(*ratio_limits)
        axes_style(ax)
    finish(fig, axs, len(MS), "V7 whole-tile scheduling vs V6 collapsed scheduling — same k, same weights (B=32)", "speedup.png")

    fig, axs = panels(5)
    for ax, (K, N) in zip(axs.flat, SHAPES):
        for M in MS:
            rs = [r for r in paired if r["pattern"] == "iid" and (r["M"], r["K"], r["N"]) == (M, K, N)]
            ax.plot(threads, [geomean([r["v7_vs_v6"] for r in rs if r["threads"] == t]) for t in threads], "o-", label=f"M={M}")
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set(title=f"K={K}, N={N}" + (" (few-group diagnostic)" if N == 32 else ""), ylabel="V7 speedup vs V6")
        axes_style(ax)
    finish(fig, axs, 5, "Effect of matrix size — geomean over sparsities, fixed k", "matrix_speedup.png")

    fig, axs = panels(len(MS))
    for ax, M in zip(axs.flat, MS):
        for field, label in [("v6_scaling", "V6"), ("v7_scaling", "V7")]:
            ax.plot(threads, [geomean([r[field] for r in regular if r["M"] == M and r["threads"] == t]) for t in threads], "o-", label=label)
        ax.set(title=f"M={M}", ylabel="speedup vs own 1-thread time")
        axes_style(ax)
    finish(fig, axs, len(MS), "Thread scaling — regular iid matrices; fixed common k", "thread_scaling.png")

    fig, axs = panels(len(MS))
    for ax, M in zip(axs.flat, MS):
        rs = [r for r in paired if r["pattern"] == "iid" and r["M"] == M and r["N"] == 32]
        for field, label in [("v6_ms", "V6"), ("v7_ms", "V7")]:
            ax.plot(threads, [1000 * geomean([r[field] for r in rs if r["threads"] == t]) for t in threads], "o-", label=label)
        ax.set(title=f"M={M}, K=1024, N=32", ylabel="inference time [microseconds], lower is better")
        axes_style(ax)
    finish(fig, axs, len(MS), "Narrow-output diagnostic: absolute latency — geomean over sparsities", "narrow_runtime.png")

    fig, axs = panels(2)
    for ax, (K, N) in zip(axs.flat, [(1024, 4096), (2048, 8192)]):
        for M in PATTERN_MS:
            for pattern in ["iid", "column_clustered"]:
                rs = [r for r in paired if (r["pattern"], r["M"], r["K"], r["N"]) == (pattern, M, K, N)]
                ax.plot(threads, [geomean([r["v7_vs_v6"] for r in rs if r["threads"] == t]) for t in threads], "o-", label=f"M={M}, {pattern}")
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set(title=f"K={K}, N={N}", ylabel="V7 speedup vs V6")
        axes_style(ax)
    finish(fig, axs, 2, "Random zeros vs contiguous zero output columns — geomean over sparsities", "pattern_speedup.png")

    fig, axs = panels(4)
    shown_threads = [t for t in [1, 4, 10, 14] if t in threads]
    for ax, (K, N) in zip(axs.flat, SHAPES[:4]):
        xs = [10 ** (-2 + i / 40) for i in range(201)]
        for j, t in enumerate(shown_threads):
            color = f"C{j}"
            pk = peaks[t]
            ax.plot(xs, [min(pk["peak_simd_gflops"], x * pk["bw_dram_gbs"]) for x in xs], color=color, alpha=0.5, linewidth=1)
            rs = [r for r in regular if r["M"] == ROOF_M and r["threads"] == t and (r["K"], r["N"]) == (K, N)]
            for field, marker, label in [("v6_gflops", "o", "V6"), ("v7_gflops", "^", "V7")]:
                ax.scatter([r["intensity"] for r in rs], [r[field] for r in rs], marker=marker, color=color, s=42, label=f"{label}, {t}T")
        ax.set(xscale="log", yscale="log", xlim=(0.02, 1000), ylim=(0.5, max(p["peak_simd_gflops"] for p in peaks.values()) * 2),
               title=f"M={ROOF_M}, K={K}, N={N}", xlabel="additions / compulsory byte", ylabel="GFLOP/s (fp32 additions)")
        ax.grid(alpha=0.25)
    finish(fig, axs, 4, "V6 vs V7 roofline — fresh M4 Pro ceilings for each thread count, fixed k", "roofline.png")

    lines = ["# SPR V6 vs V7 on the local Apple M4 Pro", "", manifest["cpu"].replace("\n", "  \n"),
             "", "Kernels: `spr.v6.b32` (collapsed tile x group scheduling) and `spr.v7.b32` (whole-tile scheduling). Both use serial preparation.",
             f"Threads: {threads}; M: {MS}; K×N: {SHAPES}; sparsities: {sps}; k: {manifest['ks']}.",
             "All weights and inputs use seed 42 and are identical between versions and thread counts.",
             "iid: independent element-wise zeros. column_clustered: floor(s*N) leading output columns are zero; remaining columns are dense random +/-1.",
             "The clustered diagnostic uses M=256/512 and K×N=1024×4096 / 2048×8192. It is deliberately imbalanced.",
             "All output rows are verified against the double-precision reference before timing.",
             f"Inference medians: {manifest['rounds']} interleaved rounds after 3 warmups. Preparation is measured once, excluded from inference, and is not used for conclusions.",
             "OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0, OMP_DYNAMIC=FALSE. macOS chooses core placement; no pinning.",
             "Main comparisons fix k to V6's fastest one-thread k per workload and use it for both versions at every thread count.",
             "`tuned_comparison.csv` additionally compares independently fastest k for each version and thread count.", "",
             "## V7 inference speedup over V6", "", "Geometric means over regular iid matrix shapes and four sparsities. >1 means V7 is faster.", "",
             "| Threads | " + " | ".join(f"M={M}" for M in MS) + " |", "|" + "---|" * (len(MS) + 1)]
    summary = []
    for t in threads:
        vals = [geomean([r["v7_vs_v6"] for r in regular if r["threads"] == t and r["M"] == M]) for M in MS]
        lines.append("| " + str(t) + " | " + " | ".join(f"{v:.2f}×" for v in vals) + " |")
        summary.append(dict(threads=t, **{f"M{M}": v for M, v in zip(MS, vals)}))
    def aggregate(dataset, M, t, field):
        return geomean([r[field] for r in dataset if r["M"] == M and r["threads"] == t])

    focus_t = 10 if 10 in threads else threads[-1]
    clustered = [r for r in paired if r["pattern"] == "column_clustered"]
    narrow = [r for r in paired if r["pattern"] == "iid" and r["N"] == 32]
    best_regular = {M: max(threads, key=lambda t: aggregate(regular, M, t, "v7_scaling")) for M in MS}
    best_narrow = {M: max(threads, key=lambda t: aggregate(narrow, M, t, "v7_scaling")) for M in MS}
    def best_label(M, t):
        return "serial at all settings" if M <= 32 else f"{t}T"

    lines += ["", "## Interpretation", "",
              "V7 schedules only input tiles: M=32/64/256/512/1024 provide 1/2/8/16/32 tiles at B=32. Requested threads can exceed the number of useful workers.",
              "At M=32, V7's parallel-region condition is false. Changing the requested thread count does not parallelize that case, and any variation in V7's absolute time is not evidence of thread scaling.",
              "When the tile count is divisible by the thread count, V6's contiguous static collapsed assignment can already give each thread complete tiles. V7 may then mostly remove indexing overhead.",
              "Having at least as many tiles as threads does not ensure perfect balance: 16 tiles on 10 threads still give some threads twice as many tiles as others.",
              f"At {focus_t} threads, regular-grid V7/V6 speedups: " + ", ".join(f"M={M}: {aggregate(regular, M, focus_t, 'v7_vs_v6'):.2f}×" for M in MS) + ".",
              "Best observed V7 thread counts over regular matrices (fixed-k geometric means): " + ", ".join(f"M={M}: {best_label(M, best_regular[M])}" for M in MS) + ".",
              f"For synthetic clustered zeros at M=256, V7/V6 is {aggregate(clustered, 256, focus_t, 'v7_vs_v6'):.2f}× at {focus_t} threads, separate from regular-grid averages.",
              f"For N=32, M=64, V7/V6 is {aggregate(narrow, 64, focus_t, 'v7_vs_v6'):.2f}× at {focus_t} threads; values below one mean V7 is slower.",
              "Best observed V7 thread counts for N=32: " + ", ".join(f"M={M}: {best_label(M, best_narrow[M])}" for M in MS) + ". See narrow_runtime.png before interpreting large relative speedups."]
    lines += ["", "## Breakdown by matrix size (iid)", "", "| M | K×N | Threads | V7/V6 speedup |", "|---|---|---|---|"]
    for M in MS:
        for K, N in SHAPES:
            for t in [1, 4, 10, 14]:
                rs = [r for r in paired if (r["pattern"], r["M"], r["K"], r["N"], r["threads"]) == ("iid", M, K, N, t)]
                if rs:
                    lines.append(f"| {M} | {K}×{N} | {t} | {geomean([r['v7_vs_v6'] for r in rs]):.2f}× |")
    lines += ["", "## Plots and data", "", "- `speedup.png`: V7/V6 by sparsity, M, threads (regular matrices).",
              "- `matrix_speedup.png`: includes the narrow-output diagnostic; this is excluded from the main aggregate.",
              "- `thread_scaling.png`: each version versus its own 1T time.", "- `pattern_speedup.png`: iid versus clustered output zeros.",
              "- `narrow_runtime.png`: absolute times in the few-group diagnostic; excessive threads can hurt both versions.",
              "- `roofline.png`: M=512, shared k; colors match measured thread-count SIMD/DRAM roofs.",
              "- `tXX/report.md`, `tXX/roofline.png`: standard harness reports for M=512 iid cases, independently best k.",
              "- `comparison.csv`: all fixed-k results; `tuned_comparison.csv`: independently tuned comparisons; `raw.csv`: full k sweep.",
              "", "Roofline intensity uses compulsory format + X + Y traffic and the kernels' existing addition-count convention; it is not measured cache traffic.",
              "Near-unity differences may reflect measurement noise. This run uses one deterministic weight sample per workload; the clustered case is a synthetic scheduling diagnostic."]
    if (out / "ownership_summary.md").exists():
        lines += ["", "Tile-count and load-balance analysis: [ownership_summary.md](ownership_summary.md)."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    checks = re.search(r"(\d+) cases passed, (\d+) failed", (out / "correctness.log").read_text())
    assert checks and int(checks[2]) == 0
    validation = dict(configurations=len(rows), failed_configurations=0, comparison_points=len(paired),
                      correctness_passed=int(checks[1]), correctness_failed=int(checks[2]),
                      maximum_absolute_error=max(float(r["max_err"]) for r in rows),
                      source_matches_current={v: (out / f"source/cpu/spr/spr_{v}.cpp").read_bytes() == (ROOT / f"cpu/spr/spr_{v}.cpp").read_bytes() for v in ["v6", "v7"]})
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print("\n".join(lines[:29]), flush=True)
    print(f"\nResults: {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", default="1,2,4,6,8,10,12,14")
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--plots-only", type=Path)
    a = ap.parse_args()
    if a.plots_only:
        summarize(a.plots_only.resolve())
        return
    threads = list(map(int, a.threads.split(",")))
    assert threads[0] == 1 and len(threads) == len(set(threads)) and all(t > 0 for t in threads)
    jobs = [dict(pattern="iid", M=M, K=K, N=N) for M in MS for K, N in SHAPES]
    jobs += [dict(pattern="column_clustered", M=M, K=K, N=N) for M in PATTERN_MS for K, N in [(1024, 4096), (2048, 8192)]]
    stamp = datetime.datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / f"{stamp}_spr_v6_v7_comparison"
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0", OMP_DYNAMIC="FALSE")
    cpu_keys = ["machdep.cpu.brand_string", "hw.physicalcpu", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu",
                "hw.cachelinesize", "hw.perflevel0.l1dcachesize", "hw.perflevel0.l2cachesize",
                "hw.perflevel1.l1dcachesize", "hw.perflevel1.l2cachesize"]
    cpu = subprocess.check_output(["sysctl", *cpu_keys], text=True)
    assert "Apple M4 Pro" in cpu, "Check hardware before using this default grid"
    manifest = dict(cpu=cpu, threads=threads, jobs=jobs, sparsities=[0.3, 0.5, 0.7, 0.9], ks=[2, 3, 4], rounds=a.rounds,
                    command=sys.argv, seed=42, prep_rounds=1,
                    cache_overrides={k: env.get(k) for k in ["SPR_L1_BYTES", "SPR_L2_BYTES", "SPR_LLC_BYTES"]},
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sources = [*ROOT.glob("cpu/spr/*.cpp"), *ROOT.glob("include/spr/*.hpp"), ROOT / "cpu/core.cpp",
               ROOT / "CMakeLists.txt", ROOT / "tests/test_correctness.cpp", ROOT / "bench/bench_cpu.cpp",
               ROOT / "bench/peaks.cpp", ROOT / "bench/roofline.py", Path(__file__).resolve(), ROOT / "build/compile_commands.json"]
    for src in sources:
        dest = out / "source" / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    (out / "source.diff").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
    (out / "binary_hashes.json").write_text(json.dumps({name: hashlib.sha256((ROOT / "build" / name).read_bytes()).hexdigest() for name in ["bench_cpu", "test_correctness", "peaks"]}, indent=2) + "\n")
    print(f"Results: {out}\nChecking V6 and V7...", flush=True)
    command([ROOT / "build/test_correctness", "spr.v6", "spr.v7"], out / "correctness.log", env)
    for t in threads:
        dest = out / f"t{t:02}"
        dest.mkdir()
        print(f"[{t} threads] Measuring fresh local machine ceilings", flush=True)
        pk = subprocess.check_output([str(ROOT / "build/peaks"), "--threads", str(t)], env=env, text=True)
        assert json.loads(pk)["threads"] == t
        (dest / "peaks.json").write_text(pk)
        for j, job in enumerate(jobs, 1):
            print(f"[{t} threads] {j}/{len(jobs)} {job['pattern']} M={job['M']}, K={job['K']}, N={job['N']}", flush=True)
            command([ROOT / "build/bench_cpu", "--M", job["M"], "--K", job["K"], "--N", job["N"],
                     "--s", ",".join(map(str, manifest["sparsities"])), "--sparsity-pattern", job["pattern"],
                     "--k", ",".join(map(str, manifest["ks"])), "--threads", t,
                     "--kernels", ",".join("=" + k for k in KERNELS), "--rounds", a.rounds,
                     "--warmup", 3, "--prep-rounds", 1, "--ref-rows", job["M"], "--seed", 42,
                     "--tag", "v6v7_" + job["pattern"], "--out", dest / "raw.csv"], dest / "benchmark.log", env)
            recent = read_csv(dest / "raw.csv")[-24:]
            assert len(recent) == 24 and all(r["ok"] == "1" and float(r["max_err"]) >= 0 for r in recent)
        refs = [r for r in read_csv(dest / "raw.csv") if r["M"] == str(ROOF_M) and r["tag"] == "v6v7_iid"]
        write_csv(dest / "reference_raw.csv", refs)
        command([sys.executable, ROOT / "bench/roofline.py", dest / "reference_raw.csv", dest / "peaks.json",
                 "--out", dest, "--ref", KERNELS[0]], dest / "plot.log", env)
    summarize(out)


if __name__ == "__main__":
    main()
