#!/usr/bin/env python3
"""Run the existing SPR benchmark and roofline tools over a CPU thread sweep.

    /usr/bin/python3 bench/spr_threads.py
    /usr/bin/python3 bench/spr_threads.py --plots-only results/RUN

Preparation and inference scaling are plotted separately. New runs use V5's
parallel inference; older manifests retain their serial-inference rooflines.
Build with CMake first. Requires Matplotlib in the selected Python environment.
"""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V5 = "spr.v5.b32"
V4 = "spr.v4.b32"
HISTORY = ["spr.naive", "spr.v0", "spr.v1", "spr.v2.b32", "spr.v3.b32.u1", V4, V5]
SHAPES = [(1024, 4096), (2048, 2048), (4096, 1024), (2048, 8192)]


def gmean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def run(cmd, log, env):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)


def summarize(out):
    manifest = json.loads((out / "manifest.json").read_text())
    parallel_inference = manifest.get("parallel_inference", False)
    inference_mode = "parallel" if parallel_inference else "serial"
    threads = manifest["threads"]
    all_rows = []
    for t in threads:
        with open(out / f"t{t:02}" / "raw.csv") as f:
            for r in csv.DictReader(f):
                if r["ok"] != "1" or float(r["median_ms"]) <= 0 or float(r["max_err"]) < 0:
                    raise RuntimeError("Failed or unverified benchmark configuration: " + str(r))
                for name in ("threads", "M", "K", "N"):
                    r[name] = int(r[name])
                for name in ("sparsity", "median_ms", "prep_ms", "adds_per_row", "bytes"):
                    r[name] = float(r[name])
                all_rows.append(r)
    with open(out / "raw.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    # Thread scaling uses exactly the same k at every thread count. Select k once
    # from V5's best one-thread inference time for each shape and sparsity.
    one = [r for r in all_rows if r["threads"] == 1 and r["kernel"] == V5]
    keys = sorted({(r["M"], r["K"], r["N"], r["sparsity"]) for r in one})
    selected = {key: min((r for r in one if (r["M"], r["K"], r["N"], r["sparsity"]) == key),
                        key=lambda r: r["median_ms"]) for key in keys}
    lookup = {(r["threads"], r["kernel"], r["M"], r["K"], r["N"], r["sparsity"], r["param"]): r
              for r in all_rows}
    scaling = []
    for key in keys:
        base = selected[key]
        for t in threads:
            r = lookup[(t, V5, *key, base["param"])]
            control = lookup[(t, V4, *key, base["param"])]
            scaling.append(dict(threads=t, M=key[0], K=key[1], N=key[2], sparsity=key[3],
                                param=base["param"], prep_ms=r["prep_ms"], inference_ms=r["median_ms"],
                                prep_speedup=base["prep_ms"] / r["prep_ms"],
                                inference_speedup=base["median_ms"] / r["median_ms"],
                                prep_vs_v4=control["prep_ms"] / r["prep_ms"],
                                v4_prep_ms=control["prep_ms"], v4_inference_ms=control["median_ms"]))
    with open(out / "scaling.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(scaling[0]))
        writer.writeheader()
        writer.writerows(scaling)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    shapes = list(dict.fromkeys((r["M"], r["K"], r["N"]) for r in one))
    sparsities = sorted({r["sparsity"] for r in one})

    def panels():
        return plt.subplots(math.ceil(len(shapes) / 2), 2, figsize=(12, 8), squeeze=False)

    def save(fig, name, title):
        fig.suptitle(title)
        handles, labels = fig.axes[0].get_legend_handles_labels()
        legend_rows = math.ceil(len(labels) / 4)
        fig.tight_layout(rect=(0, 0.055 + 0.025 * legend_rows, 1, 0.95))
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
        fig.savefig(out / name, dpi=170, bbox_inches="tight")
        plt.close(fig)

    for field, filename, title, ylabel in [
        ("prep_speedup", "speedup.png", "V5 preparation speedup on Apple M4 Pro", "speedup vs V5 at 1 thread"),
        ("inference_speedup", "inference_speedup.png", f"V5 inference scaling ({inference_mode})", "speedup vs V5 at 1 thread"),
        ("prep_ms", "preparation.png", "V5 preparation time on Apple M4 Pro", "preparation median [ms]"),
        ("inference_ms", "runtime.png", f"V5 inference time ({inference_mode})", "inference median [ms]"),
    ]:
        fig, axs = panels()
        for ax, sh in zip(axs.flat, shapes):
            for s in sparsities:
                rs = sorted((r for r in scaling if (r["M"], r["K"], r["N"]) == sh and r["sparsity"] == s),
                            key=lambda r: r["threads"])
                ax.plot([r["threads"] for r in rs], [r[field] for r in rs], "o-", label=f"{s:.0%} sparse")
            if "speedup" in field:
                ax.axhline(1, color="gray", linestyle=":", linewidth=1)
            ax.axvline(10, color="gray", linestyle="--", linewidth=0.8)
            ax.set(title=f"M={sh[0]}, K={sh[1]}, N={sh[2]}", xlabel="requested OpenMP threads", ylabel=ylabel,
                   xticks=threads)
            ax.grid(alpha=0.25)
        save(fig, filename, title + " (B=32)\nFixed k per workload; dashed line = 10 performance cores")

    # Version progression: independently best k, all measured at one thread.
    fig, axs = panels()
    for ax, sh in zip(axs.flat, shapes):
        for kernel in HISTORY:
            ys = []
            for s in sparsities:
                rs = [r for r in all_rows if r["threads"] == 1 and
                      (r["M"], r["K"], r["N"], r["sparsity"]) == (*sh, s)]
                ref = min(r["median_ms"] for r in rs if r["kernel"] == "spr.naive")
                value = min(r["median_ms"] for r in rs if r["kernel"] == kernel)
                ys.append(ref / value)
            ax.plot([100 * s for s in sparsities], ys, "o-", label=kernel)
        ax.set(title=f"M={sh[0]}, K={sh[1]}, N={sh[2]}", xlabel="weight sparsity [%]",
               ylabel="inference speedup vs spr.naive")
        ax.grid(alpha=0.25)
    save(fig, "version_speedup.png", "SPR version progression on Apple M4 Pro (1 thread, best k)")

    # Historical serial runs use one-thread roofs; new parallel runs use the
    # matching aggregate SIMD/DRAM roof for each requested thread count.
    pk = json.loads((out / "t01" / "peaks.json").read_text())
    thread_peaks = {t: json.loads((out / f"t{t:02}" / "peaks.json").read_text()) for t in threads}
    fig, axs = panels()
    for ax, sh in zip(axs.flat, shapes):
        xs = [10 ** (-2 + i * 5 / 200) for i in range(201)]
        peak = max(p["peak_simd_gflops"] for p in thread_peaks.values()) if parallel_inference else pk["peak_simd_gflops"]
        for name, style in ([] if parallel_inference else [("dram", "-"), ("l2", "--"), ("l1", ":")]):
            bw = pk[f"bw_{name}_gbs"]
            ax.plot(xs, [min(peak, x * bw) for x in xs], style, color="gray", linewidth=1,
                    label=f"{name.upper()} {bw:.0f} GB/s")
        if not parallel_inference:
            ax.axhline(pk["peak_scalar_gflops"], linestyle="-.", color="gray", linewidth=0.8,
                       label=f"scalar add peak {pk['peak_scalar_gflops']:.1f} GFLOP/s")
            ax.text(0.03, peak * 1.08, f"SIMD add peak {peak:.1f} GFLOP/s", fontsize=8, color="gray")
        for t in threads:
            rs = [lookup[(t, V5, *key, selected[key]["param"])] for key in keys if key[:3] == sh]
            work = [r["M"] * r["adds_per_row"] for r in rs]
            ai = [w / (r["bytes"] + 4 * r["M"] * (r["K"] + r["N"])) for r, w in zip(rs, work)]
            perf = [w / r["median_ms"] / 1e6 for r, w in zip(rs, work)]
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][threads.index(t) % 10]
            if parallel_inference:
                tp = thread_peaks[t]
                ax.plot(xs, [min(tp["peak_simd_gflops"], x * tp["bw_dram_gbs"]) for x in xs],
                        color=color, alpha=0.6, linewidth=1)
            ax.scatter(ai, perf, s=40, color=color, marker=["o", "s", "^", "D", "v", "P", "X", "*"][threads.index(t) % 8],
                       label=f"{t} requested thread{'s' if t != 1 else ''}", alpha=0.8)
        ax.set(xscale="log", yscale="log", xlim=(0.02, 1000), ylim=(1, peak * 2),
               title=f"M={sh[0]}, K={sh[1]}, N={sh[2]}", xlabel="additions / compulsory byte",
               ylabel="GFLOP/s (fp32 additions)")
        ax.grid(alpha=0.25)
    roof_note = "Parallel inference: matching thread-count SIMD/DRAM roofs" if parallel_inference else "Serial inference: one-thread roofs"
    save(fig, "roofline.png", f"V5 inference roofline — Apple M4 Pro (B=32)\n{roof_note}; fixed k across thread counts")

    summary = []
    for t in threads:
        rs = [r for r in scaling if r["threads"] == t]
        summary.append(dict(threads=t, prep_speedup=gmean([r["prep_speedup"] for r in rs]),
                            inference_speedup=gmean([r["inference_speedup"] for r in rs]),
                            prep_vs_v4=gmean([r["prep_vs_v4"] for r in rs])))
    lines = ["# SPR V5 thread sweep — Apple M4 Pro", "",
             "CPU: 10 performance cores + 4 efficiency cores; macOS chooses core placement (no pinning).",
             ("V5 parallelizes preparation and inference across independent groups." if parallel_inference else
              "V5 parallelizes the first preparation stage only. Its inference remains serial."), "",
             f"Scaling kernel: `{V5}`; serial controls: `{V4}` and `spr.v2.b32`.",
             f"Grid: M={manifest['M']}; K×N={SHAPES}; sparsities={manifest['sparsities']}; k={manifest['ks']}.",
             f"Preparation: median of {manifest['prep_rounds']} fresh-kernel measurements after one untimed preparation.",
             f"Inference: median of {manifest['rounds']} interleaved measurements after 3 warmups.",
             "OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0, OMP_DYNAMIC=FALSE; identical settings for every thread count.",
             "All benchmark configurations were checked against the reference on all M rows.",
             "Thread scaling fixes k to V5's fastest one-thread inference configuration for each workload.",
             "Version comparisons and per-thread standard reports use independently best k.", "",
             "| Threads | Preparation speedup vs V5/1T | Inference speedup vs V5/1T | Preparation speedup vs V4 at same threads |",
             "|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['threads']} | {r['prep_speedup']:.2f}× | {r['inference_speedup']:.2f}× | {r['prep_vs_v4']:.2f}× |")
    lines += ["", "Ratios above are geometric means over the 16 shape/sparsity workloads.", "",
              "## Plots", "", "- `speedup.png`: preparation scaling.",
              "- `preparation.png`: absolute preparation time.", f"- `inference_speedup.png`: {inference_mode} inference scaling.",
              "- `runtime.png`: absolute inference time.", "- `version_speedup.png`: inference progression from naive through V5 at one thread.",
              f"- `roofline.png`: {roof_note.lower()}.",
              "- `tXX/{report.md,roofline.png,runtime.png}`: existing infrastructure reports, with measured XX-thread machine ceilings.",
              "", "## Interpretation limits", "",
              "Roofline intensity uses compulsory format + input + output traffic, not measured cache traffic.",
              "V4/V5's additions use their existing adds_per_row() accounting convention.",
              "Requested threads describe OpenMP settings, not a guarantee that inference uses those threads.",
              "Results include normal scheduling, frequency and thermal variation; preparation excludes cold OpenMP startup.",
              "Kernel/benchmark source snapshots, commands, CPU/cache metadata, correctness logs and machine ceilings are retained here."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines[:26]), flush=True)
    print(f"\nSaved plots and report in {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", default="1,2,4,6,8,10,12,14")
    ap.add_argument("--ks", default="2,3,4,5,6,8")
    ap.add_argument("--sparsities", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--prep-rounds", type=int, default=5)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plots-only", type=Path)
    a = ap.parse_args()
    if a.plots_only:
        summarize(a.plots_only.resolve())
        return
    threads = [int(t) for t in a.threads.split(",")]
    if not threads or threads[0] != 1 or any(t < 1 for t in threads) or len(set(threads)) != len(threads):
        ap.error("threads must be distinct positive integers starting with 1")
    stamp = datetime.datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
    out = (a.out or ROOT / "results" / f"{stamp}_spr_v5_threads").resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0", OMP_DYNAMIC="FALSE",
               MPLCONFIGDIR=str(ROOT / ".cache" / "matplotlib"))
    cpu_keys = ["machdep.cpu.brand_string", "hw.physicalcpu", "hw.perflevel0.physicalcpu",
                "hw.perflevel1.physicalcpu", "hw.cachelinesize", "hw.perflevel0.l1dcachesize",
                "hw.perflevel0.l2cachesize", "hw.perflevel1.l1dcachesize", "hw.perflevel1.l2cachesize"]
    cpu = subprocess.check_output(["sysctl", *cpu_keys], text=True)
    if "Apple M4 Pro" not in cpu:
        raise RuntimeError("Default sweep targets an Apple M4 Pro; check hardware before running")
    manifest = dict(threads=threads, M=a.M, ks=a.ks, sparsities=a.sparsities,
                    parallel_inference=True,
                    rounds=a.rounds, prep_rounds=a.prep_rounds, cpu=cpu, command=sys.argv,
                    cache_overrides={k: env.get(k) for k in ("SPR_L1_BYTES", "SPR_L2_BYTES", "SPR_LLC_BYTES")},
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in [*ROOT.glob("cpu/spr/*.cpp"), *ROOT.glob("include/spr/*.hpp"),
                   ROOT / "CMakeLists.txt", ROOT / "bench/bench_cpu.cpp", ROOT / "bench/peaks.cpp",
                   ROOT / "bench/roofline.py", Path(__file__).resolve(), ROOT / "tests/test_correctness.cpp"]:
        dest = out / "source" / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    (out / "source.diff").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
    print(f"Results: {out}\nChecking all SPR kernels...", flush=True)
    run([ROOT / "build/test_correctness", "spr."], out / "correctness.log", env)
    if "\n0 cases passed" in (out / "correctness.log").read_text():
        raise RuntimeError("No kernels tested; rebuild before benchmarking")
    for t in threads:
        dest = out / f"t{t:02}"
        dest.mkdir()
        print(f"[{t} threads] Measuring fresh machine ceilings...", flush=True)
        peaks = subprocess.check_output([str(ROOT / "build/peaks"), "--threads", str(t)], env=env, text=True)
        json.loads(peaks)
        (dest / "peaks.json").write_text(peaks)
        kernels = HISTORY if t == 1 else ["spr.v2.b32", V4, V5]
        for K, N in SHAPES:
            print(f"[{t} threads] M={a.M}, K={K}, N={N}; {len(kernels)} kernels", flush=True)
            run([ROOT / "build/bench_cpu", "--M", a.M, "--K", K, "--N", N,
                 "--s", a.sparsities, "--k", a.ks, "--threads", t, "--kernels", ",".join("=" + k for k in kernels),
                 "--rounds", a.rounds, "--prep-rounds", a.prep_rounds, "--warmup", 3, "--ref-rows", a.M,
                 "--tag", "spr_v5_threads", "--out", dest / "raw.csv"], dest / "benchmark.log", env)
        run([sys.executable, ROOT / "bench/roofline.py", dest / "raw.csv", dest / "peaks.json",
             "--out", dest, "--ref", "spr.naive" if t == 1 else V4], dest / "plot.log", env)
    summarize(out)


if __name__ == "__main__":
    main()
