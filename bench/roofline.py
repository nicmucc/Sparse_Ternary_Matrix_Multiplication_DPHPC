#!/usr/bin/env python3
"""Roofline model + runtime report for the SPR kernels (spr.* only).

    python3 bench/roofline.py RAW.csv PEAKS.json --out DIR [--ref spr.naive]

RAW.csv    output of bench_cpu (bench/spr_bench.sh runs it for you)
PEAKS.json machine ceilings measured by build/<profile>/peaks (cached in build/results/machine/)
Writes DIR/roofline.png, DIR/runtime.png, DIR/report.md (the report is also printed).

Roofline quantities (one point = one kernel at its best k for one shape and sparsity):
  work       W = adds_per_row * M                     every SPR operation is one fp32 add = 1 flop
  perf       P = W / time                             [GFLOP/s]
  traffic    Q = bytes() + 4*M*K + 4*M*N              compulsory: format + X + Y, each read/written once
  intensity  I = W / Q                                [flop/byte]
  roof       min(SIMD add peak, I * DRAM bandwidth)
Also reported: "format re-read BW" = bytes() * M / time, the bandwidth the kernel would need if it streams
the whole format once per row of X (what an m-outer loop does). If that is a large fraction of the L2 / DRAM
ceiling, the loop order is the bottleneck, not the arithmetic.
"""
import argparse
import csv
import json
import math
import os
import platform
import re
import subprocess
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Datasheet ceilings, keyed by CPU brand string. PER CORE for compute, whole chip for memory.
# Add your x86 machine here (brand string from: grep -m1 "model name" /proc/cpuinfo).
#   simd/scalar: fp32 ADD throughput = GHz * add pipes * lanes   (no FMA: SPR only adds)
SPECS = {
    "Apple M1 Pro": dict(ghz=3.228, simd_gflops_core=3.228 * 4 * 4, scalar_gflops_core=3.228 * 4, dram_gbs=200.0,
                         note="Firestorm P-core: 4x 128-bit FP pipes; LPDDR5-6400 256-bit"),
}

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
# variants of one version (spr.v4.b16, spr.v4.b32.kc512.g4, ...) share the version's color and differ by marker + line
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
LINES = ["-", "--", ":", "-."]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def cpu_brand():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def natural(name):   # spr.naive first, then spr.v0 < spr.v1 < spr.v2 < spr.v10 < others
    if name == "spr.naive": return (0, [])
    return (1, [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)])


def shared_legend(fig, axs):
    """One legend for the whole figure, to the right of the panels (instead of one per panel over the data)."""
    fig.tight_layout()
    handles, labels = {}, []
    for ax in axs.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles: handles[l] = h; labels.append(l)
    if len(labels) > 1:
        fig.legend([handles[l] for l in labels], labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=8, title="kernel", title_fontsize=9)


def gmean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(raw):
    rows = []
    for r in csv.DictReader(open(raw)):
        if not r["kernel"].startswith("spr.") or r["ok"] != "1" or float(r["median_ms"]) <= 0: continue
        M, K, N = int(r["M"]), int(r["K"]), int(r["N"])
        ms, apr, b = float(r["median_ms"]), float(r["adds_per_row"]), float(r["bytes"])
        t = ms * 1e-3
        d = dict(kernel=r["kernel"], M=M, K=K, N=N, s=float(r["sparsity"]), ms=ms,
                 k=r["param"][2:] if r["param"].startswith("k=") else "-", apr=apr, bytes=b,
                 bpw=float(r["bits_per_weight"]), prep=float(r["prep_ms"]))
        d["flops"] = apr * M
        d["gflops"] = d["flops"] / t / 1e9 if apr > 0 else float("nan")
        d["traffic"] = b + 4.0 * (M * K + M * N)
        d["ai"] = d["flops"] / d["traffic"] if apr > 0 else float("nan")
        d["ns_add"] = t * 1e9 / d["flops"] if apr > 0 else float("nan")
        d["reread_gbs"] = b * M / t / 1e9
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("peaks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="spr.naive", help="kernel the speed-ups are measured against")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pk = json.load(open(a.peaks))
    T = pk["threads"]
    simd, scal = pk["peak_simd_gflops"], pk["peak_scalar_gflops"]
    bw = {"L1": pk["bw_l1_gbs"], "L2": pk["bw_l2_gbs"], "DRAM": pk["bw_dram_gbs"]}
    brand = cpu_brand()
    spec = SPECS.get(brand)

    rows = load(a.raw)
    if not rows: raise SystemExit("no successful spr.* rows in " + a.raw)
    kernels = sorted({r["kernel"] for r in rows}, key=natural)
    ref = a.ref if a.ref in kernels else kernels[0]
    shapes = sorted({(r["M"], r["K"], r["N"]) for r in rows}, key=lambda x: (x[1] * x[2], x))
    sps = sorted({r["s"] for r in rows})
    # one color per version (spr.naive, spr.v0, spr.v1, ...), follows the version, never its rank;
    # inside a version, each variant gets its own marker and line style
    family = lambda k: ".".join(k.split(".")[:2])
    fams = list(dict.fromkeys(family(k) for k in kernels))
    color, marker, lstyle = {}, {}, {}
    for k in kernels:
        f = family(k)
        j = [x for x in kernels if family(x) == f].index(k)
        color[k] = COLORS[fams.index(f) % len(COLORS)]
        marker[k] = MARKERS[j % len(MARKERS)]
        lstyle[k] = LINES[j % len(LINES)]

    best = {}   # (kernel, shape, s) -> fastest k row
    for r in rows:
        key = (r["kernel"], (r["M"], r["K"], r["N"]), r["s"])
        if key not in best or r["ms"] < best[key]["ms"]: best[key] = r

    md = []
    P = md.append
    P("# SPR kernels: runtime and roofline\n")
    P(f"- CPU **{brand}**, threads **{T}**, ISA `{pk['isa']}`; kernels: " + ", ".join(f"`{k}`" for k in kernels))
    P(f"- speed-ups are vs **`{ref}`** (> 1 = faster); every number is the kernel's best k for that shape / sparsity\n")

    # ---------------------------------------------------------------- 1. ceilings
    P("## 1. Machine ceilings (measured by `build/peaks`)\n")
    crow = [["fp32 add, SIMD", f"{simd:.1f} GFLOP/s", f"{spec['simd_gflops_core'] * T:.1f}" if spec else "-"],
            ["fp32 add, scalar", f"{scal:.1f} GFLOP/s", f"{spec['scalar_gflops_core'] * T:.1f}" if spec else "-"],
            ["read BW, L1", f"{bw['L1']:.0f} GB/s", "-"], ["read BW, L2", f"{bw['L2']:.0f} GB/s", "-"],
            ["read BW, DRAM", f"{bw['DRAM']:.0f} GB/s", f"{spec['dram_gbs']:.0f}" if spec else "-"]]
    P(table(["ceiling", "measured", "datasheet"], crow))
    P(f"\nRidge point (SIMD peak / DRAM BW): **{simd / bw['DRAM']:.2f} flop/byte**: a kernel with lower intensity is "
      f"DRAM-bound at best, higher intensity can reach the compute peak." + (f" Datasheet: {spec['note']}." if spec else "") + "\n")

    # ---------------------------------------------------------------- 2. runtime
    P("## 2. Runtime: median ms (best k)\n")
    others = [k for k in kernels if k != ref]
    for sh in shapes:
        P(f"**M={sh[0]}, K={sh[1]}, N={sh[2]}**\n")
        tr = []
        for s in sps:
            line = [f"{s:.0%}"]
            for k in kernels:
                b = best.get((k, sh, s))
                line.append(f"{b['ms']:.2f} (k={b['k']})" if b else "-")
            for k in others:
                b, r0 = best.get((k, sh, s)), best.get((ref, sh, s))
                line.append(f"**{r0['ms'] / b['ms']:.2f}×**" if b and r0 else "-")
            tr.append(line)
        P(table(["s"] + kernels + [f"{k} vs {ref}" for k in others], tr) + "\n")

    if others:
        P("**Geomean speed-up over all shapes and sparsities**\n")
        g = []
        for i, k in enumerate(kernels):
            prev = kernels[i - 1] if i else None
            vs_ref = [best[(ref, sh, s)]["ms"] / best[(k, sh, s)]["ms"] for sh in shapes for s in sps
                      if (k, sh, s) in best and (ref, sh, s) in best]
            vs_prev = [best[(prev, sh, s)]["ms"] / best[(k, sh, s)]["ms"] for sh in shapes for s in sps
                       if prev and (k, sh, s) in best and (prev, sh, s) in best]
            g.append([f"`{k}`", f"{gmean(vs_ref):.2f}×", f"{gmean(vs_prev):.2f}× vs `{prev}`" if prev else "-"])
        P(table(["kernel", f"vs {ref}", "vs previous version"], g) + "\n")

    # ---------------------------------------------------------------- 3. roofline numbers
    P("## 3. Roofline position (best k)\n")
    P("- **GFLOP/s** = additions per second; **AI** = additions / compulsory bytes (format + X + Y)")
    P("- **% scalar / % SIMD** = fraction of the scalar / SIMD add peak; **% roof** = fraction of min(SIMD peak, AI × DRAM BW)")
    P("- **re-read GB/s** = format bytes × M / time: the bandwidth needed if the format is streamed once per row of X "
      f"(compare with L2 {bw['L2']:.0f} / DRAM {bw['DRAM']:.0f} GB/s)\n")
    for k in kernels:
        P(f"`{k}`\n")
        tr = []
        for sh in shapes:
            for s in sps:
                b = best.get((k, sh, s))
                if not b or not b["gflops"] == b["gflops"]: continue
                roof = min(simd, b["ai"] * bw["DRAM"])
                tr.append([f"{sh[1]}×{sh[2]}", f"{s:.0%}", b["k"], f"{b['ms']:.2f}", f"{b['gflops']:.2f}", f"{b['ai']:.2f}",
                           f"{b['gflops'] / scal:.0%}", f"{b['gflops'] / simd:.0%}", f"{b['gflops'] / roof:.0%}",
                           "memory" if b["ai"] * bw["DRAM"] < simd else "compute", f"{b['reread_gbs']:.0f}"])
        if tr: P(table(["K×N", "s", "k", "ms", "GFLOP/s", "AI", "% scalar", "% SIMD", "% roof", "roof is", "re-read GB/s"], tr) + "\n")
        else: P("(adds_per_row() not implemented: no roofline numbers)\n")

    # ---------------------------------------------------------------- 4. k sweep
    P("## 4. k sweep: ms / ns per add\n")
    P("If ns/add rises with k, the extra time goes into loop overhead / memory, not into additions.\n")
    for k in kernels:
        ks = sorted({r["k"] for r in rows if r["kernel"] == k}, key=lambda x: int(x) if x.isdigit() else 0)
        P(f"`{k}`\n")
        tr = []
        for sh in shapes:
            for s in sps:
                cells = {r["k"]: r for r in rows if r["kernel"] == k and (r["M"], r["K"], r["N"]) == sh and r["s"] == s}
                if not cells: continue
                bk = best[(k, sh, s)]["k"]
                tr.append([f"{sh[1]}×{sh[2]}", f"{s:.0%}"] +
                          [("**" if kk == bk else "") + f"{cells[kk]['ms']:.1f} / {cells[kk]['ns_add']:.2f}" + ("**" if kk == bk else "")
                           if kk in cells else "-" for kk in ks])
        P(table(["K×N", "s"] + [f"k={kk}" for kk in ks], tr) + "\n")

    # ---------------------------------------------------------------- 5. insights
    P("## 5. Insights\n")
    for k in kernels:
        bs = [b for (kk, _, _), b in best.items() if kk == k and b["gflops"] == b["gflops"]]
        if not bs: continue
        fs = gmean([b["gflops"] / scal for b in bs]); fv = gmean([b["gflops"] / simd for b in bs])
        fr = gmean([b["gflops"] / min(simd, b["ai"] * bw["DRAM"]) for b in bs])
        mem = sum(b["ai"] * bw["DRAM"] < simd for b in bs)
        rr = max(bs, key=lambda b: b["reread_gbs"])
        P(f"- `{k}`: {fs:.0%} of the scalar add peak, {fv:.0%} of the SIMD peak, {fr:.0%} of the roofline (geomean). "
          f"{mem}/{len(bs)} points sit left of the ridge (DRAM-bound roof).")
        if fr < 0.25:
            P(f"  - Far below every ceiling: neither DRAM bandwidth nor the add units are the limit. The time goes into "
              f"latency: index loads, gathers, short loops with data-dependent trip counts (branch mispredicts).")
        if rr["reread_gbs"] > 0.5 * bw["L2"]:
            P(f"  - Format re-read traffic reaches {rr['reread_gbs']:.0f} GB/s ({rr['K']}×{rr['N']}, s={rr['s']:.0%}), "
              f"{rr['reread_gbs'] / bw['L2']:.0%} of L2 bandwidth: streaming the format once per row of X is a real cost. "
              f"Reading it once per block (block-outer loop, M-tile inner) removes it.")
        rising = 0; total = 0
        for sh in shapes:
            for s in sps:
                cs = sorted(((int(r["k"]), r["ns_add"]) for r in rows if r["kernel"] == k and r["k"].isdigit()
                             and (r["M"], r["K"], r["N"]) == sh and r["s"] == s))
                if len(cs) >= 3:
                    total += 1; rising += cs[-1][1] > 1.5 * cs[0][1]
        if total and rising > total / 2:
            P(f"  - ns/add rises strongly with k in {rising}/{total} cases: larger k saves additions but each one costs more.")
    P("")

    # ---------------------------------------------------------------- plots
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    ncol = min(2, len(shapes)); nrow = math.ceil(len(shapes) / ncol)

    # roofline
    label_points = len(kernels) <= 4
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.6 * nrow), squeeze=False)
    ai_all = [b["ai"] for b in best.values() if b["ai"] == b["ai"]] or [1.0]
    xlo, xhi = min(ai_all + [0.05]) / 3, max(ai_all + [simd / bw["L1"] * 4]) * 3
    for ax, sh in zip(axs.flat, shapes):
        xs = [xlo * (xhi / xlo) ** (i / 200) for i in range(201)]
        for name, ls in (("DRAM", "-"), ("L2", "--"), ("L1", ":")):
            ax.plot(xs, [min(simd, x * bw[name]) for x in xs], ls, color=INK2, lw=1.2)
            xr = xlo * 1.6
            ax.text(xr, xr * bw[name] * 1.12, f"{name} {bw[name]:.0f} GB/s", color=INK2, fontsize=7.5, rotation=0)
        ax.axhline(simd, color=INK2, lw=1.2)
        ax.axhline(scal, color=INK2, lw=1.0, ls="-.")
        # peak labels on the left, where there are no data points
        ax.text(xlo * 1.6, simd * 1.08, f"SIMD add peak {simd:.0f} GFLOP/s", ha="left", color=INK2, fontsize=7.5)
        ax.text(xlo * 1.6, scal * 1.08, f"scalar add peak {scal:.1f}", ha="left", color=INK2, fontsize=7.5)
        for k in kernels:
            pts = [best[(k, sh, s)] for s in sps if (k, sh, s) in best and best[(k, sh, s)]["gflops"] == best[(k, sh, s)]["gflops"]]
            if not pts: continue
            ax.scatter([p["ai"] for p in pts], [p["gflops"] for p in pts], s=42, color=color[k], marker=marker[k],
                       label=k, edgecolors="white", linewidths=1.0, zorder=3)
            for p in (pts if label_points else []):   # too many kernels: labels pile up, sparsity is in the tables
                ax.annotate(f"{p['s']:.0%}", (p["ai"], p["gflops"]), textcoords="offset points", xytext=(5, -3),
                            fontsize=7, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(xlo, xhi)
        lo = min([b["gflops"] for b in best.values() if b["gflops"] == b["gflops"]] + [scal]) / 4
        ax.set_ylim(lo, simd * 2)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_title(f"M={sh[0]}, K={sh[1]}, N={sh[2]}  ({T} thread{'s' if T > 1 else ''})", color=INK, fontsize=10)
        ax.set_xlabel("arithmetic intensity [add / byte, compulsory traffic]"); ax.set_ylabel("GFLOP/s (fp32 adds)")
    for ax in list(axs.flat)[len(shapes):]: ax.axis("off")
    fig.suptitle(f"SPR roofline, {brand} (best k per point" + (", label = sparsity)" if label_points else ")"), color=INK)
    shared_legend(fig, axs)
    fig.savefig(os.path.join(a.out, "roofline.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # runtime vs sparsity
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.8 * nrow), squeeze=False)
    for ax, sh in zip(axs.flat, shapes):
        for k in kernels:
            pts = [(s, best[(k, sh, s)]["ms"]) for s in sps if (k, sh, s) in best]
            if pts:
                ax.plot([p[0] * 100 for p in pts], [p[1] for p in pts], ls=lstyle[k], marker=marker[k], color=color[k],
                        lw=1.6, ms=6, mec="white", mew=1.0, label=k)
        ax.set_yscale("log"); ax.grid(True, color=GRID, lw=0.6)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_title(f"M={sh[0]}, K={sh[1]}, N={sh[2]}", color=INK, fontsize=10)
        ax.set_xlabel("sparsity of W [%]"); ax.set_ylabel("median ms (best k), lower is better")
    for ax in list(axs.flat)[len(shapes):]: ax.axis("off")
    shared_legend(fig, axs)
    fig.savefig(os.path.join(a.out, "runtime.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    P("Plots: `roofline.png`, `runtime.png`")
    text = "\n".join(md) + "\n"
    open(os.path.join(a.out, "report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
