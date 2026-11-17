#!/usr/bin/env python3
"""Turn a bench_cpu CSV into a markdown summary aimed at iterating on SPR.

    python3 bench/summarize.py results/<run>/raw.csv > summary.md

Every kernel is grouped into a METHOD (RSR, RSR++, SSR, TCSC, ...) and each method is
represented by its best (kernel, k) per shape and sparsity, i.e. what its authors would
report after tuning. Every spr.* kernel is kept separately, so your versions compare
against each other too. Standard library only.
"""
import csv
import math
import os
import sys
from collections import defaultdict

# Every kernel belongs to one ALGORITHM at one engineering LEVEL ("scalar" or an optimized ISA).
ORDER = ["RSR · scalar", "RSR · RSR-core", "RSR++ · scalar", "SSR · scalar", "SSR · AVX2",
         "TCSC · scalar", "TCSC · AVX2", "TCSC · AVX-512", "Dense · scalar", "Dense · BLAS"]


def method(kernel):
    if kernel.startswith("spr.v8."): return "spr.v8"  # tuning selects the variant and k
    if kernel.startswith("spr."): return kernel
    if kernel in ("rsr.rsr_inference", "rsr.vector_rsr_inference"): return "RSR · scalar"
    if kernel in ("rsr.rsr_pp_inference", "rsr.vector_rsr_pp_inference"): return "RSR++ · scalar"
    if kernel.startswith("rsrcore."): return "RSR · RSR-core"
    if kernel == "ssr.ssr_inference_patched": return "SSR · scalar"
    if kernel.startswith("ssr.ssr_inference_float_vec"): return "SSR · AVX2"
    if kernel.startswith("tcsc.") and "Direct" in kernel: return "Dense · upstream"
    if kernel.startswith("tcsc."):
        if "AVX512" in kernel: return "TCSC · AVX-512"
        return "TCSC · AVX2" if "AVX2" in kernel else "TCSC · scalar"
    return {"lib.dense_naive": "Dense · scalar", "lib.dense_blas": "Dense · BLAS", "lib.eigen_dense": "Dense · Eigen",
            "lib.eigen_sparse": "Eigen-sparse · scalar"}.get(kernel, kernel)


def gmean(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(path):
    rows = list(csv.DictReader(open(path)))
    bad = [r for r in rows if r["ok"] != "1"]
    rows = [r for r in rows if r["ok"] == "1" and float(r["max_err"]) >= 0 and float(r["median_ms"]) > 0]
    if not rows:
        raise ValueError("no verified timings to summarize")
    for r in rows:
        r["ms"] = float(r["median_ms"]); r["s"] = float(r["sparsity"]); r["meth"] = method(r["kernel"])
        r["shape"] = (int(r["M"]), int(r["K"]), int(r["N"]))
        r["adds"] = float(r.get("adds_per_row") or 0)
        r["kv"] = int(r["param"][2:]) if r["param"].startswith("k=") else None

    shapes = sorted({r["shape"] for r in rows}, key=lambda t: t[1] * t[2])
    sps = sorted({r["s"] for r in rows})
    best = {}                                    # (shape, s, method) -> fastest row
    for r in rows:
        key = (r["shape"], r["s"], r["meth"])
        if key not in best or r["ms"] < best[key]["ms"]: best[key] = r
    meths = {m for (_, _, m) in best}
    sprs = sorted(m for m in meths if m.startswith("spr."))
    bases = [m for m in ORDER if m in meths] + sorted(meths - set(ORDER) - set(sprs))
    fmt_s = lambda s: f"{100 * s:.0f}%"
    nnz = lambda sh, s: (1 - s) * sh[1] * sh[2]          # additions/row of X for TCSC (one per non-zero)
    lvl = lambda m: "scalar" if m.endswith("scalar") else "optimized"
    r0 = rows[0]

    P = print
    P("# SPR benchmark summary\n")
    P(f"- machine `{r0['machine']}`, compiler `{r0['compiler']}`, threads **{r0['threads']}**, tag `{r0['tag']}`")
    P(f"- shapes (M, K, N): {', '.join(map(str, shapes))}; sparsities: {', '.join(map(fmt_s, sps))}")
    commit = os.path.join(os.path.dirname(path), "commit.txt")
    if os.path.exists(commit): P(f"- code: `{open(commit).read().split()[0]}`" + (" (+ uncommitted changes)" if len(open(commit).read().split()) > 1 else ""))
    P("- each method = its fastest (kernel, k) per shape and sparsity. *scalar* = plain C++, the same engineering "
      "level as `spr.naive`; the others are each algorithm's most optimized published kernel. On Apple Silicon the "
      "AVX2 / AVX-512 kernels run through a NEON shim (AVX-512 timings are meaningless there); RSR-core uses native NEON.")
    if bad: P(f"- **{len(bad)} configurations gave WRONG results and were not timed:** " + ", ".join(sorted({b['kernel'] + ' ' + b['param'] for b in bad})))

    # ---------------------------------------------------------------- 1. headline
    for sp in sprs:
        P(f"\n## 1. `{sp}`: speed-up vs each method\n")
        P("baseline time / SPR time: **> 1 means SPR is faster**. Geometric mean over shapes.\n")
        hdr = ["method", "level"] + [fmt_s(s) for s in sps] + ["all", "wins"]
        out = []
        for b in bases:
            cells, allv = [], []
            for s in sps:
                v = [best[(sh, s, b)]["ms"] / best[(sh, s, sp)]["ms"] for sh in shapes if (sh, s, b) in best and (sh, s, sp) in best]
                allv += v
                cells.append(f"{gmean(v):.2f}" if v else "–")
            out.append([b, lvl(b)] + cells + [f"**{gmean(allv):.2f}**", f"{sum(x > 1 for x in allv)}/{len(allv)}"])
        P(table(hdr, out))

    # ---------------------------------------------------------------- 2. raw best times
    P("\n## 2. Best time per method: median ms (best k)\n")
    for sh in shapes:
        P(f"\n**M={sh[0]}, K={sh[1]}, N={sh[2]}**\n")
        hdr = ["s"] + sprs + bases
        out = []
        for s in sps:
            line = [fmt_s(s)]
            for m in sprs + bases:
                r = best.get((sh, s, m))
                line.append("–" if not r else f"{r['ms']:.2f}" + (f" ({r['param']})" if r["kv"] else ""))
            out.append(line)
        P(table(hdr, out))

    # ---------------------------------------------------------------- 3. SPR k-sweep diagnostics
    for sp in sprs:
        P(f"\n## 3. `{sp}` diagnostics: k sweep\n")
        P("- **adds/row**: additions `run()` performs per row of X (`adds_per_row()`); **work vs TCSC** = adds/row ÷ nnz (TCSC adds each non-zero once)")
        P("- **ns/add**: time per addition. If it rises with k while adds/row falls, the time is going into memory / loop overhead, not arithmetic")
        P("- **GOP/s** = dense-equivalent 2·M·K·N / time")
        for sh in shapes:
            P(f"\n**M={sh[0]}, K={sh[1]}, N={sh[2]}**\n")
            hdr = ["s", "k", "ms", "adds/row (M)", "work vs TCSC", "ns/add", "GOP/s", "bits/weight", "prep ms"]
            out = []
            for s in sps:
                cfg = sorted((r for r in rows if r["meth"] == sp and r["shape"] == sh and r["s"] == s), key=lambda r: r["kv"] or 0)
                fastest = min(cfg, key=lambda r: r["ms"]) if cfg else None
                for r in cfg:
                    a = r["adds"]
                    out.append([fmt_s(s), r["param"][2:] + (" ★" if r is fastest else ""), f"{r['ms']:.2f}",
                                f"{a / 1e6:.2f}" if a else "?", f"{a / nnz(sh, s):.2f}" if a and nnz(sh, s) else "–",
                                f"{r['ms'] * 1e6 / (sh[0] * a):.3f}" if a else "?", f"{2 * sh[0] * sh[1] * sh[2] / (r['ms'] * 1e6):.1f}",
                                f"{float(r['bits_per_weight']):.1f}", f"{float(r['prep_ms']):.0f}"])
            P(table(hdr, out))

    # ---------------------------------------------------------------- 4. algorithm vs engineering
    refs = [b for b in ("TCSC · scalar", "TCSC · AVX2") if b in meths]
    if refs:
        P("\n## 4. Algorithm vs engineering, against TCSC\n")
        P("TCSC does exactly one addition per non-zero, so its time per addition is a clean engineering yardstick.")
        P("- **work** = SPR adds / TCSC adds (< 1: the algorithm saves additions)")
        P("- **cost/add** = SPR ns per add / TCSC ns per add (> 1: each SPR addition is more expensive: the engineering gap)")
        P("- **net** = work × cost/add = SPR time / TCSC time (< 1: SPR wins)\n")
        for sp in sprs:
            hdr = ["shape", "s", "best k"] + [f"{x} {c}" for x in refs for c in ("work", "cost/add", "net")]
            out = []
            for sh in shapes:
                for s in sps:
                    r = best.get((sh, s, sp))
                    if not r or not r["adds"] or not nnz(sh, s): continue
                    line = [f"{sh[1]}×{sh[2]}", fmt_s(s), r["param"][2:]]
                    for x in refs:
                        t = best.get((sh, s, x))
                        if not t: line += ["–"] * 3; continue
                        w = r["adds"] / nnz(sh, s)
                        line += [f"{w:.2f}", f"{(r['ms'] / t['ms']) / w:.2f}", f"{r['ms'] / t['ms']:.2f}"]
                    out.append(line)
            P(f"`{sp}`\n"); P(table(hdr, out))

    # ---------------------------------------------------------------- 5. memory + preprocessing
    sh = shapes[-1]
    P(f"\n## 5. Memory and preprocessing (K={sh[1]}, N={sh[2]}, best config)\n")
    P("bits/weight of the compressed W + metadata (2.0 = plain 2-bit packing, 8 = int8) / preprocessing ms\n")
    hdr = ["method"] + [fmt_s(s) for s in sps]
    P(table(hdr, [[m] + [(lambda r: f"{float(r['bits_per_weight']):.1f} / {float(r['prep_ms']):.0f}" if r else "–")(best.get((sh, s, m))) for s in sps] for m in sprs + bases]))

    # ---------------------------------------------------------------- 6. automatic insights
    P("\n## 6. Insights\n")
    for sp in sprs:
        cases = []                               # (ratio vs fastest baseline, shape, s, baseline)
        for sh in shapes:
            for s in sps:
                if (sh, s, sp) not in best: continue
                for group in ("scalar", "any"):
                    cand = [b for b in bases if (sh, s, b) in best and (group == "any" or lvl(b) == "scalar")]
                    if not cand: continue
                    fb = min(cand, key=lambda b: best[(sh, s, b)]["ms"])
                    cases.append((group, best[(sh, s, fb)]["ms"] / best[(sh, s, sp)]["ms"], sh, s, fb))
        for group, label in (("scalar", "the fastest scalar baseline (same engineering level)"), ("any", "the fastest baseline overall")):
            c = [x for x in cases if x[0] == group]
            if not c: continue
            P(f"- `{sp}` vs {label}: geomean **{gmean([x[1] for x in c]):.2f}×**, faster in {sum(x[1] > 1 for x in c)}/{len(c)} cases.")
            w = min(c, key=lambda x: x[1])
            verdict = f"{w[4]} is {1 / w[1]:.2f}× faster than SPR" if w[1] < 1 else f"SPR still {w[1]:.2f}× faster than {w[4]}"
            P(f"  - worst case: K×N={w[2][1]}×{w[2][2]}, s={fmt_s(w[3])}: {verdict}.")
        # time vs work, k range
        ks = sorted({r["kv"] for r in rows if r["meth"] == sp and r["kv"]})
        mism, edge = [], []
        for sh in shapes:
            for s in sps:
                cfg = [r for r in rows if r["meth"] == sp and r["shape"] == sh and r["s"] == s and r["adds"]]
                if len(cfg) < 2: continue
                ft, fw = min(cfg, key=lambda r: r["ms"]), min(cfg, key=lambda r: r["adds"])
                if ft["kv"] != fw["kv"]:
                    mism.append(f"{sh[1]}×{sh[2]}@{fmt_s(s)}: fewest adds at k={fw['kv']} ({fw['ms'] * 1e6 / (sh[0] * fw['adds']):.2f} ns/add), "
                                f"fastest at k={ft['kv']} ({ft['ms'] * 1e6 / (sh[0] * ft['adds']):.2f} ns/add)")
                if ft["kv"] in (ks[0], ks[-1]) and len(ks) > 2: edge.append(f"{sh[1]}×{sh[2]}@{fmt_s(s)}→k={ft['kv']}")
        if mism:
            P(f"- `{sp}`: in {len(mism)} cases the k with the fewest additions is NOT the fastest, so time doesn't follow work "
              "(possible explanations include memory traffic and loop overhead; profiling is needed):")
            for m in mism[:4]: P(f"  - {m}")
        if edge: P(f"- `{sp}`: best k is at the edge of the sweep {ks} in: {', '.join(edge)}. Widen the k range there.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_cpu.csv")
