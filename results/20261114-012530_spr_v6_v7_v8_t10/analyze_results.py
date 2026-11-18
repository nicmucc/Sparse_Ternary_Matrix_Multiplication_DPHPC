"""Interpret the saved timing data; does not execute any kernels or checks."""
import csv
import json
import math
from pathlib import Path

out = Path(__file__).resolve().parent


def read(name):
    rows = list(csv.DictReader((out / name).open()))
    for r in rows:
        for key in ["M", "K", "N"]:
            r[key] = int(r[key])
        for key in ["sparsity", "v6_ms", "v7_ms", "v8_ms", "v8_vs_v6", "v8_vs_v7"]:
            r[key] = float(r[key])
    return rows


def gm(rows, field):
    return math.exp(sum(math.log(r[field]) for r in rows) / len(rows))


rows = read("comparison.csv")
regular = [r for r in rows if r["pattern"] == "iid" and r["N"] != 32]
clustered = [r for r in rows if r["pattern"] == "column_clustered"]
keys = {(r["M"], r["K"], r["N"], r["sparsity"]) for r in clustered}
matched = [r for r in regular if (r["M"], r["K"], r["N"], r["sparsity"]) in keys]
tuned = [r for r in read("tuned_comparison.csv") if r["pattern"] == "iid" and r["N"] != 32]
gain6, gain7 = gm(regular, "v8_vs_v6"), gm(regular, "v8_vs_v7")
text = ["# Interpretation of the V8 rectangle-scheduling benchmark", "",
        "All results use 10 requested threads on the local Apple M4 Pro. The main comparison fixes B=32 and k=3.",
        "Correctness checks and benchmark reference verification were skipped as requested; these are timing measurements only.", "",
        f"Across the 96 regular-matrix/sparsity workloads, V8/V6 is {gain6:.3f}× and V8/V7 is {gain7:.3f}× (geometric means).",
        f"V8's average runtime relative to V6 is {1/gain6:.3f}×: " +
        (f"{(1/gain6-1)*100:.1f}% longer." if gain6 < 1 else f"{(1-1/gain6)*100:.1f}% shorter."),
        "A speedup above 1 means V8 is faster. A speedup below 1 means V8 takes longer.", "",
        "| M | SIMD tiles | V8/V6 | V8/V7 | V8 runtime versus V6 |",
        "|---|---|---|---|---|"]
for M in sorted({r["M"] for r in regular}):
    rs = [r for r in regular if r["M"] == M]
    a, b = gm(rs, "v8_vs_v6"), gm(rs, "v8_vs_v7")
    text.append(f"| {M} | {(M+31)//32} | {a:.3f}× | {b:.3f}× | {1/a:.3f}× as long |")
text += ["", "V7 can only use one worker at M=1/32 and at most two workers at M=64. V8 can split work across groups as well. Its large gains over V7 in these cases primarily show recovered parallelism; V6 already offers that parallelism.",
         "For regular matrices, the rectangular scheduler does not establish a general improvement over V6 in this run. V8 introduces dynamic dispatch and coarser work units. Those are plausible costs, but the timing data does not isolate them from cache behavior or core placement.",
         "", "## Sparse-pattern diagnostic with matched shapes", "",
         "These two rows use matching M, K, N, target sparsity levels and seeds, with different constructions of the sparse pattern.", "",
         "| Pattern | Workloads | V8/V6 | V8/V7 |", "|---|---|---|---|"]
for label, data in [("Independent random zeros", matched), ("Contiguous zero output columns", clustered)]:
    text.append(f"| {label} | {len(data)} | {gm(data,'v8_vs_v6'):.3f}× | {gm(data,'v8_vs_v7'):.3f}× |")
text += ["", "The clustered case intentionally makes group costs unequal. Differences here indicate sensitivity to work distribution; they do not isolate the cost estimator from the dynamic scheduling policy.",
         "", "| M | Clustered V8/V6 | Clustered V8/V7 |", "|---|---|---|"]
for M in sorted({r["M"] for r in clustered}):
    rs = [r for r in clustered if r["M"] == M]
    text.append(f"| {M} | {gm(rs,'v8_vs_v6'):.3f}× | {gm(rs,'v8_vs_v7'):.3f}× |")
text += ["", "## Sensitivity to k", "",
         "Matching k isolates scheduling; independently choosing each version's best observed k is an additional tuning comparison.", "",
         "| Choice | V8/V6 | V8/V7 |", "|---|---|---|"]
all_k = read("all_k_comparison.csv")
for k in [2, 3, 4]:
    rs = [r for r in all_k if r["pattern"] == "iid" and r["N"] != 32 and int(r["k"]) == k]
    text.append(f"| Shared k={k} | {gm(rs,'v8_vs_v6'):.3f}× | {gm(rs,'v8_vs_v7'):.3f}× |")
text.append(f"| Independently fastest observed k | {gm(tuned,'v8_vs_v6'):.3f}× | {gm(tuned,'v8_vs_v7'):.3f}× |")
text += ["", "## Narrow outputs: absolute runtime", "",
         "N=32, K=1024, k=3; geometric means over sparsities. Small absolute runtimes make thread overhead important.", "",
         "| M | V6 [µs] | V7 [µs] | V8 [µs] |", "|---|---|---|---|"]
for M in sorted({r["M"] for r in rows}):
    rs = [r for r in rows if r["pattern"] == "iid" and r["N"] == 32 and r["M"] == M]
    text.append(f"| {M} | {1000*gm(rs,'v6_ms'):.2f} | {1000*gm(rs,'v7_ms'):.2f} | {1000*gm(rs,'v8_ms'):.2f} |")
text += ["", "These are 31-round medians from one timing campaign and one seed per workload; small differences need repeated campaigns to establish reliability.",
         "", "[Measurement method and complete tables](report.md) · [Speedup plot](speedup.png) · [Runtime plot](runtime.png) · [Pattern plot](pattern_speedup.png) · [Roofline](roofline.png)"]
(out / "interpretation.md").write_text("\n".join(text) + "\n")
(out / "interpretation.json").write_text(json.dumps(dict(regular_v8_vs_v6=gain6, regular_v8_vs_v7=gain7,
    matched_iid_v8_vs_v6=gm(matched,"v8_vs_v6"), clustered_v8_vs_v6=gm(clustered,"v8_vs_v6"),
    clustered_v8_vs_v7=gm(clustered,"v8_vs_v7"), tuned_v8_vs_v6=gm(tuned,"v8_vs_v6")), indent=2) + "\n")
print("\n".join(text))
