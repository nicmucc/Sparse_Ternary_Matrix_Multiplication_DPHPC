"""Validate this run and summarize V5's position relative to measured roofs.

Run after bench/spr_threads.py completes. No measurements are rerun or altered.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

out = Path(__file__).resolve().parent
rows = list(csv.DictReader((out / "raw.csv").open()))
assert len(rows) == 2688, len(rows)
assert all(r["ok"] == "1" and int(r["rounds"]) == 11 for r in rows)
assert all(0 <= float(r["max_err"]) <= 2e-3 * math.sqrt(int(r["K"])) for r in rows)

best = {}
for r in rows:
    key = tuple(r[k] for k in ("threads", "kernel", "M", "K", "N", "sparsity"))
    if key not in best or float(r["median_ms"]) < float(best[key]["median_ms"]):
        best[key] = r

def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))

details = []
for key, r in best.items():
    t, kernel, M, K, N, s = key
    if kernel != "spr.v5.b32":
        continue
    pk = json.loads((out / f"t{int(t):02}" / "peaks.json").read_text())
    work = float(r["adds_per_row"]) * int(M)
    traffic = float(r["bytes"]) + 4 * int(M) * (int(K) + int(N))
    perf = work / float(r["median_ms"]) / 1e6
    ai = work / traffic
    roof = min(pk["peak_simd_gflops"], ai * pk["bw_dram_gbs"])
    v4 = best[(t, "spr.v4.b32", M, K, N, s)]
    details.append(dict(threads=int(t), M=int(M), K=int(K), N=int(N), sparsity=float(s),
                        param=r["param"], inference_ms=float(r["median_ms"]), gflops=perf,
                        intensity_add_per_byte=ai, measured_simd_gflops=pk["peak_simd_gflops"],
                        measured_dram_gbs=pk["bw_dram_gbs"], roof_gflops=roof,
                        simd_utilization=perf / pk["peak_simd_gflops"], roof_utilization=perf / roof,
                        inference_speedup_vs_v4=float(v4["median_ms"]) / float(r["median_ms"])))
assert len(details) == 128
with (out / "roofline_positions.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(details[0]))
    writer.writeheader()
    writer.writerows(details)

lines = ["# Current V5 inference: roofline position", "",
         "Fresh measurements on the local Apple M4 Pro (10 performance + 4 efficiency cores).",
         "Each kernel uses its independently fastest measured block width for each workload.",
         "Ratios and utilization are geometric means across 16 shape/sparsity workloads.", "",
         "| Threads | Inference speedup vs serial V4 | Measured SIMD add peak [GFLOP/s] | Measured DRAM [GB/s] | % SIMD peak | % modeled roof | V5 GFLOP/s range |",
         "|---|---|---|---|---|---|---|"]
for t in sorted({r["threads"] for r in details}):
    rs = [r for r in details if r["threads"] == t]
    first = rs[0]
    lines.append(f"| {t} | {gmean([r['inference_speedup_vs_v4'] for r in rs]):.2f}× | "
                 f"{first['measured_simd_gflops']:.1f} | {first['measured_dram_gbs']:.1f} | "
                 f"{100 * gmean([r['simd_utilization'] for r in rs]):.1f}% | "
                 f"{100 * gmean([r['roof_utilization'] for r in rs]):.1f}% | "
                 f"{min(r['gflops'] for r in rs):.1f}–{max(r['gflops'] for r in rs):.1f} |")
lines += ["", "The roof is min(measured SIMD addition peak, intensity × measured DRAM read bandwidth).",
          "Intensity counts compulsory format + X + Y traffic, not hardware-counter cache traffic.",
          "Being below this roof alone does not establish which resource is the bottleneck.",
          "OpenMP scheduling, synchronization, allocations and intermediate-buffer traffic are included in inference time.",
          "Core placement is controlled by macOS; requested threads do not imply pinned performance cores.", "",
          "See `roofline_positions.csv` for individual workloads and `report.md` for fixed-k thread scaling."]
(out / "roofline_summary.md").write_text("\n".join(lines) + "\n")
validation = dict(configurations=len(rows), failed_configurations=0, inference_rounds=11,
                  maximum_absolute_error=max(float(r["max_err"]) for r in rows),
                  parallel_inference=True, roofline_points=len(details),
                  kernel_source_sha256=hashlib.sha256((out / "source/cpu/spr/spr_v5.cpp").read_bytes()).hexdigest())
(out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
print("\n".join(lines))
