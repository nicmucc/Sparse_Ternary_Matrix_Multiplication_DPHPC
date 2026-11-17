#!/usr/bin/env python3
"""Baseline tuning cache: the best (kernel, k) of every baseline method, per machine / threads / shape / sparsity.

The baselines never change between SPR iterations, so their k sweep is done ONCE (bench/run.sh does it
automatically when an entry is missing, or always with TUNE=1). Later runs only time each method's best
configuration, still interleaved with SPR.

    python3 bench/tune.py add raw_sweep.csv cache.csv               # store the best config of each method
    python3 bench/tune.py get cache.csv M K N s [all|scalar|opt]    # -> "=kernel@k,=kernel,..." ('' if missing)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from summarize import method  # noqa: E402  (same method grouping as the summary)

FIELDS = ["M", "K", "N", "sparsity", "method", "kernel", "k", "median_ms"]


def add(raw, cache):
    best = {}
    for r in csv.DictReader(open(raw)):
        if r["ok"] != "1" or float(r["median_ms"]) <= 0 or r["kernel"].startswith("spr."): continue
        key = (r["M"], r["K"], r["N"], float(r["sparsity"]), method(r["kernel"]))
        if key not in best or float(r["median_ms"]) < float(best[key]["median_ms"]): best[key] = r
    old = [r for r in csv.DictReader(open(cache))] if os.path.exists(cache) else []
    old = [r for r in old if (r["M"], r["K"], r["N"], float(r["sparsity"]), r["method"]) not in best]   # replace re-tuned entries
    with open(cache, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(old)
        for (M, K, N, s, m), r in sorted(best.items()):
            w.writerow({"M": M, "K": K, "N": N, "sparsity": s, "method": m, "kernel": r["kernel"],
                        "k": r["param"][2:] if r["param"].startswith("k=") else "-", "median_ms": r["median_ms"]})


def get(cache, M, K, N, s, level="all"):
    if not os.path.exists(cache): return ""
    out = []
    for r in csv.DictReader(open(cache)):
        if (r["M"], r["K"], r["N"]) != (M, K, N) or abs(float(r["sparsity"]) - float(s)) > 1e-9: continue
        scalar = r["method"].endswith("scalar")
        if level == "scalar" and not scalar or level == "opt" and scalar: continue
        out.append("=" + r["kernel"] + ("" if r["k"] == "-" else "@" + r["k"]))
    return ",".join(out)


if __name__ == "__main__":
    if sys.argv[1] == "add": add(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "get": print(get(*sys.argv[2:]))
    else: sys.exit(__doc__)
