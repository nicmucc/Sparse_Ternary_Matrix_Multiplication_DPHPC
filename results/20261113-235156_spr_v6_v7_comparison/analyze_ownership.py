"""Summarize tile ownership after the V6/V7 benchmark completes."""
import csv
import json
import math
from pathlib import Path

out = Path(__file__).resolve().parent
rows = list(csv.DictReader((out / "comparison.csv").open()))
regular = [r for r in rows if r["pattern"] == "iid" and int(r["N"]) != 32]


def mean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


conditions = []
for r in regular:
    t = int(r["threads"])
    tiles = (int(r["M"]) + 31) // 32
    category = ("underfilled" if tiles < t else
                "whole_tiles_in_both" if tiles % t == 0 else "enough_but_uneven")
    conditions.append(dict(M=r["M"], K=r["K"], N=r["N"], sparsity=r["sparsity"],
                           threads=t, tiles=tiles, condition=category,
                           v7_vs_v6=float(r["v7_vs_v6"]), v6_ms=float(r["v6_ms"]), v7_ms=float(r["v7_ms"])))
with (out / "ownership_conditions.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(conditions[0]))
    w.writeheader()
    w.writerows(conditions)

text = ["# Whole-tile ownership: V7 versus V6", "",
        "B=32. Identical weights, block width and prepared format within every comparison.",
        "Ratios below are geometric means over regular iid matrix shapes and sparsities; the first table excludes one-thread runs.", "",
        "| Scheduling situation | Workload points | V7/V6 speedup |",
        "|---|---|---|"]
summary = []
for category, label in [("underfilled", "Fewer tiles than requested threads"),
                        ("whole_tiles_in_both", "Tiles divide evenly across threads"),
                        ("enough_but_uneven", "Enough tiles, uneven division")]:
    rs = [r for r in conditions if r["condition"] == category and r["threads"] > 1]
    gain = mean([r["v7_vs_v6"] for r in rs])
    text.append(f"| {label} | {len(rs)} | {gain:.3f}× |")
    summary.append(dict(condition=category, points=len(rs), v7_vs_v6=gain))
text += ["", "Static V6 partitions a contiguous flattened (tile, group) iteration space. When the tile count divides evenly by the thread count, each static range can already contain complete tiles. V7 then has similar ownership while expressing it with a simpler loop.",
         "When there are 16 tiles and 10 threads, V7 assigns two tiles to some workers and one to others. V6 can split tiles at group boundaries, giving finer load balance.",
         "When there are fewer tiles than threads, V7 has no group-level parallelism to fill the remaining workers.",
         "These statements describe scheduling. Ratios alone do not establish whether cache misses or instruction overhead caused a measured difference.", "",
         "## Best observed V7 thread count by input size", "",
         "Based on regular-matrix geometric means at fixed block width. For M=32, V7's parallel-region condition is false and it runs serially at every requested thread setting.", "",
         "| M | Tiles | Best requested threads | V7 speedup vs its own 1T | V7/V6 at that setting |",
         "|---|---|---|---|---|"]
for M in sorted({int(r["M"]) for r in regular}):
    vals = []
    for t in sorted({int(r["threads"]) for r in regular}):
        rs = [r for r in regular if int(r["M"]) == M and int(r["threads"]) == t]
        vals.append((t, mean([float(r["v7_scaling"]) for r in rs]), mean([float(r["v7_vs_v6"]) for r in rs])))
    if M == 32:
        t, speed, gain = next(x for x in vals if x[0] == 1)
        label = "serial for all settings"
    else:
        t, speed, gain = max(vals, key=lambda x: x[1])
        label = str(t)
    text.append(f"| {M} | {(M + 31) // 32} | {label} | {speed:.2f}× | {gain:.2f}× |")
(out / "ownership_summary.md").write_text("\n".join(text) + "\n")
(out / "ownership_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("\n".join(text))
