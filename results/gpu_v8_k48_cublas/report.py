"""Validate raw samples and plot cuBLAS time divided by SPR time."""
import collections
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from plot_gpu_v8_campaign import Canvas, GRID, INK, MUTED, PALETTE, write_pdf

out = Path(sys.argv[1])
def rows(name):
    with (out / name).open() as f:
        return list(csv.DictReader(f))

def key(r):
    return tuple(int(r[n]) for n in ("M", "K", "N")) + (float(r["sparsity"]), r["kernel"])

samples = collections.defaultdict(list)
for r in rows("samples.csv"):
    samples[key(r)].append(float(r["ms"]))
times = {}
for r in rows("summary.csv"):
    case = key(r)
    a = samples[case]
    t = float(r["median_ms"])
    if (case in times or r["ok"] != "1" or len(a) != int(r["rounds"])
            or not a or not math.isclose(statistics.median(a), t, rel_tol=1e-5)):
        raise ValueError(f"Incomplete or inconsistent measurement: {case}")
    times[case] = t
shapes = [(1,4096,14336),(256,1024,4096),(512,4096,4096),(4096,4096,14336),
          (16384,4096,14336),(100000,4096,14336)]
fractions = (.50,.90,.99)
ratios = {}
with (out / "comparison.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["M","K","N","zero_fraction","v8_k48_ms","cublas_ms","cublas_ms_over_v8_ms"])
    for shape in shapes:
        for s in fractions:
            v = times[(*shape,s,"spr.v8.wide.cuda")]
            c = times[(*shape,s,"lib.cublas.cuda")]
            ratios[shape,s] = c / v
            w.writerow([*shape,s,v,c,c/v])

canvas = Canvas()
canvas.text(40,786,"SPR v8 (k=48) versus dense cuBLAS SGEMM",19,bold=True)
canvas.text(40,759,"RTX 5060 Ti  |  seed 42  |  synchronized inference, excluding preparation and transfers",10,MUTED)
low = min(-1, math.floor(math.log2(min(ratios.values()))))
high = max(1, math.ceil(math.log2(max(ratios.values()))))
def x(value):
    return 184 + 350*(math.log2(value)-low)/(high-low)
for power in range(low, high+1):
    pos = x(2**power)
    canvas.line(pos,180,pos,667,INK if power==0 else GRID,1.2 if power==0 else 0.5)
    canvas.text(pos,165,f"{2**power:.3g}x",8,align="center")
for i,s in enumerate(fractions):
    canvas.marker(192+100*i,705,PALETTE[3+i],i)
    canvas.text(201+100*i,701,f"{100*s:.0f}% zeros",10)
for j,shape in enumerate(shapes):
    y=630-83*j
    canvas.text(40,y+4,f"M = {shape[0]:,}",12,bold=True)
    canvas.text(40,y-12,f"K {shape[1]:,}   N {shape[2]:,}",9,MUTED)
    for i,s in enumerate(fractions):
        canvas.marker(x(ratios[shape,s]),y+9*(1-i),PALETTE[3+i],i)
canvas.text(184,134,"cuBLAS median / v8 median (log scale)",11)
canvas.text(40,91,"Above 1x: v8 is faster. Below 1x: cuBLAS is faster.",11)
write_pdf(out/"speedups.pdf",[canvas])
print(f"{len(times)} checked configurations, {sum(map(len,samples.values()))} timing samples")
