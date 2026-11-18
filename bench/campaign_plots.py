"""Plot one completed CPU campaign from its saved cluster measurements."""
import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from campaign import METHODS


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def workload(r):
    return int(r["M"]), int(r["K"]), int(r["N"]), float(r["sparsity"])


def label(w):
    m, k, n, s = w
    return f"M={m}  ·  {k}×{n}  ·  {s:.0%} zeros"


def plot(run, out):
    results = read(run/"evaluation.csv")
    context = json.loads((run/"context.json").read_text())
    cpu = next((line.split(":",1)[1].strip() for line in context["cpu"].splitlines()
                if line.startswith("Model name:")), context["hostname"])
    mode = context["settings"]
    peaks = json.loads((run/"peaks.json").read_text())
    selected, fixed = defaultdict(dict), defaultdict(dict)
    for r in results:
        w, t, seed = workload(r), int(r["threads"]), int(r["seed"])
        for keep, group, key in [(r["selected"],selected,(w,t,r["method"])),
                                 (r["fixed_scaling"],fixed,(w,t))]:
            if keep == "1":
                if seed in group[key]:
                    raise ValueError(f"Duplicate evaluation: {key}, seed {seed}")
                group[key][seed] = r
    workloads = sorted({workload(r) for r in results})
    if len(workloads) != (8 if mode=="full" else 1):
        raise ValueError("Incomplete campaign workload coverage")
    threads = (1,2,4)
    for w in workloads:
        for t in threads:
            for group in [fixed[w,t]]+[selected[w,t,m] for m in METHODS if m!="RSR++" or t==1]:
                if set(group) != {43,44}:
                    raise ValueError("Missing held-out measurement")
    out.mkdir(parents=True, exist_ok=True)
    colors = ["#136a77","#4879b5","#b35c27","#76649f","#8d7065","#57834b"]

    def save(fig, name, title, note):
        fig.suptitle(title, x=.02, ha="left", weight="bold", fontsize=15)
        footer = f"{cpu}   |   {mode}   |   {note}"
        lines = textwrap.wrap(footer, width=int(fig.get_figwidth()*14))
        fig.text(.02,.012,"\n".join(lines), fontsize=8, color="#3b4850", va="bottom")
        footer_height = (.19*len(lines)+.12)/fig.get_figheight()
        fig.tight_layout(rect=(0,footer_height,1,.94))
        fig.savefig(out/f"{name}.png", dpi=145)
        plt.close(fig)

    # Independent x ranges let a slow serial baseline remain legible; every panel states ms.
    for batch in sorted({w[0] for w in workloads}):
        cases = [w for w in workloads if w[0]==batch]
        fig, axes = plt.subplots(len(cases),3,figsize=(14,2.55*len(cases)+1.5), squeeze=False)
        for idx,w in enumerate(cases):
            for ax,t in zip(axes[idx],threads):
                names = [m for m in METHODS if t==1 or m!="RSR++"]
                for y,name in enumerate(names):
                    pair = selected[w,t,name].values()
                    v = median(float(r["median_ms"]) for r in pair)
                    all_ms = [x for r in pair for x in json.loads(r["samples_ms"])]
                    lo, hi = np.percentile(all_ms,[25,75])
                    ax.barh(y,v,color=colors[METHODS.index(name)],height=.58)
                    ax.plot([lo,hi],[y,y],color="#20333b",lw=1.6)
                    ax.annotate(f"{v:.3g}",(max(v,hi),y),xytext=(4,0),textcoords="offset points",
                                va="center",fontsize=8)
                ax.set_yticks(range(len(names)),names)
                ax.invert_yaxis()
                ax.set_xlim(0,ax.get_xlim()[1]*1.23)
                ax.set_title(f"{w[1]}×{w[2]} · {w[3]:.0%} zeros · {t}T",fontsize=10)
                ax.set_xlabel("Inference time (ms)")
        save(fig,f"runtime_M{batch}",f"Inference time   /   M = {batch}",
             "Labels: median of seed medians. Whiskers: interquartile range of 14 timed rounds. Independent axes.")

    fig,ax = plt.subplots(figsize=(9,5))
    ax.plot(threads,threads,"--",color="#999",label="Ideal")
    for w in workloads:
        ratios = [[float(fixed[w,1][seed]["median_ms"])/float(fixed[w,t][seed]["median_ms"])
                   for seed in (43,44)] for t in threads]
        v = np.asarray(ratios)
        ax.plot(threads,v.mean(axis=1),marker="o",label=label(w))
        ax.fill_between(threads,v.min(axis=1),v.max(axis=1),alpha=.08)
    ax.set(xticks=threads,xlabel="Requested OpenMP threads",ylabel="Speedup (×)",ylim=(0,None))
    ax.legend(fontsize=8,loc="center left",bbox_to_anchor=(1,.5))
    save(fig,"scaling","SPR scaling with the one-thread configuration",
         "Same input and tiling at every thread count. Band spans paired seed speedups.")

    fig,ax = plt.subplots(figsize=(8,2.7+.48*len(workloads)))
    values = [[median(100*float(r["adds_per_row"])*w[0]/(float(r["median_ms"])*1e6)
                      /peaks[str(t)]["peak_simd_gflops"] for r in selected[w,t,"SPR v8"].values())
               for t in threads] for w in workloads]
    im = ax.imshow(values,aspect="auto",cmap="Blues",vmin=0,vmax=max(map(max,values)) or 1)
    for i,row in enumerate(values):
        for j,v in enumerate(row):
            ax.text(j,i,f"{v:.1f}%",ha="center",va="center",color="white" if im.norm(v)>.6 else "black")
    ax.set_yticks(range(len(workloads)),[label(w) for w in workloads])
    ax.set_xticks(range(3),[f"{t} thread"+("s" if t>1 else "") for t in threads])
    save(fig,"peak","SPR throughput as a percentage of measured SIMD FP32 add peak",
         "Source-level FP32 adds/subtracts per second divided by the peak at the same requested thread count.")

    # Pair by evaluation seed; a factor above one favors SPR.
    fig,ax = plt.subplots(figsize=(9,2.8+.55*len(workloads)))
    rivals = ["SSR","TCSC","RSR-core","OpenBLAS"]
    ratios = [[median(float(selected[w,4,m][seed]["median_ms"]) /
                      float(selected[w,4,"SPR v8"][seed]["median_ms"]) for seed in (43,44))
               for m in rivals] for w in workloads]
    # The color crosses neutral at parity; labels give the exact speedup.
    relative = np.asarray(ratios)-1
    limit = max(abs(relative).max(), 0.01)
    im = ax.imshow(relative, aspect="auto", cmap="BrBG", vmin=-limit, vmax=limit)
    for i,j in np.ndindex(relative.shape):
        ax.text(j,i,f"{ratios[i][j]:.2f}×",ha="center",va="center",
                color="white" if abs(relative[i,j])/limit>.65 else "black")
    ax.set_yticks(range(len(workloads)),[label(w) for w in workloads])
    ax.set_xticks(range(len(rivals)),rivals)
    save(fig,"speedup","SPR speedup against each optimized method at four threads",
         "Same-workload, same-seed baseline time / SPR time. Above 1× favors SPR.")

    # Same-thread tuned comparisons, distinct from fixed-configuration scaling.
    for batch in sorted({w[0] for w in workloads}):
        cases = [w for w in workloads if w[0] == batch]
        cols = min(2, len(cases))
        rows = math.ceil(len(cases)/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6.3*cols, 3.7*rows+1.3),
                                 squeeze=False, sharey=True)
        maximum = 1.0
        for ax, w in zip(axes.flat, cases):
            ax.axhline(1, color="#666666", linestyle="--", linewidth=1)
            for method in rivals:
                values = np.array([
                    [float(selected[w,t,method][seed]["median_ms"]) /
                     float(selected[w,t,"SPR v8"][seed]["median_ms"])
                     for seed in (43,44)] for t in threads])
                color = colors[METHODS.index(method)]
                ax.plot(threads, np.median(values, axis=1), marker="o",
                        linewidth=2, color=color, label=method)
                ax.fill_between(threads, values.min(axis=1), values.max(axis=1),
                                color=color, alpha=.12)
                maximum = max(maximum, values.max())
            ax.set(xticks=threads, xlabel="OpenMP threads (both methods)",
                   ylabel="Baseline time / SPR time (x)",
                   title=f"{w[1]} x {w[2]} | {w[3]:.0%} zeros")
            ax.grid(axis="y", alpha=.22)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes.flat:
            ax.set_ylim(0, maximum*1.12)
        for ax in list(axes.flat)[len(cases):]:
            ax.set_visible(False)
        axes[0,0].legend(ncol=2, fontsize=9, loc="upper left")
        save(fig, f"speedup_lines_M{batch}",
             f"SPR speedup against optimized baselines / M = {batch}",
             "Above 1x favors SPR; below 1x favors the baseline. Each method tuned at each thread count. "
             "Lines: median paired seed ratios; bands: range over two seeds (not confidence intervals). "
             "RSR++ is serial-only and remains in the runtime plots.")

    coarse = [r for r in read(run/"tuning.csv") if r["phase"]=="coarse" and r["threads"]=="4"]
    cols = min(2,len(workloads))
    fig, axes = plt.subplots(math.ceil(len(workloads)/cols),cols,figsize=(6*cols,3.1*math.ceil(len(workloads)/cols)+1),
                             squeeze=False)
    for ax,w in zip(axes.flat,workloads):
        data = [r for r in coarse if workload(r)==w]
        bs, ks = [sorted({int(r[key]) for r in data}) for key in ("B","k")]
        arr = np.array([[min(float(r["median_ms"]) for r in data if int(r["B"])==b and int(r["k"])==k)
                         for k in ks] for b in bs])
        im=ax.imshow(arr,aspect="auto",cmap="YlGnBu")
        for i,j in np.ndindex(arr.shape):
            ax.text(j,i,f"{arr[i,j]:.3g}",ha="center",va="center",color="white" if im.norm(arr[i,j])>.6 else "black")
        choice = selected[w,4,"SPR v8"][43]
        ax.text(.5,-.31, "Selected: " + "   ".join(f"{key}={choice[key]}" for key in ("k","B","Kc","G")),
                transform=ax.transAxes, ha="center", fontsize=8)
        ax.set(xticks=range(len(ks)),xticklabels=ks,yticks=range(len(bs)),yticklabels=bs,
               xlabel="Pattern width k",ylabel="Batch tile B",title=label(w))
    save(fig,"tiling","SPR tile search at four threads",
         "Each cell: fastest coarse-grid median (ms) across Kc and G for that B and k. Color scale per panel.")


    # Individual vector panels for presentations; no composite-image cropping.
    panels = out/"slide_panels"
    panels.mkdir(exist_ok=True)
    def panel_save(fig, name):
        fig.tight_layout(pad=.7)
        fig.savefig(panels/f"{name}.pdf", bbox_inches="tight")
        fig.savefig(panels/f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    with plt.rc_context({"font.size":13, "axes.titlesize":14, "axes.labelsize":13,
                         "xtick.labelsize":12, "ytick.labelsize":12, "legend.fontsize":11}):
        for w in workloads:
            tag = f"M{w[0]}_K{w[1]}_N{w[2]}_s{int(w[3]*100)}"
            fig, ax = plt.subplots(figsize=(5.0,3.6))
            ax.axhline(1, color="#555", ls="--", lw=1.2)
            maximum = 1
            for method in rivals:
                v = np.array([[float(selected[w,t,method][seed]["median_ms"])/
                               float(selected[w,t,"SPR v8"][seed]["median_ms"])
                               for seed in (43,44)] for t in threads])
                color = colors[METHODS.index(method)]
                ax.plot(threads, np.median(v,axis=1), marker="o", lw=2.4, color=color, label=method)
                ax.fill_between(threads,v.min(axis=1),v.max(axis=1),color=color,alpha=.13)
                maximum = max(maximum,v.max())
            ax.set(xticks=threads,xlabel="Threads",ylabel="Baseline time / SPR time",ylim=(0,maximum*1.3),
                   title=f"{w[1]} x {w[2]} | {w[3]:.0%} zeros")
            ax.legend(ncol=2,loc="upper left",frameon=False)
            ax.grid(axis="y",alpha=.22)
            ax.spines[["top","right"]].set_visible(False)
            panel_save(fig,"speedup_"+tag)

            data = [r for r in coarse if workload(r)==w]
            bs, ks = [sorted({int(r[key]) for r in data}) for key in ("B","k")]
            arr = np.array([[min(float(r["median_ms"]) for r in data if int(r["B"])==b and int(r["k"])==k)
                             for k in ks] for b in bs])
            fig, ax = plt.subplots(figsize=(5.0,3.6))
            im=ax.imshow(arr,aspect="auto",cmap="YlGnBu")
            for i,j in np.ndindex(arr.shape):
                ax.text(j,i,f"{arr[i,j]:.3g}",ha="center",va="center",
                        color="white" if im.norm(arr[i,j])>.6 else "black")
            ax.set(xticks=range(len(ks)),xticklabels=ks,yticks=range(len(bs)),yticklabels=bs,
                   xlabel="Pattern width k",ylabel="Batch tile B",title=f"{w[1]} x {w[2]} | {w[3]:.0%} zeros")
            panel_save(fig,"tiling_"+tag)

            fig, ax = plt.subplots(figsize=(5.0,3.6))
            names = [m for m in METHODS if m!="RSR++"]
            for i,m in enumerate(names):
                v=[float(selected[w,4,m][seed]["median_ms"]) for seed in (43,44)]
                ax.barh(i,median(v),color=colors[METHODS.index(m)])
                ax.text(max(v),i,f" {median(v):.3g}",va="center",fontsize=12)
            ax.set(yticks=range(len(names)),yticklabels=names,xlabel="Kernel time (ms)",
                   title=f"{w[1]} x {w[2]} | {w[3]:.0%} zeros | 4 threads")
            ax.invert_yaxis(); ax.set_xlim(0,ax.get_xlim()[1]*1.2)
            panel_save(fig,"runtime_"+tag)

        for batch in sorted({w[0] for w in workloads}):
            cases=[w for w in workloads if w[0]==batch]
            fig, ax=plt.subplots(figsize=(5.0,3.6))
            ax.plot(threads,threads,"--",color="#888",label="Ideal")
            for w in cases:
                v=np.array([[float(fixed[w,1][seed]["median_ms"])/float(fixed[w,t][seed]["median_ms"])
                             for seed in (43,44)] for t in threads])
                ax.plot(threads,np.median(v,axis=1),marker="o",label=f"{w[1]}x{w[2]}, {w[3]:.0%} zeros")
                ax.fill_between(threads,v.min(axis=1),v.max(axis=1),alpha=.1)
            ax.set(xticks=threads,xlabel="Threads",ylabel="One-thread time / time",ylim=(0,5.5),title=f"M = {batch}")
            ax.legend(ncol=2,fontsize=9,loc="upper left",frameon=False); ax.grid(alpha=.2)
            panel_save(fig,f"scaling_M{batch}")
            fig, ax=plt.subplots(figsize=(5.0,3.6))
            for w in cases:
                v=[median(100*float(r["adds_per_row"])*w[0]/(float(r["median_ms"])*1e6)/
                          peaks[str(t)]["peak_simd_gflops"] for r in selected[w,t,"SPR v8"].values()) for t in threads]
                ax.plot(threads,v,marker="o",label=f"{w[1]}x{w[2]}, {w[3]:.0%} zeros")
            ax.set(xticks=threads,xlabel="Threads",ylabel="Measured SIMD add peak (%)",title=f"M = {batch}")
            ax.set_ylim(0,ax.get_ylim()[1]*1.4)
            ax.legend(ncol=2,fontsize=9,loc="upper left",frameon=False); ax.grid(alpha=.2)
            panel_save(fig,f"peak_M{batch}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--out",type=Path)
    a = p.parse_args()
    plot(a.run,a.out or a.run/"plots")
