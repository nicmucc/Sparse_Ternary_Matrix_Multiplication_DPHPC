#!/usr/bin/env python3
"""Benchmark V6/V9 without correctness checks, then plot and report.

Build bench_cpu and peaks first. Run with /usr/bin/python3.
Rebuild plots/report: --plots-only results/RUN. Timing is always sequential.
"""
import argparse
import csv
import datetime
import hashlib
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
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

KERNELS = ['spr.v6.b32', 'spr.v9.b32']
MS = [32, 64, 256, 320, 512]
SHAPES = [(1024, 4096), (4096, 1024), (2048, 8192)]

def gm(values):
    vals = list(values)
    return math.exp(statistics.mean(math.log(v) for v in vals))

def load(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def save(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def command(cmd, log, env):
    with log.open('a') as f:
        f.write('\n$ ' + ' '.join(map(str, cmd)) + '\n')
        f.flush()
        subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)

def summarize(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    ts = manifest['threads']
    rows = [r for t in ts for r in load(out / f't{t:02}/raw.csv')]
    expected = len(ts) * sum(len(j['sparsities'])*len(j.get('ks', manifest['ks'])) for j in manifest['jobs']) * 2
    if len(rows) != expected:
        raise RuntimeError(f'Incomplete benchmark: {len(rows)} != {expected}')
    # This validates data completeness only; correctness was deliberately skipped.
    assert all(float(r['max_err']) == -1 and r['ok'] == '1' and
               int(r['rounds']) == manifest['rounds'] and float(r['median_ms']) > 0 for r in rows)
    save(out / 'raw.csv', rows)
    for r in rows:
        for n in ('threads', 'M', 'K', 'N'):
            r[n] = int(r[n])
        for n in ('sparsity', 'median_ms', 'min_ms', 'bytes', 'adds_per_row'):
            r[n] = float(r[n])
        r['category'] = r['tag'].removeprefix('v6v9_')
    workload = lambda r: (r['category'], r['M'], r['K'], r['N'], r['sparsity'])
    lookup = {(r['threads'], r['kernel'], workload(r), r['param']): r for r in rows}
    assert len(lookup) == expected
    ws = sorted({workload(r) for r in rows})
    best = {}
    for r in rows:
        key = (r['threads'], r['kernel'], workload(r))
        if key not in best or r['median_ms'] < best[key]['median_ms']:
            best[key] = r
    paired, tuned = [], []
    for w in ws:
        # Common k fixed from V6/1T, avoiding per-thread tuning in scaling plots.
        param = best[1, KERNELS[0], w]['param']
        for t in ts:
            a, b = [lookup[t, kernel, w, param] for kernel in KERNELS]
            a0, b0 = [lookup[1, kernel, w, param] for kernel in KERNELS]
            assert a['bytes'] == b['bytes'] and a['adds_per_row'] == b['adds_per_row']
            work = b['M'] * b['adds_per_row']
            compulsory = b['bytes'] + 4 * b['M'] * (b['K'] + b['N'])
            paired.append(dict(category=w[0], M=w[1], K=w[2], N=w[3], sparsity=w[4],
                               threads=t, param=param, v6_ms=a['median_ms'], v9_ms=b['median_ms'],
                               v9_vs_v6=a['median_ms']/b['median_ms'],
                               v6_scaling=a0['median_ms']/a['median_ms'],
                               v9_scaling=b0['median_ms']/b['median_ms'],
                               v6_efficiency=a0['median_ms']/a['median_ms']/t,
                               v9_efficiency=b0['median_ms']/b['median_ms']/t,
                               additions=work, format_bytes=b['bytes'],
                               compulsory_bytes=compulsory, intensity=work/compulsory,
                               v6_gadds=work/a['median_ms']/1e6,
                               v9_gadds=work/b['median_ms']/1e6))
            aa, bb = [best[t, kernel, w] for kernel in KERNELS]
            tuned.append(dict(category=w[0], M=w[1], K=w[2], N=w[3], sparsity=w[4], threads=t,
                              v6_param=aa['param'], v9_param=bb['param'], v6_ms=aa['median_ms'],
                              v9_ms=bb['median_ms'], v9_vs_v6=aa['median_ms']/bb['median_ms']))
    save(out / 'comparison.csv', paired)
    save(out / 'tuned_comparison.csv', tuned)
    regular = [r for r in paired if r['category'] == 'regular']
    peaks = {t: json.loads((out / f't{t:02}/peaks.json').read_text()) for t in ts}
    colors = {'v6': '#2166ac', 'v9': '#d95f02'}
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    def finish(fig, axs, used, title, file):
        for ax in list(axs.flat)[used:]:
            ax.axis('off')
        handles, labels = {}, []
        for ax in axs.flat:
            for h, label in zip(*ax.get_legend_handles_labels()):
                if label not in handles:
                    handles[label] = h
                    labels.append(label)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, .07, 1, .95))
        fig.legend([handles[x] for x in labels], labels, loc='lower center', ncol=min(5, len(labels)), frameon=False)
        fig.savefig(out / file, dpi=170, bbox_inches='tight')
        plt.close(fig)

    def style(ax, ylabel):
        ax.set_xticks(ts)
        ax.set_xlabel('requested OpenMP threads')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.23)

    ratio_values = [gm(r['v9_vs_v6'] for r in regular if r['M']==M and r['threads']==t and r['sparsity']==s)
                    for M in MS for t in ts for s in manifest['sparsities']]
    ratio_limits = (min(.95, min(ratio_values)*.99), max(1.05, max(ratio_values)*1.01))
    fig, axs = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
    for ax, M in zip(axs.flat, MS):
        for s in manifest['sparsities']:
            vals = [gm(r['v9_vs_v6'] for r in regular if r['M'] == M and r['threads'] == t and r['sparsity'] == s) for t in ts]
            ax.plot(ts, vals, 'o-', label=f'{s:.0%} sparse')
        ax.axhline(1, color='black', ls='--', lw=.8)
        ax.set_ylim(*ratio_limits)
        ax.set_title(f'M={M} ({M//32} full tiles); geomean over 3 shapes')
        style(ax, 'V9 speedup over V6 (>1 faster)')
    finish(fig, axs, 5, 'V6 vs V9: paired speedups, common fixed k, B=32', 'speedup.png')

    fig, axs = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
    for ax, M in zip(axs.flat, MS):
        for v in ('v6', 'v9'):
            ax.plot(ts, [gm(r[v+'_scaling'] for r in regular if r['M']==M and r['threads']==t) for t in ts], 'o-', color=colors[v], label=v.upper())
        ax.plot(ts, ts, ':', color='gray', label='ideal requested-thread scaling')
        ax.set_title(f'M={M}; geomean over shapes and sparsities')
        style(ax, 'speedup over own 1-thread baseline')
    finish(fig, axs, 5, 'Strong scaling: each version uses its own 1-thread baseline', 'thread_scaling.png')

    fig, axs = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
    for ax, M in zip(axs.flat, MS):
        for v in ('v6', 'v9'):
            ax.plot(ts, [gm(r[v+'_ms'] for r in regular if r['M']==M and r['threads']==t) for t in ts], 'o-', color=colors[v], label=v.upper())
        ax.set_title(f'M={M}; geometric mean latency')
        style(ax, 'inference median [ms], lower is faster')
    finish(fig, axs, 5, 'Absolute latency on regular matrices, fixed common k', 'runtime.png')

    diagnostics = sorted({(r['category'],r['M'],r['K'],r['N']) for r in paired if r['category']!='regular'})
    fig, axs = plt.subplots(math.ceil(len(diagnostics)/3), 3, figsize=(15, 3.5*math.ceil(len(diagnostics)/3)), squeeze=False)
    for ax, (category,M,K,N) in zip(axs.flat, diagnostics):
        rs = [r for r in paired if (r['category'],r['M'],r['K'],r['N'])==(category,M,K,N)]
        for v in ('v6','v9'):
            ax.plot(ts, [1000*gm(r[v+'_ms'] for r in rs if r['threads']==t) for t in ts], 'o-', color=colors[v], label=v.upper())
        ax.set_title(f'{category}: M={M}, K={K}, N={N}')
        style(ax, 'latency [microseconds]')
    finish(fig, axs, len(diagnostics), 'Diagnostics: small work, tails, few tasks, and clustered columns', 'diagnostics.png')

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    high_values = [r['v9_vs_v6'] for r in paired if r['category'].startswith('high_k')]
    high_limits = (min(.95, min(high_values)*.98), max(1.05, max(high_values)*1.02))
    for ax, M in zip(axs.flat, (64,256,512)):
        for k in (6,8):
            ax.plot(ts, [next(r['v9_vs_v6'] for r in paired if r['category']==f'high_k{k}' and r['M']==M and r['threads']==t) for t in ts], 'o-', label=f'k={k}')
        ax.axhline(1,color='black',ls='--',lw=.8)
        ax.set_ylim(*high_limits)
        ax.set_title(f'M={M}, K=1024, N=4096, 50% sparse')
        style(ax,'V9 speedup over V6 (>1 faster)')
    finish(fig,axs,3,'Loop interchange for larger k (held at each exact k)','high_k_speedup.png')

    fig, axs = plt.subplots(5, 2, figsize=(12, 18), squeeze=False)
    xx = np.logspace(-2, 3, 250)
    for ax, t in zip(axs.flat, ts):
        pk = peaks[t]
        ax.plot(xx, np.minimum(pk['peak_simd_gflops'], xx*pk['bw_dram_gbs']), color='black', lw=1.3, label='measured SIMD / DRAM roof')
        ax.plot(xx, np.minimum(pk['peak_simd_gflops'], xx*pk['bw_l2_gbs']), color='gray', lw=.9, ls='--', label='measured SIMD / L2 read roof')
        for j,M in enumerate(MS):
            rs = [r for r in regular if r['threads']==t and r['M']==M]
            for v,marker in [('v6','o'),('v9','^')]:
                ax.scatter([r['intensity'] for r in rs], [r[v+'_gadds'] for r in rs],
                           s=24, marker=marker, color=f'C{j}', alpha=.78,
                           facecolors='none' if v=='v6' else f'C{j}', label=f'{v.upper()}, M={M}')
        ax.set(xscale='log',yscale='log',xlim=(.1,200),ylim=(.1,max(p['peak_simd_gflops'] for p in peaks.values())*1.4),
               title=f'{t} requested threads',xlabel='additions / compulsory byte',ylabel='Gadd/s (FP32 adds/subtracts)')
        ax.grid(alpha=.2)
    # Compact explicit legend rather than repeating ten workload labels.
    for ax in axs.flat:
        h,l=ax.get_legend_handles_labels()
        ax.legend(h[:2],l[:2],fontsize=6,loc='lower right',frameon=False)
    fig.suptitle('V6 circles / V9 triangles: optimistic compulsory-traffic roofline\nColors: M=32 blue, 64 orange, 256 green, 320 red, 512 purple',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(out/'roofline.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

    by_t=[]
    for t in ts:
        rs=[r for r in regular if r['threads']==t]
        by_t.append(dict(threads=t, v9_vs_v6=gm(r['v9_vs_v6'] for r in rs),
                         v6_scaling=gm(r['v6_scaling'] for r in rs), v9_scaling=gm(r['v9_scaling'] for r in rs),
                         v6_efficiency=gm(r['v6_efficiency'] for r in rs), v9_efficiency=gm(r['v9_efficiency'] for r in rs)))
    save(out/'thread_summary.csv',by_t)
    median_k=[r for r in paired if r['threads']==10]
    faster=sum(r['v9_vs_v6']>1.05 for r in median_k)
    slower=sum(r['v9_vs_v6']<1/1.05 for r in median_k)
    tiny = next(r for r in paired if r['threads']==10 and r['category']=='tiny')
    lines=['# SPR V6 versus V9: 1–10 thread benchmarks','',
           'Measured '+manifest['date']+'. Both versions use B=32 and identical deterministic inputs/weights (seed 42). Correctness checks were **not run**, as requested.','',
           '## Implementation','',
           '[V9 source](source/cpu/spr/spr_v9.cpp) copies V6 and interchanges only the two inference task loops. Preprocessing, process_group(), static collapse(2), scratch allocation, arithmetic and tile selection are unchanged. All matching B=16/B=32, L2 and fixed-Kc/G variants are registered; the benchmark measures the automatic-L1 B=32 variants.',
           '', '## Findings','',
           f"- **Regular-matrix loop-order comparison:** V9/V6 geometric-mean speedup ranges from {min(r['v9_vs_v6'] for r in by_t):.3f}× to {max(r['v9_vs_v6'] for r in by_t):.3f}× across all ten thread counts. At ten threads it is {by_t[-1]['v9_vs_v6']:.3f}×.",
           f"- **Tiny workload:** V6 and V9 take {tiny['v6_ms']*1000:.1f} and {tiny['v9_ms']*1000:.1f} microseconds respectively at ten requested threads ({tiny['v9_vs_v6']:.2f}×). Neither version introduces a serial-work threshold.",
           f"- **Measured scaling:** at ten threads the regular-grid speedup over one thread is {by_t[-1]['v6_scaling']:.2f}× for V6 and {by_t[-1]['v9_scaling']:.2f}× for V9, about {by_t[-1]['v9_efficiency']:.0%} nominal efficiency.",
           '- **Only task traversal order changes.** V6 traverses tiles then groups; V9 traverses groups then tiles. Both retain static collapse(2), the same OpenMP team condition, scratch lifetime, preprocessing and arithmetic. Roofline pairs have the same estimated intensity.',
           '', '## Setup','',
           manifest['cpu'].replace('\n','  \n'),'',
           f"Exactly {len(ts)} requested thread counts: {ts}. Main grid: M={MS}; K×N={SHAPES}; sparsity={manifest['sparsities']}; k sweep={manifest['ks']}.",
           f"{len(rows):,} timed configurations × {manifest['rounds']} rounds = {len(rows)*manifest['rounds']:,} timed inference calls, after three warmups per configuration. Kernels are timed interleaved with rotating order; benchmarks and ceiling calibration run sequentially.",
           'The primary grid uses k=2/3/4. Additional diagnostics hold k=6 and k=8 separately at M=64/256/512 and K=1024,N=4096,s=0.5. Kernels are spr.v6.b32 and spr.v9.b32 with automatic L1 cache tiling.',
           'Regular and most diagnostic weights are IID ternary, with P(0)=s and equal probabilities for +/-1. The clustered diagnostic instead zeroes floor(s*N) leading columns and fills the remaining columns with dense +/-1 weights. It is deliberately uneven and excluded from regular-grid means.',
           'Preparation, layout conversion/bind, and any finish operation are outside inference timing. We run `bench_cpu --verify none`; `max_err=-1` means unchecked, and `ok=1` in raw data must not be interpreted as verification.',
           'Compiled with the repository Release flags: Apple Clang, `-O3 -ffast-math -DNDEBUG -mcpu=native` and Homebrew OpenMP. Sources and binary hashes are saved in the run directory.',
           'OMP_DYNAMIC=FALSE, OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0. Binding to cores was requested; macOS placement is not guaranteed (the preceding investigation reported zero available OpenMP places). Both versions request the same OpenMP team when there is more than one task; no adaptive team limit is used.',
           'Main comparisons select V6’s fastest k at one thread once per workload, then hold that same k for both versions and every thread count. Independently tuned results are saved separately in `tuned_comparison.csv`. This isolates scheduling rather than mixing per-thread k choices.',
           'This is one campaign on a shared laptop, using one random weight sample per workload. Small differences around unity can be noise; medians are not confidence intervals. Minima and per-configuration round counts are preserved in raw data.','',
           '## Speedup and scaling','',
           '`V9/V6 speedup = time(V6,t) / time(V9,t)`; values above one favor V9. Each version’s scaling is its own one-thread time divided by its t-thread time. Nominal efficiency divides that scaling by the requested thread count. Regular-grid summaries are geometric means with equal weight per workload; diagnostics are excluded.','',
           '| Threads | V9/V6 speedup | V6 scaling | V9 scaling | V6 efficiency | V9 efficiency |',
           '|---:|---:|---:|---:|---:|---:|']
    for r in by_t:
        lines.append(f"| {r['threads']} | {r['v9_vs_v6']:.3f}× | {r['v6_scaling']:.2f}× | {r['v9_scaling']:.2f}× | {r['v6_efficiency']:.0%} | {r['v9_efficiency']:.0%} |")
    lines+=['','### Speedup by batch size','', '| Threads | '+' | '.join(f'M={M}' for M in MS)+' |','|'+'---:|'*(len(MS)+1)]
    for t in ts:
        lines.append('| '+str(t)+' | '+' | '.join(f"{gm(r['v9_vs_v6'] for r in regular if r['threads']==t and r['M']==M):.3f}×" for M in MS)+' |')
    lines+=['','![V9 speedup vs V6](speedup.png)','','![Thread scaling](thread_scaling.png)','','![Absolute latency](runtime.png)','',
            '## Interpretation','',
            'V6 flattens tasks as t*num_groups + g. V9 flattens the identical task set as g*num_tiles + t. Static scheduling assigns contiguous chunks of the flattened space to workers. A worker can therefore process consecutive tiles of a group in V9, whereas V6 tends to process consecutive groups of a tile.',
            'V9 can reuse a group’s metadata across nearby tiles, but it moves through a larger portion of X before changing groups. V6 favors reuse of an input tile across groups. Tail costs and clustered column sparsity are also redistributed across static worker chunks. These timing results measure the combined effects; they do not isolate cache misses, false sharing, or load balance.',
            f"On the regular grid at ten requested threads, V9/V6 is **{by_t[-1]['v9_vs_v6']:.3f}×** (geometric mean). Across all {len(median_k)} fixed-k workloads at ten threads, {faster} favor V9 by more than 5%, {slower} favor V6 by more than 5%, and {len(median_k)-faster-slower} lie between those thresholds. These are descriptive counts, not significance tests.",
            'With one input tile (M<=32), or one output group, swapping the loops preserves the logical task sequence, so differences there should be treated as code-generation/layout effects or noise. Larger M exposes the actual locality and assignment tradeoff. The k=6 and k=8 diagnostics are held separately at each exact k to expose tree-heavy cases rather than hiding them behind a best-k choice.','',
            '### Diagnostic workloads at 10 requested threads','',
            '| Category | M | K×N | Sparsity | Common k | V6 ms | V9 ms | V9/V6 |',
            '|---|---:|---|---:|---|---:|---:|---:|']
    for r in paired:
        if r['threads']==10 and r['category']!='regular':
            lines.append(f"| {r['category']} | {r['M']} | {r['K']}×{r['N']} | {r['sparsity']:.0%} | {r['param']} | {r['v6_ms']:.4f} | {r['v9_ms']:.4f} | {r['v9_vs_v6']:.2f}× |")
    lines+=['','![Diagnostic latency](diagnostics.png)','','### Larger k: explicit k=6 and k=8 comparisons','','These cases are excluded from the regular-grid average and keep each exact k fixed. Values below one are regressions.','','![High-k loop-order speedups](high_k_speedup.png)','', '## Roofline comparison','',
            'Fresh local ceilings were measured using `build/peaks --threads t` for **each** t=1…10. Compute ceilings count FP32 additions, not FMA throughput; bandwidth ceilings are the existing streaming read microbenchmarks. These are empirical microbenchmark ceilings, not guaranteed strict bounds for every access pattern.',
            'The plotted operation count follows the existing kernel accounting: `A = M * adds_per_row()`. Subtractions count as one arithmetic operation; clears, stores, integer/index work, and control flow are excluded. `performance = A / (median_ms * 10^6)` Gadd/s.',
            'Intensity is **A / (compressed-format bytes + 4*M*(K+N))**, the optimistic compulsory-byte convention used by the repository. Both versions have identical metadata and arithmetic counts at common k, so each V6/V9 pair shares the same horizontal position; scheduling changes its vertical performance.',
            'The black roof is min(measured SIMD-add throughput, intensity × measured DRAM-read bandwidth); the dashed gray roof substitutes L2-read bandwidth. Regular-grid workloads only are plotted, one panel per requested thread count.',
            'Metadata is actually traversed per batch tile, and X/scratch/Y traffic can repeat through caches. The intensity estimate does not measure actual DRAM traffic, write allocation, coherence, or cache misses. Distance below a roof is headroom, not proof of a particular bottleneck; these plots alone cannot establish “DRAM-bound” versus “compute-bound.” Both versions use the same requested-thread ceiling; core placement and cache behavior remain unmeasured.','',
            '| Requested threads | SIMD add peak Gadd/s | L1 read GB/s | L2 read GB/s | DRAM read GB/s |','|---:|---:|---:|---:|---:|']
    for t in ts:
        pk=peaks[t]
        lines.append(f"| {t} | {pk['peak_simd_gflops']:.1f} | {pk['bw_l1_gbs']:.1f} | {pk['bw_l2_gbs']:.1f} | {pk['bw_dram_gbs']:.1f} |")
    lines+=['','![Roofline at every thread count](roofline.png)','',
            '## Per-shape results at 10 threads','',
            '| M | K×N | V6 ms (geomean over sparsities) | V9 ms | V9/V6 |','|---:|---|---:|---:|---:|']
    for M in MS:
        for K,N in SHAPES:
            rs=[r for r in regular if (r['threads'],r['M'],r['K'],r['N'])==(10,M,K,N)]
            lines.append(f"| {M} | {K}×{N} | {gm(r['v6_ms'] for r in rs):.3f} | {gm(r['v9_ms'] for r in rs):.3f} | {gm(r['v9_vs_v6'] for r in rs):.3f}× |")
    lines+=['','## Reproduction and data','',
            'Build: `cmake --build build --target bench_cpu peaks -j 4`. Run: `/usr/bin/python3 bench/compare_v6_v9.py`. Replot: `/usr/bin/python3 bench/compare_v6_v9.py --plots-only '+str(out.relative_to(ROOT))+'`.',
            'No correctness command is invoked by this script; every kernel benchmark explicitly passes `--verify none`.',
            '- [manifest.json](manifest.json): complete jobs, environment, flags context, source/binary hashes.',
            '- [raw.csv](raw.csv): all k values and requested threads; unchecked results marked max_err=-1.',
            '- [comparison.csv](comparison.csv): fixed common-k paired times, scaling, efficiency, roofline coordinates.',
            '- [tuned_comparison.csv](tuned_comparison.csv): independently best k per version and thread count.',
            '- [thread_summary.csv](thread_summary.csv): regular-grid aggregates.',
            '- `t01` … `t10`: benchmark logs, raw CSVs, and freshly measured ceilings.',
            '- `source/`: source snapshots, including the new V9 variant and the comparison runner. V6 was left unchanged.',
            '','Earlier bottleneck investigation: [previous report](../v6_investigation/report.md). CSV files remain subject to the repository’s existing `*.csv` ignore rule.']
    rendered=[]
    for line in lines:
        if rendered and line and rendered[-1]:
            previous=rendered[-1]
            same_table=line.startswith('|') and previous.startswith('|')
            same_list=line.startswith('- ') and previous.startswith('- ')
            if not (same_table or same_list):
                rendered.append('')
        rendered.append(line)
    report='\n'.join(rendered)+'\n'
    (out/'report.md').write_text(report)
    # Publish the same full report at the requested root path, fixing relative links.
    import re
    prefix=out.relative_to(ROOT).as_posix()
    report=re.sub(r'\]\(([^)]+)\)', lambda m: ']('+str((out/m[1]).resolve().relative_to(ROOT))+')', report)
    (ROOT/'report.md').write_text(report)
    (out/'validation.json').write_text(json.dumps(dict(configurations=len(rows),timed_calls=len(rows)*manifest['rounds'],
        correctness_run=False,unchecked_configurations=len(rows),comparison_points=len(paired),
        source_matches_current={v:(out/f'source/cpu/spr/spr_{v}.cpp').read_bytes()==(ROOT/f'cpu/spr/spr_{v}.cpp').read_bytes() for v in ('v6','v9')}),indent=2)+'\n')
    print('Completed:',out,flush=True)
    print('Regular-grid V9/V6 at 10 threads:',by_t[-1]['v9_vs_v6'],flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds',type=int,default=21)
    ap.add_argument('--plots-only',type=Path)
    a=ap.parse_args()
    if a.plots_only:
        summarize(a.plots_only.resolve());return
    stamp=datetime.datetime.now(ZoneInfo('Europe/Zurich')).strftime('%Y%m%d-%H%M%S')
    out=ROOT/'results'/f'{stamp}_spr_v6_v9_threads_1_10'
    out.mkdir(parents=True)
    env=dict(os.environ,OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0',OMP_PROC_BIND='close',OMP_PLACES='cores')
    jobs=[dict(category='regular',pattern='iid',M=M,K=K,N=N,sparsities=[.5,.9]) for M in MS for K,N in SHAPES]
    jobs += [dict(category='tiny',pattern='iid',M=1,K=128,N=128,sparsities=[.5]),
             dict(category='decode',pattern='iid',M=1,K=1024,N=4096,sparsities=[.5,.9])]
    jobs += [dict(category='tail',pattern='iid',M=M,K=1024,N=4096,sparsities=[.5,.9]) for M in (63,65)]
    jobs += [dict(category='narrow',pattern='iid',M=M,K=1024,N=32,sparsities=[.5,.9]) for M in (32,64,320)]
    jobs += [dict(category='clustered',pattern='column_clustered',M=M,K=1024,N=4096,sparsities=[.5]) for M in (64,320)]
    jobs += [dict(category=f'high_k{k}',pattern='iid',M=M,K=1024,N=4096,sparsities=[.5],ks=[k]) for k in (6,8) for M in (64,256,512)]
    cpu=subprocess.check_output(['sysctl','machdep.cpu.brand_string','hw.physicalcpu','hw.logicalcpu','hw.perflevel0.physicalcpu','hw.perflevel1.physicalcpu','hw.cachelinesize','hw.perflevel0.l1dcachesize','hw.perflevel0.l2cachesize'],text=True)
    paths=['cpu/spr/spr_v6.cpp','cpu/spr/spr_v9.cpp','cpu/core.cpp','include/spr/cache.hpp','include/spr/harness.hpp','include/spr/kernel.hpp','bench/bench_cpu.cpp','bench/peaks.cpp','bench/compare_v6_v9.py','CMakeLists.txt']
    for name in paths:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    manifest=dict(date=datetime.datetime.now(ZoneInfo('Europe/Zurich')).isoformat(),cpu=cpu,threads=list(range(1,11)),
                  jobs=jobs,sparsities=[.5,.9],ks=[2,3,4],rounds=a.rounds,seed=42,correctness_run=False,
                  environment={k:env.get(k) for k in ('OMP_DYNAMIC','OMP_WAIT_POLICY','KMP_BLOCKTIME','OMP_PROC_BIND','OMP_PLACES','SPR_L1_BYTES','SPR_L2_BYTES','SPR_LLC_BYTES')},
                  command=sys.argv,compiler=subprocess.check_output(['clang++','--version'],text=True),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in paths},
                  binary_hashes={n:hashlib.sha256((ROOT/'build'/n).read_bytes()).hexdigest() for n in ('bench_cpu','peaks')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'working_tree.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    print('Results:',out,flush=True)
    for t in manifest['threads']:
        dest=out/f't{t:02}';dest.mkdir()
        print(f'[{t}/10 threads] Fresh ceiling calibration',flush=True)
        pk=subprocess.check_output([str(ROOT/'build/peaks'),'--threads',str(t)],env=env,text=True,cwd=ROOT)
        assert json.loads(pk)['threads']==t
        (dest/'peaks.json').write_text(pk)
        for j,job in enumerate(jobs,1):
            print(f"[{t}/10 threads] {j}/{len(jobs)} {job['category']} M={job['M']} K={job['K']} N={job['N']}",flush=True)
            command([ROOT/'build/bench_cpu','--M',job['M'],'--K',job['K'],'--N',job['N'],
                     '--s',','.join(map(str,job['sparsities'])),'--sparsity-pattern',job['pattern'],
                     '--k',','.join(map(str,job.get('ks',manifest['ks']))),'--threads',t,'--kernels',','.join('='+k for k in KERNELS),
                     '--rounds',a.rounds,'--warmup',3,'--prep-rounds',1,'--verify','none','--ref-rows',0,
                     '--seed',42,'--tag','v6v9_'+job['category'],'--out',dest/'raw.csv'],dest/'benchmark.log',env)
    summarize(out)

if __name__=='__main__':
    main()
