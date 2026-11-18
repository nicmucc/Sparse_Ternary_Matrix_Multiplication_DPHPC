#!/usr/bin/env python3
"""Benchmark V6/V8 without correctness checks, then plot and report.

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

KERNELS = ['spr.v6.b32', 'spr.v8.b32']
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
    expected = len(ts) * sum(len(j['sparsities']) for j in manifest['jobs']) * len(manifest['ks']) * 2
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
        r['category'] = r['tag'].removeprefix('v6v8_')
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
                               threads=t, param=param, v6_ms=a['median_ms'], v8_ms=b['median_ms'],
                               v8_vs_v6=a['median_ms']/b['median_ms'],
                               v6_scaling=a0['median_ms']/a['median_ms'],
                               v8_scaling=b0['median_ms']/b['median_ms'],
                               v6_efficiency=a0['median_ms']/a['median_ms']/t,
                               v8_efficiency=b0['median_ms']/b['median_ms']/t,
                               additions=work, format_bytes=b['bytes'],
                               compulsory_bytes=compulsory, intensity=work/compulsory,
                               v6_gadds=work/a['median_ms']/1e6,
                               v8_gadds=work/b['median_ms']/1e6))
            aa, bb = [best[t, kernel, w] for kernel in KERNELS]
            tuned.append(dict(category=w[0], M=w[1], K=w[2], N=w[3], sparsity=w[4], threads=t,
                              v6_param=aa['param'], v8_param=bb['param'], v6_ms=aa['median_ms'],
                              v8_ms=bb['median_ms'], v8_vs_v6=aa['median_ms']/bb['median_ms']))
    save(out / 'comparison.csv', paired)
    save(out / 'tuned_comparison.csv', tuned)
    regular = [r for r in paired if r['category'] == 'regular']
    peaks = {t: json.loads((out / f't{t:02}/peaks.json').read_text()) for t in ts}
    colors = {'v6': '#2166ac', 'v8': '#d95f02'}
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

    fig, axs = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
    for ax, M in zip(axs.flat, MS):
        for s in manifest['sparsities']:
            vals = [gm(r['v8_vs_v6'] for r in regular if r['M'] == M and r['threads'] == t and r['sparsity'] == s) for t in ts]
            ax.plot(ts, vals, 'o-', label=f'{s:.0%} sparse')
        ax.axhline(1, color='black', ls='--', lw=.8)
        ax.set_title(f'M={M} ({M//32} full tiles); geomean over 3 shapes')
        style(ax, 'V8 speedup over V6 (>1 faster)')
    finish(fig, axs, 5, 'V6 vs V8: paired speedups, common fixed k, B=32', 'speedup.png')

    fig, axs = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
    for ax, M in zip(axs.flat, MS):
        for v in ('v6', 'v8'):
            ax.plot(ts, [gm(r[v+'_scaling'] for r in regular if r['M']==M and r['threads']==t) for t in ts], 'o-', color=colors[v], label=v.upper())
        ax.plot(ts, ts, ':', color='gray', label='ideal requested-thread scaling')
        ax.set_title(f'M={M}; geomean over shapes and sparsities')
        style(ax, 'speedup over own 1-thread baseline')
    finish(fig, axs, 5, 'Strong scaling: each version uses its own 1-thread baseline', 'thread_scaling.png')

    fig, axs = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
    for ax, M in zip(axs.flat, MS):
        for v in ('v6', 'v8'):
            ax.plot(ts, [gm(r[v+'_ms'] for r in regular if r['M']==M and r['threads']==t) for t in ts], 'o-', color=colors[v], label=v.upper())
        ax.set_title(f'M={M}; geometric mean latency')
        style(ax, 'inference median [ms], lower is faster')
    finish(fig, axs, 5, 'Absolute latency on regular matrices, fixed common k', 'runtime.png')

    diagnostics = sorted({(r['category'],r['M'],r['K'],r['N']) for r in paired if r['category']!='regular'})
    fig, axs = plt.subplots(math.ceil(len(diagnostics)/3), 3, figsize=(15, 3.5*math.ceil(len(diagnostics)/3)), squeeze=False)
    for ax, (category,M,K,N) in zip(axs.flat, diagnostics):
        rs = [r for r in paired if (r['category'],r['M'],r['K'],r['N'])==(category,M,K,N)]
        for v in ('v6','v8'):
            ax.plot(ts, [1000*gm(r[v+'_ms'] for r in rs if r['threads']==t) for t in ts], 'o-', color=colors[v], label=v.upper())
        ax.set_title(f'{category}: M={M}, K={K}, N={N}')
        style(ax, 'latency [microseconds]')
    finish(fig, axs, len(diagnostics), 'Diagnostics: small work, tails, few tasks, and clustered columns', 'diagnostics.png')

    fig, axs = plt.subplots(5, 2, figsize=(12, 18), squeeze=False)
    xx = np.logspace(-2, 3, 250)
    for ax, t in zip(axs.flat, ts):
        pk = peaks[t]
        ax.plot(xx, np.minimum(pk['peak_simd_gflops'], xx*pk['bw_dram_gbs']), color='black', lw=1.3, label='measured SIMD / DRAM roof')
        ax.plot(xx, np.minimum(pk['peak_simd_gflops'], xx*pk['bw_l2_gbs']), color='gray', lw=.9, ls='--', label='measured SIMD / L2 read roof')
        for j,M in enumerate(MS):
            rs = [r for r in regular if r['threads']==t and r['M']==M]
            for v,marker in [('v6','o'),('v8','^')]:
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
    fig.suptitle('V6 circles / V8 triangles: optimistic compulsory-traffic roofline\nColors: M=32 blue, 64 orange, 256 green, 320 red, 512 purple',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(out/'roofline.png',dpi=170,bbox_inches='tight')
    plt.close(fig)

    by_t=[]
    for t in ts:
        rs=[r for r in regular if r['threads']==t]
        by_t.append(dict(threads=t, v8_vs_v6=gm(r['v8_vs_v6'] for r in rs),
                         v6_scaling=gm(r['v6_scaling'] for r in rs), v8_scaling=gm(r['v8_scaling'] for r in rs),
                         v6_efficiency=gm(r['v6_efficiency'] for r in rs), v8_efficiency=gm(r['v8_efficiency'] for r in rs)))
    save(out/'thread_summary.csv',by_t)
    median_k=[r for r in paired if r['threads']==10]
    faster=sum(r['v8_vs_v6']>1.05 for r in median_k)
    slower=sum(r['v8_vs_v6']<1/1.05 for r in median_k)
    lines=['# SPR V6 versus V8: 1–10 thread benchmarks','',
           'Measured '+manifest['date']+'. Both versions use B=32 and identical deterministic inputs/weights (seed 42). Correctness checks were **not run**, as requested.','',
           '## Setup','',
           manifest['cpu'].replace('\n','  \n'),'',
           f"Exactly {len(ts)} requested thread counts: {ts}. Main grid: M={MS}; K×N={SHAPES}; sparsity={manifest['sparsities']}; k sweep={manifest['ks']}.",
           f"{len(rows):,} timed configurations × {manifest['rounds']} rounds = {len(rows)*manifest['rounds']:,} timed inference calls, after three warmups per configuration. Kernels are timed interleaved with rotating order; benchmarks and ceiling calibration run sequentially.",
           'Preparation, layout conversion/bind, and any finish operation are outside inference timing. We run `bench_cpu --verify none`; `max_err=-1` means unchecked, and `ok=1` in raw data must not be interpreted as verification.',
           'Compiled with the repository Release flags: Apple Clang, `-O3 -ffast-math -DNDEBUG -mcpu=native` and Homebrew OpenMP. Sources and binary hashes are saved in the run directory.',
           'OMP_DYNAMIC=FALSE, OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0. Binding to cores was requested; macOS placement is not guaranteed (the preceding investigation reported zero available OpenMP places). Requested threads are not necessarily active V8 workers.',
           'Main comparisons select V6’s fastest k at one thread once per workload, then hold that same k for both versions and every thread count. Independently tuned results are saved separately in `tuned_comparison.csv`. This isolates scheduling rather than mixing per-thread k choices.',
           'This is one campaign on a shared laptop, using one random weight sample per workload. Small differences around unity can be noise; medians are not confidence intervals. Minima and per-configuration round counts are preserved in raw data.','',
           '## Speedup and scaling','',
           '`V8/V6 speedup = time(V6,t) / time(V8,t)`; values above one favor V8. Each version’s scaling is its own one-thread time divided by its t-thread time. Nominal efficiency divides that scaling by the requested thread count. Regular-grid summaries are geometric means with equal weight per workload; diagnostics are excluded.','',
           '| Threads | V8/V6 speedup | V6 scaling | V8 scaling | V6 efficiency | V8 efficiency |',
           '|---:|---:|---:|---:|---:|---:|']
    for r in by_t:
        lines.append(f"| {r['threads']} | {r['v8_vs_v6']:.3f}× | {r['v6_scaling']:.2f}× | {r['v8_scaling']:.2f}× | {r['v6_efficiency']:.0%} | {r['v8_efficiency']:.0%} |")
    lines+=['','### Speedup by batch size','', '| Threads | '+' | '.join(f'M={M}' for M in MS)+' |','|'+'---:|'*(len(MS)+1)]
    for t in ts:
        lines.append('| '+str(t)+' | '+' | '.join(f"{gm(r['v8_vs_v6'] for r in regular if r['threads']==t and r['M']==M):.3f}×" for M in MS)+' |')
    lines+=['','![V8 speedup vs V6](speedup.png)','','![Thread scaling](thread_scaling.png)','','![Absolute latency](runtime.png)','',
            '## Interpretation','',
            'V8 caps its requested team by available tasks and floor(estimated additions / 32768), with a minimum of one worker. It uses tile-only scheduling only for full tiles with at least 90% wave efficiency; otherwise it retains static collapsed tile/group scheduling.',
            'At B=32, M=32/64/256/320/512 expose 1/2/8/10/16 tiles. At ten active workers, M=320 has a perfect tile wave; M=256 and M=512 fall back to 2D. Many divisible cases already give V6 whole tiles under static collapse, so the V8 policy need not produce large gains there.',
            f"On the regular grid at ten requested threads, V8/V6 is **{by_t[-1]['v8_vs_v6']:.3f}×** (geometric mean). Across all {len(median_k)} fixed-k workloads at ten threads, {faster} favor V8 by more than 5%, {slower} favor V6 by more than 5%, and {len(median_k)-faster-slower} lie between those thresholds. These are descriptive counts, not significance tests.",
            'The largest gains should be read together with absolute diagnostic latency: avoiding excessive workers on a tiny problem can yield a large ratio without saving much absolute time. Tail and clustered-column cases still use static scheduling; V8 does not implement the dynamic/guided scheduling, stride padding, or full-tile padding explored in the earlier report.','',
            '### Diagnostic workloads at 10 requested threads','',
            '| Category | M | K×N | Sparsity | Common k | V6 ms | V8 ms | V8/V6 |',
            '|---|---:|---|---:|---|---:|---:|---:|']
    for r in paired:
        if r['threads']==10 and r['category']!='regular':
            lines.append(f"| {r['category']} | {r['M']} | {r['K']}×{r['N']} | {r['sparsity']:.0%} | {r['param']} | {r['v6_ms']:.4f} | {r['v8_ms']:.4f} | {r['v8_vs_v6']:.2f}× |")
    lines+=['','![Diagnostic latency](diagnostics.png)','', '## Roofline comparison','',
            'Fresh local ceilings were measured using `build/peaks --threads t` for **each** t=1…10. Compute ceilings count FP32 additions, not FMA throughput; bandwidth ceilings are the existing streaming read microbenchmarks. These are empirical microbenchmark ceilings, not guaranteed strict bounds for every access pattern.',
            'The plotted operation count follows the existing kernel accounting: `A = M * adds_per_row()`. Subtractions count as one arithmetic operation; clears, stores, integer/index work, and control flow are excluded. `performance = A / (median_ms * 10^6)` Gadd/s.',
            'Intensity is **A / (compressed-format bytes + 4*M*(K+N))**, the optimistic compulsory-byte convention used by the repository. Both versions have identical metadata and arithmetic counts at common k, so each V6/V8 pair shares the same horizontal position; scheduling changes its vertical performance.',
            'The black roof is min(measured SIMD-add throughput, intensity × measured DRAM-read bandwidth); the dashed gray roof substitutes L2-read bandwidth. Regular-grid workloads only are plotted, one panel per requested thread count.',
            'Metadata is actually traversed per batch tile, and X/scratch/Y traffic can repeat through caches. The intensity estimate does not measure actual DRAM traffic, write allocation, coherence, or cache misses. Distance below a roof is headroom, not proof of a particular bottleneck; these plots alone cannot establish “DRAM-bound” versus “compute-bound.” V8 can reduce its active team, so requested-thread ceilings may overstate the resources it uses.','',
            '| Requested threads | SIMD add peak Gadd/s | L1 read GB/s | L2 read GB/s | DRAM read GB/s |','|---:|---:|---:|---:|---:|']
    for t in ts:
        pk=peaks[t]
        lines.append(f"| {t} | {pk['peak_simd_gflops']:.1f} | {pk['bw_l1_gbs']:.1f} | {pk['bw_l2_gbs']:.1f} | {pk['bw_dram_gbs']:.1f} |")
    lines+=['','![Roofline at every thread count](roofline.png)','',
            '## Per-shape results at 10 threads','',
            '| M | K×N | V6 ms (geomean over sparsities) | V8 ms | V8/V6 |','|---:|---|---:|---:|---:|']
    for M in MS:
        for K,N in SHAPES:
            rs=[r for r in regular if (r['threads'],r['M'],r['K'],r['N'])==(10,M,K,N)]
            lines.append(f"| {M} | {K}×{N} | {gm(r['v6_ms'] for r in rs):.3f} | {gm(r['v8_ms'] for r in rs):.3f} | {gm(r['v8_vs_v6'] for r in rs):.3f}× |")
    lines+=['','## Reproduction and data','',
            'Build: `cmake --build build --target bench_cpu peaks -j 4`. Run: `/usr/bin/python3 bench/compare_v6_v8.py`. Replot: `/usr/bin/python3 bench/compare_v6_v8.py --plots-only '+str(out.relative_to(ROOT))+'`.',
            'No correctness command is invoked by this script; every kernel benchmark explicitly passes `--verify none`.',
            '- [manifest.json](manifest.json): complete jobs, environment, flags context, source/binary hashes.',
            '- [raw.csv](raw.csv): all k values and requested threads; unchecked results marked max_err=-1.',
            '- [comparison.csv](comparison.csv): fixed common-k paired times, scaling, efficiency, roofline coordinates.',
            '- [tuned_comparison.csv](tuned_comparison.csv): independently best k per version and thread count.',
            '- [thread_summary.csv](thread_summary.csv): regular-grid aggregates.',
            '- `t01` … `t10`: benchmark logs, raw CSVs, and freshly measured ceilings.',
            '- `source/`: source snapshots, including the comparison script. No kernel changes were made for this campaign.',
            '','Earlier bottleneck investigation: [previous report](../v6_investigation/report.md). CSV files remain subject to the repository’s existing `*.csv` ignore rule.']
    report='\n'.join(lines)+'\n'
    (out/'report.md').write_text(report)
    # Publish the same full report at the requested root path, fixing relative links.
    import re
    prefix=out.relative_to(ROOT).as_posix()
    report=re.sub(r'\]\(([^)]+)\)', lambda m: ']('+str((out/m[1]).resolve().relative_to(ROOT))+')', report)
    (ROOT/'report.md').write_text(report)
    (out/'validation.json').write_text(json.dumps(dict(configurations=len(rows),timed_calls=len(rows)*manifest['rounds'],
        correctness_run=False,unchecked_configurations=len(rows),comparison_points=len(paired),
        source_matches_current={v:(out/f'source/cpu/spr/spr_{v}.cpp').read_bytes()==(ROOT/f'cpu/spr/spr_{v}.cpp').read_bytes() for v in ('v6','v8')}),indent=2)+'\n')
    print('Completed:',out,flush=True)
    print('Regular-grid V8/V6 at 10 threads:',by_t[-1]['v8_vs_v6'],flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds',type=int,default=21)
    ap.add_argument('--plots-only',type=Path)
    a=ap.parse_args()
    if a.plots_only:
        summarize(a.plots_only.resolve());return
    stamp=datetime.datetime.now(ZoneInfo('Europe/Zurich')).strftime('%Y%m%d-%H%M%S')
    out=ROOT/'results'/f'{stamp}_spr_v6_v8_threads_1_10'
    out.mkdir(parents=True)
    env=dict(os.environ,OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0',OMP_PROC_BIND='close',OMP_PLACES='cores')
    jobs=[dict(category='regular',pattern='iid',M=M,K=K,N=N,sparsities=[.5,.9]) for M in MS for K,N in SHAPES]
    jobs += [dict(category='tiny',pattern='iid',M=1,K=128,N=128,sparsities=[.5]),
             dict(category='decode',pattern='iid',M=1,K=1024,N=4096,sparsities=[.5,.9])]
    jobs += [dict(category='tail',pattern='iid',M=M,K=1024,N=4096,sparsities=[.5,.9]) for M in (63,65)]
    jobs += [dict(category='narrow',pattern='iid',M=M,K=1024,N=32,sparsities=[.5,.9]) for M in (32,64,320)]
    jobs += [dict(category='clustered',pattern='column_clustered',M=M,K=1024,N=4096,sparsities=[.5]) for M in (64,320)]
    cpu=subprocess.check_output(['sysctl','machdep.cpu.brand_string','hw.physicalcpu','hw.logicalcpu','hw.perflevel0.physicalcpu','hw.perflevel1.physicalcpu','hw.cachelinesize','hw.perflevel0.l1dcachesize','hw.perflevel0.l2cachesize'],text=True)
    paths=['cpu/spr/spr_v6.cpp','cpu/spr/spr_v8.cpp','cpu/core.cpp','include/spr/cache.hpp','include/spr/harness.hpp','include/spr/kernel.hpp','bench/bench_cpu.cpp','bench/peaks.cpp','bench/compare_v6_v8.py','CMakeLists.txt']
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
                     '--k','2,3,4','--threads',t,'--kernels',','.join('='+k for k in KERNELS),
                     '--rounds',a.rounds,'--warmup',3,'--prep-rounds',1,'--verify','none','--ref-rows',0,
                     '--seed',42,'--tag','v6v8_'+job['category'],'--out',dest/'raw.csv'],dest/'benchmark.log',env)
    summarize(out)

if __name__=='__main__':
    main()
