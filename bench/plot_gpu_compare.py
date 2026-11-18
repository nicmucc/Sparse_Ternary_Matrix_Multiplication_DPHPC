#!/usr/bin/env python3
"""Plot recorded SPR GPU benchmarks; requires numpy and matplotlib."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'spr-matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

METRICS = ['gpu_event', 'run_wall', 'end_to_end_wall']
LABELS = {'gpu_event': 'GPU kernel', 'run_wall': 'Synchronized run()', 'end_to_end_wall': 'End-to-end inference'}
COLORS = {'v1': '#2563eb', 'v2': '#ea580c'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--job-log', type=Path, required=True)
    args = p.parse_args()
    root = args.run.resolve()
    out = root / 'plots'
    out.mkdir(exist_ok=True)
    failures = json.loads((root / 'failures.json').read_text())
    assert not failures, 'Run has failures; inspect failures.json before comparing'
    raw = list(csv.DictReader((root / 'raw_samples.csv').open()))
    groups = collections.defaultdict(list)
    errors = []
    for r in raw:
        key = (r['version'], int(r['M']), int(r['K']), int(r['N']), int(r['k']), float(r['zero_fraction']), r['metric'])
        groups[key].append(float(r['ms']))
        errors.append(float(r['max_abs_error']))
    env = json.loads((root / 'environment.json').read_text())
    n = env['samples']
    assert all(len(v) == (1 if key[-1] == 'prepare_wall' else n) for key, v in groups.items())
    summary = {key: (np.percentile(v, 25), statistics.median(v), np.percentile(v, 75)) for key, v in groups.items()}
    recorded = list(csv.DictReader((root / 'timing_summary.csv').open()))
    assert len(recorded) == len(summary)
    for r in recorded:
        key = (r['version'], int(r['M']), int(r['K']), int(r['N']), int(r['k']), float(r['zero_fraction']), r['metric'])
        assert abs(float(r['median_ms']) - summary[key][1]) < 1e-8
    log = args.job_log.read_text()
    passes = re.findall(r'\[(\d+)/126\].*?: PASS', log)
    assert len(passes) == 252 and 'failed version/configurations: 0' in log
    profile = json.loads((root / 'profiling_status.json').read_text())
    assert all(r.get('returncode') == 0 for r in profile)
    Ms = sorted({key[1] for key in summary})
    shapes = sorted({(key[2], key[3]) for key in summary})
    ks = sorted({key[4] for key in summary})
    zeros = sorted({key[5] for key in summary})
    x = np.arange(len(Ms))
    def stat(version, M, shape, k, z, metric):
        return summary[(version, M, *shape, k, z, metric)]
    def speed(M, shape, k, z, metric):
        return stat('v1', M, shape, k, z, metric)[1] / stat('v2', M, shape, k, z, metric)[1]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': '#ffffff',
                         'axes.grid': True, 'grid.alpha': .18, 'savefig.facecolor': '#ffffff'})
    assets = []
    def save(fig, stem, caption):
        fig.savefig(out / (stem + '.png'), dpi=170, bbox_inches='tight')
        fig.savefig(out / (stem + '.pdf'), bbox_inches='tight')
        plt.close(fig)
        assets.append((stem, caption))

    fig, axes = plt.subplots(4, 3, figsize=(17, 12), layout='constrained')
    for si, shape in enumerate(shapes):
        for ti, metric in enumerate(['gpu_event', 'end_to_end_wall']):
            for zi, z in enumerate(zeros):
                ax = axes[si * 2 + ti, zi]
                vals = np.array([[speed(m, shape, k, z, metric) for m in Ms] for k in ks])
                image = ax.imshow(np.log2(vals), aspect='auto', cmap='RdBu', norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=6))
                ax.grid(False)
                for i in range(len(ks)):
                    for j in range(len(Ms)):
                        ax.text(j, i, f'{vals[i,j]:.2f}', ha='center', va='center',
                                color='white' if np.log2(vals[i,j]) > 3.8 else '#111827', fontsize=10)
                ax.set_xticks(x, Ms)
                ax.set_yticks(range(len(ks)), [f'k={k}' for k in ks])
                ax.set_title(f'{shape[0]} × {shape[1]} weights · {int(z*100)}% zeros\n{LABELS[metric]}', fontsize=11)
                ax.set_xlabel('Input rows M')
    cb = fig.colorbar(image, ax=axes, shrink=.7, pad=.015, ticks=[-1, 0, 1, 2, 3, 4, 5, 6])
    cb.ax.set_yticklabels(['0.5×', '1×', '2×', '4×', '8×', '16×', '32×', '64×'])
    cb.set_label('Speedup v1 / v2; >1 favors v2 (log color scale)')
    fig.suptitle('V2 wins on GPU time everywhere; setup costs reduce end-to-end gains', fontsize=19)
    save(fig, '01_speedup_overview', 'All 126 workloads, for GPU-event and end-to-end timing. Numbers are ratios of medians; red means v2 is slower.')

    for number, metric in [(2, 'gpu_event'), (3, 'end_to_end_wall')]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
        for si, shape in enumerate(shapes):
            for ki, k in enumerate(ks):
                ax = axes[si, ki]
                for version in ['v1', 'v2']:
                    a = np.array([stat(version, m, shape, k, .5, metric) for m in Ms])
                    ax.plot(x, a[:,1], 'o-', color=COLORS[version], label=version, lw=2, ms=4)
                    ax.fill_between(x, a[:,0], a[:,2], color=COLORS[version], alpha=.16)
                ax.set_yscale('log')
                ax.set_xticks(x, Ms)
                ax.set_title(f'K=N={shape[0]}, k={k}')
                ax.set_xlabel('Input rows M (equally spaced categories)')
                ax.set_ylabel('Latency (ms, log scale)')
                ax.legend()
        fig.suptitle(f'{LABELS[metric]} latency · 50% zero weights\nMedian and interquartile range over 30 samples', fontsize=17)
        save(fig, f'{number:02d}_{metric}_latency', 'Absolute latency matters: a large speedup at one k does not imply that k is fastest. Bands show sample IQR, not confidence intervals.')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for si, shape in enumerate(shapes):
        for zi, z in enumerate([.5, .9]):
            ax = axes[si, zi]
            for vi, version in enumerate(['v1', 'v2']):
                run = stat(version, 512, shape, 4, z, 'run_wall')[1]
                e2e = stat(version, 512, shape, 4, z, 'end_to_end_wall')[1]
                extra = e2e - run
                assert extra >= 0
                ax.bar(vi, run, color='#2563eb', label='Synchronized run()' if vi == 0 else None, width=.55)
                ax.bar(vi, extra, bottom=run, color='#fbbf24', label='Estimated time outside run()' if vi == 0 else None, width=.55)
                ax.text(vi, e2e, f'{e2e:.2f} ms', ha='center', va='bottom', fontsize=11)
                ax.text(vi, run + extra/2, f'{100*extra/e2e:.0f}%\noutside run', ha='center', va='center', fontsize=9)
            ax.set_xticks([0,1], ['v1', 'v2'])
            ax.set_ylabel('Latency (ms)')
            ax.set_title(f'K=N={shape[0]} · {int(z*100)}% zeros\nM=512, k=4')
            ax.set_ylim(0, ax.get_ylim()[1]*1.13)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=10)
    fig.suptitle('Why GPU speedup is not the same as end-to-end speedup', fontsize=17)
    save(fig, '04_setup_and_transfer_cost', 'Yellow is median(end-to-end) minus median(run), an estimate including allocations and transfers plus v2 input transposition. It is not a direct measurement of transpose alone.')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    best_rows = []
    for si, shape in enumerate(shapes):
        for ti, metric in enumerate(['gpu_event','end_to_end_wall']):
            ax = axes[si,ti]
            best_by_version = {}
            for version in ['v1','v2']:
                best = [min((stat(version,m,shape,k,.5,metric)[1],k) for k in ks) for m in Ms]
                best_by_version[version] = best
                ax.plot(x, [a[0] for a in best], 'o-', label=f'{version}: best tested k', color=COLORS[version], lw=2)
                for m, (ms,k) in zip(Ms,best):
                    best_rows.append([version,m,*shape,.5,metric,k,ms])
            for i in [0,3,5,6]:
                a,b=best_by_version['v1'][i][0],best_by_version['v2'][i][0]
                ax.annotate(f'{a/b:.2f}×', (i,b), xytext=(0,-17),textcoords='offset points',ha='center',color=COLORS['v2'],fontsize=9)
            ax.set_yscale('log')
            ax.margins(y=.20)
            ax.set_xticks(x,Ms)
            ax.set_xlabel('Input rows M (equally spaced categories)')
            ax.set_ylabel('Latency (ms, log scale)')
            ax.set_title(f'K=N={shape[0]} · {LABELS[metric]}')
            ax.legend(fontsize=9)
    fig.suptitle('Compare the fastest tested k for each version · 50% zero weights', fontsize=17)
    save(fig,'05_best_k_comparison','Each point independently selects the lowest median among k=4,8,16. Labels show v1/v2 speedup; selection uses the same timing samples, not a separate tuning run.')
    with (out/'best_k.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['version','M','K','N','zero_fraction','metric','best_k','median_ms']);writer.writerows(best_rows)

    counts = {}
    for metric in METRICS:
        vals=[speed(m,shape,k,z,metric) for shape in shapes for m in Ms for k in ks for z in zeros]
        counts[metric]={'wins':sum(v>1 for v in vals),'count':len(vals),'min':min(vals),'median':statistics.median(vals),'max':max(vals)}
    report = ['# SPR GPU comparison analysis', '',
              'Run: RTX 2080 Ti, CUDA 12.9, student cluster job 4443. The GPU and log metadata are read from the supplied result folders.', '',
              f'Validated {len(raw):,} raw rows, 30 samples per inference metric, all 252 version/configuration PASS entries, no failed cases, and four successful profiler invocations. Recomputed medians match timing_summary.csv. Maximum recorded initial validation error: {max(errors):.6g}. Large cases use sampled reference validation.', '',
              '| Timing | V2 wins | Median speedup across workloads | Speedup range |',
              '|---|---:|---:|---:|']
    for metric,c in counts.items():
        report.append(f"| {LABELS[metric]} | {c['wins']}/{c['count']} | {c['median']:.2f}× | {c['min']:.2f}–{c['max']:.2f}× |")
    report += ['', '## Interpretation', '',
               '- GPU speedup includes changes to launch geometry and warp utilization as well as memory access. These timings alone cannot isolate coalescing as the cause. V2 also improves M=1, where there are no neighboring input rows to coalesce across.',
               '- The two end-to-end regressions occur at K=N=1024, k=4, 90% zeros: M=128 gives 0.877× and M=512 gives 0.804×. The inexpensive kernel is outweighed by setup/conversion/transfer costs.',
               '- The largest matched-k GPU speedup is 38.56× at M=512, K=N=4096, k=8, 50% zeros: 1112.69 ms versus 28.86 ms. End-to-end is 1127.18 ms versus 47.15 ms (23.91×).',
               '- That largest ratio is not the fastest configuration. At the same M, shape and sparsity, k=4 gives end-to-end 57.55 ms versus 18.58 ms (3.10×). Use the best-k plot when comparing tuned versions.',
               '- Median(end-to-end) minus median(run) estimates combined non-run cost. It does not isolate transpose, allocation, upload or download, and separately timed phases can experience different cache states.',
               '- The 31/32/33 input-row cases show warp-boundary behavior, but differences can also reflect cache footprint, alignment and measurement variation. Bands are IQR within one run, not cross-job uncertainty.',
               '- Nsight reports are present, but are binary .ncu-repz files. No custom roofline or DRAM-traffic conclusion is inferred without exporting the hardware metrics. Their logs warn of uncontrolled caches; profile timings should not be merged with ordinary benchmark samples.',
               '- Results use repeated identical inputs, no explicit cache flush, and unlocked GPU clocks. Repeat on additional seeds/jobs before treating the largest ratios as stable general performance.', '',
               '## Figures', '']
    for stem,caption in assets:
        report += [f'### {stem}', '', caption, '', f'![{caption}]({stem}.png)', '']
    (out/'analysis.md').write_text('\n'.join(report))
    document=['<!doctype html><meta charset="utf-8"><title>SPR GPU benchmark plots</title>',
              '<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#172033}img{width:100%;height:auto}section{margin:40px 0}a{color:#2563eb}</style>',
              '<h1>SPR v1 versus v2: GPU benchmark plots</h1>',
              '<p>RTX 2080 Ti · 126 workloads per version · 30 samples per metric · all validation checks passed.</p>',
              '<p>V2 wins all 126 GPU-time comparisons and 124/126 end-to-end comparisons. Kernel-time median speedup across workloads is 3.93×; end-to-end median speedup is 2.45×. See <a href="analysis.md">analysis notes</a> for limitations and best-k comparisons.</p>']
    for stem,caption in assets:
        document.append(f'<section><h2>{html_escape(stem)}</h2><p>{html_escape(caption)}</p><img src="{stem}.png" alt="{html_escape(caption)}"><p><a href="{stem}.pdf">PDF</a> · <a href="{stem}.png">PNG</a></p></section>')
    (out/'index.html').write_text('\n'.join(document))
    (out/'validation.json').write_text(json.dumps({'raw_rows':len(raw),'pass_log_entries':len(passes),'sample_count':n,'max_initial_error':max(errors),'speedups':counts,'input_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'raw_samples.csv',root/'timing_summary.csv',args.job_log]}},indent=2))
    print(json.dumps(counts,indent=2)); print(f'Wrote {len(assets)} PNG/PDF figures and analysis to {out}')


def html_escape(text):
    import html
    return html.escape(text)


if __name__=='__main__':
    main()
