#!/usr/bin/env python3
"""Audit and interpret the helper-0 combine experiment against the previous v3 job."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OLD = NEW.parent / "gpu_compare_v2_v3_4493"
sys.path.insert(0, str(NEW / "source" / "bench"))
from plot_gpu_compare import load_results

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def audit(root):
    env, versions, workloads, stats, raw = load_results(root)
    assert versions == ("v2", "v3") and env["suite"] == "smoke"
    assert env["samples"] == 5 and len(workloads) == 6 and len(raw) == 192
    for relative, expected in env["sources"].items():
        assert hashlib.sha256((root / "source" / relative).read_bytes()).hexdigest() == expected, relative
    checks = json.loads((root / "correctness.json").read_text())
    assert len(checks) == 4 and all(c["status"] == "PASS" for c in checks)
    cases = {}
    for version in versions:
        text = (root / f"correctness_{version}_test_correctness.log").read_text()
        common = re.search(r"(\d+) cases passed, (\d+) failed", text)
        assert common and common[2] == "0"
        edge_text = (root / f"correctness_{version}_test_gpu_correctness.log").read_text()
        edge = re.search(r"PASS .*: (\d+) cases", edge_text)
        assert edge
        cases[version] = int(common[1]) + int(edge[1])
    logs = sorted(root.glob("sanitize_*.log"))
    assert len(logs) == 12
    for p in logs:
        expected = "0 hazards displayed (0 errors, 0 warnings)" if "racecheck" in p.name else "ERROR SUMMARY: 0 errors"
        assert expected in p.read_text(), p
    logs = sorted(root.glob("v[23]_M*.log"))
    assert len(logs) == 12 and all("PASS, final max error" in p.read_text() for p in logs)
    return env, stats, raw, cases


def main():
    old_env, old, old_raw, old_cases = audit(OLD)
    env, new, raw, cases = audit(NEW)
    changed = [p for p, digest in old_env["sources"].items() if env["sources"].get(p) != digest]
    assert changed == ["gpu/spr/spr_v3.cu"], changed
    assert env["nvcc"] == old_env["nvcc"]
    old_code = (OLD / "source/gpu/spr/spr_v3.cu").read_text()
    new_code = (NEW / "source/gpu/spr/spr_v3.cu").read_text()
    assert "s >>= 1" in old_code and "s >>= 1" not in new_code
    assert "total += partial[h * blockDim.x + threadIdx.x]" in new_code
    assert "dim3 threads(32, 8)" in old_code and "dim3 threads(32, 8)" in new_code
    Ms, ks = [1, 33, 257], [4, 8]
    metrics = ("gpu_event", "run_wall", "end_to_end_wall")
    def value(data, version, m, k, metric):
        return data[version, m, 257, 257, k, .5, metric][1] * 1000
    rows = []
    for m in Ms:
        for k in ks:
            r = dict(M=m, k=k)
            for metric in metrics:
                for version in ("v2", "v3"):
                    before, after = value(old, version, m, k, metric), value(new, version, m, k, metric)
                    r[f"old_{version}_{metric}_us"] = before
                    r[f"new_{version}_{metric}_us"] = after
                    r[f"{version}_{metric}_change_pct"] = 100 * (after / before - 1)
                r[f"old_{metric}_speedup"] = value(old, "v2", m, k, metric) / value(old, "v3", m, k, metric)
                r[f"new_{metric}_speedup"] = value(new, "v2", m, k, metric) / value(new, "v3", m, k, metric)
            rows.append(r)
    with (HERE / "old_new_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, metric, title in zip(axes, ("gpu_event", "end_to_end_wall"), ("GPU kernel", "End-to-end")):
        vals = np.array([[next(r[f"new_{metric}_speedup"] for r in rows if r["M"] == m and r["k"] == k)
                          for m in Ms] for k in ks])
        im = ax.imshow(np.log2(vals), cmap="RdBu", norm=TwoSlopeNorm(vmin=-1.1, vcenter=0, vmax=1.1), aspect="auto")
        for j, k in enumerate(ks):
            for i, m in enumerate(Ms):
                ax.text(i, j, f"{vals[j,i]:.2f}×", ha="center", va="center", fontsize=17,
                        fontweight="bold", color="white" if abs(math.log2(vals[j,i])) > .6 else "#172033")
        ax.set_xticks(range(3), [str(m) for m in Ms])
        ax.set_yticks(range(2), [f"k={k}" for k in ks])
        ax.set_xlabel("Input rows M")
        ax.set_title(title)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=.85, pad=.03, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["0.5×", "1×", "2×"])
    cb.set_label("v2 / updated v3: above 1 favors v3")
    fig.suptitle("Helper-0 combine: updated v3 versus v2 in job 4548", fontsize=16)
    fig.supxlabel("RTX 5060 Ti · K=N=257 · 50% zeros · ratios of medians from 5 samples", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"01_new_vs_v2.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained", sharex=True)
    positions = np.arange(len(rows))
    for ax, metric, title in zip(axes, ("gpu_event", "end_to_end_wall"), ("GPU kernel", "End-to-end")):
        all_values = []
        for vi, (version, color, label) in enumerate((("v2", "#94a3b8", "Unchanged v2 (run-to-run context)"),
                                                     ("v3", "#ea580c", "v3: tree → helper-0 combine"))):
            vals = [r[f"{version}_{metric}_change_pct"] for r in rows]
            all_values += vals
            bars = ax.bar(positions + (vi-.5)*.36, vals, width=.34, color=color, label=label, zorder=3)
            for bar, val in zip(bars, vals):
                ax.annotate(f"{val:+.1f}%", (bar.get_x()+bar.get_width()/2, val), xytext=(0, 3 if val >= 0 else -3),
                            textcoords="offset points", ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
        lo, hi = min(all_values), max(all_values)
        ax.set_ylim(lo - max(1.5, (hi-lo)*.2), hi + max(2, (hi-lo)*.35))
        ax.axhline(0, color="#333333", linewidth=1)
        ax.grid(True, axis="y", alpha=.2, zorder=0)
        ax.set_ylabel("Median time change (%)\nnegative = lower latency")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    axes[-1].set_xticks(positions, [f"M={r['M']}\nk={r['k']}" for r in rows])
    fig.suptitle("Old job 4493 → new job 4548: small GPU changes, larger end-to-end changes", fontsize=15)
    fig.supxlabel("Separate jobs, 5 samples each; bars are changes in medians, not isolated optimization effects or confidence intervals", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"02_old_new_change.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    buses = [re.search(r"00000000:([0-9A-Fa-f:.]+)", e["gpu"])[1] for e in (old_env, env)]
    report = ["# Interpretation: helper-0 final reduction experiment (job 4548)", "",
        "The updated v3 retains cooperative contribution accumulation and replaces the shared-memory tree reduction with a sequential combine of eight partial sums by helper 0. It removes three intermediate block-wide barriers per level, leaving two; it is not free of all reduction or synchronization.", "",
        "## Scope and validation", "",
        "Both jobs are smoke runs: RTX 5060 Ti, CUDA compiler 12.9, sm_120, K=N=257, 50% zeros, M={1,33,257}, k={4,8}, seed 42, five warm-ups and five samples per inference metric.", "",
        f"Each version passed {cases['v3']} reference/edge cases in job 4548. All 12 sanitizer invocations reported zero errors, with zero racecheck warnings. All 12 benchmark executions passed validation and failures.json is empty. Medians and quartiles were recomputed from 192 raw records. Source-snapshot hashes match metadata in both runs; only gpu/spr/spr_v3.cu changed among the recorded source files. V2 and the benchmark harness are unchanged.", "",
        "Large smoke workloads (M=257) use sampled FP64 reference checks plus a finite-output scan; the dedicated correctness suites check every output of their own test cases.", "",
        "## Updated v3 against v2: same-job comparison", "",
        "Speedup is median(v2)/median(updated v3), above 1 favoring updated v3.", "",
        "| M | k | GPU v2 (µs) | GPU updated v3 (µs) | GPU speedup | End-to-end speedup |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['M']} | {r['k']} | {r['new_v2_gpu_event_us']:.3f} | {r['new_v3_gpu_event_us']:.3f} | {r['new_gpu_event_speedup']:.3f}× | {r['new_end_to_end_wall_speedup']:.3f}× |")
    report += ["", "![Same-job speedups](01_new_vs_v2.png)", "",
        "- Small-batch GPU wins remain: 1.64–2.01× at M=1 and 1.49–1.52× at M=33.",
        "- At M=257,k=4, updated v3 takes 60.128 µs versus v2's 50.752 µs, or 18.5% longer. The former 24.4% gap narrows but the larger-batch kernel regression remains.",
        "- At M=257,k=8, GPU-event medians differ by only 0.8% in v3's favor, while synchronized run() is 1.3% slower. Treat this as near parity rather than a convincing win.",
        "- End-to-end medians favor v3 in all six cases, from 1.02× to 1.54×. These include allocations, transposition and transfers, so they are not pure reduction-kernel effects. The M=1,k=4 end-to-end IQRs overlap.",
        "- k=4 is still the fastest tested width for both versions at every M. The 2.01× matched-k speedup at M=1,k=8 is not the fastest configuration.", "",
        "## Did removing the tree help relative to the previous v3?", "",
        "| M | k | Previous v3 GPU (µs) | Updated v3 GPU (µs) | v3 time change | Unchanged v2 time change |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['M']} | {r['k']} | {r['old_v3_gpu_event_us']:.3f} | {r['new_v3_gpu_event_us']:.3f} | {r['v3_gpu_event_change_pct']:+.2f}% | {r['v2_gpu_event_change_pct']:+.2f}% |")
    report += ["", "![Cross-job median changes](02_old_new_change.png)", "",
        "The observed GPU change is modest: five medians improve by about 2.2–3.8%, and one is unchanged. Old/new v3 GPU IQRs overlap in five of six cases. The unchanged v2 GPU baseline also moves by approximately -2.3% to +2.2%, which is comparable to these gains. This suggests a small potential benefit, not a demonstrated stable speedup from the optimization alone.", "",
        f"The jobs report the same hostname and GPU model, but different PCI bus addresses ({buses[0]} versus {buses[1]}). The metadata does not establish that the same physical GPU was used. Clocks were not locked, and this is a cross-job comparison, not an interleaved comparison of the two v3 implementations.", "",
        "The end-to-end changes need particular care:", "",
        "- At M=257,k=4, v3 changes from 466.235 to 455.556 µs (-2.3%), while unchanged v2 changes from 466.050 to 499.265 µs (+7.1%). Much of the new 1.10× v2/v3 end-to-end ratio comes from the baseline becoming slower.",
        "- At M=257,k=8, v3 end-to-end time falls 13.4%, but its GPU time falls only 2.2%. That larger total-latency improvement cannot be assigned solely to the removed reduction barriers; setup/transfer phases and cross-run variation were not separately measured.", "",
        "## Interpretation and next step", "",
        "The simpler final combine is a reasonable candidate to keep for further testing, but it has not solved v3's larger-batch kernel disadvantage. Parallel segment sums and contribution accumulation remain, as do global scratch accesses, two block-wide barriers per level and sequential merges. Their relative runtime costs have not been profiled; fewer barriers alone do not guarantee proportional speedup.", "",
        "For a clean next experiment, compile both old and updated v3 from their recorded snapshots and benchmark them in the same allocation, alternating their order with at least 30 samples and repeated measurement batches. Keep M,K,N,k,helper count and timing boundaries fixed. Then run the larger full campaign. Do not select an optimal helper count or universal dispatch threshold from this five-sample, one-shape experiment.", "",
        "Additional latency/IQR plots are in K257_N257_z0.5.png. The reproducible numerical comparison is old_new_comparison.csv.", ""]
    (HERE / "interpretation.md").write_text("\n".join(report))
    document = ["<!doctype html><meta charset='utf-8'><title>Helper-0 combine experiment</title>",
        "<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;color:#172033}img{width:100%}</style>",
        "<h1>Helper-0 combine experiment: v2 versus updated v3</h1>",
        "<p>Job 4548 · RTX 5060 Ti · smoke run · five samples per metric. All recorded correctness/sanitizer checks passed.</p>",
        "<p>The updated v3 retains small-batch wins. Across jobs, GPU medians are only 0–3.8% lower than previous v3; unchanged v2 also varies. This is evidence for a small possible benefit, the data do not establish an isolated optimization effect.</p>",
        "<p><a href='interpretation.md'>Detailed interpretation</a> · <a href='old_new_comparison.csv'>Numerical comparison</a></p>"]
    for stem, title in (("01_new_vs_v2", "Same-job comparison against v2"),
                        ("02_old_new_change", "Cross-job changes with unchanged v2 as context"),
                        ("K257_N257_z0.5", "Absolute latency and sample IQR")):
        document += [f"<h2>{html.escape(title)}</h2><img src='{stem}.png' alt='{html.escape(title)}'>",
                     f"<p><a href='{stem}.pdf'>PDF</a></p>"]
    (HERE / "report.html").write_text("\n".join(document))
    inputs = [r / name for r in (OLD, NEW) for name in ("environment.json", "raw_samples.csv", "timing_summary.csv", "correctness.json", "failures.json")]
    (HERE / "experiment_validation.json").write_text(json.dumps({
        "jobs": [4493, 4548], "changed_sources": changed, "correctness_cases_per_version": cases,
        "sanitizer_checks_per_job": 12, "samples_per_metric": 5, "workloads_per_version": 6,
        "input_hashes": {str(p.relative_to(NEW.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    }, indent=2))
    print("Validated both runs, source hashes, correctness and sanitizer logs; generated plots and interpretation.")
    print(HERE / "interpretation.md")


if __name__ == "__main__":
    main()
