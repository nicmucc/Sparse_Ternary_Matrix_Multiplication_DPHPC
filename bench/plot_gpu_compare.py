#!/usr/bin/env python3
"""Validate and plot any recorded two-version SPR GPU comparison (PNG/PDF/HTML)."""
import argparse
import collections
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spr-matplotlib-cache"))

METRICS = ("gpu_event", "run_wall", "end_to_end_wall")
LABELS = {"gpu_event": "GPU kernel", "run_wall": "Synchronized run()", "end_to_end_wall": "End-to-end inference"}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return (row["version"], int(row["M"]), int(row["K"]), int(row["N"]),
            int(row["k"]), float(row["zero_fraction"]), row["metric"])


def load_results(root):
    """Reject failed, incomplete, duplicated or inconsistent timing records."""
    env = json.loads((root / "environment.json").read_text())
    if env.get("status", "complete") != "complete":
        raise ValueError("Campaign is incomplete; inspect environment.json and the job log")
    if json.loads((root / "failures.json").read_text()):
        raise ValueError("Campaign has failed cases; inspect failures.json")
    raw = read_csv(root / "raw_samples.csv")
    versions = tuple(env.get("versions", sorted({r["version"] for r in raw})))
    if len(versions) != 2 or len(set(versions)) != 2:
        raise ValueError("Expected two distinct versions")
    if {r["version"] for r in raw} != set(versions):
        raise ValueError("Recorded versions differ from the requested comparison")
    groups = collections.defaultdict(list)
    for r in raw:
        if not math.isfinite(float(r["ms"])) or float(r["ms"]) <= 0:
            raise ValueError("Timing samples must be finite and positive")
        if not math.isfinite(float(r["max_abs_error"])) or float(r["max_abs_error"]) > 2e-3 * math.sqrt(int(r["K"])):
            raise ValueError("Recorded validation error exceeds the reference tolerance")
        groups[key(r)].append(r)
    workloads = sorted({k[1:-1] for k in groups})
    expected = env.get("configurations_per_version", 6 if env["suite"] == "smoke" else 126)
    if len(workloads) != expected:
        raise ValueError(f"Expected {expected} workloads, found {len(workloads)}")
    summary = {}
    for version in versions:
        for workload in workloads:
            for metric in (*METRICS, "prepare_wall"):
                k = (version, *workload, metric)
                records = groups.get(k, [])
                count = 1 if metric == "prepare_wall" else env["samples"]
                if len(records) != count or {int(r["sample"]) for r in records} != set(range(count)):
                    raise ValueError(f"Missing or duplicated samples for {k}")
                values = [float(r["ms"]) for r in records]
                q = statistics.quantiles(values, n=4, method="inclusive") if count > 1 else values * 3
                summary[k] = (q[0], statistics.median(values), q[2])
    recorded = read_csv(root / "timing_summary.csv")
    if len(recorded) != len(summary) or len({key(r) for r in recorded}) != len(summary):
        raise ValueError("Timing summary has missing or duplicated groups")
    for r in recorded:
        actual = summary.get(key(r))
        if actual is None or any(not math.isclose(float(r[field]), value, rel_tol=1e-8, abs_tol=1e-9)
                                 for field, value in zip(("q1_ms", "median_ms", "q3_ms"), actual)):
            raise ValueError("Timing summary differs from raw samples")
    return env, versions, workloads, summary, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--job-log", type=Path, help="Optional job log to include in provenance hashes")
    args = parser.parse_args()
    root = args.run.resolve()
    env, versions, workloads, summary, raw = load_results(root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    baseline, candidate = versions
    out = root / "plots"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "savefig.facecolor": "white"})
    assets, best_rows = [], []
    shapes = sorted({(w[1], w[2], w[4]) for w in workloads})
    for K, N, zero in shapes:
        selected = [w for w in workloads if w[1:3] == (K, N) and w[4] == zero]
        Ms = sorted({w[0] for w in selected})
        ks = sorted({w[3] for w in selected})
        x = np.arange(len(Ms))
        fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
        for row, metric in enumerate(METRICS):
            latency, speed = axes[row]
            for ki, k in enumerate(ks):
                valid = [i for i, m in enumerate(Ms) if (m, K, N, k, zero) in selected]
                points = x[valid]
                color = f"C{ki % 10}"
                for vi, version in enumerate(versions):
                    values = np.array([summary[(version, Ms[i], K, N, k, zero, metric)] for i in valid])
                    latency.plot(points, values[:, 1], "o-" if vi == 0 else "s--", color=color,
                                 label=f"{version}, k={k}", markersize=4)
                    latency.fill_between(points, values[:, 0], values[:, 2], color=color, alpha=.1)
                ratios = [summary[(baseline, Ms[i], K, N, k, zero, metric)][1] /
                          summary[(candidate, Ms[i], K, N, k, zero, metric)][1] for i in valid]
                speed.plot(points, ratios, "o-", color=color, label=f"k={k}")
            latency.set_yscale("log")
            latency.set_ylabel("Latency (ms, log scale)")
            latency.set_title(f"{LABELS[metric]}: medians and sample IQR")
            speed.axhline(1, color="black", linestyle="--", linewidth=1)
            speed.set_yscale("log", base=2)
            speed.set_ylabel(f"Speedup {baseline}/{candidate} (log scale)")
            speed.set_title(f"Above 1 favors {candidate}; below 1 favors {baseline}")
            for ax in (latency, speed):
                ax.set_xticks(x, Ms)
                ax.set_xlabel("Input rows M (equally spaced categories)")
                ax.legend(fontsize=8)
            for m in Ms:
                for version in versions:
                    choices = [(summary[(version, m, K, N, k, zero, metric)][1], k)
                               for k in ks if (m, K, N, k, zero) in selected]
                    ms, best_k = min(choices)
                    best_rows.append([version, m, K, N, zero, metric, best_k, ms])
        fig.suptitle(f"SPR {baseline} versus {candidate}: K={K}, N={N}, {zero:.0%} zeros\n"
                     f"{env['samples']} samples per inference metric; speedups compare matching k")
        stem = f"K{K}_N{N}_z{zero:g}"
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{stem}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        assets.append(stem)
    counts = {}
    for metric in METRICS:
        ratios = [summary[(baseline, *w, metric)][1] / summary[(candidate, *w, metric)][1] for w in workloads]
        counts[metric] = {"candidate_wins": sum(r > 1 for r in ratios), "count": len(ratios),
                          "min": min(ratios), "median": statistics.median(ratios), "max": max(ratios)}
    report = [f"# SPR {baseline} versus {candidate}", "",
              f"{len(workloads)} workloads per version; {env['samples']} samples per inference metric.", "",
              "Hardware, compiler, commands and source hashes are in ../environment.json.", "",
              f"Speedup = median({baseline}) / median({candidate}); above 1 favors {candidate}.", "",
              f"| Timing | {candidate} wins | Median speedup across workloads | Range |",
              "|---|---:|---:|---:|"]
    for metric, c in counts.items():
        report.append(f"| {LABELS[metric]} | {c['candidate_wins']}/{c['count']} | {c['median']:.3f}x | {c['min']:.3f}–{c['max']:.3f}x |")
    report += ["", "Interpretation:", "",
               "- GPU-event time isolates the instrumented launch; synchronized run() includes host launch/synchronization overhead.",
               "- End-to-end time includes bind, run and finish, including input transposition, allocations and transfers.",
               "- Matched-k plots compare the same SPR width. best_k.csv selects the fastest tested k separately for each version using these same timing samples; it is not held-out tuning.",
               "- IQR bands describe variation within this run, not confidence intervals across jobs. Inputs repeat, caches are warm and GPU clocks are not locked.",
               "- Dedicated correctness suites check every output on their test cases. Large benchmark cases use sampled FP64 reference checks and a full finite-output scan.", ""]
    for stem in assets:
        report += [f"![Latency and speedup]({stem}.png)", ""]
    (out / "analysis.md").write_text("\n".join(report))
    with (out / "best_k.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "M", "K", "N", "zero_fraction", "metric", "best_k", "median_ms"])
        writer.writerows(best_rows)
    document = ["<!doctype html><meta charset='utf-8'><title>SPR GPU comparison plots</title>",
                "<style>body{font:16px system-ui;max-width:1200px;margin:30px auto}img{width:100%}</style>",
                f"<h1>SPR {baseline} versus {candidate}</h1><p>See <a href='analysis.md'>analysis notes</a>.</p>"]
    for stem in assets:
        document.append(f"<h2>{html.escape(stem)}</h2><img src='{stem}.png' alt='Latency and speedup plots'>"
                        f"<p><a href='{stem}.pdf'>PDF</a></p>")
    (out / "index.html").write_text("\n".join(document))
    paths = [root / name for name in ("raw_samples.csv", "timing_summary.csv", "environment.json")]
    if args.job_log:
        paths.append(args.job_log)
    (out / "validation.json").write_text(json.dumps({"versions": versions, "raw_rows": len(raw),
        "workloads": len(workloads), "speedups": counts,
        "input_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}, indent=2))
    print(json.dumps(counts, indent=2))
    print(f"Wrote {len(assets)} PNG/PDF figures and analysis to {out}")


if __name__ == "__main__":
    main()
