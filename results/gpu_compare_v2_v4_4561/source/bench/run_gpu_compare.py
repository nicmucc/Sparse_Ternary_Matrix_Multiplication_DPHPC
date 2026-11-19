#!/usr/bin/env python3
"""Build and compare the current CUDA SPR sources; Python standard library only."""
import argparse
import csv
import datetime
import hashlib
import html
import io
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(argv, **kwargs):
    return subprocess.run([str(a) for a in argv], text=True, **kwargs)


def checked_command(argv, log, output, metadata):
    """Persist the command and its log, and stop the campaign on failure."""
    metadata["commands"].append([str(a) for a in argv])
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    result = command(argv, capture_output=True, cwd=ROOT)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Command failed (exit {result.returncode}); inspect {log}")


def check_correctness(output, build, versions, args, metadata):
    """Run both full-output reference suites before any benchmark samples."""
    tests = [build / "test_correctness", build / "test_gpu_correctness"]
    checked_command(["make", "-j2", f"GPU_BUILD_DIR={build}", f"CUDA_ARCH={args.arch}",
                     *tests], output / "compile_correctness.log", output, metadata)
    status = []
    for version in versions:
        for binary in tests:
            log = output / f"correctness_{version}_{binary.name}.log"
            checked_command([binary, f"spr.{version}.cuda"], log, output, metadata)
            status.append({"version": version, "suite": binary.name, "status": "PASS", "log": log.name})
            print(f"Correctness {version} {binary.name}: PASS", flush=True)
    (output / "correctness.json").write_text(json.dumps(status, indent=2))


def instrument(source):
    """Add event markers to a build-only copy, leaving the original source intact."""
    lines = source.splitlines(keepends=True)
    launches = stops = 0
    result = []
    for line in lines:
        if "<<<" in line:
            result.append("        spr_bench::mark_start();\n")
            launches += 1
        if "cudaGetLastError()" in line:
            result.append("        spr_bench::mark_stop();\n")
            stops += 1
        result.append(line)
    if launches != 1 or stops != 1:
        raise RuntimeError("Expected exactly one launch and launch-error check in each kernel source")
    return "".join(result)


def quartiles(values):
    if len(values) == 1:
        return values[0], values[0], values[0]
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[0], statistics.median(values), q[2]


def summarize(rows, output, versions=("v1", "v2")):
    baseline, candidate = versions
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("version", "M", "K", "N", "k", "zero_fraction", "metric"))
        groups.setdefault(key, []).append(float(row["ms"]))
    stats = {key: quartiles(values) for key, values in groups.items()}
    with (output / "timing_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "M", "K", "N", "k", "zero_fraction", "metric", "q1_ms", "median_ms", "q3_ms"])
        for key, values in sorted(stats.items()):
            writer.writerow([*key, *values])
    speedups = []
    for key, base in stats.items():
        if key[0] != baseline or key[-1] == "prepare_wall":
            continue
        other = stats.get((candidate, *key[1:]))
        if other and other[1] > 0:
            speedups.append(dict(zip(
                ("M", "K", "N", "k", "zero_fraction", "metric", f"{baseline}_ms", f"{candidate}_ms", "speedup"),
                (*key[1:], base[1], other[1], base[1] / other[1]))))
    fields = ["M", "K", "N", "k", "zero_fraction", "metric", f"{baseline}_ms", f"{candidate}_ms", "speedup"]
    with (output / "speedups.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(speedups)

    # Standalone plots for every sparsity; no plotting packages required.
    parts = ["<!doctype html><meta charset='utf-8'><title>SPR GPU comparison</title>",
             "<style>body{font:16px system-ui;max-width:1000px;margin:30px auto}svg{max-width:100%;height:auto}"
             "table{border-collapse:collapse}td,th{padding:5px 10px;border:1px solid #ddd}</style>",
             f"<h1>SPR {baseline} / {candidate} GPU comparison</h1><p>Speedup = median({baseline}) / median({candidate}). "
             f"Above 1 favors {candidate}. Full measurements and quartiles are in the CSV files.</p>"]
    shapes = sorted({(int(r["K"]), int(r["N"])) for r in speedups})
    for K, N in shapes:
        for zero, metric in itertools.product(sorted({float(r["zero_fraction"]) for r in speedups}),
                                             ("gpu_event", "run_wall", "end_to_end_wall")):
            selected = [r for r in speedups if int(r["K"]) == K and int(r["N"]) == N
                        and float(r["zero_fraction"]) == zero and r["metric"] == metric]
            if not selected:
                continue
            ms = sorted({int(r["M"]) for r in selected})
            logs = [math.log2(r["speedup"]) for r in selected]
            lo, hi = min(-1, math.floor(min(logs))), max(1, math.ceil(max(logs)))
            def x(m): return 70 + (math.log2(m) / max(1, math.log2(max(ms)))) * 620
            def y(value): return 290 - (math.log2(value) - lo) / (hi - lo) * 240
            parts.append(f"<h2>{html.escape(metric)}: K={K}, N={N}, {zero:.0%} zeros</h2><svg viewBox='0 0 800 340' role='img' "
                         "aria-label='Speedup versus input row count'>")
            for exponent in range(lo, hi + 1):
                value = 2 ** exponent
                parts.append(f"<path d='M70 {y(value)}H690' stroke='#ddd'/><text x='10' y='{y(value)+5}'>{value:g}×</text>")
            for m in ms:
                parts.append(f"<text x='{x(m)}' y='315' text-anchor='middle'>{m}</text>")
            parts.append("<text x='350' y='338'>Input rows M (log scale)</text>")
            colors = ["#2563eb", "#c2410c", "#15803d"]
            for index, k in enumerate(sorted({int(r["k"]) for r in selected})):
                series = sorted((int(r["M"]), r["speedup"]) for r in selected if int(r["k"]) == k)
                points = " ".join(f"{x(m):.2f},{y(s):.2f}" for m, s in series)
                color = colors[index % len(colors)]
                parts.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'/>")
                for m, s in series:
                    parts.append(f"<circle cx='{x(m)}' cy='{y(s)}' r='4' fill='{color}'><title>M={m}, k={k}: {s:.3f}×</title></circle>")
                parts.append(f"<text x='715' y='{70 + index * 25}' fill='{color}'>k={k}</text>")
            parts.append("</svg>")
    parts.append("<p>Kernel-event timing excludes bind/finish. End-to-end includes allocation, "
                 "input conversion/upload, synchronized run, and output download. Sources are unchanged; "
                 "event hooks exist only in generated build copies.</p>")
    (output / "comparison.html").write_text("\n".join(parts))
    return speedups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "full"), default="full")
    parser.add_argument("--versions", nargs=2, choices=("v1", "v2", "v3", "v4"), default=("v2", "v3"),
                        metavar=("BASELINE", "CANDIDATE"), help="Speedup is baseline / candidate")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arch", default="native")
    parser.add_argument("--profile", action="store_true", help="Also attempt representative Nsight Compute profiles")
    parser.add_argument("--sanitize", action="store_true", help="Require Compute Sanitizer checks before timing")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.versions[0] == args.versions[1]:
        parser.error("--versions must select two different implementations")
    versions = tuple(args.versions)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "results" / f"gpu_compare_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    build = output / "build"
    build.mkdir()
    metadata = {"utc": stamp, "suite": args.suite, "samples": args.samples,
                "versions": versions, "status": "preparing",
                "arch": args.arch, "seed": 42, "warmups": 5,
                "validation": "Full reference when M*K*N <= 1e7; otherwise 8 rows x 64 columns, plus all-output finite check",
                "cache_policy": "Repeated identical X and W; no explicit cache flush",
                "sources": {}, "commands": []}
    for label, argv in (("gpu", ["nvidia-smi"]), ("nvcc", ["nvcc", "--version"]),
                        ("host", ["hostname"]), ("allocation", ["printenv", "SLURM_JOB_ID"])):
        result = command(argv, capture_output=True)
        metadata[label] = result.stdout + result.stderr
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    for relative in ("Makefile", "include/spr/kernel.hpp", "include/spr/harness.hpp", "cpu/core.cpp",
                     "tests/test_correctness.cpp", "tests/test_gpu_correctness.cpp",
                     "bench/bench_gpu_compare.cu", "bench/run_gpu_compare.py",
                     "bench/plot_gpu_compare.py", "bench/gpu_compare.sbatch"):
        path = ROOT / relative
        metadata["sources"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for source in sorted((ROOT / "gpu" / "spr").glob("*.cu")):
        relative = str(source.relative_to(ROOT))
        metadata["sources"][relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    check_correctness(output, build, versions, args, metadata)
    binaries = {}
    for version in versions:
        source = ROOT / "gpu" / "spr" / f"spr_{version}.cu"
        content = source.read_text()
        metadata["sources"][str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, output / source.name)
        generated = build / f"instrumented_{version}.cu"
        generated.write_text(instrument(content))
        binary = build / f"bench_{version}"
        argv = ["nvcc", "-O3", "-std=c++17", "-lineinfo", f"-arch={args.arch}",
                f"-I{ROOT / 'include'}", f'-DSPR_BENCH_SOURCE="{generated}"',
                ROOT / "bench" / "bench_gpu_compare.cu", "-o", binary]
        checked_command(argv, output / f"compile_{version}.log", output, metadata)
        binaries[version] = binary

    if args.sanitize:
        sanitizer = shutil.which("compute-sanitizer")
        if not sanitizer:
            raise RuntimeError("--sanitize requires compute-sanitizer in PATH")
        for version, tool, M in itertools.product(versions, ("memcheck", "racecheck", "synccheck"), (1, 33)):
            log = output / f"sanitize_{version}_{tool}_M{M}.log"
            checked_command([sanitizer, "--tool", tool, "--error-exitcode", "1",
                             binaries[version], version, M, 129, 17, 8, .5, 42, 1, 1],
                            log, output, metadata)
            print(f"Sanitizer {version} {tool} M={M}: PASS", flush=True)
    metadata["sanitizer_requested"] = args.sanitize
    metadata["status"] = "benchmarking"
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))

    shapes = [(257, 257)] if args.suite == "smoke" else [(1024, 1024), (4096, 4096)]
    batches = [1, 33, 257] if args.suite == "smoke" else [1, 8, 31, 32, 33, 128, 512]
    widths = [4, 8] if args.suite == "smoke" else [4, 8, 16]
    sparsities = [.5] if args.suite == "smoke" else [0., .5, .9]
    configs = list(itertools.product(shapes, batches, widths, sparsities))
    metadata["configurations_per_version"] = len(configs)
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    rows, failures = [], []
    for index, ((K, N), M, k, zero) in enumerate(configs):
        # Alternate version order; both use identical seeded inputs.
        order = versions if index % 2 == 0 else versions[::-1]
        for version in order:
            argv = [binaries[version], version, M, K, N, k, zero, 42, args.samples, 0]
            result = command(argv, capture_output=True)
            label = f"{version}_M{M}_K{K}_N{N}_k{k}_z{zero}"
            (output / f"{label}.log").write_text(result.stderr)
            if result.returncode:
                failures.append({"case": label, "returncode": result.returncode, "error": result.stderr})
                print(f"FAIL {label}: {result.stderr.strip()}", flush=True)
                continue
            new_rows = list(csv.DictReader(io.StringIO(result.stdout)))
            if len(new_rows) != 1 + 3 * args.samples:
                raise RuntimeError(f"Unexpected result row count for {label}")
            rows.extend(new_rows)
            with (output / "raw_samples.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[{index + 1}/{len(configs)}] {label}: PASS", flush=True)
    (output / "failures.json").write_text(json.dumps(failures, indent=2))
    metadata["status"] = "failed" if failures else "complete"
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} benchmark cases failed; inspect {output / 'failures.json'}. No comparison plots generated.")
    speedups = summarize(rows, output, versions)

    if args.profile:
        profiler = shutil.which("ncu")
        profile_status = []
        if profiler:
            for M in (1, 128):
                for version in versions:
                    prefix = output / f"profile_{version}_M{M}"
                    argv = [profiler, "--set", "full", "--cache-control", "none",
                            "--launch-skip", "5", "--launch-count", "1",
                            "--force-overwrite", "--export", prefix, binaries[version], version,
                            M, 4096, 4096, 8, .5, 42, 1, 1]
                    result = command(argv, capture_output=True)
                    prefix.with_suffix(".log").write_text(result.stdout + result.stderr)
                    profile_status.append({"version": version, "M": M, "returncode": result.returncode})
        else:
            profile_status.append({"error": "ncu is not available; no hardware metrics/roofline collected"})
        (output / "profiling_status.json").write_text(json.dumps(profile_status, indent=2))
    print(f"Results: {output}")
    print(f"Matched speedup measurements: {len(speedups)}; failed version/configurations: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
