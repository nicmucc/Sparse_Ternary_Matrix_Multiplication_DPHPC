#!/usr/bin/env python3
"""Baseline tuning cache: the best (kernel, k) of every baseline method, per machine / threads / shape / sparsity.

bench/run.sh fingerprints the build, CPU, affinity and search grid before reusing choices.
Only verified measurements enter the cache. --include-spr also selects SPR configurations.
Final measurements are separate from tuning, including on independent input seeds.

    python3 bench/tune.py add raw_sweep.csv cache.csv               # store the best config of each method
    python3 bench/tune.py get cache.csv M K N s [all|scalar|opt]    # -> "=kernel@k,=kernel,..." ('' if missing)
"""
import csv
import os
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(__file__))
from summarize import method  # noqa: E402  (same method grouping as the summary)

FIELDS = ["M", "K", "N", "sparsity", "method", "kernel", "k", "median_ms"]


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def check(raw):
    rows = read_rows(raw)
    if not rows or any(r["ok"] != "1" or not math.isfinite(float(r["max_err"])) or not float(r["max_err"]) >= 0 or
                       not math.isfinite(float(r["median_ms"])) or not float(r["median_ms"]) > 0 for r in rows):
        raise ValueError("empty, wrong or unverified benchmark: " + str(raw))
    contexts = {(r["machine"], r["compiler"], r["threads"], r["tag"]) for r in rows}
    if len(contexts) != 1:
        raise ValueError("mixed benchmark contexts: " + str(raw))
    return rows


def add(raw, cache, include_spr=False):
    best = {}
    for r in check(raw):
        if r["kernel"].startswith("spr.") and not include_spr: continue
        key = (r["M"], r["K"], r["N"], float(r["sparsity"]), method(r["kernel"]))
        if key not in best or float(r["median_ms"]) < float(best[key]["median_ms"]): best[key] = r
    old = read_rows(cache) if os.path.exists(cache) else []
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
    for r in read_rows(cache):
        if (r["M"], r["K"], r["N"]) != (M, K, N) or abs(float(r["sparsity"]) - float(s)) > 1e-9: continue
        scalar = r["method"].endswith("scalar")
        if not r["kernel"].startswith("spr.") and (level == "scalar" and not scalar or level == "opt" and scalar): continue
        out.append("=" + r["kernel"] + ("" if r["k"] == "-" else "@" + r["k"]))
    return ",".join(out)


def context(build, out, settings):
    """Fingerprint actual executable/libraries/CPU/affinity plus explicit tuning settings."""
    def command(*args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    binary = Path(build) / "bench_cpu"
    libraries = command("ldd", str(binary)) if platform.system() == "Linux" else command("otool", "-L", str(binary))
    library_paths = sorted({word for line in libraries.splitlines() for word in line.split()
                            if word.startswith("/") and Path(word).is_file()})
    state = dict(settings=settings, platform=platform.platform(), hostname=platform.node(),
                 cpu=command("lscpu") if platform.system() == "Linux" else command("sysctl", "-n", "machdep.cpu.brand_string"),
                 affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                 build=Path(build, "config.txt").read_text(),
                 hashes={str(p): sha(p) for p in [binary, Path(build)/"test_correctness",
                         Path("bench/run.sh"), Path("bench/tune.py"), Path("bench/summarize.py"),
                         Path("bench/papers_cpu.sh"), Path("bench/spr_bench.sh"), *map(Path, library_paths)]},
                 environment={k: os.environ.get(k, "") for k in
                     ["OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "GOMP_CPU_AFFINITY",
                      "SPR_L1_BYTES", "SPR_L2_BYTES", "SPR_LLC_BYTES", "SPR_LINE_BYTES"]})
    state["cpu"] = "\n".join(line for line in state["cpu"].splitlines() if not any(x in line for x in ("BogoMIPS:", "CPU(s) scaling MHz:", "CPU MHz:")))
    identity = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:20]
    state.update(identity=identity, commit=command("git", "rev-parse", "HEAD"),
                 status=command("git", "status", "--porcelain"), libraries=libraries,
                 allocation={k: os.environ.get(k, "") for k in
                     ["SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
                      "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_NTASKS", "SLURM_MEM_PER_NODE",
                      "SLURM_MEM_PER_CPU", "SLURM_JOB_GPUS"]},
                 slurm_job=command("scontrol", "show", "job", os.environ["SLURM_JOB_ID"])
                           if os.environ.get("SLURM_JOB_ID") else "not a Slurm job")
    Path(out).write_text(json.dumps(state, indent=2) + "\n")
    print(identity)


def edges(raw):
    """Flag optima at either edge of the tested k grid; this is a bounded optimum."""
    groups = {}
    for r in check(raw):
        if r["param"].startswith("k="):
            groups.setdefault(method(r["kernel"]), []).append(r)
    for group, rows in groups.items():
        best = min(rows, key=lambda r: float(r["median_ms"]))
        ks = sorted({int(r["param"][2:]) for r in rows})
        k = int(best["param"][2:])
        if k == ks[-1] or (k == ks[0] and k > 1):
            print(f"{group}: best {best['kernel']} k={k} at boundary of {ks}; widen grid before claiming an optimum")


def scaling(root):
    """Fixed-configuration scaling: freeze v8 selected by independent one-thread tuning."""
    root = Path(root)
    selected = {}
    def shape(r):
        return (r["M"], r["K"], r["N"], float(r["sparsity"]))
    for path in sorted((root / "comparison").glob("*/selected.csv")):
        if not (path.parent / "summary.md").exists():
            continue
        metadata = json.loads((path.parent / "context.json").read_text())
        if not metadata["settings"].startswith("threads=1|"):
            continue
        for r in read_rows(path):
            if r["kernel"].startswith("spr.v8."):
                selected.setdefault(shape(r), (r["kernel"], "k=" + r["k"]))
    measured = {}
    for path in sorted((root / "roofline").glob("*/raw.csv")):
        if not all((path.parent / name).exists() for name in ("report.md", "roofline.png", "runtime.png")):
            continue  # interrupted campaigns retain raw data but are not complete measurements
        for r in check(path):
            if selected.get(shape(r)) == (r["kernel"], r["param"]):
                key = (shape(r), r["threads"])
                measured[key] = r  # latest complete run after a possible interrupted restart
    output = []
    for (sh, thread), r in sorted(measured.items()):
        base = measured.get((sh, "1"))
        if not base:
            continue
        output.append(dict(M=sh[0], K=sh[1], N=sh[2], sparsity=sh[3], threads=thread,
                           kernel=r["kernel"], param=r["param"], median_ms=r["median_ms"],
                           reference_ms=base["median_ms"],
                           speedup=float(base["median_ms"])/float(r["median_ms"]),
                           selection="independent one-thread tuning"))
    if not output:
        print("Fixed-configuration scaling needs completed comparison tuning and roofline runs including one thread.")
        return
    with (root / "scaling.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print("fixed-configuration scaling: " + str(root / "scaling.csv"))


if __name__ == "__main__":
    if sys.argv[1] == "add": add(sys.argv[2], sys.argv[3], "--include-spr" in sys.argv[4:])
    elif sys.argv[1] == "get": print(get(*sys.argv[2:]))
    elif sys.argv[1] == "check": check(sys.argv[2])
    elif sys.argv[1] == "context": context(*sys.argv[2:])
    elif sys.argv[1] == "edges": edges(sys.argv[2])
    elif sys.argv[1] == "scaling": scaling(sys.argv[2])
    else: sys.exit(__doc__)
