"""Fresh SPR v8 tuning and held-out CPU comparison. Run via papers_cpu.sh."""
import argparse
import csv
import sys
import socket
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time
from collections import defaultdict

THREADS = (1, 2, 4)
SHAPES = ((1024, 4096), (2048, 8192))
SPR = re.compile(r"spr\.v8\.b(\d+)\.kc(\d+)\.g(\d+)")
METHODS = ("SPR v8", "SSR", "TCSC", "RSR-core", "RSR++", "OpenBLAS")


def method(name):
    families = dict(spr="SPR v8", ssr="SSR", tcsc="TCSC", rsrcore="RSR-core", rsr="RSR++", lib="OpenBLAS")
    return families[name.split(".")[0]]


def spr(k, b, kc, g):
    return (f"spr.v8.b{b}.kc{kc}.g{g}", k)


def config(row):
    return row["kernel"], int(row["k"] or 0)


def append(path, rows):
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        if new:
            writer.writeheader()
        writer.writerows(rows)


def thread_env(t):
    return dict(os.environ, OMP_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t),
                OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close", OMP_PLACES="cores")


class Workload:
    def __init__(self, build, out, tmp, dims):
        self.exe = build / "bench_cpu"
        self.out, self.tmp, self.dims = out, tmp, dims

    def measure(self, phase, candidates, t, seed, rounds, warmup, *, final=False):
        m, K, n, sparsity = self.dims
        candidates = list(dict.fromkeys(candidates))
        random.Random(314 + t).shuffle(candidates)
        rows = []
        start = time.monotonic()
        # Tuning keeps at most four prepared formats resident. All finalists in
        # one invocation share rotating round order on the identical input.
        batch = len(candidates) if final else 4
        for pos in range(0, len(candidates), batch):
            group = candidates[pos:pos+batch]
            patterns = defaultdict(list)
            for name, k in group:
                patterns[name].append(str(k or 1))
            filters = ",".join(f"={name}@{'/'.join(ks)}" for name, ks in patterns.items())
            raw, samples = self.tmp / "batch.csv", self.tmp / "samples.csv"
            raw.unlink(missing_ok=True)
            samples.unlink(missing_ok=True)
            cmd = [self.exe, "--M", m, "--K", K, "--N", n, "--s", sparsity, "--seed", seed,
                   "--threads", t, "--case-dir", self.tmp, "--ref-rows", -1, "--kernels", filters,
                   "--rounds", rounds, "--warmup", warmup, "--out", raw]
            if final:
                cmd += ["--samples", samples]
            subprocess.run(list(map(str, cmd)), check=True, stdout=subprocess.DEVNULL, env=thread_env(t))
            timings = defaultdict(list)
            if final:
                with samples.open() as f:
                    for sample in csv.DictReader(f):
                        timings[sample["kernel"], sample["param"]].append(float(sample["ms"]))
            with raw.open() as f:
                measured = list(csv.DictReader(f))
            batch_rows = []
            for r in measured:
                if (r["ok"] != "1" or not (0 <= float(r["max_err"]) <= 2e-3 * math.sqrt(K))
                        or not math.isfinite(float(r["median_ms"])) or float(r["median_ms"]) <= 0 or int(r["rounds"]) != rounds):
                    raise ValueError(f"Invalid measurement: {r}")
                name = r["kernel"]
                tile = SPR.fullmatch(name)
                b, kc, g = tile.groups() if tile else ("", "", "")
                row = dict(M=m, K=K, N=n, sparsity=sparsity, threads=t, seed=seed,
                           phase=phase, method=method(name), kernel=name, k=r["param"][2:],
                           B=b, Kc=kc, G=g, median_ms=float(r["median_ms"]), prep_ms=float(r["prep_ms"]),
                           verify_ms=float(r["verify_ms"]), max_err=float(r["max_err"]),
                           adds_per_row=float(r["adds_per_row"]) if final else "",
                           team=r["effective_threads"] if tile and int(r["effective_threads"]) else "",
                           rounds=rounds, warmup=warmup)
                if final:
                    values = timings[name, r["param"]]
                    if len(values) != rounds or any(not math.isfinite(v) or v <= 0 for v in values):
                        raise ValueError(f"Incomplete samples for {name}")
                    row["samples_ms"] = json.dumps(values)
                batch_rows.append(row)
            missing = set(group) - {config(r) for r in batch_rows}
            # TCSC dispatch checks native ISA and tile divisibility in supports().
            if any(not name.startswith("tcsc.") or final for name, k in missing):
                raise ValueError(f"Missing requested configurations: {missing}")
            if not final:
                append(self.out / "tuning.csv", batch_rows)
            rows.extend(batch_rows)
        elapsed = time.monotonic() - start
        append(self.out / "phase_times.csv", [dict(M=m, K=K, N=n, sparsity=sparsity, threads=t,
               seed=seed, phase=phase, candidates=len(rows), elapsed_s=round(elapsed, 3))])
        print(f"  {phase}, {t} threads: {len(rows)} candidates in {elapsed:.1f}s", flush=True)
        return sorted(rows, key=lambda r: r["median_ms"])


def refinement(best, dims):
    m, K, n, _ = dims
    result = set()
    for row in best:
        k, b, kc, g = (int(row[key]) for key in ("k", "B", "Kc", "G"))
        for kk in (k-1, k+1, k+2 if k == 8 else k):
            if 1 <= kk <= 16:
                result.add(spr(kk, b, kc, g))
        for chunk in (kc//2, kc*2):
            if 1 <= chunk <= K:
                result.add(spr(k, b, chunk, g))
        for group in (g//2, g*2):
            if 1 <= group <= math.ceil(n/k):
                result.add(spr(k, b, kc, group))
    return sorted(result)


def baseline_grid(names, t, smoke):
    result = []
    for name in names:
        if name.startswith("ssr.ssr_inference_float_vec"):
            ks = range(2, 9) if name.endswith("_L") else (1,2,3,4,5,6,7,8,10)
        elif name.startswith("tcsc.GEMM") and "_TCSC_" in name and "_AVX" in name and "OpenMP" in name:
            ks = (0,)
        elif name.startswith("rsrcore.rsr_ternary_"):
            ks = range(1,9)
        elif name == "rsr.rsr_pp_inference" and t == 1:
            ks = (1,2,3,4,5,6,7,8,10)
        elif name == "lib.dense_blas":
            ks = (0,)
        else:
            continue
        if smoke:
            if name.startswith("tcsc.") and "16x1_AVX2_OpenMP" not in name:
                continue
            ks = (4,) if any(ks) else (0,)
        result.extend((name, k) for k in ks)
    return result


def run_case(case, names, smoke):
    m, K, n, _ = case.dims
    grid = [spr(*p) for p in itertools.product((2,4) if smoke else (2,4,6,8),
            (8,16) if smoke else (8,16,32,64), (64,K) if smoke else (64,256,K),
            (1,4) if smoke else (1,4,16))]
    selected = {}
    for t in THREADS:
        coarse = case.measure("coarse", grid, t, 42, 3, 1)
        neighbors = sorted(set(refinement(coarse[:3], case.dims)) - set(grid))
        candidates = coarse
        if neighbors:
            candidates = coarse + case.measure("neighbors", neighbors, t, 42, 3, 1)
        best = sorted(candidates, key=lambda r: r["median_ms"])[:3]
        ranked = case.measure("finalists", [config(r) for r in best], t, 42, 7, 1)
        baselines = case.measure("baselines", baseline_grid(names,t,smoke), t, 42, 3, 1)
        selected[t] = {"SPR v8": ranked[0]}
        for row in baselines:
            selected[t].setdefault(row["method"], row)
        required = set(METHODS) - ({"RSR++"} if t != 1 else set())
        if set(selected[t]) != required:
            raise RuntimeError(f"Missing baselines: {required-set(selected[t])}")
        append(case.out / "selected.csv", list(selected[t].values()))
    fixed = config(selected[1]["SPR v8"])
    for seed in (43,44):
        for t in THREADS:
            winners = {config(r) for r in selected[t].values()}
            rows = case.measure("evaluation", sorted(winners | {fixed}), t, seed, 7, 2, final=True)
            for row in rows:
                row["selected"] = int(config(row) in winners)
                row["fixed_scaling"] = int(config(row) == fixed)
            append(case.out / "evaluation.csv", rows)


def record_machine(mode, out):
    cpus = sorted(os.sched_getaffinity(0))
    topology = [Path(f"/sys/devices/system/cpu/cpu{c}/topology") for c in cpus]
    cores = {(p.joinpath("physical_package_id").read_text(), p.joinpath("core_id").read_text())
             for p in topology}
    if min(len(cores), int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) < 4:
        raise RuntimeError("This campaign requires four allocated physical cores")
    cpu = subprocess.check_output(["lscpu"], text=True)
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        if "Gold 6240" not in cpu or "avx512f" not in cpu or "SPR_AVX512_EMULATED" in (out/"build-config.txt").read_text():
            raise RuntimeError("Expected the Xeon Gold 6240 with a native AVX-512 build")
        # libgomp may bind the loading thread to its first core. Probe OpenBLAS
        # in a child so all benchmark processes inherit the full job affinity.
        probe = "import ctypes; print(ctypes.CDLL('libopenblas.so.0').openblas_get_parallel())"
        if subprocess.check_output([sys.executable, "-c", probe], text=True).strip() != "2":
            raise RuntimeError("Expected OpenBLAS with the OpenMP backend")
    metadata = dict(cpu=cpu, hostname=socket.gethostname(), settings=mode, cpus=cpus, job=job)
    (out/"context.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot","full","smoke"))
    parser.add_argument("build", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    record_machine(a.mode, a.out)
    names = subprocess.check_output([a.build/"bench_cpu","--list"], text=True).splitlines()
    workloads = list(itertools.product((64,256), SHAPES, (0.5,0.9)))
    if a.mode == "pilot":
        workloads = [(256,SHAPES[-1],0.5)]
    elif a.mode == "smoke":
        workloads = [(64,(128,256),0.5)]
    peaks = {}
    for t in THREADS:
        peaks[t] = json.loads(subprocess.check_output([a.build/"peaks","--threads",str(t),"--add-only"],
                                                       env=thread_env(t)))
    (a.out/"peaks.json").write_text(json.dumps(peaks, indent=2))
    for m,(K,n),s in workloads:
        print(f"M={m} K={K} N={n} sparsity={s}", flush=True)
        with tempfile.TemporaryDirectory(prefix="spr-matrices-", dir=os.environ.get("SLURM_TMPDIR") or None) as tmp:
            run_case(Workload(a.build,a.out,Path(tmp),(m,K,n,s)), names, a.mode=="smoke")
    with (a.out/"phase_times.csv").open() as f:
        stages = list(csv.DictReader(f))
    seconds = sum(float(r["elapsed_s"]) for r in stages)
    print(f"Benchmark phases: {seconds/60:.2f} minutes (includes data and full reference generation).")
    with (a.out/"tuning.csv").open() as f:
        tuned = list(csv.DictReader(f))
    print(f"Tuning: {sum(float(r['prep_ms']) for r in tuned)/1000:.1f}s preparation, "
          f"{sum(float(r['verify_ms']) for r in tuned)/1000:.1f}s verification.")
    print(f"Measurements complete: {a.out}", flush=True)


if __name__ == "__main__":
    main()
