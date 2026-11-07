"""Run reproducible benchmark sweeps and save CSV plus build/machine metadata."""
import argparse
import csv
import datetime
import io
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command(args, **kwargs):
    try:
        return subprocess.run(args, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["release", "scalar", "sanitizer", "openmp"], default="release")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("build/results"))
    parser.add_argument("--shapes", nargs="+", default=["1x64x64", "17x257x37", "64x512x256"])
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.95])
    parser.add_argument("--widths", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--pattern-rows", nargs="+", type=int, default=[0, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--input-counts", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--min-ms", type=float, default=10)
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads))
    if args.build_dir is None:
        args.build_dir = Path("build") / args.profile
    binary = args.build_dir / "benchmark"
    command(["make", f"PROFILE={args.profile}", f"BUILD_DIR={args.build_dir}", str(binary)], env=env)
    args.out.mkdir(parents=True, exist_ok=True)
    cpu_model = platform.processor()
    if platform.system() == "Darwin":
        cpu_model = command(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True).stdout.strip()
    elif Path("/proc/cpuinfo").exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), cpu_model)
    metadata = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(), "cpu": cpu_model,
        "machine": platform.machine(), "logical_cpus": os.cpu_count(),
        "commit": command(["git", "rev-parse", "HEAD"], capture_output=True).stdout.strip(),
        "git_status": command(["git", "status", "--porcelain"], capture_output=True).stdout,
        "build_config": (args.build_dir / "config.txt").read_text(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "thread_environment": {key: env[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "commands": [],
    }
    shapes = list(args.shapes)
    if args.large:
        shapes.append("32x4096x4096")
    output = args.out / "benchmark.csv"
    with output.open("w", newline="") as stream:
        writer = None
        for shape, sparsity, width, patterns, seed, inputs in itertools.product(
                shapes, args.sparsities, args.widths, args.pattern_rows,
                args.seeds, args.input_counts):
            m, k, n = map(int, shape.lower().split("x"))
            cmd = [str(binary), "--m", str(m), "--k", str(k), "--n", str(n),
                   "--sparsity", str(sparsity), "--width", str(width),
                   "--pattern-rows", str(patterns), "--seed", str(seed),
                   "--inputs", str(inputs), "--samples", str(args.samples),
                   "--min-ms", str(args.min_ms)]
            print(" ".join(cmd), flush=True)
            metadata["commands"].append(cmd)
            (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
            result = command(cmd, env=env, capture_output=True)
            reader = csv.DictReader(io.StringIO(result.stdout))
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                writer.writeheader()
            writer.writerows(reader)
            stream.flush()
    print(output)


if __name__ == "__main__":
    main()
