"""Run every saved dataset listed in the generator's manifest, once per baseline."""
import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if not manifest["datasets"]:
        raise ValueError("The dataset manifest is empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = None
        for dataset in manifest["datasets"]:
            path = args.data_dir / dataset["path"]
            print(f"Benchmarking {path}", file=sys.stderr)
            result = subprocess.run(
                [str(args.benchmark.resolve()), *args.benchmark_args, "--dataset", str(path)],
                stdout=subprocess.PIPE, text=True, check=True,
            )
            rows = csv.DictReader(io.StringIO(result.stdout))
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=[*rows.fieldnames, "dataset"])
                writer.writeheader()
            for row in rows:
                writer.writerow({**row, "dataset": dataset["path"]})

    temporary.replace(args.output)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
