#!/usr/bin/env python3
"""Benchmark Python/PyTorch backends on one saved repository dataset."""

import argparse
import csv
import importlib
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from stmm_benchmark import (  # noqa: E402
    BenchmarkConfig,
    TorchDenseBackend,
    benchmark,
    load_dataset,
)


def load_plugin(specification: str) -> Iterable[object]:
    module_name, separator, factory_name = specification.partition(":")
    if not separator:
        factory_name = "create_backends"
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    result = factory()
    if hasattr(result, "prepare"):
        return (result,)
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--min-ms", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE[:FACTORY]",
        help="Add backends returned by a Python factory function",
    )
    args = parser.parse_args()

    backends: list[object] = [TorchDenseBackend()]
    for plugin in args.plugin:
        backends.extend(load_plugin(plugin))

    rows = benchmark(
        load_dataset(args.dataset),
        backends,
        BenchmarkConfig(
            width=args.width,
            samples=args.samples,
            min_ms=args.min_ms,
            threads=args.threads,
        ),
    )
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
