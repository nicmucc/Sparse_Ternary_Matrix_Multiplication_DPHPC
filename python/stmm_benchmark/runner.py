"""Calibrated runner mirroring the structure of src/benchmark.cpp."""

from dataclasses import dataclass
import os
import random
from statistics import median
from time import perf_counter_ns
from typing import Iterable

import torch

from .core import MatmulBackend, Suite, reference_and_tolerances, validate
from .data import BenchmarkDataset


@dataclass(frozen=True)
class BenchmarkConfig:
    width: int = 6
    samples: int = 15
    min_ms: float = 10.0
    threads: int = 1

    def check(self) -> None:
        if not 1 <= self.width <= 8:
            raise ValueError("width must be between 1 and 8")
        if self.samples < 3:
            raise ValueError("samples must be at least 3")
        if self.min_ms <= 0:
            raise ValueError("min_ms must be positive")
        if self.threads < 1:
            raise ValueError("threads must be positive")


def _percentile(sorted_values: list[float], fraction: float) -> float:
    return sorted_values[int((len(sorted_values) - 1) * fraction)]


def _pattern_count(W: torch.Tensor, width: int) -> int:
    K, N = W.shape
    count = 0
    for column in range(0, N, width):
        patterns = {
            tuple(W[row, column:min(N, column + width)].tolist())
            for row in range(K)
            if W[row, column:min(N, column + width)].ne(0).any()
        }
        count += len(patterns)
    return count


def benchmark(
    dataset: BenchmarkDataset,
    backends: Iterable[MatmulBackend],
    config: BenchmarkConfig = BenchmarkConfig(),
) -> list[dict[str, object]]:
    config.check()
    torch.set_num_threads(config.threads)

    inputs = (dataset.X,)
    suite = Suite(inputs, dataset.W, config.width)
    for backend in backends:
        suite.add_backend(backend)
    if not suite.kernels:
        raise ValueError("No benchmark kernels were registered")

    with torch.inference_mode():
        for input_index, X in enumerate(inputs):
            reference, tolerances = reference_and_tolerances(
                X, dataset.W, dataset.atol, dataset.rtol
            )
            for kernel in suite.kernels:
                validate(kernel, input_index, reference, tolerances)

        def batch(kernel_index: int, count: int) -> float:
            kernel = suite.kernels[kernel_index]
            start = perf_counter_ns()
            for iteration in range(count):
                kernel.run(iteration % len(inputs))
            elapsed_us = (perf_counter_ns() - start) / 1_000.0

            # Python and bound operators execute eagerly, so no per-call
            # checksum is needed to prevent dead-code elimination.
            return elapsed_us

        iterations = [len(inputs)] * len(suite.kernels)
        timings: list[list[float]] = [[] for _ in suite.kernels]

        # Calibration also performs warmup, as in the C++ harness.
        for index in range(len(suite.kernels)):
            while batch(index, iterations[index]) < config.min_ms * 1_000:
                if iterations[index] >= 2**62:
                    raise RuntimeError("Benchmark calibration overflow")
                iterations[index] *= 2

        order = list(range(len(suite.kernels)))
        rng = random.Random(dataset.seed)
        for _ in range(config.samples):
            rng.shuffle(order)
            for index in order:
                timings[index].append(
                    batch(index, iterations[index]) / iterations[index]
                )

    medians = [median(values) for values in timings]
    names = [kernel.name for kernel in suite.kernels]
    reference_name = "BLAS" if "BLAS" in names else "PyTorch dense"
    reference_index = names.index(reference_name)

    nonzero = int(torch.count_nonzero(dataset.W))
    actual_sparsity = 1.0 - nonzero / dataset.W.numel()
    patterns = _pattern_count(dataset.W, config.width)
    omp_threads = os.environ.get("OMP_NUM_THREADS", "unset")
    blas_threads = os.environ.get("OPENBLAS_NUM_THREADS", "unset")

    rows: list[dict[str, object]] = []
    for index, kernel in enumerate(suite.kernels):
        ordered = sorted(timings[index])
        speedup_reference = medians[reference_index] / medians[index]
        speedup_blas: object = speedup_reference if reference_name == "BLAS" else ""
        rows.append({
            "implementation": kernel.name,
            "m": dataset.M,
            "k": dataset.K,
            "n": dataset.N,
            "target_sparsity": dataset.target_sparsity,
            "sparsity": actual_sparsity,
            "width": config.width,
            "seed": dataset.seed,
            "nonzero_patterns": patterns,
            "mode": "warm",
            "inputs": len(inputs),
            "input_pool_bytes": sum(X.numel() * X.element_size() for X in inputs),
            "backend": kernel.backend,
            "omp_threads": omp_threads,
            "blas_threads": blas_threads,
            "torch_threads": config.threads,
            "preprocess_us": kernel.preprocess_us,
            "weight_bytes": kernel.weight_bytes,
            "median_us": medians[index],
            "p10_us": _percentile(ordered, 0.1),
            "p90_us": _percentile(ordered, 0.9),
            "speedup_blas": speedup_blas,
            "reference": reference_name,
            "speedup_reference": speedup_reference,
            "iterations": iterations[index],
            "samples": config.samples,
            "scope": "api_call_xw",
            "sample_us": ";".join(str(value) for value in timings[index]),
        })
    return rows
