"""Python benchmark interface for sparse ternary matrix multiplication."""

from .backends import TorchDenseBackend
from .core import Kernel, MatmulBackend, PreparedMatmul, Suite
from .data import BenchmarkDataset, load_dataset
from .runner import BenchmarkConfig, benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkDataset",
    "Kernel",
    "MatmulBackend",
    "PreparedMatmul",
    "Suite",
    "TorchDenseBackend",
    "benchmark",
    "load_dataset",
]
