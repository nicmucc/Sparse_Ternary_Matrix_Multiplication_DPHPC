from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from stmm_benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    Suite,
    TorchDenseBackend,
    benchmark,
)
from stmm_benchmark.core import reference_and_tolerances, validate


class PythonBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = torch.tensor(
            [[0.5, -2.0, 3.0], [-1.5, 4.0, -0.25]],
            dtype=torch.float32,
        )
        self.W = torch.tensor(
            [[-1, 0, 1, 0], [0, 0, 0, 0], [1, 0, -1, 1]],
            dtype=torch.int8,
        )

    def test_dense_backend_matches_fp64_reference(self) -> None:
        suite = Suite((self.X,), self.W, width=2)
        suite.add_backend(TorchDenseBackend())
        reference, tolerance = reference_and_tolerances(
            self.X, self.W, atol=1e-6, rtol=1e-5
        )
        validate(suite.kernels[0], 0, reference, tolerance)

    def test_duplicate_kernel_names_are_rejected(self) -> None:
        suite = Suite((self.X,), self.W, width=2)
        suite.add_backend(TorchDenseBackend())
        with self.assertRaises(ValueError):
            suite.add_backend(TorchDenseBackend())

    def test_runner_reports_cpp_compatible_scope(self) -> None:
        dataset = BenchmarkDataset(
            path=Path("."),
            case_id="unit",
            M=2,
            K=3,
            N=4,
            seed=42,
            target_sparsity=0.5,
            atol=1e-6,
            rtol=1e-5,
            X=self.X,
            W=self.W,
        )
        rows = benchmark(
            dataset,
            (TorchDenseBackend(),),
            BenchmarkConfig(width=2, samples=3, min_ms=0.01, threads=1),
        )
        self.assertEqual(rows[0]["implementation"], "PyTorch dense")
        self.assertEqual(rows[0]["scope"], "api_call_xw")
        self.assertEqual(rows[0]["reference"], "PyTorch dense")
        self.assertGreater(rows[0]["median_us"], 0)


if __name__ == "__main__":
    unittest.main()
