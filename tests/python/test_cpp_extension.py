from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

try:
    from stmm_benchmark.cpp_backends import create_backends
except ImportError:
    create_backends = None

from stmm_benchmark import Suite
from stmm_benchmark.core import reference_and_tolerances, validate


@unittest.skipIf(create_backends is None, "C++ extension is not built")
class CppExtensionTests(unittest.TestCase):
    def test_every_cpp_backend_matches_reference(self) -> None:
        X = torch.tensor(
            [[0.5, -2.0, 3.0], [-1.5, 4.0, -0.25]],
            dtype=torch.float32,
        )
        W = torch.tensor(
            [[-1, 0, 1, 0], [0, 0, 0, 0], [1, 0, -1, 1]],
            dtype=torch.int8,
        )
        suite = Suite((X,), W, width=2)
        for backend in create_backends():
            suite.add_backend(backend)

        reference, tolerance = reference_and_tolerances(
            X, W, atol=1e-6, rtol=1e-5
        )
        for kernel in suite.kernels:
            validate(kernel, 0, reference, tolerance)


if __name__ == "__main__":
    unittest.main()
