"""Adapters that register the compiled C++ baselines in the Python suite."""

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .core import PreparedMatmul

try:
    from . import _C
except ImportError as error:
    raise ImportError(
        "The C++ extension is not built. Run `make python-extension` first."
    ) from error


@dataclass(frozen=True)
class CppBackend:
    factory: Callable[..., PreparedMatmul]
    uses_width: bool = False

    def prepare(
        self,
        W: torch.Tensor,
        width: int,
    ) -> Sequence[PreparedMatmul]:
        operator = self.factory(W, width) if self.uses_width else self.factory(W)
        return (operator,)


def create_backends() -> tuple[CppBackend, ...]:
    """Return one benchmark backend for every exposed C++ implementation."""
    return (
        CppBackend(_C.prepare_naive),
        CppBackend(_C.prepare_rsr, uses_width=True),
        CppBackend(_C.prepare_rsrpp, uses_width=True),
        CppBackend(_C.prepare_ssr_basic, uses_width=True),
        CppBackend(_C.prepare_ssr_unrolled, uses_width=True),
        CppBackend(_C.prepare_ssr_tree, uses_width=True),
        CppBackend(_C.prepare_tcsc_basic),
        CppBackend(_C.prepare_tcsc_merged),
        CppBackend(_C.prepare_blas),
        CppBackend(_C.prepare_eigen_csr),
        CppBackend(_C.prepare_eigen_csc),
    )
