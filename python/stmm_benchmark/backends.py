"""Built-in Python backends."""

from dataclasses import dataclass
from typing import Sequence

import torch

from .core import PreparedMatmul


@dataclass
class _TorchDenseOperator:
    W: torch.Tensor
    name: str = "PyTorch dense"
    backend: str = "pytorch_cpu"

    @property
    def K(self) -> int:
        return int(self.W.shape[0])

    @property
    def N(self) -> int:
        return int(self.W.shape[1])

    @property
    def weight_bytes(self) -> int:
        return self.W.numel() * self.W.element_size()

    def forward_out(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        if X.device.type != "cpu" or Y.device.type != "cpu":
            raise ValueError("PyTorch dense currently supports CPU tensors only")
        if X.dtype != torch.float32 or Y.dtype != torch.float32:
            raise ValueError("X and Y must use float32")
        if not X.is_contiguous() or not Y.is_contiguous():
            raise ValueError("X and Y must be contiguous")
        if X.ndim != 2 or X.shape[1] != self.K:
            raise ValueError("X has the wrong shape")
        if Y.shape != (X.shape[0], self.N):
            raise ValueError("Y has the wrong shape")

        torch.mm(X, self.W, out=Y)


class TorchDenseBackend:
    """Dense PyTorch reference; W conversion is preprocessing."""

    def prepare(
        self,
        W: torch.Tensor,
        width: int,
    ) -> Sequence[PreparedMatmul]:
        del width
        return (_TorchDenseOperator(W.to(torch.float32).contiguous()),)
