"""Backend contracts and the Python equivalent of the C++ baseline suite."""

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class PreparedMatmul(Protocol):
    """One backend with W already converted into its reusable representation."""

    name: str
    backend: str
    weight_bytes: int
    K: int
    N: int

    def forward_out(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Overwrite Y with X @ W."""


@runtime_checkable
class MatmulBackend(Protocol):
    """Factory for one or more kernels that share weight preparation."""

    def prepare(
        self,
        W: torch.Tensor,
        width: int,
    ) -> Sequence[PreparedMatmul]:
        """Prepare W once and return the available implementation variants."""


@dataclass
class Kernel:
    name: str
    run: Callable[[int], None]
    output: Callable[[], torch.Tensor]
    preprocess_us: float
    weight_bytes: int
    backend: str


class Suite:
    """Own inputs, outputs, prepared weights, and registered kernel callbacks."""

    def __init__(
        self,
        inputs: Sequence[torch.Tensor],
        W: torch.Tensor,
        width: int,
    ) -> None:
        if not inputs or not 1 <= width <= 8:
            raise ValueError("Invalid input pool or block width")
        if W.device.type != "cpu" or W.dtype != torch.int8:
            raise ValueError("W must be a CPU int8 tensor")
        if W.ndim != 2 or not W.is_contiguous():
            raise ValueError("W must be a contiguous K x N matrix")

        self.inputs = tuple(inputs)
        self.W = W
        self.K, self.N = W.shape
        self.M = int(inputs[0].shape[0])
        self.width = width
        self.kernels: list[Kernel] = []
        self._operators: list[PreparedMatmul] = []
        self._outputs: list[torch.Tensor] = []

        for X in self.inputs:
            if X.device.type != "cpu" or X.dtype != torch.float32:
                raise ValueError("X must be a CPU float32 tensor")
            if X.shape != (self.M, self.K) or not X.is_contiguous():
                raise ValueError("Every X must be contiguous with shape M x K")

    def add_backend(self, factory: MatmulBackend) -> None:
        start = perf_counter_ns()
        operators = tuple(factory.prepare(self.W, self.width))
        preprocess_us = (perf_counter_ns() - start) / 1_000.0
        if not operators:
            raise ValueError("A backend must register at least one operator")

        for operator in operators:
            if operator.K != self.K or operator.N != self.N:
                raise ValueError(f"{operator.name}: prepared dimensions are wrong")

            output = torch.empty((self.M, self.N), dtype=torch.float32)
            self._operators.append(operator)
            self._outputs.append(output)

            def run(index: int, op=operator, out=output) -> None:
                op.forward_out(self.inputs[index], out)

            self.kernels.append(Kernel(
                name=operator.name,
                run=run,
                output=lambda out=output: out,
                preprocess_us=preprocess_us,
                weight_bytes=int(operator.weight_bytes),
                backend=operator.backend,
            ))

        self.validate_registry()

    def validate_registry(self) -> None:
        names: set[str] = set()
        for kernel in self.kernels:
            if (
                not kernel.name
                or kernel.name in names
                or not kernel.backend
                or kernel.weight_bytes < 0
                or kernel.preprocess_us < 0
            ):
                raise ValueError("Invalid or duplicate kernel registration")
            names.add(kernel.name)


def reference_and_tolerances(
    X: torch.Tensor,
    W: torch.Tensor,
    atol: float,
    rtol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    X64 = X.to(torch.float64)
    W64 = W.to(torch.float64)
    reference = X64 @ W64
    sum_abs = X64.abs() @ W.ne(0).to(torch.float64)

    unit_roundoff = torch.finfo(torch.float32).eps / 2
    ku = (X.shape[1] + 4.0) * unit_roundoff
    if ku >= 1:
        raise ValueError("K is too large for the FP32 error model")
    tolerance = (
        atol
        + rtol * reference.abs()
        + (2 * ku / (1 - ku)) * sum_abs
    )
    return reference, tolerance


def validate(
    kernel: Kernel,
    input_index: int,
    reference: torch.Tensor,
    tolerance: torch.Tensor,
) -> None:
    output = kernel.output()
    if output.shape != reference.shape or output.dtype != torch.float32:
        raise RuntimeError(f"{kernel.name}: wrong output shape or dtype")

    output.fill_(float("nan"))
    kernel.run(input_index)
    output = kernel.output()
    if output.shape != reference.shape:
        raise RuntimeError(f"{kernel.name}: wrong output shape")

    error = (output.to(torch.float64) - reference).abs()
    failures = ~torch.isfinite(output) | (error > tolerance)
    if failures.any():
        index = int(failures.reshape(-1).nonzero()[0, 0])
        raise RuntimeError(f"{kernel.name}: mismatch at index {index}")
