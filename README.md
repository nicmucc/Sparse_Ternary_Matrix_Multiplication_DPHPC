# TODO WEEK 1

## To Implement

**Baselines:**
- **Ahmed, Matteo:** Think about the best APIs to have and set up the benchmarking infrastructure in C++.
- **Nicola:** Simple baseline with BLAS GEMM.
- **Alexandre:** Simple CSR and CSC baselines.
- **Riccardo:** Use RSR / RSR++ on vector-matrix multiplication to implement sparse matrix-matrix multiplication.
- Testing infrastructure.

**Research:**
- Start thinking about an RSR-like algorithm for sparse GEMM.
- Do some research about CSC and CSR and whether there are papers discussing ternary sparse GEMM.

---


# Sparse Ternary Matrix Multiplication

Efficient computation of `Y = XW + b` and `Y = PReLU(XW + b)`, where `W` is a sparse ternary matrix with entries in `{-1, 0, +1}`.

## Problem

| Symbol | Shape | Description |
|--------|-------|-------------|
| `X` | `(M, K)` | Dense input matrix |
| `W` | `(K, N)` | Sparse ternary weight matrix |
| `b` | `N` | Dense bias vector |
| `Y` | `(M, N)` | Output |

Because `W` only contains `-1`, `0` and `+1`, multiplications reduce to additions and subtractions, and zeros can be skipped entirely. `PReLU` follows the [PyTorch implementation](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html).

## Goal

Accelerate the computation of `Y` by exploiting the structure of the ternary matrix (not by speeding up naive GEMM), using either:

- **OpenMP / MPI** (CPU), or
- **CUDA** (GPU)

Candidate approaches: a Ternary CSC/CSR format with a sparse kernel, or an algorithm derived from [RSR++](https://arxiv.org/html/2411.06360v3). We are free to design our own sparse format and preprocessing step.

## Planned Structure

```
.
├── src/          # Implementation (formats, kernels, PReLU fusion)
├── baselines/    # Naive GEMM / reference implementations
├── tests/        # Correctness tests against a reference (e.g. PyTorch)
├── bench/        # Benchmark scripts and results
├── docs/         # Algorithm design notes
└── README.md
```

## Project Breakdown

- **20%** Algorithm design
- **40%** Implementation
- **40%** Benchmarking

## Team

- Nicola Mucciaccio
- Matteo Meciani
- Riccardo Polo
- Alexandre Raybaut
- Ahmed Tlili

**Supervisor:** Tanja Srindran (tanja.srindran@inf.ethz.ch)

## References

1. [PyTorch documentation: PReLU](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html)
2. [An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks (RSR++)](https://arxiv.org/html/2411.06360v3)
