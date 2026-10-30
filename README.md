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


# TODO WEEK 28/10 -> 04/11

## To Implement

- **Matteo & Nicola & Riccardo:** Import missing code, set up benchmarking, c++ exposure, tracking performance, architecture design, roofline plots
- **Alexandre & Ahmed:** Pytorch implementation of the algorithm

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


## Imported CPU baselines

SSR and TCSC follow the same header/source layout as the other baselines:

```text
include/ssr_baseline.h          Public SSR interface and preprocessed weights
include/tcsc_baseline.h         Public TCSC interface and preprocessed weights
src/ssr_baseline.cpp            SSR preprocessing, basic, AVX2 unrolled and tree
src/tcsc_baseline.cpp           TCSC formats, preprocessing, basic and AVX2 merged
src/ternary_baseline_common.h   Internal dimension checks and AVX2 detection
tests/test_ternary_baselines.cpp Correctness and edge-case checks
licenses/ternaryllm.LICENSE     Original upstream MIT license
```

Each algorithm's implementation is self-contained in its source file. Imported
helpers are private to that file; test and benchmark code use the public headers.
All variants compute row-major `Y = XW`, with FP32 activations and ternary weights.
Bias and activation are applied by the caller. Preprocess weights once and reuse
the returned structure; keep its fields unchanged and use distinct input/output
vectors. Dimensions and flattened offsets must fit signed 32-bit indexing.

```sh
make test
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make benchmark RUNS=101 WARMUP_RUNS=20
# SSR segment width: 1..8; default 6; specialized tree kernels cover 2..6.
make benchmark SSR_BLOCK_WIDTH=4
# Use separate build directories when changing compiler/SIMD/threading flags.
make BUILD_DIR=build/scalar AVX2=0 test
OMP_NUM_THREADS=2 make BUILD_DIR=build/openmp OPENMP=1 test
```

The benchmark adds SSR basic, SSR unrolled, SSR tree, TCSC basic, and TCSC merged
alongside Naive, BLAS, Eigen CSR/CSC, RSR, and RSR++. It prints the selected SIMD
backend and SSR tree path. GCC/Clang x86 builds select AVX2 at runtime; other
machines and `AVX2=0` builds use the basic scalar implementations. Tree widths
1, 7, and 8 use the AVX2 unrolled kernel when available. OpenMP is disabled by
default; set BLAS and OpenMP thread counts explicitly for comparisons.

Weight preprocessing and input conversion happen outside timing. SSR's upstream
temporary allocations and output copy remain inside timing; TCSC reuses its
output buffer. All variants use the same inputs, weights, and bias. The correctness
suite covers the existing reference dataset plus zero/sign-only weights, empty
rows/columns, partial SIMD batches and column groups, all supported SSR widths,
repeated dirty outputs, invalid inputs, and a row index of 32768.

### Upstream source and local changes

Selected code comes from
[fpgasystems/ternaryLLM](https://github.com/fpgasystems/ternaryLLM/tree/68d384425261d60bfbde5ff079c33a474c1eb28e),
commit `68d384425261d60bfbde5ff079c33a474c1eb28e`.
The original [MIT license](licenses/ternaryllm.LICENSE) and copyright are preserved:
Copyright (c) 2025 FPGA @ Systems Group, ETH Zurich.

| Local source | Upstream source and imported implementation |
| --- | --- |
| `src/ssr_baseline.cpp` | `SSR/SSR/ssr.cpp`: `handle_block_ssr`, `ssr_preprocess`; `SSR/SSR/ssr.h`: scalar SSR; `SSR/SSR/ssr_vec.cpp`: FP32 unroll16 and tree specializations L2 through L6 |
| `src/tcsc_baseline.cpp` | `ternaryLLM_CPU/src/TCSC.hpp`: `SparseFormat`, first `MergedTCSC_Group` constructor; `GEMM_CPU_FP32.cpp`: row-major `TCSC_Naive_oneFor` and `TCSC_Merged_GroupMin_1xG8_AVX2` |
| `src/ternary_baseline_common.h` | Local validation, compiler feature guards, and runtime AVX2 detection |

SSR contributors: Adeline Pittet, Valerie Verdan, Shien Zhu.
CPU implementation contributors: Mila Kjoseva, Shien Zhu.
See the upstream README for the SSR DATE 2026 and ternaryLLM TECS 2026 citations.

The imported arithmetic is retained with these compatibility/correctness changes:

- Adapt to the repository's flat row-major types and keep imported helpers private.
- Correct scalar SSR input/permutation strides from segment width `L` to inner
  dimension `K`, accept the activation vector by const reference, and share its
  row helper with the SIMD batch cleanup.
- Process complete eight-row batches in the generated SSR tree kernels and use
  scalar cleanup for remaining rows. Crop padded output columns to `M x N`.
- Use function-level AVX2 target attributes and aligned SIMD scratch storage.
- Widen TCSC row indices from signed 16-bit to signed 32-bit and use unaligned
  SIMD loads/stores for ordinary vector buffers and arbitrary output strides.
- Build merged metadata for complete groups of eight columns; compute the
  remaining columns with scalar TCSC. Use GroupMin with non-interleaved indices.
- Guard OpenMP pragmas so threading is opt-in.

The import includes the selected FP32 CPU implementations. INT8, AVX-512,
Uniform TCSC, hardware-specific parameter lookup tables, model code, and upstream
benchmark executables are outside this subset. The retained TCSC constructor
contains other grouping branches, which the public interface does not expose.
