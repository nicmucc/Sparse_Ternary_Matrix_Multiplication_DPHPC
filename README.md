# Sparse Ternary Matrix Multiplication

CPU computation of `Y = XW`, with FP32 activations and weights in `{-1, 0, +1}`.
Sparse pattern reduction (SPR) groups input rows with matching patterns in small column
blocks, computes each group sum once, and reuses it across the matching outputs.

Every method (our SPR versions, the published kernels of the SSR, TCSC and RSR papers,
and the dense libraries) implements one kernel API, so the same test and the same
benchmark driver run all of them with identical verification and timing.

## Layout

```text
include/spr/kernel.hpp     THE kernel API: Kernel, Params, IO, Layout, registry
include/spr/harness.hpp    problems, saved datasets, verification, layout conversion
include/spr/cache.hpp      cache sizes detected at runtime (tiled SPR versions)
cpu/core.cpp               registry, FP64 reference, machine info
cpu/spr/                   our SPR kernels: spr.naive, spr.v0 ... spr.v9
cpu/orig/                  thin wrappers around the papers' kernels (see cpu/orig/README.md)
cpu/libs.cpp               dense baselines: naive loop, BLAS, Eigen dense, Eigen sparse
third_party/               the papers' code, unmodified (see third_party/README.md)
tests/test_correctness.cpp every registered kernel vs an FP64 reference
bench/bench_cpu.cpp        interleaved benchmark driver -> CSV
bench/run.sh               SPR vs the best kernel of every method (make compare)
bench/spr_bench.sh         SPR versions only: runtime + roofline (make spr-bench)
bench/peaks.cpp            machine ceilings: fp32 add peak (SIMD / scalar), L1 / L2 / DRAM bandwidth
bench/summarize.py, tune.py, roofline.py, spr_threads.py   reports, tuning cache, thread sweep
scripts/generate_data.py   saved datasets (make data), readable by bench_cpu --dataset
```

## Kernels

| Prefix | What | Source |
|---|---|---|
| `spr.*` | our SPR versions (naive and v0 ... v9: compiled tree, flat format, tiling, OpenMP) | `cpu/spr/` |
| `tcsc.*` | the FP32 CPU kernels of the TCSC paper (88: naive ladder, merged, GroupMin/GroupMid in col- and row-major, AVX2 and AVX-512, GEMV) | `third_party/SSR-ternaryLLM-repo/ternaryLLM_CPU` |
| `ssr.*` | SSR scalar (and a patched copy of it), AVX2 `vec_L2..L8`, `unroll8/16_v2` | `third_party/SSR-ternaryLLM-repo/SSR` |
| `rsr.*` | RSR / RSR++ in the fp32 version the SSR paper benchmarked | `third_party/SSR-ternaryLLM-repo/SSR/SSR/original_rsr` |
| `rsrcore.*` | RSR-core, the RSR authors' optimized kernels: ternary v3.1 / v3.3, binary v4.2 | `third_party/RSR-core-repo` |
| `lib.*` | dense baselines: naive loop, BLAS (Accelerate / OpenBLAS), Eigen dense / sparse | `cpu/libs.cpp` |

`cpu/orig/README.md` lists every wrapped function with its ISA, threading, unrolling, weight format
and shape limits, and the upstream functions that are not wrapped and why.

## Build and test

Requires a C++17 compiler with OpenMP and Make. BLAS (Accelerate on macOS, OpenBLAS elsewhere)
and Eigen are optional: without them the `lib.dense_blas` / `lib.eigen_*` kernels are not built.
The dataset generator requires NumPy and the plots Matplotlib (`python3 -m pip install -r requirements.txt`).
On macOS: `brew install libomp eigen` (Apple clang uses Homebrew's libomp).

```sh
make -j8                                   # build/release/{test_correctness, bench_cpu, peaks}
make test                                  # every kernel vs an FP64 reference
make test TEST_ARGS="spr. tcsc."           # only kernels whose name contains one of these
make PROFILE=sanitizer test                # ASan + UBSan
```

Every kernel is built with the same flags: `-O3 -ffast-math -march=native` (`-mcpu=native` on ARM).
Each profile builds in its own `build/<profile>/`; the compiler, flags and source list are
recorded in `config.txt`, and a change rebuilds that profile.

**Benchmark on x86.** The SSR and TCSC kernels are written with AVX2 / AVX-512 intrinsics.
- On x86 they compile natively, through `cpu/orig/x86_compat/`, a wrapper around the real
  `<immintrin.h>`. It turns aligned index loads that the TCSC code issues at unaligned
  addresses into unaligned loads, as the authors' MSVC build did. On a CPU without AVX-512 it also
  lets the unmodified TCSC file compile; its AVX-512 kernels then report themselves unsupported
  and are skipped.
- On Apple Silicon, `cpu/orig/shim/` maps the intrinsics to NEON. Results are correct, so tests run on
  a Mac, but SSR / TCSC timings there say nothing about x86.

`make test` runs each kernel on the shapes it supports (ragged sizes, M = 1 ... 128, K up to
2050, 0 ... 100 % zeros, k = 2 ... 8). SPR kernels also run exact sparse-tree examples, repeated
calls into dirty outputs and, for the parallel versions, 1 ... 14 threads. Unsupported shapes are
skipped. One upstream kernel, `ssr.ssr_inference`, is wrong in the authors' code. It is reported as
XFAIL and does not fail the test. `ssr.ssr_inference_patched` is the fixed copy.

## Benchmark

```sh
make compare                     # SPR vs the best kernel of every method, quick grid
MODE=paper THREADS=8 make compare
make spr-bench                   # SPR versions only: runtime + roofline report
make bench BENCH_ARGS="--M 64 --K 2048 --N 8192 --s 0.5,0.9 --k 2,3,4 --threads 1 --kernels spr.v8.b32,tcsc.GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_"
```

- **`bench_cpu`**:
  - builds every (kernel, k) configuration;
  - verifies it against the FP64 reference (wrong results get `ok=0` and are never timed);
  - times all configurations interleaved (round-robin with a rotating start);
  - appends one CSV row per configuration: median / min ms, preprocessing ms, format bytes, bits per weight, max error, additions per row.

  Layout conversion (TCSC and SPR use column-major X / Y) and the copies in and out of the
  `std::vector` APIs of SSR / RSR happen outside timing.
- **`make compare`** (`bench/run.sh`):
  - tunes every other method once per (machine, threads, shape, sparsity): each of its kernels at every k;
  - caches each method's fastest (kernel, k) in `build/results/tuning/`;
  - times SPR against those configurations;
  - writes `build/results/<date>_<mode>/summary.md`.

  `TUNE=1` redoes the tuning, and `METHODS=opt|scalar` restricts it to the optimized or the scalar level.
- **`make spr-bench`** (`bench/spr_bench.sh`):
  - measures the machine ceilings once per thread count (`peaks`);
  - times the SPR versions at every k;
  - writes `report.md`, `roofline.png` and `runtime.png`.

  Settings: `SPR`, `REF`, `MODE=quick|full`, `M`, `SHAPES`, `S`, `KS`, `THREADS`, `PEAKS=1`.

Tiled SPR versions (v4+) size their tiles from the cache sizes detected at runtime. Override them
to model another machine: `SPR_L1_BYTES`, `SPR_L2_BYTES`, `SPR_LLC_BYTES`, `SPR_LINE_BYTES`.

### Saved datasets

```sh
make data                                                   # data/benchmarks (gitignored)
make bench BENCH_ARGS="--dataset data/benchmarks/m16_k1024_n1024_s0.95_seed42 --k 2,3,4 --kernels spr.v8.b32,ssr."
```

`--dataset` takes one or more dataset directories (comma separated) instead of random matrices.
M, K, N and the sparsity come from each `metadata.json`. Bias and PReLU are not part of the timed `Y = XW`.

## The kernel API

```cpp
struct Kernel {
    virtual std::string name() const = 0;
    virtual Layout layout() const;                    // RowMajor (default) or ColMajor X / Y
    virtual bool uses_k() const;                      // true: the bench sweeps Params::k
    virtual bool supports(int M, int K, int N, const Params&) const;   // shape limits (unroll, GEMV ...)
    virtual void prepare(const int8_t* W, int K, int N, const Params&) = 0;  // preprocessing (prep_ms)
    virtual void bind(const IO&);                     // copy X into own containers  (not timed)
    virtual void run(IO&) = 0;                        // inference                    (TIMED)
    virtual void finish(IO&);                         // move the result into io.Y   (not timed)
    virtual size_t bytes() const = 0;                 // compressed W + metadata that run() reads
    virtual double adds_per_row() const;              // optional: additions per row of X
};
```

To add a kernel, write a `Kernel` subclass in a `.cpp` under `cpu/spr/` (our methods) or
`cpu/orig/` (a wrapper around third-party code), and register it with
`SPR_REGISTER_KERNEL("name", Class);`. The Makefile picks up the file, `make test` checks the
kernel, and the bench finds it by name. Third-party code goes unmodified into `third_party/`
(see its README).

## Provenance and licenses

- **SSR, TCSC and the fp32 RSR / RSR++:** `third_party/SSR-ternaryLLM-repo`, from
  [fpgasystems/ternaryLLM](https://github.com/fpgasystems/ternaryLLM/tree/68d384425261d60bfbde5ff079c33a474c1eb28e)
  at commit `68d384425261d60bfbde5ff079c33a474c1eb28e`.
  - Copyright (c) 2025 FPGA @ Systems Group, ETH Zurich, [MIT license](licenses/ternaryllm.LICENSE).
  - SSR contributors: Adeline Pittet, Valerie Verdan, Shien Zhu.
  - CPU implementation contributors: Mila Kjoseva, Shien Zhu.
- **RSR-core:** `third_party/RSR-core-repo`, from
  [UIC-InDeXLab/RSR-core](https://github.com/UIC-InDeXLab/RSR-core/tree/99bcccded70416839f1ef1fc6d58726c507d1fb9)
  at commit `99bcccded70416839f1ef1fc6d58726c507d1fb9`, [MIT license](licenses/rsr-core.LICENSE).
  See [RSR++](https://arxiv.org/html/2411.06360v3) for the original algorithm.

The vendored files are byte-identical to those commits; local changes are only wrappers
(`cpu/orig/`) and one documented patched copy (`cpu/orig/patched/`).

## Team

Nicola Mucciaccio, Matteo Meciani, Riccardo Polo, Alexandre Raybaut, Ahmed Tlili.
Supervisor: Tanja Srindran.
