# Sparse Ternary Matrix Multiplication

CPU and CUDA computation of `Y = XW`, with FP32 activations and weights in `{-1, 0, +1}`.
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
gpu/spr/                   CUDA SPR kernels: v0, v1 and v2
cpu/orig/                  thin wrappers around the papers' kernels (see cpu/orig/README.md)
cpu/libs.cpp               dense baselines: naive loop, BLAS, Eigen dense, Eigen sparse
third_party/               the papers' code, unmodified (see third_party/README.md)
tests/test_correctness.cpp every registered kernel vs an FP64 reference
tests/test_gpu_correctness.cpp  CUDA edge cases, rebinding and repeated inference
bench/bench.cpp            shared CPU/GPU interleaved benchmark driver -> CSV
bench/GPU_COMPARE.md       dedicated CUDA v1/v2 comparison and profiling workflow
bench/peaks.cpp            machine compute and bandwidth ceilings
bench/papers_cpu.sh        smoke, pilot or full D-INFK campaign
bench/campaign.py, campaign_plots.py   bounded tuning, held-out evaluation and plots
bench/v8_adapter.cpp       instantiate fixed tiles from the unmodified SPR v8 source
scripts/generate_data.py   saved datasets (make data), readable by bench_cpu --dataset
results/                  recorded measurements, plots and historical source snapshots
build/<profile>/          generated executables and objects (ignored by Git)
```

## Kernels

| Prefix | What | Source |
|---|---|---|
| `spr.*` | our SPR versions (naive and v0 ... v9: compiled tree, flat format, tiling, OpenMP) | `cpu/spr/` |
| `spr.cuda.v0`, `spr.v1.cuda`, `spr.v2.cuda` | CUDA SPR: initial implementation, compact metadata, coalesced input/scratch | `gpu/spr/` |
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

Every CPU kernel is built with the same flags: `-O3 -ffast-math -march=native` (`-mcpu=native` on ARM).
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

### CUDA build and test

Requires an NVIDIA GPU, CUDA toolkit and an OpenMP-capable host compiler on Linux.
CPU and GPU implementations are linked into separate executables using the same registry
and kernel API. `cpu/core.cpp` supplies the shared registry and reference utilities.

```sh
make build-gpu                          # build/release/gpu/{test_correctness,test_gpu_correctness,bench_gpu}
make test-gpu                           # both suites, all three CUDA kernels
make test-gpu TEST_ARGS="spr.v2.cuda"     # only v2, in both suites
make bench-gpu BENCH_ARGS="--M 128 --K 1024 --N 1024 --s 0.5 --k 8 --kernels .cuda"
```

Override `NVCC`, `CXX`, `NVCCFLAGS` or `CUDA_ARCH` as needed; CUDA defaults to
`-O3 -lineinfo -std=c++17 -arch=native`. The GPU edge suite covers empty dimensions,
all-zero blocks, ragged shapes, k=1/3/8/16, the 16-bit K boundary, two input bindings,
and three runs per binding. It supplements the shared correctness suite.

The v0 source and registry name retain their original spelling (`spr_gpu_v0.cu`,
`spr.cuda.v0`); v1 and v2 use `spr_v1.cu` / `spr_v2.cu` and `spr.v1.cuda` /
`spr.v2.cuda`. Filter with `.cuda` to select all CUDA versions.
The dedicated [v1/v2 comparison](bench/GPU_COMPARE.md) additionally measures CUDA-event
and end-to-end timings, including v2's CPU transpose in `bind()`.

`Makefile` is the current build entry point. The tracked `build-fixed/` directory
and source copies under `results/` are historical artifacts, not active build inputs.

## Benchmark

[Published Xeon results and plots](results/xeon-gold-6240/README.md) include
the full tuning sweep, held-out evaluation, selected kernels and machine metadata
from Slurm job 4329. The six original figures are embedded there.

~~~sh
make plots PYTHON=.venv/bin/python RESULTS=results/xeon-gold-6240 PLOT_DIR=build/plots/xeon-gold-6240
~~~

Use the cluster campaign below for a fresh pilot or full run. For individual
kernel measurements, use the existing driver directly.

~~~sh
make bench BENCH_ARGS="--M 64 --K 2048 --N 8192 --s 0.5 --k 2,4 --threads 4 --kernels spr.v8.b32"
~~~

Layout conversion and preparation are outside the inference timer. The campaign
verifies every timed candidate against the FP64 reference.

### SPR v8 on the D-INFK student cluster

The pilot runs M=256, K=2048, N=8192, 50% zeros. Full mode covers M={64,256},
K x N={1024 x 4096,2048 x 8192}, and {50%,90%} zeros. Both use the Xeon Gold 6240
with native AVX-512, OpenBLAS OpenMP, and 1, 2 and 4 threads on four physical
cores. Each campaign creates a fresh results directory and tunes on seed 42.
The pilot exercises the full procedure on its single workload.

SPR screens the Cartesian product of k={2,4,6,8}, B={8,16,32,64},
Kc={64,256,K}, G={1,4,16} independently at 1, 2 and 4 threads (144 combinations each). One warm-up and three rounds
per candidate. Around the fastest three it tests neighboring k (including
1 and 10 at the edges), and half/double Kc and G within bounds. The best three
from both stages get seven confirmation rounds. Tuning runs in batches of up
to four prepared kernels per process. Each workload/seed has one generated
matrix pair and full FP64-derived reference shared across batches on disk.

We tune published SSR AVX2, optimized TCSC OpenMP GEMM (native AVX2/512),
both RSR-core ternary kernels (k=1..8), and scalar RSR++ (one thread only).
OpenBLAS SGEMM is the dense FP32 comparison. RSR-core applies its GEMV once
per batch row. After selection every winner and the one-thread SPR configuration
run together on seeds 43/44 with two warm-ups and seven interleaved timed rounds.
Every candidate is verified over all rows. Preparation and layout conversion
are outside inference timing. With GCC on Linux, the benchmark observes the actual OpenMP team during verification.
The separately compiled adapter instantiates the original v8 template with fixed tiles.
Measured SIMD FP32 add peak uses the same requested thread budget as inference.

**One-time environment setup.** Create the Python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PYTHON="$PWD/.venv/bin/python"
```

On the Ubuntu student cluster, OpenBLAS OpenMP and Eigen can be installed without root.
Skip this block if `build/cluster-deps/env-openmp.sh` is already available:

```sh
mkdir -p build/cluster-deps/packages-openmp build/cluster-deps/root
(
  set -e
  cd build/cluster-deps/packages-openmp
  apt-get download libopenblas0-openmp libopenblas-openmp-dev libeigen3-dev
  for package in ./*.deb; do dpkg-deb -x "$package" ../root; done
)
cat > build/cluster-deps/env-openmp.sh <<'EOF'
export SPR_DEPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/root" && pwd)"
export PKG_CONFIG_PATH="$SPR_DEPS_ROOT/usr/lib/x86_64-linux-gnu/openblas-openmp/pkgconfig:$SPR_DEPS_ROOT/usr/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PKG_CONFIG_SYSROOT_DIR="$SPR_DEPS_ROOT"
export LD_LIBRARY_PATH="$SPR_DEPS_ROOT/usr/lib/x86_64-linux-gnu/openblas-openmp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
```

**Submission.** From the updated repository root on the login node:

~~~sh
sbatch bench/papers_cpu.sh pilot
# Follow the job (replace JOBID with the returned ID):
tail -f slurm-JOBID.out
# After the pilot finishes successfully:
sbatch bench/papers_cpu.sh full
~~~

The script loads build/cluster-deps/env-openmp.sh when present, checks the
allocation, builds on the compute node, and uses .venv/bin/python (override
with PYTHON). Slurm directives select the same node class and request four CPU
slots, 36 GiB and 24 hours. The GPU only selects the node class.
The job log prints the exact results directory and elapsed time per phase.
Use the pilot cost to budget the full run; matrix sizes and sparsity affect
the cost of individual workloads. Wait for the final Completed line.

The results directory contains context.json, build-config.txt, libraries.txt,
correctness.log, tuning.csv, selected.csv, evaluation.csv (individual rounds
in samples_ms), phase_times.csv, peaks.json, and plots/. The full-run figures
show absolute runtimes at each batch size, scaling with a fixed configuration,
speedup against each optimized baseline, percentage of measured add peak, and tuning sensitivity.
Regenerate from copied results with `make plots RESULTS=path/to/run`.


~~~sh
# Small local integration check (Linux, OpenMP, OpenBLAS, four available CPUs)
bash bench/papers_cpu.sh smoke
~~~

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

Published [results and plots](results/xeon-gold-6240/README.md) include speedup curves versus thread count for each workload, alongside the original campaign figures. Reproduction instructions are included.
