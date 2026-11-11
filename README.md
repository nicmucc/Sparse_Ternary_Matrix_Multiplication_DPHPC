# Sparse Ternary Matrix Multiplication

CPU computation of `Y = XW`, with FP32 activations and weights in `{-1, 0, +1}`.
Sparse pattern reduction groups input rows with matching patterns in small column
blocks, computes each group sum once, and reuses it across the matching outputs.

## Layout

```text
include/                 Public headers and shared test/benchmark adapters
src/baselines/           Naive, BLAS, Eigen CSR/CSC, RSR/RSR++, SSR, TCSC
src/utils/               Dataset reader and pattern utilities
src/benchmark.cpp          Calibrated benchmark executable
src/test_baselines.cpp     Self-contained correctness and API checks
scripts/run_benchmarks.py  Runs the generated dataset manifest
scripts/plot_benchmarks.py Plots for `make sweep` results
scripts/generate_data.py   Reference samples and benchmark datasets
configs/                   Dataset configurations
build/                     Build profiles and sweep results
out/plots/                 Generated plots
```

## Build and test

Requires a C++17 compiler, Make, pkg-config, Eigen, OpenBLAS, and nlohmann-json.
On macOS: `brew install eigen openblas nlohmann-json pkg-config`.
Plotting requires Matplotlib. The dataset generator requires NumPy
(`python3 -m pip install -r requirements.txt`).

```sh
make -j4
make test
make PROFILE=native test                 # -march=native / -mcpu=native
make PROFILE=scalar test                 # imported AVX2 kernels disabled
make PROFILE=sanitizer CXX=clang++ test  # ASan + UBSan
make PROFILE=openmp THREADS=2 test
```

Each profile builds in its own `build/<profile>/`. Compiler, flags, and source
list are recorded in `config.txt`; if any of them change, that profile's objects
are rebuilt. `make clean` removes the selected profile's directory.
OpenMP requires compiler/runtime support. On macOS, use a compiler with OpenMP
support for that profile. AVX2 dispatch is runtime-selected on x86; ARM uses scalar
fallbacks. Sanitizers must be supported by the selected compiler.

Tests need no generated files. Every implementation uses the same deterministic
cases: rectangular and padded shapes, zero/sign-only weights, empty rows and
columns, repeated calls into dirty outputs, general FP32 values, cancellation,
all supported block widths, and a row index of 32768. Larger FP32 and cancellation
cases run with multiple seeds. Harness checks reject duplicate registrations,
no-op kernels, partial writes, and incorrect output lengths. Focused Eigen,
RSR++, and SSR/TCSC API checks are retained.

The reference accumulates serialized FP32 inputs in FP64. Comparisons use
`atol + rtol*abs(reference) + 2*gamma(K+4)*sum(abs(terms))`,
where `gamma(n)=n*u/(1-n*u)` and `u=2^-24`. This allows accumulation rounding near
zero without choosing a fixed larger absolute tolerance. Bias contributes to the
absolute sum in dataset tests. Exact dyadic cases also run with strict comparisons.

Generate benchmark data, then optionally include one dataset in correctness tests:

```sh
make data PYTHON=.venv/bin/python
make test TEST_DATA=data/benchmarks/m16_k1024_n1024_s0.95_seed42
```

The legacy generator includes bias and PReLU files. Tests compare its linear result;
PReLU is outside the current kernel benchmark.

## Benchmark

```sh
make benchmark BENCHMARK_ARGS="--dataset data/benchmarks/m16_k1024_n1024_s0.95_seed42"
build/release/benchmark --dataset data/benchmarks/m16_k1024_n1024_s0.95_seed42 \
  --width 6 --samples 15 --min-ms 10
```

The benchmark requires a saved dataset; it never generates activations or weights.
Dimensions, generation seed, and requested sparsity are read from its metadata.

The C++ executable prints CSV to stdout and errors to stderr. It validates every
implementation and input against FP64 before timing, checking output lengths
and poisoning outputs before each validation call. All kernels measure
`Y=XW`; bias and activation are excluded. Inputs, row adapters, Eigen conversions,
outputs, and available reusable workspace are prepared before timing. RSR/RSR++
and SSR still allocate temporaries inside their public APIs. All results are
labelled `api_call_xw` and include each API's internal allocations and necessary
output copies. They are not allocation-free inner-loop measurements.

Each implementation calibrates a batch to at least `--min-ms`; calibration also
warms up the complete input pool. Each measured repetition shuffles implementation
order. CSV reports median, p10, p90, calls per batch, preprocessing time, stored
weight/metadata bytes, and speedup over BLAS. Times and percentiles describe batch
averages per call, including harness dispatch and output consumption. The
semicolon-separated `sample_us` column preserves samples in measurement order.
Backend labels come from registration and distinguish SIMD paths with scalar
tails from entirely scalar fallbacks. Thread columns record requested environment
settings, not observed worker counts. The scalar profile disables imported ternary
AVX2 kernels; BLAS and Eigen remain optimized. Preprocessing is measured once per
format, including weight conversion, and shared across variants. It excludes
input adaptation and is a setup-cost estimate, not a repeated preprocessing
microbenchmark. Stored bytes count vector payloads, excluding allocator overhead,
inputs, output/scratch, and discarded conversion intermediates. TCSC's public
format retains both basic and merged metadata, so both are counted.

Each run repeatedly reuses one saved input and its weights. This measures warm
reuse without forcibly flushing caches. The CSV records requested
`target_sparsity`, actual `sparsity`, and distinct nonzero ternary patterns across
column blocks. Both RSR and SSR use `--width`. Dataset bias and activation are
excluded from timing.

## Sweeps

```sh
make data             # once: generate the benchmark suite under data/benchmarks
make sweep            # benchmark every entry in its manifest, sequentially
make plot             # measure if missing or stale, then write out/plots/*.pdf
```

The generator defines the complete dataset list in
`data/benchmarks/manifest.json`. By default it contains 30 datasets: the
16x1024x1024 sparsity sweep from 50% to 99% zeros and the K=N size sweep from 256
to 2048 at 95% zeros, each with seeds 17, 42, and 123. The shared 1024/95% case
is measured once per seed. Generated data is gitignored.

`make sweep` reads every manifest entry and writes one row per dataset and
implementation to `build/results/benchmarks.csv`, including the dataset name and
its seed. Shape and sparsity settings live in the generator rather than Make.
`SWEEP_ARGS` controls measurement settings (default
`--width 6 --samples 15 --min-ms 10`), and `THREADS` sets
`OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` (default 1).
`make plot` re-measures if data files, the manifest, benchmark, runner, or settings
changed. `make sweep` always re-measures.

```sh
make sweep DATA_DIR=data/benchmarks RESULTS_DIR=build/run2
make plot RESULTS_DIR=build/run2 PLOT_DIR=out/plots/run2
```

The plotter identifies the fixed-shape sparsity sweep and fixed-sparsity size
sweep from the measured configurations. Curves show the median of dataset medians
across seeds. Bands show one sample standard deviation of batch averages pooled
across seeds, including both timing noise and variation between datasets.
Speedups pair each implementation with BLAS on the same dataset before combining
seeds. LaTeX renders text when installed. Run timing experiments without competing
builds or other benchmark processes.

## Baseline provenance

SSR and TCSC adapt selected FP32 CPU code from
[fpgasystems/ternaryLLM](https://github.com/fpgasystems/ternaryLLM/tree/68d384425261d60bfbde5ff079c33a474c1eb28e),
commit `68d384425261d60bfbde5ff079c33a474c1eb28e`.
Copyright (c) 2025 FPGA @ Systems Group, ETH Zurich,
[MIT license](licenses/ternaryllm.LICENSE).
SSR contributors: Adeline Pittet, Valerie Verdan, Shien Zhu.
CPU implementation contributors: Mila Kjoseva, Shien Zhu.

Local adaptations include rectangular row-major interfaces, corrected SSR strides
and batch tails, padded-output cropping, runtime AVX2 dispatch, 32-bit TCSC indices,
unaligned SIMD accesses, column-group tails, and opt-in OpenMP.
See [RSR++](https://arxiv.org/html/2411.06360v3) for the original algorithm.

The original RSR and RSR++ use positive and negative binary planes.
The three optimized GEMV kernels in `src/baselines/rsr.cpp` are adapted from
[UIC-InDeXLab/RSR-core](https://github.com/UIC-InDeXLab/RSR-core/tree/99bcccded70416839f1ef1fc6d58726c507d1fb9/kernels)
under the [MIT license](licenses/rsr-core.LICENSE).
V4.2 runs once per binary plane. V3.1 and v3.3 group signed patterns directly,
using scatter lists and masks respectively. Both require K <= 65535.
V3.3 selects SIMD based on build flags and block width (use `PROFILE=native`
for host ISA support).

## Team

Nicola Mucciaccio, Matteo Meciani, Riccardo Polo, Alexandre Raybaut, Ahmed Tlili.
Supervisor: Tanja Srindran.

## Adding an implementation

To add an implementation, compile its source under `src/baselines/` and append a
`Kernel` registration in `include/baseline_suite.h`: name, input-index callback,
output view (pointer and element count), preprocessing estimate, weight bytes,
and backend label. The shared tests and benchmark visit every registration.
Keep inputs and weights alive and unchanged while the noncopyable suite exists.
Use owned state captured by callbacks when additional preparation is needed.
