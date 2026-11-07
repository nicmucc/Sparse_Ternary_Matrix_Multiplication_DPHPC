# Sparse Ternary Matrix Multiplication

CPU computation of `Y = XW`, with FP32 activations and weights in `{-1, 0, +1}`.
Sparse pattern reduction groups input rows with matching patterns in small column
blocks, computes each group sum once, and reuses it across the matching outputs.

## Layout

```text
include/                 Public headers and shared test/benchmark adapters
src/baselines/           Naive, BLAS, Eigen CSR/CSC, RSR/RSR++, SSR, TCSC
src/utils/               Dataset reader and pattern utilities
src/benchmark.cpp        Calibrated benchmark executable
tests/test_baselines.cpp Self-contained correctness and API checks
scripts/benchmark.py     Reproducible parameter sweeps
scripts/generate_data.py Optional file-based reference datasets
configs/                 Dataset configurations
build/                   Build profiles, results, and generated files
```

## Build and test

Requires a C++17 compiler, Make, pkg-config, Eigen, OpenBLAS, and nlohmann-json.
On macOS: `brew install eigen openblas nlohmann-json pkg-config`.
Python sweeps use only the standard library. The optional dataset generator
requires NumPy (`python3 -m pip install -r requirements.txt`).

```sh
make -j4
make test
make PROFILE=scalar test
make PROFILE=sanitizer CXX=clang++ test
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 make PROFILE=openmp test
```

Profiles use separate directories under `build/`. Compiler identity, flags, and
the source-file list are recorded in `config.txt` and select configuration-specific
object paths. Changing
flags rebuilds as needed; returning to cached flags relinks the correct objects.
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

Optional external reference data:

```sh
make data PYTHON=.venv/bin/python
make test TEST_DATA=data/sample_001
```

The legacy generator includes bias and PReLU files. Tests compare its linear result;
PReLU is outside the current kernel benchmark.

## Benchmark

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make benchmark
build/release/benchmark --m 64 --k 512 --n 256 --sparsity 0.95 --width 6 \
  --samples 15 --min-ms 10 --inputs 1 --seed 42 --pattern-rows 0
python3 scripts/benchmark.py --out build/results/run-001
```

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

`--inputs 1` measures warm reuse. Larger pools rotate distinct prepared inputs;
weights remain reused. This does not forcibly flush caches. The logical FP32 pool
size is reported; row and Eigen adapters retain additional input copies. Select a
pool larger than the target machine's cache to study changing-input workloads.

`--pattern-rows 0` generates independent weight rows. A positive value repeats
that many prototype rows, exposing pattern reuse at the same expected sparsity.
CSV records requested `target_sparsity`, actual `sparsity`, and the total distinct
nonzero ternary patterns across column blocks. Both RSR and SSR use `--width`, enabling explicit block-size sweeps.
An optional `--dataset path` uses existing X/W files for the first input, with
additional inputs generated from the seed. Dataset bias/activation are excluded;
`target_sparsity` is empty for file-based datasets.

## Sweeps

`scripts/benchmark.py` builds and runs correctness tests before sweeping shapes,
sparsity, width, prototype-row count, seeds, and input-pool sizes. It writes
`benchmark.csv`, `samples.csv`, and `run.json`. Summary and sample rows join on
`case` and `implementation`. Metadata records CPU/platform information, compiler
flags, commit and dirty status, requested thread settings, and executed commands.
The default output is a timestamped directory under `build/results`. An explicit
`--out` must name a new directory; existing results are never overwritten.

```sh
python3 scripts/benchmark.py --shapes 1x64x64 17x257x37 \
  --sparsities 0.5 0.95 --widths 4 6 --pattern-rows 0 8 \
  --seeds 42 43 --input-counts 1 8 --threads 1
python3 scripts/benchmark.py --large --out build/large-results
```

`--large` adds a 32x4096x4096 case whose dense integer weights alone occupy
64 MiB. Prepared formats need substantially more memory. Compare this case with
the small presets on the target CPU; cache sizes vary. Run timing experiments
without competing builds or other benchmark processes. Single-thread comparisons
are the default; use `--profile openmp` for thread-scaling experiments.

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

## Team

Nicola Mucciaccio, Matteo Meciani, Riccardo Polo, Alexandre Raybaut, Ahmed Tlili.
Supervisor: Tanja Srindran.

## Plot results

```sh
python3 scripts/plot_benchmarks.py --csv build/results/run-001/benchmark.csv
```

Requires Matplotlib and NumPy. Writes up to four PNG plots, a combined PDF,
and a short summary under the CSV directory's `plots/`. The overview holds width
6, one input, and independent rows fixed; override with `--width`, `--inputs`,
and `--pattern-rows`. Different input counts are never pooled. The effects plot
compares the smallest and largest measured values for each varying setting,
pairing configurations before aggregating. Seed ratios use geometric means.
Target sparsity is read directly from each CSV row; for file-based datasets,
actual sparsity is used. Missing combinations are omitted from the summary.
Older CSV files without `target_sparsity` must be regenerated.

To add an implementation, compile its source under `src/baselines/` and append a
`Kernel` registration in `include/baseline_suite.h`: name, input-index callback,
output view (pointer and element count), preprocessing estimate, weight bytes,
and backend label. The shared tests and benchmark visit every registration.
Keep inputs and weights alive and unchanged while the noncopyable suite exists.
Use owned state captured by callbacks when additional preparation is needed.

