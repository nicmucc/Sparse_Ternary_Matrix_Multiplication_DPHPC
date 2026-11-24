# SPR CUDA v8 benchmark campaign

This directory holds a reproducible comparison of `spr.v8.cuda` with GPU
versions v0-v7 and a complete `k=1..16` sweep for v8 on the large shape.
The presentation artifact is `gpu_v8_campaign.pdf` (vector graphics, A4 pages).

## Hardware and timing

- NVIDIA GeForce RTX 5060 Ti, 16,311 MiB reported VRAM, CUDA 13.4 compiler,
  driver 615.71.09, native `sm_120` build. Jobs ran on `studgpu-node01`.
  Each job log records its assigned GPU UUID; Slurm can choose a different
  physical 5060 Ti for a later group.
- Independent ternary weights, with zero probability 0.5 for version
  comparisons and 0.5 or 0.9 for v8 width sweeps. Seed 42 throughout.
  The harness seeds W independently of M and X, so each M sweep reuses the
  same weights and the same X prefix at fixed K, N, and sparsity.
- All times are synchronized `run()` medians in milliseconds. Preparation,
  layout conversion, H2D input transfer, and D2H output transfer are excluded.
  `prep_ms` remains available separately in `summary.csv`.
- Each configuration runs in its own process to prevent simultaneous X, Y,
  and scratch allocations from exhausting VRAM. Warmups are 2 at M <= 4096
  and 1 otherwise; timed rounds are 9 at M <= 4096, 5 at M = 16384, and 3 at
  M = 100000. The PDF plots the median and observed min-max range.
- The benchmark checks 4 output rows against FP64 at M <= 4096, 2 rows at
  M = 16384, and 1 row at M = 100000 before timing. The native build also
  passed 361 shared correctness cases and 108 GPU edge cases for v8.

## Measured configurations

| Family | K | N | M values |
|---|---:|---:|---|
| Small | 1024 | 4096 | 256, 4096, 16384 |
| Medium | 2048 | 8192 | 256, 4096, 16384 |
| Large | 4096 | 14336 | 256, 4096, 16384, 100000 |

The version comparison uses `k=4` for v0-v5 and v7-v8. V6 has a fixed
two-column algorithm, so it is explicitly labeled `k=2`. A second v8 series
at `k=16` shows the effect of widening the weight block; it is not an
equal-width comparison with the older kernels. The width sweep tests every
integer k from 1 through 16 on the large family at M=4096, 16384, 100000
for sparsity 0.5, and M=4096, 100000 for sparsity 0.9.

A supplemental comparison measures `k=2` for v2, v5, and v7 on every shape
where it fits, then plots the faster of their tested k=2/4 configurations on
separate pages. This avoids treating a k=4 memory exclusion as proof that a
kernel cannot run at any width. The full-width v8 sweep is used for the
large-family selected-width page at M >= 4096; elsewhere that page selects
the faster of the measured v8 k=4/16 configurations.

At large-family M=16384, the k=4 global scratch of v0-v4 exceeds device
memory. At M=100000, the k=4 global scratch of v0-v4 and v7 exceeds device
memory; v2 also cannot fit at k=2 there. `excluded.csv` records these planned exclusions explicitly. Missing
points in the PDF are not timings, and the version comparison does not claim
that k=4 is the best width for every older kernel.

## Results

All 198 measured configurations passed sampled FP64 correctness checks. The
merged tables contain 1,354 timing samples. On the large family, `k=16` was
the fastest v8 width among all tested widths in each complete sweep:

| Sparsity | M | v8 k=1 | v8 k=4 | v8 k=16 | k=16 speedup over k=4 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 4096 | 600.3 ms | 282.5 ms | 148.9 ms | 1.90x |
| 0.5 | 16384 | 4157.3 ms | 1938.6 ms | 654.7 ms | 2.96x |
| 0.5 | 100000 | 27817.5 ms | 13124.2 ms | 4167.6 ms | 3.15x |
| 0.9 | 4096 | 118.6 ms | 103.4 ms | 66.4 ms | 1.56x |
| 0.9 | 100000 | 5570.6 ms | 4825.1 ms | 2896.5 ms | 1.67x |

At large-family M=100000 and sparsity 0.5, the fastest measured older
configuration was v5 k=4 at 14.276 s; v8 k=16 took 4.169 s, a 3.42x speedup.
At M=4096 with equal k=4, v2 took 363.8 ms, v7 took 297.1 ms, and v8 took
282.5 ms. Thus the large gain at M=100000 also depends on selecting a wider
block. At small and medium shapes with M=256 or 4096, v8 k=4 can beat k=16;
the optimum is shape dependent. These are kernel-only comparisons among the
tested SPR CUDA versions; they do not include input/output transfers or a
dense GEMM baseline.

## Files and regeneration

- `manifest.csv`: one row per planned configuration, including exact shape,
  sparsity, kernel, and k.
- `raw/*.csv` and `raw/*.samples.csv`: untouched summary and per-round output
  from each isolated benchmark process.
- `summary.csv`, `samples.csv`, `best_k.csv`: combined tables for external
  plotting and analysis.
- `group_gpu.csv`: assigned physical GPU UUID for each benchmark group.
- `logs/*.log`, `group-*.out`, `build-*.out`: exact benchmark commands, compiler
  and GPU information, and correctness/build output.
- `source_sha256.txt`: hashes of the benchmark binary, GPU kernels, harness,
  plotting script, and build settings used for this campaign.
- `gpu_v8_campaign.pdf`: standalone vector report with a legend on each graph.

From the repository root, on the cluster:

```sh
python3 bench/run_gpu_v8_campaign.py plan
sbatch results/gpu_v8_campaign_20261124/build_and_test.sbatch
# After the build succeeds, submit one run_group.sbatch job for each group:
# compare_small, compare_medium, compare_large, sweep_s05, sweep_s09,
# baseline_k2.
# Run them sequentially or use Slurm --dependency=afterok:<previous-job-id>.
python3 bench/run_gpu_v8_campaign.py merge
python3 bench/plot_gpu_v8_campaign.py
```

The `run_group.sbatch` script uses the first argument as the group name.
Its account, GPU node, and partition directives are cluster-specific; edit
those directives on another system. The Python runner resumes completed
configuration CSVs by default. The PDF plotter uses only the Python standard
library and does not require Matplotlib or a browser.
