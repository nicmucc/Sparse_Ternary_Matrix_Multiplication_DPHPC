# SPR v1 versus v2 on the student GPU cluster

Run from an extracted project root containing the files below. Original kernels
are not edited. Each version is compiled into its own executable to avoid their
duplicate helper symbols. Requires Python 3 and a CUDA toolkit compatible with
the allocated GPU; the supplied Slurm script requests one 2080 Ti and CUDA 12.9.

## Submit

```bash
sbatch bench/gpu_compare.sbatch
```

This submits account `dphpc`, requests 30 minutes, and runs the full comparison
plus four optional Nsight Compute profiles. The job log is `spr-gpu-JOBID.out`.
Monitor with `squeue -u "$USER"` and `tail -f spr-gpu-JOBID.out`.

For a quick check inside an existing GPU allocation:

```bash
module add cuda/12.9
python3 bench/run_gpu_compare.py --suite smoke --samples 5
```

For the full run inside an existing allocation:

```bash
python3 bench/run_gpu_compare.py --suite full --samples 30 --profile
```

## Scope and timing

Full sweep: M = 1,8,31,32,33,128,512; (K,N) = (1024,1024),(4096,4096);
k = 4,8,16; zero fraction = 0,0.5,0.9. This is 126 configurations per version.
Weights and inputs use the same fixed seed. There are five warmup runs and 30
samples per timing metric. Version order alternates between configurations.
The same input/weights are reused: this is a warm-cache, steady-state comparison.
GPU clocks are not locked by this script.

- `gpu_event`: CUDA events immediately surrounding the launch, before the
  existing device synchronization. The hooks exist only in generated build copies.
- `run_wall`: synchronized `run()` time on the CPU. Event hooks are disabled.
- `end_to_end_wall`: bind + run + finish. Includes device allocations, input
  upload, output download, and v2's CPU transpose. Excludes weight preprocessing.
- `prepare_wall`: one preparation per version/configuration, including metadata
  allocation/upload, after CUDA context initialization. Not a repeated statistic.

The two disabled event-hook branches are also present in the wall-time executable;
the kernel source and GPU arithmetic are unchanged. For very short kernels, event
and launch overhead can be significant. Use profiler durations as additional evidence.

Correctness is checked before/after timing against a dense double-precision CPU
reference. Small problems (M*K*N <= 1e7) check all outputs. Larger problems check
eight evenly spaced rows by 64 columns, and all outputs are checked for NaN/Inf.
The tolerance is the project's existing 2e-3*sqrt(K). This benchmark does not replace
the separate edge-case correctness tests or Compute Sanitizer.

## Results

Each run creates `results/gpu_compare_TIMESTAMP/` containing:

- `raw_samples.csv`: all successful samples and arithmetic/storage accounting.
- `timing_summary.csv`: first quartile, median, third quartile.
- `speedups.csv`: median(v1)/median(v2), with matching workload and timing boundary.
- `comparison.html`: standalone speedup plots at 50% zero weights; no Python plotting packages needed.
- `environment.json`: GPU/compiler/host information, seed, commands, source hashes.
- Kernel/source snapshots, compilation logs, per-configuration validation logs.
- `failures.json`: failures; failed cases are excluded from timing summaries.

Nsight Compute, when available and permitted, profiles M=1 and M=128, K=N=4096,
k=8, 50% zero weights. Reports are `profile_VERSION_M*.ncu-rep` and companion logs.
`profiling_status.json` records unavailable tools or failures, including restricted
performance counters. Profiling is separate from normal timing; cache flushing is
disabled to approximate the repeated-input workload, but replay/clock policies can
still differ. Open the reports in Nsight Compute to inspect memory transactions,
DRAM traffic, occupancy, stalls, and roofline sections.

For a manually constructed SPR roofline, use actual additions:
`A = M * adds_per_row`, `performance = A / kernel_seconds`, and
`intensity = A / (measured DRAM read bytes + write bytes)`.
Do not use compressed storage bytes as dynamic DRAM traffic or label dense
`2*M*K*N` work as executed SPR FLOPs. No DRAM bandwidth/roofline values are
fabricated when hardware-counter collection is unavailable.
