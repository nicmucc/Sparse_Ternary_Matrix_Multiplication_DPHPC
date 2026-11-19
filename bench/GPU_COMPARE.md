# Compare CUDA SPR versions on the D-INFK student cluster

Run from the updated project root. The workflow compiles the current kernel files;
it does not change them. Default comparison: v2 baseline versus v3 candidate.
The Slurm job requests the RTX 5060 Ti (`--gpus=5060ti:1`), account `dphpc`,
three CPU slots, 16 GiB RAM and one hour. It loads CUDA 12.9 and targets `sm_120`.
GPU selection and module setup follow the [ETH cluster instructions](https://isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs).

Ensure the new `gpu/spr/spr_v3.cu` is copied to the cluster too. If transferring
with Git, include it in your commit: an untracked local file will not arrive via pull.
V3 is benchmarked with its current launch configuration (currently 32 rows by
8 helpers); this workflow does not tune or modify helper counts.

## V2 versus v4

The runner accepts v1, v2, v3 and v4. Select the baseline and candidate using the
third and fourth batch-script arguments, after suite and sample count:

```bash
sbatch bench/gpu_compare.sbatch smoke 30 v2 v4
# After the smoke job completes successfully:
sbatch bench/gpu_compare.sbatch full 30 v2 v4
```

Both submissions run correctness and sanitizer checks for **v2 and v4** before
timing. These examples use 30 samples even for the short six-workload smoke suite.
The full suite still has 126 workloads per version. V4 uses cooperative segment
sums, one barrier and helper-0 tree evaluation; its current launch is `(32,8)`.

Results are `results/gpu_compare_v2_v4_JOBID/`. Speedup is median(v2)/median(v4),
so values above 1 favor v4. Existing v2/v3 commands without pair arguments continue
to use v2/v3. For a direct run inside an allocated GPU shell:

```bash
python3 bench/run_gpu_compare.py --versions v2 v4 --suite smoke --samples 30 --arch sm_120 --sanitize
```

For standalone v4 correctness:

```bash
make test-gpu CUDA_ARCH=sm_120 TEST_ARGS="spr.v2.cuda spr.v4.cuda"
```

Generate plots locally after downloading the result directory:

```bash
.venv/bin/python bench/plot_gpu_compare.py results/gpu_compare_v2_v4_JOBID
```

The plotter reads the version labels and sample count from metadata; its CSV,
PNG/PDF figures and analysis report therefore use v2/v4 labels automatically.

## Submit

First run the short campaign:

```bash
sbatch bench/gpu_compare.sbatch smoke
```

Wait for `Completed:` in `spr-gpu-JOBID.out`. Then submit the full campaign:

```bash
sbatch bench/gpu_compare.sbatch full
```

Monitor with:

```bash
squeue -u "$USER"
tail -f spr-gpu-JOBID.out
```

The log prints the results directory, normally
`results/gpu_compare_v2_v3_JOBID/`. Both versions run on the same allocated GPU.
You can change the sample count, for example `sbatch bench/gpu_compare.sbatch full 50`.
A failed compilation, correctness test, sanitizer check or benchmark case fails the job.

## Correctness runs before timing

The runner builds and executes both suites for **each selected version**:

- `tests/test_correctness.cpp`: exact tree examples and randomized cases, all outputs
  compared to the FP64-derived reference, including repeated runs.
- `tests/test_gpu_correctness.cpp`: empty dimensions, all-zero/empty first blocks,
  ragged shapes, k=1/3/8/16, K=65535, two input bindings and three runs per binding,
  all outputs checked against a double-precision reference.

The batch job also requires Compute Sanitizer `memcheck`, `racecheck` and
`synccheck` to pass for each version on M=1 and M=33, K=129, N=17, k=8,
50% zero weights. These small cases exercise partial blocks and synchronization;
they supplement the reference tests. Sanitizer executions are separate from timing.
Logs are retained as `correctness_VERSION_*.log` and `sanitize_VERSION_*.log`.

For standalone correctness inside a GPU allocation:

```bash
. /etc/profile.d/modules.sh
module add cuda/12.9
make test-gpu CUDA_ARCH=sm_120 TEST_ARGS="spr.v2.cuda spr.v3.cuda"
```

## Running interactively

From the login node, request a GPU shell:

```bash
srun --pty -A dphpc --gpus=5060ti:1 --cpus-per-task=3 --mem=16G -t 01:00:00 bash --login
```

Inside it, change to the project root and run:

```bash
. /etc/profile.d/modules.sh
module add cuda/12.9
python3 bench/run_gpu_compare.py --versions v2 v3 --suite smoke --samples 5 --arch sm_120 --sanitize
# After smoke succeeds:
python3 bench/run_gpu_compare.py --versions v2 v3 --suite full --samples 30 --arch sm_120 --sanitize
```

The Python runner always runs both correctness suites. `--sanitize` additionally
requires Compute Sanitizer to be available and all its checks to pass.
`--profile` optionally collects representative Nsight Compute reports; profiling
failures are recorded separately and do not invalidate successful benchmark timings.
The previous v1/v2 experiment remains available with `--versions v1 v2`.

## Workloads and timing

Smoke: M={1,33,257}, K=N=257, k={4,8}, zero fraction=0.5: six workloads per version.
Full: M={1,8,31,32,33,128,512}, (K,N)={(1024,1024),(4096,4096)},
k={4,8,16}, zero fraction={0,0.5,0.9}: 126 workloads per version.

Versions use identical seed-42 input/weight matrices. There are five warm-up
launches and 30 samples per inference metric in the full campaign. Version order
alternates between workloads, and timing-phase order rotates between samples.

- `gpu_event`: CUDA events around the launch, before the existing synchronization.
- `run_wall`: synchronized `run()` time measured on the CPU, with event hooks disabled.
- `end_to_end_wall`: bind + run + finish, including allocations, CPU input transposition,
  input upload and output download. Weight preparation is excluded.
- `prepare_wall`: one preparation per workload/version, including metadata upload,
  after CUDA context initialization. This is not a repeated timing distribution.

Event hooks are added only to generated source copies. Both disabled event-hook
branches remain in the wall-time executable. Small kernel timings can be sensitive
to launch/event overhead. Inputs repeat, caches are warm and GPU clocks are not locked.

Each benchmark checks output correctness before and after timing. Cases with
M*K*N <= 1e7 check every output against FP64 sums. Larger cases check eight evenly
spaced rows by 64 columns; every output is also checked for NaN/Inf. Tolerance is
2e-3*sqrt(K). Thus the dedicated suites fully check their test cases, while large
benchmark workloads receive sampled reference checks.

## Plots and results

Every successful run produces, without extra Python packages:

- `comparison.html`: standalone SVG speedup plots for all sparsities and three inference metrics.
- `raw_samples.csv`, `timing_summary.csv`: samples, medians and quartiles.
- `speedups.csv`: median(v2)/median(v3), matching M, K, N, k and sparsity.
- `correctness.json`, correctness/sanitizer logs and `failures.json`.
- `environment.json`: actual GPU/compiler information, commands, source hashes and campaign status.
- Kernel, test and benchmark source snapshots plus compilation and per-case logs.

Open `comparison.html` in a browser after copying the results to your laptop.
**Speedup above 1 means v3 is faster; below 1 means v2 is faster.** Compare GPU-event
speedup with end-to-end speedup to distinguish kernel improvements from total latency.

For PNG/PDF latency plots with IQR bands, speedup curves, and data-derived analysis:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python bench/plot_gpu_compare.py results/gpu_compare_v2_v3_JOBID
```

This can run on your laptop after downloading the results; it requires no GPU.
Plots are saved in `plots/`, with `index.html`, `analysis.md`, `best_k.csv` and
`validation.json`. The plotter rejects failed/incomplete runs and verifies summary
statistics against raw samples. It reads the version pair and sample count from
metadata and makes no assumptions about which version wins.

If NumPy/Matplotlib are already installed in the batch job's Python environment,
the job generates PNG/PDF plots automatically. For a cluster-created virtual environment:

```bash
PYTHON="$PWD/.venv/bin/python" sbatch bench/gpu_compare.sbatch full
```

`best_k.csv` selects the fastest measured k per version/workload using these same
samples. It is exploratory selection, not a held-out tuning result. IQR bands are
within-run variability, not confidence intervals across independent jobs.

Optional Nsight reports are `profile_VERSION_M*.ncu-rep`; `profiling_status.json`
records unavailable tools or restricted performance counters. No DRAM bandwidth
or roofline values are inferred without hardware counters. Actual SPR additions
are `M * adds_per_row`; dense `2*M*K*N` is not the work executed by SPR.
