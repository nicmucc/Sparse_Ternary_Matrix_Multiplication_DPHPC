# Interpretation of the V8 rectangle-scheduling benchmark

All results use 10 requested threads on the local Apple M4 Pro. The main comparison fixes B=32 and k=3.
Correctness checks and benchmark reference verification were skipped as requested; these are timing measurements only.

Across the 96 regular-matrix/sparsity workloads, V8/V6 is 0.949× and V8/V7 is 2.270× (geometric means).
V8's average runtime relative to V6 is 1.054×: 5.4% longer.
A speedup above 1 means V8 is faster. A speedup below 1 means V8 takes longer.

| M | SIMD tiles | V8/V6 | V8/V7 | V8 runtime versus V6 |
|---|---|---|---|---|
| 1 | 1 | 0.968× | 5.291× | 1.033× as long |
| 32 | 1 | 0.992× | 5.657× | 1.008× as long |
| 64 | 2 | 0.957× | 3.614× | 1.045× as long |
| 256 | 8 | 0.936× | 1.085× | 1.069× as long |
| 512 | 16 | 0.906× | 1.067× | 1.103× as long |
| 1024 | 32 | 0.937× | 1.092× | 1.067× as long |

V7 can only use one worker at M=1/32 and at most two workers at M=64. V8 can split work across groups as well. Its large gains over V7 in these cases primarily show recovered parallelism; V6 already offers that parallelism.
For regular matrices, the rectangular scheduler does not establish a general improvement over V6 in this run. V8 introduces dynamic dispatch and coarser work units. Those are plausible costs, but the timing data does not isolate them from cache behavior or core placement.

## Sparse-pattern diagnostic with matched shapes

These two rows use matching M, K, N, target sparsity levels and seeds, with different constructions of the sparse pattern.

| Pattern | Workloads | V8/V6 | V8/V7 |
|---|---|---|---|
| Independent random zeros | 32 | 0.922× | 1.476× |
| Contiguous zero output columns | 32 | 1.211× | 1.445× |

The clustered case intentionally makes group costs unequal. Differences here indicate sensitivity to work distribution; they do not isolate the cost estimator from the dynamic scheduling policy.

| M | Clustered V8/V6 | Clustered V8/V7 |
|---|---|---|
| 64 | 1.843× | 3.385× |
| 256 | 1.093× | 1.097× |
| 512 | 1.059× | 1.085× |
| 1024 | 1.009× | 1.081× |

## Sensitivity to k

Matching k isolates scheduling; independently choosing each version's best observed k is an additional tuning comparison.

| Choice | V8/V6 | V8/V7 |
|---|---|---|
| Shared k=2 | 0.922× | 2.110× |
| Shared k=3 | 0.949× | 2.270× |
| Shared k=4 | 0.972× | 2.277× |
| Independently fastest observed k | 0.976× | 2.227× |

## Narrow outputs: absolute runtime

N=32, K=1024, k=3; geometric means over sparsities. Small absolute runtimes make thread overhead important.

| M | V6 [µs] | V7 [µs] | V8 [µs] |
|---|---|---|---|
| 1 | 31.57 | 8.88 | 12.72 |
| 32 | 35.58 | 11.80 | 15.20 |
| 64 | 60.54 | 64.67 | 31.87 |
| 256 | 81.92 | 79.89 | 81.27 |
| 512 | 100.76 | 104.73 | 99.22 |
| 1024 | 157.52 | 170.73 | 159.02 |

These are 31-round medians from one timing campaign and one seed per workload; small differences need repeated campaigns to establish reliability.

[Measurement method and complete tables](report.md) · [Speedup plot](speedup.png) · [Runtime plot](runtime.png) · [Pattern plot](pattern_speedup.png) · [Roofline](roofline.png)
