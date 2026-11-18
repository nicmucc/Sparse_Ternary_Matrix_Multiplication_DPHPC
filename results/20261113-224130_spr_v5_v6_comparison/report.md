# SPR V5 vs V6 on the local Apple M4 Pro

machdep.cpu.brand_string: Apple M4 Pro  
hw.physicalcpu: 14  
hw.perflevel0.physicalcpu: 10  
hw.perflevel1.physicalcpu: 4  
hw.cachelinesize: 128  
hw.perflevel0.l1dcachesize: 131072  
hw.perflevel0.l2cachesize: 16777216  
hw.perflevel1.l1dcachesize: 65536  
hw.perflevel1.l2cachesize: 4194304  


Kernels: `spr.v5.b32` (group parallelism) and `spr.v6.b32` (tile × group parallelism).
Threads: [1, 2, 4, 6, 8, 10, 12, 14]; M: [1, 32, 64, 256]; K×N: [(1024, 4096), (2048, 2048), (4096, 1024), (2048, 8192), (1024, 32)]; sparsities: [0.3, 0.5, 0.7, 0.9]; k: [2, 3, 4].
All weights and inputs use seed 42 and are identical between versions and thread counts.
iid: independent element-wise zeros. column_clustered: floor(s*N) leading output columns are zero; remaining columns are dense random +/-1.
The clustered diagnostic uses M=64/256 and K×N=1024×4096 / 2048×8192. It is deliberately imbalanced.
All output rows are verified against the double-precision reference before timing.
Inference medians: 21 interleaved rounds after 3 warmups. Preparation is measured once, excluded from inference, and is not used for conclusions.
OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0, OMP_DYNAMIC=FALSE. macOS chooses core placement; no pinning.
Main comparisons fix k to V5's fastest one-thread k per workload and use it for both versions at every thread count.
`tuned_comparison.csv` additionally compares independently fastest k for each version and thread count.

## V6 inference speedup over V5

Geometric means over regular iid matrix shapes and four sparsities. >1 means V6 is faster.

| Threads | M=1 | M=32 | M=64 | M=256 |
|---|---|---|---|---|
| 1 | 1.00× | 1.00× | 1.00× | 0.99× |
| 2 | 1.00× | 0.99× | 1.02× | 1.08× |
| 4 | 1.00× | 1.00× | 1.03× | 1.16× |
| 6 | 1.00× | 0.99× | 1.07× | 1.15× |
| 8 | 1.00× | 1.00× | 1.14× | 1.32× |
| 10 | 1.00× | 0.99× | 1.21× | 1.35× |
| 12 | 1.00× | 1.00× | 1.13× | 1.27× |
| 14 | 1.00× | 1.00× | 1.18× | 1.37× |

## Interpretation

M=1 and M=32 contain at most one B=32 tile, so there is no extra M parallelism to exploit.
At 10 threads, regular-grid V6/V5 speedups are 1.21× at M=64 and 1.35× at M=256.
Best observed V6 thread counts over regular matrices (fixed-k geometric means): M=1: 8T, M=32: 10T, M=64: 10T, M=256: 10T.
For synthetic clustered zeros at M=256, V6/V5 is 2.50× at 10 threads, separate from regular-grid averages.
For N=32, M=64, V6/V5 is 0.55× at 10 threads; values below one mean V6 is slower.
Best observed V6 thread counts for N=32: M=1: 1T, M=32: 1T, M=64: 1T, M=256: 4T. See narrow_runtime.png before interpreting large relative speedups.

## Breakdown by matrix size (iid)

| M | K×N | Threads | V6/V5 speedup |
|---|---|---|---|
| 1 | 1024×4096 | 1 | 1.00× |
| 1 | 1024×4096 | 4 | 1.00× |
| 1 | 1024×4096 | 10 | 0.97× |
| 1 | 1024×4096 | 14 | 1.01× |
| 1 | 2048×2048 | 1 | 1.06× |
| 1 | 2048×2048 | 4 | 1.02× |
| 1 | 2048×2048 | 10 | 1.00× |
| 1 | 2048×2048 | 14 | 1.01× |
| 1 | 4096×1024 | 1 | 0.98× |
| 1 | 4096×1024 | 4 | 1.01× |
| 1 | 4096×1024 | 10 | 1.00× |
| 1 | 4096×1024 | 14 | 0.99× |
| 1 | 2048×8192 | 1 | 0.97× |
| 1 | 2048×8192 | 4 | 0.97× |
| 1 | 2048×8192 | 10 | 1.05× |
| 1 | 2048×8192 | 14 | 1.00× |
| 1 | 1024×32 | 1 | 1.01× |
| 1 | 1024×32 | 4 | 1.00× |
| 1 | 1024×32 | 10 | 1.03× |
| 1 | 1024×32 | 14 | 1.00× |
| 32 | 1024×4096 | 1 | 1.01× |
| 32 | 1024×4096 | 4 | 0.99× |
| 32 | 1024×4096 | 10 | 1.00× |
| 32 | 1024×4096 | 14 | 1.00× |
| 32 | 2048×2048 | 1 | 0.99× |
| 32 | 2048×2048 | 4 | 0.99× |
| 32 | 2048×2048 | 10 | 0.98× |
| 32 | 2048×2048 | 14 | 0.99× |
| 32 | 4096×1024 | 1 | 1.01× |
| 32 | 4096×1024 | 4 | 1.01× |
| 32 | 4096×1024 | 10 | 1.00× |
| 32 | 4096×1024 | 14 | 0.99× |
| 32 | 2048×8192 | 1 | 0.98× |
| 32 | 2048×8192 | 4 | 0.99× |
| 32 | 2048×8192 | 10 | 1.00× |
| 32 | 2048×8192 | 14 | 1.00× |
| 32 | 1024×32 | 1 | 1.00× |
| 32 | 1024×32 | 4 | 0.99× |
| 32 | 1024×32 | 10 | 1.01× |
| 32 | 1024×32 | 14 | 0.98× |
| 64 | 1024×4096 | 1 | 0.99× |
| 64 | 1024×4096 | 4 | 1.03× |
| 64 | 1024×4096 | 10 | 1.21× |
| 64 | 1024×4096 | 14 | 1.21× |
| 64 | 2048×2048 | 1 | 1.00× |
| 64 | 2048×2048 | 4 | 1.06× |
| 64 | 2048×2048 | 10 | 1.22× |
| 64 | 2048×2048 | 14 | 1.20× |
| 64 | 4096×1024 | 1 | 1.00× |
| 64 | 4096×1024 | 4 | 1.04× |
| 64 | 4096×1024 | 10 | 1.19× |
| 64 | 4096×1024 | 14 | 1.23× |
| 64 | 2048×8192 | 1 | 0.99× |
| 64 | 2048×8192 | 4 | 0.98× |
| 64 | 2048×8192 | 10 | 1.22× |
| 64 | 2048×8192 | 14 | 1.08× |
| 64 | 1024×32 | 1 | 0.99× |
| 64 | 1024×32 | 4 | 0.96× |
| 64 | 1024×32 | 10 | 0.55× |
| 64 | 1024×32 | 14 | 0.41× |
| 256 | 1024×4096 | 1 | 0.99× |
| 256 | 1024×4096 | 4 | 1.18× |
| 256 | 1024×4096 | 10 | 1.40× |
| 256 | 1024×4096 | 14 | 1.45× |
| 256 | 2048×2048 | 1 | 0.99× |
| 256 | 2048×2048 | 4 | 1.16× |
| 256 | 2048×2048 | 10 | 1.47× |
| 256 | 2048×2048 | 14 | 1.50× |
| 256 | 4096×1024 | 1 | 1.00× |
| 256 | 4096×1024 | 4 | 1.14× |
| 256 | 4096×1024 | 10 | 1.39× |
| 256 | 4096×1024 | 14 | 1.39× |
| 256 | 2048×8192 | 1 | 0.99× |
| 256 | 2048×8192 | 4 | 1.15× |
| 256 | 2048×8192 | 10 | 1.18× |
| 256 | 2048×8192 | 14 | 1.18× |
| 256 | 1024×32 | 1 | 1.02× |
| 256 | 1024×32 | 4 | 4.04× |
| 256 | 1024×32 | 10 | 6.58× |
| 256 | 1024×32 | 14 | 7.25× |

## Plots and data

- `speedup.png`: V6/V5 by sparsity, M, threads (regular matrices).
- `matrix_speedup.png`: includes the narrow-output diagnostic; this is excluded from the main aggregate.
- `thread_scaling.png`: each version versus its own 1T time.
- `pattern_speedup.png`: iid versus clustered output zeros.
- `narrow_runtime.png`: absolute times in the few-group diagnostic; excessive threads can hurt both versions.
- `roofline.png`: M=256, shared k; colors match measured thread-count SIMD/DRAM roofs.
- `tXX/report.md`, `tXX/roofline.png`: standard harness reports for M=64 iid cases, independently best k.
- `comparison.csv`: all fixed-k results; `tuned_comparison.csv`: independently tuned comparisons; `raw.csv`: full k sweep.

Roofline intensity uses compulsory format + X + Y traffic and the kernels' existing addition-count convention; it is not measured cache traffic.
Near-unity differences may reflect measurement noise. This run uses one deterministic weight sample per workload; the clustered case is a synthetic scheduling diagnostic.
