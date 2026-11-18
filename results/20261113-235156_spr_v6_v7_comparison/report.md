# SPR V6 vs V7 on the local Apple M4 Pro

machdep.cpu.brand_string: Apple M4 Pro  
hw.physicalcpu: 14  
hw.perflevel0.physicalcpu: 10  
hw.perflevel1.physicalcpu: 4  
hw.cachelinesize: 128  
hw.perflevel0.l1dcachesize: 131072  
hw.perflevel0.l2cachesize: 16777216  
hw.perflevel1.l1dcachesize: 65536  
hw.perflevel1.l2cachesize: 4194304  


Kernels: `spr.v6.b32` (collapsed tile x group scheduling) and `spr.v7.b32` (whole-tile scheduling). Both use serial preparation.
Threads: [1, 2, 4, 6, 8, 10, 12, 14]; M: [32, 64, 256, 512, 1024]; K×N: [(1024, 4096), (2048, 2048), (4096, 1024), (2048, 8192), (1024, 32)]; sparsities: [0.3, 0.5, 0.7, 0.9]; k: [2, 3, 4].
All weights and inputs use seed 42 and are identical between versions and thread counts.
iid: independent element-wise zeros. column_clustered: floor(s*N) leading output columns are zero; remaining columns are dense random +/-1.
The clustered diagnostic uses M=256/512 and K×N=1024×4096 / 2048×8192. It is deliberately imbalanced.
All output rows are verified against the double-precision reference before timing.
Inference medians: 21 interleaved rounds after 3 warmups. Preparation is measured once, excluded from inference, and is not used for conclusions.
OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0, OMP_DYNAMIC=FALSE. macOS chooses core placement; no pinning.
Main comparisons fix k to V6's fastest one-thread k per workload and use it for both versions at every thread count.
`tuned_comparison.csv` additionally compares independently fastest k for each version and thread count.

## V7 inference speedup over V6

Geometric means over regular iid matrix shapes and four sparsities. >1 means V7 is faster.

| Threads | M=32 | M=64 | M=256 | M=512 | M=1024 |
|---|---|---|---|---|---|
| 1 | 1.01× | 1.00× | 1.00× | 1.00× | 1.00× |
| 2 | 0.56× | 1.01× | 0.99× | 1.00× | 1.00× |
| 4 | 0.30× | 0.53× | 0.99× | 1.00× | 1.00× |
| 6 | 0.27× | 0.38× | 0.70× | 0.91× | 0.92× |
| 8 | 0.25× | 0.32× | 1.00× | 0.99× | 1.00× |
| 10 | 0.22× | 0.28× | 0.84× | 0.85× | 0.86× |
| 12 | 0.23× | 0.35× | 1.06× | 1.06× | 1.01× |
| 14 | 0.25× | 0.33× | 0.95× | 0.99× | 1.03× |

## Interpretation

V7 schedules only input tiles: M=32/64/256/512/1024 provide 1/2/8/16/32 tiles at B=32. Requested threads can exceed the number of useful workers.
At M=32, V7's parallel-region condition is false. Changing the requested thread count does not parallelize that case, and any variation in V7's absolute time is not evidence of thread scaling.
When the tile count is divisible by the thread count, V6's contiguous static collapsed assignment can already give each thread complete tiles. V7 may then mostly remove indexing overhead.
Having at least as many tiles as threads does not ensure perfect balance: 16 tiles on 10 threads still give some threads twice as many tiles as others.
At 10 threads, regular-grid V7/V6 speedups: M=32: 0.22×, M=64: 0.28×, M=256: 0.84×, M=512: 0.85×, M=1024: 0.86×.
Best observed V7 thread counts over regular matrices (fixed-k geometric means): M=32: serial at all settings, M=64: 10T, M=256: 10T, M=512: 14T, M=1024: 14T.
For synthetic clustered zeros at M=256, V7/V6 is 1.00× at 10 threads, separate from regular-grid averages.
For N=32, M=64, V7/V6 is 0.95× at 10 threads; values below one mean V7 is slower.
Best observed V7 thread counts for N=32: M=32: serial at all settings, M=64: 1T, M=256: 4T, M=512: 8T, M=1024: 8T. See narrow_runtime.png before interpreting large relative speedups.

## Breakdown by matrix size (iid)

| M | K×N | Threads | V7/V6 speedup |
|---|---|---|---|
| 32 | 1024×4096 | 1 | 1.02× |
| 32 | 1024×4096 | 4 | 0.30× |
| 32 | 1024×4096 | 10 | 0.25× |
| 32 | 1024×4096 | 14 | 0.25× |
| 32 | 2048×2048 | 1 | 1.02× |
| 32 | 2048×2048 | 4 | 0.31× |
| 32 | 2048×2048 | 10 | 0.25× |
| 32 | 2048×2048 | 14 | 0.28× |
| 32 | 4096×1024 | 1 | 1.02× |
| 32 | 4096×1024 | 4 | 0.32× |
| 32 | 4096×1024 | 10 | 0.24× |
| 32 | 4096×1024 | 14 | 0.29× |
| 32 | 2048×8192 | 1 | 0.97× |
| 32 | 2048×8192 | 4 | 0.28× |
| 32 | 2048×8192 | 10 | 0.15× |
| 32 | 2048×8192 | 14 | 0.18× |
| 32 | 1024×32 | 1 | 1.00× |
| 32 | 1024×32 | 4 | 1.19× |
| 32 | 1024×32 | 10 | 1.84× |
| 32 | 1024×32 | 14 | 2.20× |
| 64 | 1024×4096 | 1 | 1.01× |
| 64 | 1024×4096 | 4 | 0.55× |
| 64 | 1024×4096 | 10 | 0.28× |
| 64 | 1024×4096 | 14 | 0.34× |
| 64 | 2048×2048 | 1 | 1.00× |
| 64 | 2048×2048 | 4 | 0.54× |
| 64 | 2048×2048 | 10 | 0.29× |
| 64 | 2048×2048 | 14 | 0.34× |
| 64 | 4096×1024 | 1 | 1.00× |
| 64 | 4096×1024 | 4 | 0.51× |
| 64 | 4096×1024 | 10 | 0.30× |
| 64 | 4096×1024 | 14 | 0.35× |
| 64 | 2048×8192 | 1 | 1.00× |
| 64 | 2048×8192 | 4 | 0.52× |
| 64 | 2048×8192 | 10 | 0.24× |
| 64 | 2048×8192 | 14 | 0.29× |
| 64 | 1024×32 | 1 | 1.00× |
| 64 | 1024×32 | 4 | 0.95× |
| 64 | 1024×32 | 10 | 0.95× |
| 64 | 1024×32 | 14 | 0.99× |
| 256 | 1024×4096 | 1 | 0.99× |
| 256 | 1024×4096 | 4 | 1.00× |
| 256 | 1024×4096 | 10 | 0.83× |
| 256 | 1024×4096 | 14 | 0.93× |
| 256 | 2048×2048 | 1 | 1.00× |
| 256 | 2048×2048 | 4 | 0.99× |
| 256 | 2048×2048 | 10 | 0.82× |
| 256 | 2048×2048 | 14 | 0.95× |
| 256 | 4096×1024 | 1 | 1.00× |
| 256 | 4096×1024 | 4 | 1.00× |
| 256 | 4096×1024 | 10 | 0.85× |
| 256 | 4096×1024 | 14 | 1.00× |
| 256 | 2048×8192 | 1 | 1.00× |
| 256 | 2048×8192 | 4 | 0.99× |
| 256 | 2048×8192 | 10 | 0.85× |
| 256 | 2048×8192 | 14 | 0.94× |
| 256 | 1024×32 | 1 | 0.98× |
| 256 | 1024×32 | 4 | 0.99× |
| 256 | 1024×32 | 10 | 1.00× |
| 256 | 1024×32 | 14 | 0.99× |
| 512 | 1024×4096 | 1 | 1.02× |
| 512 | 1024×4096 | 4 | 1.00× |
| 512 | 1024×4096 | 10 | 0.81× |
| 512 | 1024×4096 | 14 | 0.93× |
| 512 | 2048×2048 | 1 | 1.00× |
| 512 | 2048×2048 | 4 | 1.00× |
| 512 | 2048×2048 | 10 | 0.85× |
| 512 | 2048×2048 | 14 | 1.01× |
| 512 | 4096×1024 | 1 | 1.00× |
| 512 | 4096×1024 | 4 | 1.00× |
| 512 | 4096×1024 | 10 | 0.88× |
| 512 | 4096×1024 | 14 | 1.06× |
| 512 | 2048×8192 | 1 | 1.00× |
| 512 | 2048×8192 | 4 | 1.00× |
| 512 | 2048×8192 | 10 | 0.85× |
| 512 | 2048×8192 | 14 | 0.96× |
| 512 | 1024×32 | 1 | 0.96× |
| 512 | 1024×32 | 4 | 0.99× |
| 512 | 1024×32 | 10 | 0.98× |
| 512 | 1024×32 | 14 | 1.03× |
| 1024 | 1024×4096 | 1 | 1.00× |
| 1024 | 1024×4096 | 4 | 0.99× |
| 1024 | 1024×4096 | 10 | 0.86× |
| 1024 | 1024×4096 | 14 | 1.03× |
| 1024 | 2048×2048 | 1 | 1.00× |
| 1024 | 2048×2048 | 4 | 1.00× |
| 1024 | 2048×2048 | 10 | 0.86× |
| 1024 | 2048×2048 | 14 | 1.06× |
| 1024 | 4096×1024 | 1 | 1.00× |
| 1024 | 4096×1024 | 4 | 0.99× |
| 1024 | 4096×1024 | 10 | 0.86× |
| 1024 | 4096×1024 | 14 | 1.03× |
| 1024 | 2048×8192 | 1 | 1.01× |
| 1024 | 2048×8192 | 4 | 1.00× |
| 1024 | 2048×8192 | 10 | 0.86× |
| 1024 | 2048×8192 | 14 | 0.99× |
| 1024 | 1024×32 | 1 | 0.99× |
| 1024 | 1024×32 | 4 | 0.98× |
| 1024 | 1024×32 | 10 | 1.00× |
| 1024 | 1024×32 | 14 | 1.03× |

## Plots and data

- `speedup.png`: V7/V6 by sparsity, M, threads (regular matrices).
- `matrix_speedup.png`: includes the narrow-output diagnostic; this is excluded from the main aggregate.
- `thread_scaling.png`: each version versus its own 1T time.
- `pattern_speedup.png`: iid versus clustered output zeros.
- `narrow_runtime.png`: absolute times in the few-group diagnostic; excessive threads can hurt both versions.
- `roofline.png`: M=512, shared k; colors match measured thread-count SIMD/DRAM roofs.
- `tXX/report.md`, `tXX/roofline.png`: standard harness reports for M=512 iid cases, independently best k.
- `comparison.csv`: all fixed-k results; `tuned_comparison.csv`: independently tuned comparisons; `raw.csv`: full k sweep.

Roofline intensity uses compulsory format + X + Y traffic and the kernels' existing addition-count convention; it is not measured cache traffic.
Near-unity differences may reflect measurement noise. This run uses one deterministic weight sample per workload; the clustered case is a synthetic scheduling diagnostic.

Tile-count and load-balance analysis: [ownership_summary.md](ownership_summary.md).
