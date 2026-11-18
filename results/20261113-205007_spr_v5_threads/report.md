# SPR V5 thread sweep — Apple M4 Pro

CPU: 10 performance cores + 4 efficiency cores; macOS chooses core placement (no pinning).
V5 parallelizes the first preparation stage only. Its inference remains serial.

Scaling kernel: `spr.v5.b32`; serial controls: `spr.v4.b32` and `spr.v2.b32`.
Grid: M=64; K×N=[(1024, 4096), (2048, 2048), (4096, 1024), (2048, 8192)]; sparsities=0.3,0.5,0.7,0.9; k=2,3,4,5,6,8.
Preparation: median of 5 fresh-kernel measurements after one untimed preparation.
Inference: median of 11 interleaved measurements after 3 warmups.
OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0, OMP_DYNAMIC=FALSE; identical settings for every thread count.
All benchmark configurations were checked against the reference on all M rows.
Thread scaling fixes k to V5's fastest one-thread inference configuration for each workload.
Version comparisons and per-thread standard reports use independently best k.

| Threads | Preparation speedup vs V5/1T | Inference speedup vs V5/1T | Preparation speedup vs V4 at same threads |
|---|---|---|---|
| 1 | 1.00× | 1.00× | 0.96× |
| 2 | 1.45× | 1.00× | 1.39× |
| 4 | 1.86× | 1.00× | 1.82× |
| 6 | 2.10× | 1.00× | 2.04× |
| 8 | 2.24× | 0.99× | 2.19× |
| 10 | 2.32× | 0.99× | 2.29× |
| 12 | 2.32× | 0.99× | 2.26× |
| 14 | 2.27× | 0.98× | 2.26× |

Ratios above are geometric means over the 16 shape/sparsity workloads.

## Plots

- `speedup.png`: preparation scaling.
- `preparation.png`: absolute preparation time.
- `inference_speedup.png`: inference scaling (serial control).
- `runtime.png`: absolute inference time.
- `version_speedup.png`: inference progression from naive through V5 at one thread.
- `roofline.png`: V5 across requested thread counts, using measured one-thread roofs because inference is serial.
- `tXX/{report.md,roofline.png,runtime.png}`: existing infrastructure reports, with measured XX-thread machine ceilings.

## Interpretation limits

Roofline intensity uses compulsory format + input + output traffic, not measured cache traffic.
V4/V5's additions use their existing adds_per_row() accounting convention.
Requested threads describe OpenMP settings, not a guarantee that inference uses those threads.
Results include normal scheduling, frequency and thermal variation; preparation excludes cold OpenMP startup.
Kernel/benchmark source snapshots, commands, CPU/cache metadata, correctness logs and machine ceilings are retained here.
