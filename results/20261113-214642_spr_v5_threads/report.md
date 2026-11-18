# SPR V5 thread sweep — Apple M4 Pro

CPU: 10 performance cores + 4 efficiency cores; macOS chooses core placement (no pinning).
V5 parallelizes preparation and inference across independent groups.

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
| 1 | 1.00× | 1.00× | 0.95× |
| 2 | 1.53× | 1.80× | 1.45× |
| 4 | 2.07× | 3.25× | 1.99× |
| 6 | 2.32× | 4.20× | 2.28× |
| 8 | 2.54× | 5.17× | 2.50× |
| 10 | 2.72× | 5.56× | 2.61× |
| 12 | 2.65× | 4.60× | 2.57× |
| 14 | 2.71× | 4.76× | 2.61× |

Ratios above are geometric means over the 16 shape/sparsity workloads.

## Plots

- `speedup.png`: preparation scaling.
- `preparation.png`: absolute preparation time.
- `inference_speedup.png`: parallel inference scaling.
- `runtime.png`: absolute inference time.
- `version_speedup.png`: inference progression from naive through V5 at one thread.
- `roofline.png`: parallel inference: matching thread-count simd/dram roofs.
- `tXX/{report.md,roofline.png,runtime.png}`: existing infrastructure reports, with measured XX-thread machine ceilings.

## Interpretation limits

Roofline intensity uses compulsory format + input + output traffic, not measured cache traffic.
V4/V5's additions use their existing adds_per_row() accounting convention.
Requested threads describe OpenMP settings, not a guarantee that inference uses those threads.
Results include normal scheduling, frequency and thermal variation; preparation excludes cold OpenMP startup.
Kernel/benchmark source snapshots, commands, CPU/cache metadata, correctness logs and machine ceilings are retained here.

Detailed roofline utilization and comparison with serial V4: [roofline_summary.md](roofline_summary.md).
