# Current V5 inference: roofline position

Fresh measurements on the local Apple M4 Pro (10 performance + 4 efficiency cores).
Each kernel uses its independently fastest measured block width for each workload.
Ratios and utilization are geometric means across 16 shape/sparsity workloads.

| Threads | Inference speedup vs serial V4 | Measured SIMD add peak [GFLOP/s] | Measured DRAM [GB/s] | % SIMD peak | % modeled roof | V5 GFLOP/s range |
|---|---|---|---|---|---|---|
| 1 | 1.00× | 71.3 | 95.0 | 35.2% | 35.2% | 16.4–28.6 |
| 2 | 1.79× | 135.9 | 164.4 | 33.9% | 33.9% | 40.9–52.9 |
| 4 | 3.28× | 249.2 | 237.5 | 34.7% | 34.7% | 73.5–102.9 |
| 6 | 4.40× | 373.0 | 244.2 | 28.6% | 28.6% | 65.0–135.3 |
| 8 | 5.48× | 489.6 | 218.3 | 26.7% | 26.7% | 87.9–190.3 |
| 10 | 5.83× | 607.7 | 227.9 | 24.2% | 24.2% | 89.6–220.5 |
| 12 | 4.70× | 569.9 | 246.9 | 21.0% | 21.0% | 76.8–166.7 |
| 14 | 4.79× | 615.5 | 245.6 | 19.3% | 19.3% | 71.8–184.8 |

The roof is min(measured SIMD addition peak, intensity × measured DRAM read bandwidth).
Intensity counts compulsory format + X + Y traffic, not hardware-counter cache traffic.
Being below this roof alone does not establish which resource is the bottleneck.
OpenMP scheduling, synchronization, allocations and intermediate-buffer traffic are included in inference time.
Core placement is controlled by macOS; requested threads do not imply pinned performance cores.

See `roofline_positions.csv` for individual workloads and `report.md` for fixed-k thread scaling.
