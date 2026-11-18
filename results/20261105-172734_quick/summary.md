# SPR benchmark summary

- machine `Tlilis-MacBook-Pro.local`, compiler `clang 15.0.0 (clang-1500.0.40.1)`, threads **1**, tag `quick`
- shapes (M, K, N): (64, 1024, 4096), (64, 2048, 8192); sparsities: 30%, 50%, 70%, 90%
- code: `e70a484` (+ uncommitted changes)
- each method = its fastest (kernel, k) per shape and sparsity. *scalar* = plain C++, the same engineering level as `spr.naive`; the others are each algorithm's most optimized published kernel. On Apple Silicon the AVX2 / AVX-512 kernels run through a NEON shim (AVX-512 timings are meaningless there); RSR-core uses native NEON.

## 1. `spr.naive`: speed-up vs each method

baseline time / SPR time: **> 1 means SPR is faster**. Geometric mean over shapes.

| method | level | 30% | 50% | 70% | 90% | all | wins |
|---|---|---|---|---|---|---|---|
| RSR · scalar | scalar | 2.51 | 3.01 | 4.21 | 9.08 | **4.12** | 8/8 |
| RSR · RSR-core | optimized | 1.13 | 1.18 | 1.47 | 2.65 | **1.51** | 8/8 |
| RSR++ · scalar | scalar | 2.29 | 2.76 | 3.81 | 7.66 | **3.68** | 8/8 |
| SSR · scalar | scalar | 1.18 | 1.19 | 1.18 | 1.40 | **1.24** | 8/8 |
| SSR · AVX2 | optimized | 0.50 | 0.49 | 0.48 | 0.46 | **0.48** | 0/8 |
| TCSC · scalar | scalar | 1.35 | 1.02 | 0.74 | 0.44 | **0.82** | 3/8 |
| TCSC · AVX2 | optimized | 0.28 | 0.22 | 0.18 | 0.14 | **0.20** | 0/8 |
| TCSC · AVX-512 | optimized | 0.28 | 0.23 | 0.18 | 0.14 | **0.20** | 0/8 |
| Dense · scalar | scalar | 0.67 | 0.72 | 0.89 | 1.81 | **0.94** | 2/8 |
| Dense · BLAS | optimized | 0.09 | 0.10 | 0.13 | 0.26 | **0.13** | 0/8 |

## 2. Best time per method: median ms (best k)


**M=64, K=1024, N=4096**

| s | spr.naive | RSR · scalar | RSR · RSR-core | RSR++ · scalar | SSR · scalar | SSR · AVX2 | TCSC · scalar | TCSC · AVX2 | TCSC · AVX-512 | Dense · scalar | Dense · BLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30% | 39.36 (k=2) | 89.61 (k=5) | 43.27 (k=2) | 85.10 (k=6) | 42.55 (k=4) | 17.70 (k=6) | 45.41 | 9.44 | 9.54 | 23.03 | 2.63 |
| 50% | 34.02 (k=2) | 99.15 (k=5) | 42.26 (k=2) | 93.67 (k=6) | 39.75 (k=3) | 15.94 (k=5) | 31.50 | 7.04 | 7.19 | 22.72 | 2.41 |
| 70% | 24.55 (k=2) | 111.56 (k=5) | 41.51 (k=2) | 100.84 (k=6) | 28.83 (k=3) | 12.38 (k=4) | 18.27 | 4.50 | 4.58 | 22.78 | 2.55 |
| 90% | 14.48 (k=2) | 121.29 (k=6) | 36.63 (k=3) | 102.30 (k=6) | 17.86 (k=4) | 5.93 (k=5) | 5.04 | 1.71 | 1.62 | 22.72 | 2.60 |

**M=64, K=2048, N=8192**

| s | spr.naive | RSR · scalar | RSR · RSR-core | RSR++ · scalar | SSR · scalar | SSR · AVX2 | TCSC · scalar | TCSC · AVX2 | TCSC · AVX-512 | Dense · scalar | Dense · BLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30% | 115.92 (k=3) | 319.60 (k=6) | 134.81 (k=3) | 280.70 (k=6) | 149.83 (k=5) | 64.40 (k=6) | 182.81 | 37.62 | 38.28 | 89.93 | 15.19 |
| 50% | 115.68 (k=3) | 359.60 (k=6) | 130.38 (k=3) | 320.72 (k=6) | 141.05 (k=4) | 58.42 (k=6) | 129.43 | 27.84 | 28.09 | 89.34 | 15.63 |
| 70% | 102.70 (k=3) | 401.37 (k=6) | 130.63 (k=3) | 362.28 (k=6) | 121.43 (k=4) | 46.26 (k=5) | 76.00 | 17.99 | 18.22 | 87.55 | 16.03 |
| 90% | 42.74 (k=2) | 420.35 (k=6) | 118.88 (k=4) | 354.83 (k=8) | 68.15 (k=4) | 21.71 (k=5) | 23.64 | 7.07 | 7.29 | 89.37 | 16.23 |

## 3. `spr.naive` diagnostics: k sweep

- **adds/row**: additions `run()` performs per row of X (`adds_per_row()`); **work vs TCSC** = adds/row ÷ nnz (TCSC adds each non-zero once)
- **ns/add**: time per addition. If it rises with k while adds/row falls, the time is going into memory / loop overhead, not arithmetic
- **GOP/s** = dense-equivalent 2·M·K·N / time

**M=64, K=1024, N=4096**

| s | k | ms | adds/row (M) | work vs TCSC | ns/add | GOP/s | bits/weight | prep ms |
|---|---|---|---|---|---|---|---|---|
| 30% | 2 ★ | 39.36 | 1.93 | 0.66 | 0.318 | 13.6 | 15.1 | 94 |
| 30% | 3 | 40.46 | 1.42 | 0.48 | 0.444 | 13.3 | 11.5 | 73 |
| 30% | 4 | 74.37 | 1.20 | 0.41 | 0.972 | 7.2 | 10.4 | 56 |
| 30% | 5 | 139.16 | 1.22 | 0.42 | 1.782 | 3.9 | 12.3 | 47 |
| 30% | 6 | 185.56 | 1.48 | 0.50 | 1.960 | 2.9 | 17.0 | 41 |
| 30% | 8 | 117.67 | 1.94 | 0.66 | 0.947 | 4.6 | 22.3 | 39 |
| 50% | 2 ★ | 34.02 | 1.60 | 0.76 | 0.333 | 15.8 | 12.5 | 80 |
| 50% | 3 | 39.87 | 1.29 | 0.61 | 0.484 | 13.5 | 10.4 | 65 |
| 50% | 4 | 74.08 | 1.14 | 0.54 | 1.017 | 7.2 | 10.0 | 54 |
| 50% | 5 | 134.17 | 1.16 | 0.55 | 1.801 | 4.0 | 11.6 | 46 |
| 50% | 6 | 164.71 | 1.32 | 0.63 | 1.947 | 3.3 | 14.8 | 41 |
| 50% | 8 | 113.34 | 1.64 | 0.78 | 1.083 | 4.7 | 19.3 | 37 |
| 70% | 2 ★ | 24.55 | 1.09 | 0.87 | 0.351 | 21.9 | 8.7 | 60 |
| 70% | 3 | 35.15 | 0.98 | 0.78 | 0.560 | 15.3 | 8.1 | 53 |
| 70% | 4 | 57.20 | 0.93 | 0.74 | 0.964 | 9.4 | 8.2 | 46 |
| 70% | 5 | 80.90 | 0.93 | 0.74 | 1.364 | 6.6 | 8.9 | 42 |
| 70% | 6 | 100.58 | 0.96 | 0.76 | 1.635 | 5.3 | 9.9 | 39 |
| 70% | 8 | 100.49 | 1.07 | 0.85 | 1.461 | 5.3 | 12.2 | 34 |
| 90% | 2 ★ | 14.48 | 0.42 | 1.01 | 0.536 | 37.1 | 3.5 | 25 |
| 90% | 3 | 19.37 | 0.42 | 1.00 | 0.722 | 27.7 | 3.6 | 27 |
| 90% | 4 | 24.47 | 0.42 | 1.00 | 0.911 | 21.9 | 3.8 | 25 |
| 90% | 5 | 28.89 | 0.42 | 1.01 | 1.069 | 18.6 | 3.9 | 24 |
| 90% | 6 | 33.03 | 0.43 | 1.02 | 1.211 | 16.3 | 4.1 | 24 |
| 90% | 8 | 38.65 | 0.44 | 1.04 | 1.381 | 13.9 | 4.4 | 22 |

**M=64, K=2048, N=8192**

| s | k | ms | adds/row (M) | work vs TCSC | ns/add | GOP/s | bits/weight | prep ms |
|---|---|---|---|---|---|---|---|---|
| 30% | 2 | 164.75 | 7.68 | 0.65 | 0.335 | 13.0 | 14.8 | 411 |
| 30% | 3 ★ | 115.92 | 5.57 | 0.47 | 0.325 | 18.5 | 10.9 | 313 |
| 30% | 4 | 176.20 | 4.47 | 0.38 | 0.616 | 12.2 | 9.2 | 252 |
| 30% | 5 | 349.89 | 4.12 | 0.35 | 1.325 | 6.1 | 9.4 | 206 |
| 30% | 6 | 568.90 | 4.66 | 0.40 | 1.907 | 3.8 | 12.4 | 175 |
| 30% | 8 | 499.81 | 6.76 | 0.58 | 1.154 | 4.3 | 19.8 | 154 |
| 50% | 2 | 140.22 | 6.34 | 0.76 | 0.346 | 15.3 | 12.3 | 348 |
| 50% | 3 ★ | 115.68 | 5.02 | 0.60 | 0.360 | 18.6 | 9.9 | 292 |
| 50% | 4 | 175.32 | 4.24 | 0.51 | 0.646 | 12.2 | 8.7 | 237 |
| 50% | 5 | 331.06 | 4.01 | 0.48 | 1.290 | 6.5 | 9.1 | 201 |
| 50% | 6 | 529.41 | 4.35 | 0.52 | 1.900 | 4.1 | 11.4 | 177 |
| 50% | 8 | 492.25 | 5.71 | 0.68 | 1.346 | 4.4 | 16.8 | 156 |
| 70% | 2 | 105.97 | 4.33 | 0.86 | 0.383 | 20.3 | 8.4 | 262 |
| 70% | 3 ★ | 102.70 | 3.80 | 0.75 | 0.422 | 20.9 | 7.5 | 227 |
| 70% | 4 | 149.45 | 3.48 | 0.69 | 0.672 | 14.4 | 7.2 | 201 |
| 70% | 5 | 222.32 | 3.35 | 0.67 | 1.038 | 9.7 | 7.5 | 182 |
| 70% | 6 | 298.32 | 3.39 | 0.67 | 1.375 | 7.2 | 8.2 | 170 |
| 70% | 8 | 380.02 | 3.76 | 0.75 | 1.580 | 5.7 | 10.3 | 145 |
| 90% | 2 ★ | 42.74 | 1.64 | 0.98 | 0.407 | 50.2 | 3.3 | 116 |
| 90% | 3 | 55.61 | 1.61 | 0.96 | 0.541 | 38.6 | 3.3 | 114 |
| 90% | 4 | 70.45 | 1.58 | 0.94 | 0.695 | 30.5 | 3.3 | 111 |
| 90% | 5 | 87.61 | 1.57 | 0.94 | 0.871 | 24.5 | 3.4 | 105 |
| 90% | 6 | 104.52 | 1.57 | 0.94 | 1.040 | 20.5 | 3.5 | 104 |
| 90% | 8 | 131.76 | 1.59 | 0.95 | 1.297 | 16.3 | 3.8 | 100 |

## 4. Algorithm vs engineering, against TCSC

TCSC does exactly one addition per non-zero, so its time per addition is a clean engineering yardstick.
- **work** = SPR adds / TCSC adds (< 1: the algorithm saves additions)
- **cost/add** = SPR ns per add / TCSC ns per add (> 1: each SPR addition is more expensive: the engineering gap)
- **net** = work × cost/add = SPR time / TCSC time (< 1: SPR wins)

`spr.naive`

| shape | s | best k | TCSC · scalar work | TCSC · scalar cost/add | TCSC · scalar net | TCSC · AVX2 work | TCSC · AVX2 cost/add | TCSC · AVX2 net |
|---|---|---|---|---|---|---|---|---|
| 1024×4096 | 30% | 2 | 0.66 | 1.32 | 0.87 | 0.66 | 6.33 | 4.17 |
| 1024×4096 | 50% | 2 | 0.76 | 1.42 | 1.08 | 0.76 | 6.34 | 4.83 |
| 1024×4096 | 70% | 2 | 0.87 | 1.55 | 1.34 | 0.87 | 6.28 | 5.46 |
| 1024×4096 | 90% | 2 | 1.01 | 2.85 | 2.87 | 1.01 | 8.40 | 8.45 |
| 2048×8192 | 30% | 3 | 0.47 | 1.34 | 0.63 | 0.47 | 6.50 | 3.08 |
| 2048×8192 | 50% | 3 | 0.60 | 1.49 | 0.89 | 0.60 | 6.95 | 4.16 |
| 2048×8192 | 70% | 3 | 0.75 | 1.79 | 1.35 | 0.75 | 7.56 | 5.71 |
| 2048×8192 | 90% | 2 | 0.98 | 1.85 | 1.81 | 0.98 | 6.18 | 6.05 |

## 5. Memory and preprocessing (K=2048, N=8192, best config)

bits/weight of the compressed W + metadata (2.0 = plain 2-bit packing, 8 = int8) / preprocessing ms

| method | 30% | 50% | 70% | 90% |
|---|---|---|---|---|
| spr.naive | 10.9 / 313 | 9.9 / 292 | 7.5 / 227 | 3.3 / 116 |
| RSR · scalar | 11.0 / 737 | 11.0 / 678 | 11.0 / 582 | 11.0 / 420 |
| RSR · RSR-core | 5.7 / 65 | 5.7 / 66 | 5.7 / 70 | 4.4 / 74 |
| RSR++ · scalar | 11.0 / 729 | 11.0 / 676 | 11.0 / 576 | 9.0 / 399 |
| SSR · scalar | 13.0 / 733 | 16.1 / 674 | 16.1 / 588 | 16.1 / 455 |
| SSR · AVX2 | 11.0 / 767 | 11.0 / 671 | 13.0 / 588 | 13.0 / 446 |
| TCSC · scalar | 11.2 / 131 | 8.0 / 107 | 4.8 / 80 | 1.6 / 61 |
| TCSC · AVX2 | 11.2 / 148 | 8.0 / 119 | 4.8 / 88 | 1.6 / 64 |
| TCSC · AVX-512 | 11.2 / 148 | 8.0 / 115 | 4.8 / 87 | 1.6 / 64 |
| Dense · scalar | 32.0 / 7 | 32.0 / 6 | 32.0 / 7 | 32.0 / 6 |
| Dense · BLAS | 32.0 / 7 | 32.0 / 7 | 32.0 / 7 | 32.0 / 7 |

## 6. Insights

- `spr.naive` vs the fastest scalar baseline (same engineering level): geomean **0.63×**, faster in 0/8 cases.
  - worst case: K×N=1024×4096, s=90%: TCSC · scalar is 2.87× faster than SPR.
- `spr.naive` vs the fastest baseline overall: geomean **0.11×**, faster in 0/8 cases.
  - worst case: K×N=1024×4096, s=30%: Dense · BLAS is 14.95× faster than SPR.
- `spr.naive`: in 8 cases the k with the fewest additions is NOT the fastest, so time doesn't follow work (memory traffic / short unpredictable loops dominate). Fixing this unlocks the larger-k savings the theory predicts:
  - 1024×4096@30%: fewest adds at k=4 (0.97 ns/add), fastest at k=2 (0.32 ns/add)
  - 1024×4096@50%: fewest adds at k=4 (1.02 ns/add), fastest at k=2 (0.33 ns/add)
  - 1024×4096@70%: fewest adds at k=5 (1.36 ns/add), fastest at k=2 (0.35 ns/add)
  - 1024×4096@90%: fewest adds at k=3 (0.72 ns/add), fastest at k=2 (0.54 ns/add)
- `spr.naive`: best k is at the edge of the sweep [2, 3, 4, 5, 6, 8] in: 1024×4096@30%→k=2, 1024×4096@50%→k=2, 1024×4096@70%→k=2, 1024×4096@90%→k=2, 2048×8192@90%→k=2. Widen the k range there.
