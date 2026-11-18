# SPR benchmark summary

- machine `Tlilis-MacBook-Pro.local`, compiler `clang 15.0.0 (clang-1500.0.40.1)`, threads **1**, tag `quick`
- shapes (M, K, N): (64, 1024, 4096), (64, 2048, 8192); sparsities: 30%, 50%, 70%, 90%
- code: `e70a484` (+ uncommitted changes)
- each method = its fastest (kernel, k) per shape and sparsity. *scalar* = plain C++, the same engineering level as `spr.naive`; the others are each algorithm's most optimized published kernel. On Apple Silicon the AVX2 / AVX-512 kernels run through a NEON shim (AVX-512 timings are meaningless there); RSR-core uses native NEON.

## 1. `spr.naive`: speed-up vs each method

baseline time / SPR time: **> 1 means SPR is faster**. Geometric mean over shapes.

| method | level | 30% | 50% | 70% | 90% | all | wins |
|---|---|---|---|---|---|---|---|
| RSR · scalar | scalar | 2.54 | 2.97 | 4.13 | 8.91 | **4.08** | 8/8 |
| RSR · RSR-core | optimized | 1.15 | 1.18 | 1.44 | 2.60 | **1.50** | 8/8 |
| RSR++ · scalar | scalar | 2.32 | 2.74 | 3.74 | 7.61 | **3.67** | 8/8 |
| SSR · scalar | scalar | 1.19 | 1.19 | 1.16 | 1.41 | **1.24** | 8/8 |
| SSR · AVX2 | optimized | 0.51 | 0.49 | 0.47 | 0.45 | **0.48** | 0/8 |
| TCSC · scalar | scalar | 1.36 | 1.02 | 0.73 | 0.43 | **0.81** | 3/8 |
| TCSC · AVX2 | optimized | 0.28 | 0.22 | 0.18 | 0.14 | **0.20** | 0/8 |
| TCSC · AVX-512 | optimized | 0.29 | 0.22 | 0.18 | 0.14 | **0.20** | 0/8 |
| Dense · scalar | scalar | 0.68 | 0.73 | 0.89 | 1.83 | **0.95** | 2/8 |
| Dense · BLAS | optimized | 0.10 | 0.10 | 0.13 | 0.26 | **0.13** | 0/8 |

## 2. Best time per method: median ms (best k)


**M=64, K=1024, N=4096**

| s | spr.naive | RSR · scalar | RSR · RSR-core | RSR++ · scalar | SSR · scalar | SSR · AVX2 | TCSC · scalar | TCSC · AVX2 | TCSC · AVX-512 | Dense · scalar | Dense · BLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30% | 38.91 (k=2) | 90.73 (k=5) | 43.34 (k=2) | 86.06 (k=6) | 42.43 (k=4) | 17.59 (k=6) | 45.64 | 9.49 | 9.65 | 23.74 | 2.69 |
| 50% | 34.76 (k=2) | 100.45 (k=5) | 43.21 (k=2) | 94.33 (k=6) | 40.90 (k=3) | 16.61 (k=5) | 32.23 | 7.17 | 7.30 | 23.95 | 2.73 |
| 70% | 25.54 (k=2) | 112.08 (k=5) | 42.29 (k=2) | 102.84 (k=6) | 29.60 (k=3) | 12.42 (k=4) | 18.53 | 4.59 | 4.74 | 23.00 | 2.63 |
| 90% | 15.13 (k=2) | 124.28 (k=6) | 36.96 (k=3) | 105.39 (k=6) | 18.34 (k=4) | 5.99 (k=5) | 5.13 | 1.72 | 1.68 | 23.58 | 2.70 |

**M=64, K=2048, N=8192**

| s | spr.naive | RSR · scalar | RSR · RSR-core | RSR++ · scalar | SSR · scalar | SSR · AVX2 | TCSC · scalar | TCSC · AVX2 | TCSC · AVX-512 | Dense · scalar | Dense · BLAS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30% | 114.63 (k=3) | 316.86 (k=6) | 135.18 (k=3) | 278.85 (k=6) | 149.23 (k=5) | 64.74 (k=6) | 180.55 | 37.39 | 37.79 | 87.30 | 15.42 |
| 50% | 116.45 (k=3) | 354.61 (k=6) | 129.97 (k=3) | 321.57 (k=6) | 141.14 (k=4) | 59.25 (k=6) | 129.55 | 27.98 | 28.04 | 90.57 | 16.15 |
| 70% | 105.88 (k=3) | 412.38 (k=6) | 132.04 (k=3) | 367.89 (k=6) | 122.98 (k=4) | 47.40 (k=5) | 77.34 | 18.61 | 18.50 | 94.13 | 16.22 |
| 90% | 43.54 (k=2) | 421.03 (k=6) | 120.24 (k=4) | 362.05 (k=8) | 71.49 (k=4) | 22.28 (k=5) | 23.99 | 7.31 | 7.66 | 93.60 | 16.30 |

## 3. `spr.naive` diagnostics: k sweep

- **adds/row**: additions `run()` performs per row of X (`adds_per_row()`); **work vs TCSC** = adds/row ÷ nnz (TCSC adds each non-zero once)
- **ns/add**: time per addition. If it rises with k while adds/row falls, the time is going into memory / loop overhead, not arithmetic
- **GOP/s** = dense-equivalent 2·M·K·N / time

**M=64, K=1024, N=4096**

| s | k | ms | adds/row (M) | work vs TCSC | ns/add | GOP/s | bits/weight | prep ms |
|---|---|---|---|---|---|---|---|---|
| 30% | 2 ★ | 38.91 | 1.93 | 0.66 | 0.314 | 13.8 | 15.1 | 96 |
| 30% | 3 | 40.88 | 1.42 | 0.48 | 0.449 | 13.1 | 11.5 | 73 |
| 30% | 4 | 74.88 | 1.20 | 0.41 | 0.978 | 7.2 | 10.4 | 57 |
| 30% | 5 | 140.55 | 1.22 | 0.42 | 1.800 | 3.8 | 12.3 | 47 |
| 30% | 6 | 189.76 | 1.48 | 0.50 | 2.004 | 2.8 | 17.0 | 42 |
| 30% | 8 | 119.98 | 1.94 | 0.66 | 0.965 | 4.5 | 22.3 | 39 |
| 50% | 2 ★ | 34.76 | 1.60 | 0.76 | 0.340 | 15.4 | 12.5 | 81 |
| 50% | 3 | 40.86 | 1.29 | 0.61 | 0.496 | 13.1 | 10.4 | 67 |
| 50% | 4 | 76.20 | 1.14 | 0.54 | 1.046 | 7.0 | 10.0 | 55 |
| 50% | 5 | 135.68 | 1.16 | 0.55 | 1.821 | 4.0 | 11.6 | 47 |
| 50% | 6 | 165.51 | 1.32 | 0.63 | 1.956 | 3.2 | 14.8 | 42 |
| 50% | 8 | 115.59 | 1.64 | 0.78 | 1.104 | 4.6 | 19.3 | 38 |
| 70% | 2 ★ | 25.54 | 1.09 | 0.87 | 0.365 | 21.0 | 8.7 | 60 |
| 70% | 3 | 35.82 | 0.98 | 0.78 | 0.571 | 15.0 | 8.1 | 54 |
| 70% | 4 | 57.97 | 0.93 | 0.74 | 0.977 | 9.3 | 8.2 | 47 |
| 70% | 5 | 82.17 | 0.93 | 0.74 | 1.386 | 6.5 | 8.9 | 42 |
| 70% | 6 | 102.40 | 0.96 | 0.76 | 1.665 | 5.2 | 9.9 | 39 |
| 70% | 8 | 102.56 | 1.07 | 0.85 | 1.491 | 5.2 | 12.2 | 34 |
| 90% | 2 ★ | 15.13 | 0.42 | 1.01 | 0.560 | 35.5 | 3.5 | 26 |
| 90% | 3 | 19.64 | 0.42 | 1.00 | 0.731 | 27.3 | 3.6 | 25 |
| 90% | 4 | 24.96 | 0.42 | 1.00 | 0.929 | 21.5 | 3.8 | 25 |
| 90% | 5 | 30.07 | 0.42 | 1.01 | 1.113 | 17.9 | 3.9 | 25 |
| 90% | 6 | 34.10 | 0.43 | 1.02 | 1.250 | 15.7 | 4.1 | 25 |
| 90% | 8 | 39.32 | 0.44 | 1.04 | 1.405 | 13.7 | 4.4 | 22 |

**M=64, K=2048, N=8192**

| s | k | ms | adds/row (M) | work vs TCSC | ns/add | GOP/s | bits/weight | prep ms |
|---|---|---|---|---|---|---|---|---|
| 30% | 2 | 165.21 | 7.68 | 0.65 | 0.336 | 13.0 | 14.8 | 410 |
| 30% | 3 ★ | 114.63 | 5.57 | 0.47 | 0.322 | 18.7 | 10.9 | 312 |
| 30% | 4 | 174.54 | 4.47 | 0.38 | 0.610 | 12.3 | 9.2 | 252 |
| 30% | 5 | 350.17 | 4.12 | 0.35 | 1.327 | 6.1 | 9.4 | 204 |
| 30% | 6 | 574.57 | 4.66 | 0.40 | 1.926 | 3.7 | 12.4 | 174 |
| 30% | 8 | 497.12 | 6.76 | 0.58 | 1.148 | 4.3 | 19.8 | 155 |
| 50% | 2 | 138.10 | 6.34 | 0.76 | 0.340 | 15.5 | 12.3 | 347 |
| 50% | 3 ★ | 116.45 | 5.02 | 0.60 | 0.363 | 18.4 | 9.9 | 288 |
| 50% | 4 | 173.85 | 4.24 | 0.51 | 0.640 | 12.4 | 8.7 | 235 |
| 50% | 5 | 331.96 | 4.01 | 0.48 | 1.294 | 6.5 | 9.1 | 199 |
| 50% | 6 | 530.45 | 4.35 | 0.52 | 1.903 | 4.0 | 11.4 | 178 |
| 50% | 8 | 495.24 | 5.71 | 0.68 | 1.354 | 4.3 | 16.8 | 156 |
| 70% | 2 | 106.43 | 4.33 | 0.86 | 0.384 | 20.2 | 8.4 | 265 |
| 70% | 3 ★ | 105.88 | 3.80 | 0.75 | 0.435 | 20.3 | 7.5 | 226 |
| 70% | 4 | 154.32 | 3.48 | 0.69 | 0.694 | 13.9 | 7.2 | 201 |
| 70% | 5 | 227.10 | 3.35 | 0.67 | 1.060 | 9.5 | 7.5 | 182 |
| 70% | 6 | 310.78 | 3.39 | 0.67 | 1.433 | 6.9 | 8.2 | 171 |
| 70% | 8 | 391.31 | 3.76 | 0.75 | 1.627 | 5.5 | 10.3 | 148 |
| 90% | 2 ★ | 43.54 | 1.64 | 0.98 | 0.414 | 49.3 | 3.3 | 121 |
| 90% | 3 | 54.29 | 1.61 | 0.96 | 0.528 | 39.6 | 3.3 | 116 |
| 90% | 4 | 71.85 | 1.58 | 0.94 | 0.709 | 29.9 | 3.3 | 113 |
| 90% | 5 | 88.87 | 1.57 | 0.94 | 0.883 | 24.2 | 3.4 | 110 |
| 90% | 6 | 106.78 | 1.57 | 0.94 | 1.062 | 20.1 | 3.5 | 107 |
| 90% | 8 | 130.54 | 1.59 | 0.95 | 1.285 | 16.5 | 3.8 | 102 |

## 4. Algorithm vs engineering, against TCSC

TCSC does exactly one addition per non-zero, so its time per addition is a clean engineering yardstick.
- **work** = SPR adds / TCSC adds (< 1: the algorithm saves additions)
- **cost/add** = SPR ns per add / TCSC ns per add (> 1: each SPR addition is more expensive: the engineering gap)
- **net** = work × cost/add = SPR time / TCSC time (< 1: SPR wins)

`spr.naive`

| shape | s | best k | TCSC · scalar work | TCSC · scalar cost/add | TCSC · scalar net | TCSC · AVX2 work | TCSC · AVX2 cost/add | TCSC · AVX2 net |
|---|---|---|---|---|---|---|---|---|
| 1024×4096 | 30% | 2 | 0.66 | 1.29 | 0.85 | 0.66 | 6.23 | 4.10 |
| 1024×4096 | 50% | 2 | 0.76 | 1.42 | 1.08 | 0.76 | 6.36 | 4.84 |
| 1024×4096 | 70% | 2 | 0.87 | 1.59 | 1.38 | 0.87 | 6.41 | 5.57 |
| 1024×4096 | 90% | 2 | 1.01 | 2.93 | 2.95 | 1.01 | 8.77 | 8.82 |
| 2048×8192 | 30% | 3 | 0.47 | 1.34 | 0.63 | 0.47 | 6.47 | 3.07 |
| 2048×8192 | 50% | 3 | 0.60 | 1.50 | 0.90 | 0.60 | 6.96 | 4.16 |
| 2048×8192 | 70% | 3 | 0.75 | 1.81 | 1.37 | 0.75 | 7.54 | 5.69 |
| 2048×8192 | 90% | 2 | 0.98 | 1.85 | 1.81 | 0.98 | 6.08 | 5.95 |

## 5. Memory and preprocessing (K=2048, N=8192, best config)

bits/weight of the compressed W + metadata (2.0 = plain 2-bit packing, 8 = int8) / preprocessing ms

| method | 30% | 50% | 70% | 90% |
|---|---|---|---|---|
| spr.naive | 10.9 / 312 | 9.9 / 288 | 7.5 / 226 | 3.3 / 121 |
| RSR · scalar | 11.0 / 721 | 11.0 / 668 | 11.0 / 579 | 11.0 / 429 |
| RSR · RSR-core | 5.7 / 64 | 5.7 / 65 | 5.7 / 71 | 4.4 / 77 |
| RSR++ · scalar | 11.0 / 729 | 11.0 / 673 | 11.0 / 584 | 9.0 / 413 |
| SSR · scalar | 13.0 / 727 | 16.1 / 665 | 16.1 / 592 | 16.1 / 452 |
| SSR · AVX2 | 11.0 / 716 | 11.0 / 665 | 13.0 / 587 | 13.0 / 446 |
| TCSC · scalar | 11.2 / 133 | 8.0 / 106 | 4.8 / 81 | 1.6 / 62 |
| TCSC · AVX2 | 11.2 / 147 | 8.0 / 117 | 4.8 / 91 | 1.6 / 64 |
| TCSC · AVX-512 | 11.2 / 152 | 8.0 / 115 | 4.8 / 91 | 1.6 / 64 |
| Dense · scalar | 32.0 / 6 | 32.0 / 7 | 32.0 / 6 | 32.0 / 6 |
| Dense · BLAS | 32.0 / 6 | 32.0 / 7 | 32.0 / 6 | 32.0 / 7 |

## 6. Insights

- `spr.naive` vs the fastest scalar baseline (same engineering level): geomean **0.63×**, faster in 0/8 cases.
  - worst case: K×N=1024×4096, s=90%: TCSC · scalar is 2.95× faster than SPR.
- `spr.naive` vs the fastest baseline overall: geomean **0.11×**, faster in 0/8 cases.
  - worst case: K×N=1024×4096, s=30%: Dense · BLAS is 14.44× faster than SPR.
- `spr.naive`: in 8 cases the k with the fewest additions is NOT the fastest, so time doesn't follow work (memory traffic / short unpredictable loops dominate). Fixing this unlocks the larger-k savings the theory predicts:
  - 1024×4096@30%: fewest adds at k=4 (0.98 ns/add), fastest at k=2 (0.31 ns/add)
  - 1024×4096@50%: fewest adds at k=4 (1.05 ns/add), fastest at k=2 (0.34 ns/add)
  - 1024×4096@70%: fewest adds at k=5 (1.39 ns/add), fastest at k=2 (0.36 ns/add)
  - 1024×4096@90%: fewest adds at k=3 (0.73 ns/add), fastest at k=2 (0.56 ns/add)
- `spr.naive`: best k is at the edge of the sweep [2, 3, 4, 5, 6, 8] in: 1024×4096@30%→k=2, 1024×4096@50%→k=2, 1024×4096@70%→k=2, 1024×4096@90%→k=2, 2048×8192@90%→k=2. Widen the k range there.
