# SPR benchmark summary

- machine `Tlilis-MacBook-Pro.local`, compiler `clang 15.0.0 (clang-1500.0.40.1)`, threads **1**, tag `quick`
- shapes (M, K, N): (64, 1024, 4096), (64, 2048, 8192); sparsities: 30%, 50%, 70%, 90%
- code: `e70a484` (+ uncommitted changes)
- each method = its fastest (kernel, k) per shape and sparsity. *scalar* methods are at the same engineering level as `spr.naive`; *SIMD* ones use hand-written AVX2 (NEON through the shim on Apple Silicon).

## 1. `spr.naive`: speed-up vs each method

baseline time / SPR time: **> 1 means SPR is faster**. Geometric mean over shapes.

| method | level | 30% | 50% | 70% | 90% | all | wins |
|---|---|---|---|---|---|---|---|
| RSR | scalar | 2.42 | 3.05 | 4.08 | 9.03 | **4.06** | 8/8 |
| RSR++ | scalar | 2.21 | 2.80 | 3.71 | 7.66 | **3.64** | 8/8 |
| TCSC-scalar | scalar | 4.76 | 3.59 | 2.41 | 1.21 | **2.65** | 7/8 |
| SSR | SIMD | 0.48 | 0.48 | 0.46 | 0.44 | **0.46** | 0/8 |
| TCSC | SIMD | 0.27 | 0.22 | 0.17 | 0.14 | **0.19** | 0/8 |
| BLAS-dense | SIMD | 0.07 | 0.07 | 0.09 | 0.18 | **0.09** | 0/8 |

## 2. Best time per method: median ms (best k)


**M=64, K=1024, N=4096**

| s | spr.naive | RSR | RSR++ | TCSC-scalar | SSR | TCSC | BLAS-dense |
|---|---|---|---|---|---|---|---|
| 30% | 39.78 (k=2) | 89.47 (k=5) | 82.85 (k=6) | 160.60 | 17.29 (k=6) | 9.27 | 2.56 |
| 50% | 34.30 (k=2) | 103.65 (k=5) | 96.56 (k=6) | 109.19 | 15.83 (k=5) | 6.86 | 2.57 |
| 70% | 25.43 (k=2) | 110.06 (k=5) | 100.70 (k=6) | 56.62 | 12.01 (k=5) | 4.43 | 2.51 |
| 90% | 14.33 (k=2) | 124.15 (k=6) | 106.32 (k=6) | 14.20 | 5.57 (k=5) | 1.67 | 2.52 |

**M=64, K=2048, N=8192**

| s | spr.naive | RSR | RSR++ | TCSC-scalar | SSR | TCSC | BLAS-dense |
|---|---|---|---|---|---|---|---|
| 30% | 120.34 (k=3) | 314.29 (k=6) | 281.24 (k=6) | 675.63 | 65.02 (k=6) | 36.73 | 8.31 |
| 50% | 116.90 (k=3) | 359.81 (k=6) | 324.41 (k=6) | 472.43 | 57.88 (k=6) | 27.44 | 8.31 |
| 70% | 103.14 (k=3) | 395.93 (k=6) | 359.24 (k=6) | 268.67 | 45.81 (k=5) | 17.74 | 7.98 |
| 90% | 43.75 (k=2) | 411.83 (k=6) | 346.07 (k=8) | 64.17 | 21.57 (k=5) | 7.20 | 8.33 |

## 3. `spr.naive` diagnostics: k sweep

- **adds/row**: additions `run()` performs per row of X (`adds_per_row()`); **work vs TCSC** = adds/row ÷ nnz (TCSC adds each non-zero once)
- **ns/add**: time per addition. If it rises with k while adds/row falls, the time is going into memory / loop overhead, not arithmetic
- **GOP/s** = dense-equivalent 2·M·K·N / time

**M=64, K=1024, N=4096**

| s | k | ms | adds/row (M) | work vs TCSC | ns/add | GOP/s | bits/weight | prep ms |
|---|---|---|---|---|---|---|---|---|
| 30% | 2 ★ | 39.78 | 1.93 | 0.66 | 0.321 | 13.5 | 15.1 | 91 |
| 30% | 3 | 41.82 | 1.42 | 0.48 | 0.459 | 12.8 | 11.5 | 69 |
| 30% | 4 | 73.74 | 1.20 | 0.41 | 0.964 | 7.3 | 10.4 | 55 |
| 30% | 5 | 136.05 | 1.22 | 0.42 | 1.742 | 3.9 | 12.3 | 46 |
| 30% | 6 | 190.04 | 1.48 | 0.50 | 2.007 | 2.8 | 17.0 | 40 |
| 30% | 8 | 121.03 | 1.94 | 0.66 | 0.974 | 4.4 | 22.3 | 38 |
| 50% | 2 ★ | 34.30 | 1.60 | 0.76 | 0.335 | 15.7 | 12.5 | 77 |
| 50% | 3 | 41.60 | 1.29 | 0.61 | 0.505 | 12.9 | 10.4 | 64 |
| 50% | 4 | 73.26 | 1.14 | 0.54 | 1.006 | 7.3 | 10.0 | 52 |
| 50% | 5 | 131.50 | 1.16 | 0.55 | 1.765 | 4.1 | 11.6 | 44 |
| 50% | 6 | 160.93 | 1.32 | 0.63 | 1.902 | 3.3 | 14.8 | 42 |
| 50% | 8 | 115.06 | 1.64 | 0.78 | 1.099 | 4.7 | 19.3 | 37 |
| 70% | 2 ★ | 25.43 | 1.09 | 0.87 | 0.363 | 21.1 | 8.7 | 56 |
| 70% | 3 | 36.41 | 0.98 | 0.78 | 0.580 | 14.7 | 8.1 | 49 |
| 70% | 4 | 55.67 | 0.93 | 0.74 | 0.938 | 9.6 | 8.2 | 45 |
| 70% | 5 | 80.30 | 0.93 | 0.74 | 1.354 | 6.7 | 8.9 | 40 |
| 70% | 6 | 98.85 | 0.96 | 0.76 | 1.607 | 5.4 | 9.9 | 37 |
| 70% | 8 | 100.38 | 1.07 | 0.85 | 1.459 | 5.3 | 12.2 | 33 |
| 90% | 2 ★ | 14.33 | 0.42 | 1.01 | 0.531 | 37.5 | 3.5 | 24 |
| 90% | 3 | 19.62 | 0.42 | 1.00 | 0.731 | 27.4 | 3.6 | 23 |
| 90% | 4 | 24.84 | 0.42 | 1.00 | 0.924 | 21.6 | 3.8 | 24 |
| 90% | 5 | 28.85 | 0.42 | 1.01 | 1.068 | 18.6 | 3.9 | 24 |
| 90% | 6 | 33.20 | 0.43 | 1.02 | 1.217 | 16.2 | 4.1 | 23 |
| 90% | 8 | 38.14 | 0.44 | 1.04 | 1.363 | 14.1 | 4.4 | 22 |

**M=64, K=2048, N=8192**

| s | k | ms | adds/row (M) | work vs TCSC | ns/add | GOP/s | bits/weight | prep ms |
|---|---|---|---|---|---|---|---|---|
| 30% | 2 | 171.33 | 7.68 | 0.65 | 0.348 | 12.5 | 14.8 | 398 |
| 30% | 3 ★ | 120.34 | 5.57 | 0.47 | 0.338 | 17.8 | 10.9 | 311 |
| 30% | 4 | 176.21 | 4.47 | 0.38 | 0.616 | 12.2 | 9.2 | 247 |
| 30% | 5 | 340.04 | 4.12 | 0.35 | 1.288 | 6.3 | 9.4 | 203 |
| 30% | 6 | 568.49 | 4.66 | 0.40 | 1.906 | 3.8 | 12.4 | 172 |
| 30% | 8 | 505.98 | 6.76 | 0.58 | 1.169 | 4.2 | 19.8 | 155 |
| 50% | 2 | 141.76 | 6.34 | 0.76 | 0.349 | 15.1 | 12.3 | 337 |
| 50% | 3 ★ | 116.90 | 5.02 | 0.60 | 0.364 | 18.4 | 9.9 | 275 |
| 50% | 4 | 172.26 | 4.24 | 0.51 | 0.634 | 12.5 | 8.7 | 241 |
| 50% | 5 | 325.96 | 4.01 | 0.48 | 1.270 | 6.6 | 9.1 | 200 |
| 50% | 6 | 527.49 | 4.35 | 0.52 | 1.893 | 4.1 | 11.4 | 175 |
| 50% | 8 | 501.30 | 5.71 | 0.68 | 1.371 | 4.3 | 16.8 | 155 |
| 70% | 2 | 109.10 | 4.33 | 0.86 | 0.394 | 19.7 | 8.4 | 255 |
| 70% | 3 ★ | 103.14 | 3.80 | 0.75 | 0.424 | 20.8 | 7.5 | 220 |
| 70% | 4 | 145.80 | 3.48 | 0.69 | 0.656 | 14.7 | 7.2 | 194 |
| 70% | 5 | 216.97 | 3.35 | 0.67 | 1.013 | 9.9 | 7.5 | 174 |
| 70% | 6 | 290.80 | 3.39 | 0.67 | 1.341 | 7.4 | 8.2 | 160 |
| 70% | 8 | 376.32 | 3.76 | 0.75 | 1.565 | 5.7 | 10.3 | 143 |
| 90% | 2 ★ | 43.75 | 1.64 | 0.98 | 0.416 | 49.1 | 3.3 | 115 |
| 90% | 3 | 58.55 | 1.61 | 0.96 | 0.569 | 36.7 | 3.3 | 111 |
| 90% | 4 | 71.20 | 1.58 | 0.94 | 0.702 | 30.2 | 3.3 | 111 |
| 90% | 5 | 86.18 | 1.57 | 0.94 | 0.857 | 24.9 | 3.4 | 103 |
| 90% | 6 | 102.91 | 1.57 | 0.94 | 1.024 | 20.9 | 3.5 | 99 |
| 90% | 8 | 126.16 | 1.59 | 0.95 | 1.242 | 17.0 | 3.8 | 96 |

## 4. Algorithm vs engineering, against TCSC

TCSC does exactly one addition per non-zero, so its time per addition is a clean engineering yardstick.
- **work** = SPR adds / TCSC adds (< 1: the algorithm saves additions)
- **cost/add** = SPR ns per add / TCSC ns per add (> 1: each SPR addition is more expensive: the engineering gap)
- **net** = work × cost/add = SPR time / TCSC time (< 1: SPR wins)

`spr.naive`

| shape | s | best k | TCSC-scalar work | TCSC-scalar cost/add | TCSC-scalar net | TCSC work | TCSC cost/add | TCSC net |
|---|---|---|---|---|---|---|---|---|
| 1024×4096 | 30% | 2 | 0.66 | 0.38 | 0.25 | 0.66 | 6.51 | 4.29 |
| 1024×4096 | 50% | 2 | 0.76 | 0.41 | 0.31 | 0.76 | 6.56 | 5.00 |
| 1024×4096 | 70% | 2 | 0.87 | 0.52 | 0.45 | 0.87 | 6.61 | 5.74 |
| 1024×4096 | 90% | 2 | 1.01 | 1.00 | 1.01 | 1.01 | 8.53 | 8.58 |
| 2048×8192 | 30% | 3 | 0.47 | 0.38 | 0.18 | 0.47 | 6.91 | 3.28 |
| 2048×8192 | 50% | 3 | 0.60 | 0.41 | 0.25 | 0.60 | 7.12 | 4.26 |
| 2048×8192 | 70% | 3 | 0.75 | 0.51 | 0.38 | 0.75 | 7.70 | 5.81 |
| 2048×8192 | 90% | 2 | 0.98 | 0.70 | 0.68 | 0.98 | 6.21 | 6.08 |

## 5. Memory and preprocessing (K=2048, N=8192, best config)

bits/weight of the compressed W + metadata (2.0 = plain 2-bit packing, 8 = int8) / preprocessing ms

| method | 30% | 50% | 70% | 90% |
|---|---|---|---|---|
| spr.naive | 10.9 / 311 | 9.9 / 275 | 7.5 / 220 | 3.3 / 115 |
| RSR | 11.0 / 731 | 11.0 / 669 | 11.0 / 571 | 11.0 / 413 |
| RSR++ | 11.0 / 720 | 11.0 / 667 | 11.0 / 578 | 9.0 / 405 |
| TCSC-scalar | 11.2 / 131 | 8.0 / 108 | 4.8 / 92 | 1.6 / 58 |
| SSR | 11.0 / 720 | 11.0 / 668 | 13.0 / 589 | 13.0 / 434 |
| TCSC | 11.2 / 145 | 8.0 / 123 | 4.8 / 90 | 1.6 / 63 |
| BLAS-dense | 32.0 / 6 | 32.0 / 12 | 32.0 / 2 | 32.0 / 2 |

## 6. Insights

- `spr.naive` vs the fastest scalar baseline (same engineering level): geomean **2.06×**, faster in 7/8 cases.
  - worst case: K×N=1024×4096, s=90%: TCSC-scalar is 1.01× faster than SPR.
- `spr.naive` vs the fastest baseline overall: geomean **0.09×**, faster in 0/8 cases.
  - worst case: K×N=1024×4096, s=30%: BLAS-dense is 15.53× faster than SPR.
- `spr.naive`: in 8 cases the k with the fewest additions is NOT the fastest, so time doesn't follow work (memory traffic / short unpredictable loops dominate). Fixing this unlocks the larger-k savings the theory predicts:
  - 1024×4096@30%: fewest adds at k=4 (0.96 ns/add), fastest at k=2 (0.32 ns/add)
  - 1024×4096@50%: fewest adds at k=4 (1.01 ns/add), fastest at k=2 (0.34 ns/add)
  - 1024×4096@70%: fewest adds at k=5 (1.35 ns/add), fastest at k=2 (0.36 ns/add)
  - 1024×4096@90%: fewest adds at k=3 (0.73 ns/add), fastest at k=2 (0.53 ns/add)
- `spr.naive`: best k is at the edge of the sweep [2, 3, 4, 5, 6, 8] in: 1024×4096@30%→k=2, 1024×4096@50%→k=2, 1024×4096@70%→k=2, 1024×4096@90%→k=2, 2048×8192@90%→k=2. Widen the k range there.
