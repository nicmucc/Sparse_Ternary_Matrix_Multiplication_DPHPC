The results answer two different questions:

1. **Does v2 make the GPU computation faster?** Yes, in every tested configuration.
2. **Does v2 make a complete inference call faster?** Usually, but its input conversion and setup costs reduce the benefit.

I’ll explain this using actual measurements, then connect them to the plots.

**1. What one “workload” means**

Consider:

```text
M = 512
K = N = 4096
k = 4
50% zero weights
```

This computes:

```text
X: 512 × 4096
W: 4096 × 4096
Y: 512 × 4096
```

`k=4` means SPR groups four weight columns together during preprocessing.

V1 and v2 receive exactly the same `X` and `W`. What changes is the input/scratch layout and how threads are assigned work.

**2. What the three measured times mean**

For that workload:

| Measurement | V1 | V2 | Speedup |
|---|---:|---:|---:|
| GPU computation | 50.68 ms | 7.58 ms | **6.69×** |
| Complete synchronized `run()` | 50.70 ms | 7.58 ms | **6.69×** |
| `bind()` + `run()` + `finish()` | 57.55 ms | 18.58 ms | **3.10×** |

The GPU computation became much faster:

```text
50.68 / 7.58 ≈ 6.69×
```

But the complete call improved by less:

```text
57.55 / 18.58 ≈ 3.10×
```

Why? Because the complete call also includes:

```text
Allocate GPU buffers
Transpose X on the CPU in v2
Upload X
Run the GPU computation
Download Y
```

Those operations do not disappear when the kernel gets faster.

For this workload, the estimated time outside `run()` is:

```text
V1: 57.55 − 50.70 =  6.85 ms
V2: 18.58 −  7.58 = 11.00 ms
```

V2 saves approximately **43 ms in computation**, while adding approximately **4 ms outside computation**. It still wins comfortably.

These differences estimate combined setup and transfer costs; we did not measure transpose alone.

**3. How to read the speedup heatmap**

Open [the overview](/Users/alexray/Documents/ETH/DPHPC/Project_Two/SPR/results/gpu_compare_20261118T140411Z/plots/01_speedup_overview.png).

Each small panel fixes:

```text
Weight matrix size
Percentage of zero weights
Timing being compared
```

Inside a panel:

- Horizontal axis: input row count `M`.
- Vertical axis: SPR block width `k`.
- Cell number: `v1 time / v2 time`.

Interpret a cell as follows:

```text
6.69 → v2 takes about one-sixth of v1's time
1.00 → equal time
0.80 → v2 takes longer
```

For the example above, find:

```text
4096 × 4096 weights
50% zeros
GPU kernel
row k=4, column M=512
```

You will see **6.69**.

In the corresponding end-to-end panel, that cell shows **3.10**.

The blue cells indicate v2 wins. The reddish cells indicate v2 loses.

**4. How can v2 have a faster kernel but lose overall?**

Here is one actual example:

```text
M=512, K=N=1024, k=4, 90% zeros
```

| Measurement | V1 | V2 |
|---|---:|---:|
| GPU computation | 1.10 ms | 0.18 ms |
| Complete inference call | 2.59 ms | 3.22 ms |

V2 saves approximately **0.92 ms on the GPU**, but its combined setup and transfer costs increase enough to outweigh that saving.

The end-to-end speedup is:

```text
2.59 / 3.22 ≈ 0.80×
```

Equivalently, **v2 takes about 24% longer overall** for this case.

That is what the red cells and the [setup-cost plot](/Users/alexray/Documents/ETH/DPHPC/Project_Two/SPR/results/gpu_compare_20261118T140411Z/plots/04_setup_and_transfer_cost.png) illustrate.

In the setup-cost plot:

- Blue is synchronized `run()` time.
- Yellow estimates time outside `run()`.
- The total bar height is end-to-end time.

A small blue section means further kernel optimization alone may have limited impact on the complete call.

**5. How to read the latency curves**

The GPU and end-to-end latency plots show **actual milliseconds**, rather than ratios.

Each panel fixes the matrix shape and `k`, using 50% zero weights:

```text
Horizontal axis → M
Vertical axis   → time in milliseconds
Blue line       → v1
Orange line     → v2
```

**Lower is better.**

The vertical axis is logarithmic: the distance from `0.1` to `1` ms represents the same factor as the distance from `1` to `10` ms.

The `M` values are displayed at equal horizontal spacing. Therefore, the distance from `31` to `32` represents a much smaller numerical change than the distance from `128` to `512`.

The shaded bands contain the middle 50% of the 30 measured times. Narrow bands mean those samples were consistent. They do not tell us how much results would vary across different jobs or GPUs.

**6. Why the largest speedup is not necessarily the best configuration**

For the same large workload, changing `k` gives:

| `k` | V1 GPU time | V2 GPU time | Speedup |
|---|---:|---:|---:|
| 4 | 50.68 ms | **7.58 ms** | 6.69× |
| 8 | 1112.69 ms | 28.86 ms | **38.56×** |

At `k=8`, v2 achieves a much larger speedup because **v1 performs particularly poorly there**.

However, if you want the fastest v2 execution, choose `k=4` for this workload:

```text
7.58 ms is faster than 28.86 ms
```

That is why I included the [best-tested-`k` plot](/Users/alexray/Documents/ETH/DPHPC/Project_Two/SPR/results/gpu_compare_20261118T140411Z/plots/05_best_k_comparison.png). It selects the fastest tested `k` independently for each version and workload.

Use:

- The heatmap to understand how the optimization behaves at a fixed `k`.
- The best-`k` plot to compare the fastest configurations you found.

**7. What the overall numbers tell you**

Across the 126 workloads:

```text
GPU time:   v2 wins 126/126, median speedup 3.93×
End-to-end: v2 wins 124/126, median speedup 2.45×
```

“Median speedup” means we calculate one speedup per workload, sort those ratios, and take the middle. It is not total benchmark time divided by total benchmark time.

The measurements support that **v2 is an effective optimization on this GPU and these inputs**. They do not yet establish how much of the improvement comes specifically from coalescing: the thread mapping also changes launch geometry and warp utilization. The profiler reports are needed to investigate that explanation.