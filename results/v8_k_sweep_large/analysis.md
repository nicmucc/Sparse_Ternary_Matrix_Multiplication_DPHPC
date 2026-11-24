# v8 block-width sweep and k=16 profile

Device: RTX 5060 Ti, `M=K=4096`, `N=14336`, 50% random zeros, seed 42.
The benchmark ran nine interleaved rounds for k=4–8 and seven for k=9–16;
all configurations passed the sampled correctness check. Preparation and
transfers are excluded from synchronized `run()` time.

| k | Median run (ms) | Metadata (bytes) | Additions per input row |
|---:|---:|---:|---:|
| 4 | 286.402 | 29,330,400 | 14,536,200 |
| 5 | 234.890 | 26,978,464 | 13,692,200 |
| 6 | 202.834 | 28,989,392 | 15,911,100 |
| 7 | 177.806 | 34,955,870 | 21,269,200 |
| 8 | 197.953 | 41,095,888 | 27,196,900 |
| 9 | 184.758 | 44,041,080 | 31,285,800 |
| 10 | 205.515 | 43,664,452 | 33,254,300 |
| 11 | 182.748 | 41,473,352 | 33,901,600 |
| 12 | 177.885 | 38,720,022 | 33,941,100 |
| 13 | 168.840 | 36,003,698 | 33,756,900 |
| 14 | 157.076 | 33,521,278 | 33,507,300 |
| 15 | 157.299 | 31,328,162 | 33,256,100 |
| **16** | **153.656** | **29,374,618** | **33,021,100** |

The fastest tested k is 16, 1.86× faster than k=4 on this shape. Results are
not monotonic: the balance of input reads, observed pattern count, metadata,
register output updates, and instruction issue changes with k.

Full Nsight Compute reports (`--set full --cache-control none`) compare k=4
and k=16 on the same shape. They are one profiled launch each; the benchmark
table is stronger evidence for runtime.

| NCU metric | k=4 | k=16 |
|---|---:|---:|
| Kernel duration | 283.05 ms | 157.65 ms |
| DRAM read | 118.72 GB | 31.91 GB |
| DRAM write | 0.243 GB | 0.242 GB |
| DRAM peak utilization | 95.25% | 46.23% |
| Compute (SM) throughput | 27.77% | 88.54% |
| Scheduler issue active | 14.86% | 88.62% |
| L1 hit rate | 20.75% | 49.13% |
| L2 hit rate | 47.48% | 47.59% |
| Registers per thread | 40 | 40 |
| Achieved occupancy | 99.46% | 99.44% |
| Warp instructions executed | 15.56 billion | 51.68 billion |

At k=4 the kernel is effectively DRAM-bandwidth limited. Widening k cuts the
number of weight blocks from 3,584 to 896 and the input traffic enough to
make k=16 faster despite more than doubling its reported addition count.
At k=16 the limiter has shifted to instruction issue: the SM is 89% busy,
DRAM is only 46% busy, and no register spills were recorded. SASS execution
counts are dominated by FADD (~30%) and LOP3 (~29%). The SASS source export
records 15.497 billion executed FADD warp instructions but only 27.28% of
their thread instructions predicated on. The source's positive/negative mask
conditions are uniform across a warp, yet the generated scatter path issues
many masked-off FADD instructions. Reducing both mask decoding and issued
inactive adds is a concrete target. A compact update list, grouped mask
dispatch, or a different output mapping needs benchmarking because dynamic
indexing or extra branches can erase the benefit.

At k=16 and K=4096, nearly every observed row has a distinct 16-column
ternary pattern under this random-weight model. SPR grouping then saves few
row-sum operations; a representation that directly stores row masks, or a
hybrid kernel, is worth comparing. For large M, input reuse across weight
blocks also remains important, since the entire X does not fit in L2.

Artifacts: `comparison.csv`, `samples.csv`, `wide_comparison.csv`,
`wide_samples.csv`, `v8_K4096_N14336_M4096_k16_z0.5.ncu-repz`,
`k16_metrics.csv`, `k16_details.txt`, and `k16_source_sass.csv`.
