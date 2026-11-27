# Experimental CUDA v8 wide-block test

`gpu/spr/spr_v8_wide.cu` is a separate experimental version registered as
`spr.v8.wide.cuda`. It preserves v8's thread mapping and direct accumulation,
replacing 16-bit pattern masks with 32-bit masks and 32-bit base-3 keys with
64-bit keys. It retains v8's widths 1–16 and adds widths 24 and 32. The original
`spr.v8.cuda` source was not changed.

## Correctness and resources

On an RTX 5060 Ti using native `sm_120`, Slurm job 5796 completed with exit 0:

- Shared FP64-reference suite: 361 passed, 0 failed.
- GPU edge FP64-reference suite: 108 passed, 0 failed.
- Exact-integer full-output suite: 760 passed, 0 failed; each case had two
  bindings and two repeated runs. Widths 24 and 32 cover all-zero, all-positive,
  all-negative, alternating, random, and empty-first-block weights; ragged
  output blocks; M around 128-thread boundaries and M=100000; K=65535; and the
  maximum grid.y of 65535 blocks.
- CUDA `memcheck` and `initcheck`: zero errors on 12 focused cases including
  widths 24 and 32 and K=65535.
- Compiled resource usage: 40, 48, and 56 registers/thread at k=16, 24, and
  32 respectively; zero stack and local memory for all three.

The full output is in `test-5796.out`; compiler resources are in
`resource_usage.txt`.

## Timings

Slurm job 5798 completed with exit 0. It used the same physical RTX 5060 Ti
for all configurations, seed 42, iid ternary weights, synchronized `run()`
medians, and FP64 sampled-output verification. Preparation and host-device
transfers are outside timing. `k=16` is the original `spr.v8.cuda`; k=24/32 are
`spr.v8.wide.cuda`. Times are milliseconds.

| M | K | N | Zero fraction | k=16 | k=24 | k=32 | Fastest tested | Speedup over k=16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4096 | 4096 | 0.5 | 5.399 | 4.930 | 4.828 | 32 | 1.12x |
| 4096 | 4096 | 14336 | 0.5 | 144.314 | 129.485 | 122.377 | 32 | 1.18x |
| 16384 | 4096 | 14336 | 0.5 | 641.471 | 558.760 | 518.836 | 32 | 1.24x |
| 100000 | 4096 | 14336 | 0.5 | 4093.107 | 3495.346 | 3215.325 | 32 | 1.27x |
| 4096 | 4096 | 14336 | 0.9 | 65.859 | 84.374 | 102.042 | 16 | 1.00x |
| 100000 | 4096 | 14336 | 0.9 | 2895.547 | 2444.087 | 2710.552 | 24 | 1.18x |

The M=512 comparison here starts at k=16. A separate earlier matched run on
the same physical GPU measured original v8 at k=5 as 3.101 ms, so the wider
widths do not establish a new best width for that smaller shape.

Widths were interleaved within one process at M<=16384. At M=100000, each
configuration ran in its own process because three simultaneous X/Y bindings
exceed GPU memory. Warmup/timed rounds were 5/31 at M=512, 2/9 at M=4096,
1/5 at M=16384, and 1/3 at M=100000. All 18 benchmark configurations passed
their FP64 sampled-row check. `summary-5798.csv` and `samples-5798.csv` contain
all 180 timing samples.

## Reproduce on the cluster

```sh
sbatch results/gpu_v8_wide_20261125/test.sbatch
# Once the test job passes:
sbatch results/gpu_v8_wide_20261125/bench.sbatch
```
