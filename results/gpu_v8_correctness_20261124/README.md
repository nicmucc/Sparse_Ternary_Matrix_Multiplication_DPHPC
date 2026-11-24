# CUDA SPR v8 correctness audit

Run on `studgpu-node01` with CUDA 13.4, `sm_120`, and an RTX 5060 Ti.
The completed Slurm job is 5413; its output is `verify-5413.out`.

| Check | Result |
|---|---|
| Shared correctness suite against FP64 reference | 361 passed, 0 failed |
| GPU edge suite against FP64 reference | 108 passed, 0 failed |
| Independent exact-integer reference suite | 671 passed, 0 failed |
| CUDA `memcheck` on all 671 exact cases | 0 errors |
| CUDA `initcheck` on all 671 exact cases | 0 errors |
| Slurm job | COMPLETED, exit 0 |

The exact suite checks **every output element** for each supported `k=1..16`,
with two bindings and two repeated runs per case. It covers all-zero,
all-positive, all-negative, alternating, random, and empty-first-block weights;
ragged final blocks; M around warp and 128-thread boundaries; empty dimensions;
K=65535 (including a slot of length 65535); M=100000; and the maximum grid.y of
65535 blocks. Integer X values keep the reference and computed results exactly
representable in FP32, so the test requires equality rather than a tolerance.

The shared and GPU edge suites additionally use noninteger FP32 inputs and
compare against an FP64 dense reference with their documented tolerances.

## Indexing argument

For each weight block, `prepare()` maps every row's k ternary weights to one
base-3 key. It drops precisely the all-zero rows, groups equal keys into one
slot, and stores each surviving row ID once. The slot masks decode the same
key, so the `+1` and `-1` bits match the corresponding columns. For each output
row m, the kernel sums `X[r,m]` over each slot and adds or subtracts that sum
to the masked output accumulators. Thus, for each column j, the accumulated
value is the sum of `X[r,m] * W[r,j]` over all rows r. Only the actual width of
the final weight block is stored to Y.

The implementation's index limits are consistent: `3^16-1` fits in its
`uint32_t` key, `K<=65535` fits in the `uint16_t` row IDs and slot lengths, and
`N<=65535*k` keeps grid.y within 65535 blocks. Zero M or N skips launch; zero
K launches and writes zero outputs.

## Reproduce

From the repository root, submit `verify.sbatch` on this cluster:

```sh
sbatch results/gpu_v8_correctness_20261124/verify.sbatch
```

The standalone exact suite is available as:

```sh
make test-gpu-v8 CUDA_ARCH=sm_120
```

This establishes correctness for the documented ternary-weight contract and
tested FP32 behavior. As with any finite test suite, it is not a formal proof
of all possible floating-point inputs or allocation failures.
