# `cpu/orig/`: the published kernels, plugged into our harness

Nothing in here reimplements an algorithm. The Makefile compiles the authors' source files
**directly from `third_party/`, unmodified** (see `../../third_party/README.md` for the commits). The `.cpp` files in this folder are only thin
wrappers that:

1. build the weight format the way the authors' own `main.cpp` does;
2. hand X / Y over in the layout the function expects;
3. call the function.

Every kernel name is the upstream function name with a prefix (`tcsc.`, `ssr.`, `rsr.`), so
any number in a CSV can be traced back to one function in one file.

| Wrapper | Upstream files compiled (unmodified) | Paper |
|---|---|---|
| `tcsc.cpp` | `ternaryLLM_CPU/src/GEMM_CPU_FP32.cpp`, `TCSC.hpp` | Zhu et al., *Efficient Addition-Based Sparse GEMM…*, ACM TECS 2026 |
| `ssr.cpp`  | `SSR/SSR/ssr.cpp`, `ssr.h`, `ssr_vec.cpp` | Pittet et al., *SSR: Sparse Segment Reduction…*, DATE 2026 |
| `rsr.cpp`  | `SSR/SSR/original_rsr/{rsr.cpp, rsr.h, rsrpp.h, utils.cpp, naive.cpp}` | Dehghankar et al. 2025 (RSR / RSR++), in the fp32 matrix version the SSR authors benchmarked |
| `rsrcore.cpp` | `RSR-core-repo/kernels/bit_1_58/cpu/{rsr_ternary_prep_nonsquare.c, rsr_ternary_v3_1.c, rsr_ternary_v3_3.c}`, `RSR-core-repo/kernels/bit_1/cpu/{rsr_prep_nonsquare.c, rsr_v4_2.c}` | RSR-core, the RSR authors' optimized kernel engine (github.com/UIC-InDeXLab/RSR-core @ 99bcccd) |
| `patched/ssr_inference_patched.h` | copy of `ssr.h::ssr_inference` with the indexing bug fixed (2 lines) | SSR scalar |

All paths are relative to `third_party/SSR-ternaryLLM-repo/` (commit `68d3844`) unless they start with `RSR-core-repo/`.

**Status:** `make test` passes for every wrapped kernel except
`ssr.ssr_inference`, which fails because of a bug in the authors' code (see the SSR section).

---

## How to read the tables

| Column | Meaning |
|---|---|
| **ISA** | `scalar` = plain C++; `AVX2` = hand-written 256-bit intrinsics; `AVX-512` = 512-bit intrinsics |
| **Threads** | `OpenMP` = `#pragma omp parallel for` inside the function; `1` = single-threaded |
| **Unroll M×N** | rows of X × columns of W computed per inner iteration. Our `supports()` requires `M % mu == 0` and `N % nu == 0` |
| **Format** | weight data structure, built in `prepare()` (timed as `prep_ms`) |
| **X / Y** | `col` = column-major (`X[k*M + m]`, `Y[n*M + m]`); `row` = row-major. The harness converts outside timing |

Weight formats (`TCSC.hpp`):
- **Naive**: `SparseFormat`. Two CSC lists (+1 rows, −1 rows), int16 row indices, int32 column pointers.
- **Merged**: `MergedTCSC`. Per column, +1/−1 indices interleaved in pairs, then the remainder; 4 ints of metadata per column.
- **GroupMin(G)**: `MergedTCSC_Group(G, "min", interleaved)`. Groups of G columns share the aligned part (`min` over the group of `min(#+1, #−1)`); each column keeps its own remainder loops.
- **GroupMid(G)**: `MergedTCSC_Group(G, "mid", true)`. Shorter columns are padded with `a − a = 0` pairs so each column has a single remainder loop.

> **Benchmark on x86.** On x86 the real `<immintrin.h>` is used and every kernel runs natively.
> - **Apple Silicon (development only):** `shim/immintrin.h` maps AVX2 and AVX-512 intrinsics to
>   portable vectors (NEON). Results are correct, so `make test` works on a Mac, but the AVX2 / AVX-512
>   timings there say nothing about x86.
> - **x86 (every build):** the Makefile puts `x86_compat/immintrin.h` before the real header. The rowMajor
>   GroupMin GEMM and all GEMV kernels read row indices with *aligned* loads (`_mm256_load_si256`,
>   `_mm512_load_si512`) at offsets that are not multiples of the vector width (found by UBSan). MSVC,
>   which the authors used, emits unaligned moves for these; GCC / clang emit aligned moves that fault.
>   The header maps both to their unaligned forms (same speed on aligned data). The colMajor kernels are
>   not affected. The wrapper also passes 64-byte aligned copies of the row indices.
> - **x86 without AVX-512:** `GEMM_CPU_FP32.cpp` defines its AVX-512 kernels without `#ifdef`, so the
>   unmodified file would not compile. `x86_compat/immintrin.h` then also provides the seven AVX-512
>   intrinsics that file uses, as plain 512-bit vectors. Those kernels report themselves unsupported
>   (`SPR_AVX512_EMULATED`), so tests and benchmarks skip them on that machine.

---

## TCSC (`tcsc.cpp`): `ternaryLLM_CPU/src/GEMM_CPU_FP32.cpp`

### Dense baselines (the authors' own)

| Kernel (`tcsc.` + name) | Line | ISA | Threads | Unroll | Format | X/Y |
|---|---|---|---|---|---|---|
| `GEMM_CPU_FP32_colMajor_Direct_OpenMP` | 8 | scalar + `omp simd` | OpenMP | 1×1 | dense int8 W | col |
| `GEMM_CPU_FP32_rowMajor_Direct_OpenMP` | 16839 | scalar + `omp simd` | OpenMP | 1×1 | dense int8 W | row |

### Naive TCSC: the optimization ladder of the TCSC paper (Fig. 4)

| Kernel | Line | ISA | Threads | Unroll | Format | X/Y | What changes |
|---|---|---|---|---|---|---|---|
| `GEMM_CPU_FP32_colMajor_TCSC_Naive` | 22 | scalar | 1 | 1×1 | Naive | col | one loop over +1 rows, one over −1 rows |
| `GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor` | 48 | scalar | 1 | 1×1 | Naive | col | one shared loop over `min(#+1,#−1)`, then remainders |
| `GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_4x4_Unroll` | 82 | scalar | 1 | 4×4 | Naive | col | + loop unrolling |
| `GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2` | 247 | AVX2 | 1 | 8×1 | Naive | col | + SIMD over 8 rows of X |
| `GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2_OpenMP` | 286 | AVX2 | OpenMP | 8×1 | Naive | col | + threads |
| `GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2` | 325 | AVX2 | 1 | 8×4 | Naive | col | + unroll 4 columns |
| `GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP` | 417 | AVX2 | OpenMP | 8×4 | Naive | col | + threads |
| `GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_OpenMP` | 16853 | scalar | OpenMP | 1×1 | Naive | row | row-major X |
| `GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_4x4_Unroll_OpenMP` | 16887 | scalar | OpenMP | 4×4 | Naive | row | row-major X + unroll |

### Merged TCSC, one column at a time (paper Figs. 4–5)

| Kernel | Line | ISA | Threads | Unroll | Format | X/Y | Notes |
|---|---|---|---|---|---|---|---|
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_8x1_AVX2_OpenMP` | 509 | AVX2 | OpenMP | 8×1 | Merged | col | |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_OpenMP` | 541 | AVX2 | OpenMP | 16×1 | Merged | col | |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_if_OpenMP` | 582 | AVX2 | OpenMP | 16×1 | Merged | col | remainder via `if` (Fig. 5) |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMP` | 628 | AVX2 | OpenMP | 8×4 | Merged | col | |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMPi` | 710 | AVX2 | OpenMP (inner i) | 8×4 | Merged | col | loop order / OpenMP placement study (Fig. 5) |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMP` | 791 | AVX2 | OpenMP | 8×4 | Merged | col | same study |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMPj` | 873 | AVX2 | OpenMP (inner j) | 8×4 | Merged | col | same study |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_16xG1_AVX512_OpenMP` | 7364 | AVX-512 | OpenMP | 16×1 | Merged | col | x86 only for timing |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_32xG1_AVX512_OpenMP` | 8094 | AVX-512 | OpenMP | 32×1 | Merged | col | x86 only for timing |

### Merged TCSC with column grouping, column-major GEMM: **the paper's main CPU kernels**

All of these use AVX2 or AVX-512, OpenMP, the GroupMin(G) format built with `interleaved=true`, and col X/Y.

| Kernel pattern | Lines | Variants (M-unroll × G) |
|---|---|---|
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_{MU}xG{G}_AVX2_OpenMP` | 956–4781 | MU ∈ {8, 16, 32}, G ∈ {4, 8, 16, 32}, plus 64xG4 |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_{MU}xG{G}_AVX512_OpenMP` | 7390–8664 | MU ∈ {16, 32}, G ∈ {4, 8, 16, 32}. x86 only for timing |
| `GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMid_8xG4_AVX2_OpenMP` | 5033 | GroupMid(4) format |

Constraints: `M % MU == 0` and `N % G == 0`. The case study in the TCSC paper (vs. SSR / RSR++)
uses GroupMin with G = 8 and AVX2.

### Merged TCSC with column grouping, row-major GEMM

All use AVX2, OpenMP, and row X/Y. The format is GroupMin(G) with `interleaved=false`, with
row indices widened to int32 (as in their `main.cpp`).

| Kernel pattern | Lines | Variants |
|---|---|---|
| `GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_{MU}xG{G}_AVX2_OpenMP` | 10547–14171 | MU ∈ {1, 2, 4, 8, 16}, G ∈ {8, 16, 32, 64} |
| `GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_1xG8_AVX2_OpenMP{ij,ijj,ji,jii}` | 10360–10499 | loop order / OpenMP placement study |

### GEMV (M = 1, token generation)

All use OpenMP and row X/Y, with the GroupMin(G) format built with `interleaved=false` and int32
indices. `supports()` returns true only for M = 1.

| Kernel pattern | Lines | ISA | Variants |
|---|---|---|---|
| `GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G{8,16,32,64}_AVX2_OpenMP` | 14879–15008 | AVX2 | |
| `GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G{16,32,64,128}_CS8_AVX2_OpenMP` | 15083–15461 | AVX2 | "CS8": format built with `SIMD_SIZE=8` (chunked layout) |
| `GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G64_CS8_SIMD{1,2,3}_AVX2_OpenMP` | 15238–15386 | AVX2 | SIMD strategy variants |
| `GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G{16,32,64,128}_AVX512_OpenMP`, `G32_SIMD3_AVX512` | 15592–15731 | AVX-512 | x86 only for timing |
| `GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G{32,64,128}_CS16_AVX512_OpenMP`, `G128_CS16_SIMD{1,2,3}` | 15806–16108 | AVX-512 | x86 only for timing |

### TCSC functions NOT wrapped, and why

| Upstream | Why |
|---|---|
| `*_TCSC_Uniform_*` (23 GEMM, 15 GEMV) | They need **structured** weights: every column must have the same number of +1 and −1, which their generator `sparseWeight(..., uniform=true)` produces. They give wrong results on general ternary matrices, so they aren't comparable to SPR, SSR or GroupMin. |
| `GEMM_CPU_FP32_colMajor_TCSC_Aligned_GroupMax_8xG4_AVX2_OpenMP` | Same reason: needs "aligned" (semi-structured) weights. |
| `GEMM_CPU_INT8.cpp` (all) | INT8 activations. We benchmark FP32 activations only, for now. |
| templates `GEMM` / `GEMM_CPU_colMajor_Direct_OpenMP` (`GEMM_CPU_FP32.hpp`), the direct template at the end of `TCSC.hpp` | Generic helpers; the FP32 Direct kernels above cover them. |

---

## SSR (`ssr.cpp`): `SSR/SSR/`

All of them use the format from `ssr_preprocess(mat, L, K, N)` (`ssr.cpp:45`). Per block of L
columns and per half (+1 / −1), it stores a permutation of all K rows (K ints) and a
segment list (2^L ints). The zero-pattern rows are skipped through the segment offsets.

All use row X/Y. The functions take X as a `std::vector` and return a new
`std::vector<float>` (M × N padded to a multiple of L). The copy-in happens in `bind()` and the
copy-out in `finish()`, both outside timing. `k` in our CSVs = their `L`.

| Kernel (`ssr.` + name) | Where | ISA | Threads | Unroll | k range | Notes |
|---|---|---|---|---|---|---|
| `ssr_inference` | `ssr.h:159` | scalar | OpenMP over rows | 1 row | 1–12 | **"SSR Naive" of the paper.** Segmented sum, then the tree from Alg. 2 as loops. **Broken upstream**: indexes `v[row * L + permutations[b_idx * L + i]]`, which should be `row * K` and `b_idx * K`. Wrong results except in degenerate cases, so `test_correctness` reports FAIL and the bench never times it. |
| `ssr_inference_float_vec_L` | `ssr_vec.cpp:17151` (L2) … `23228` (L8) | AVX2 | OpenMP over rows | 8 rows / `__m256` | 2–8 | **The optimized SSR of the paper** (their `SSR/src/main.cpp` calls `_L4` and `_L8`). One fully unrolled function per L. **No run-time tree**: each pattern's segment sum is added directly into every output column whose bit is set, so cost per block is about L·2^(L−1) adds, all 2^L−1 patterns paid even if absent. X gathered with `_mm256_set_ps` (8 scalar loads). Requires `M % 8 == 0`. |
| `ssr_inference_float_vec_unroll8_v2` | `ssr_vec.cpp:16787` | AVX2 | OpenMP | 8 rows | 1–12 | Generic-L kernel. Parameter names are `(L, M, N, K)`, but the authors' `main.cpp` passes `(L, K_LEN, N_COL, M_ROW)`, i.e. inside, "M" is the inner dimension. We call it exactly like `main.cpp`. Handles any M (scalar remainder). |
| `ssr_inference_float_vec_unroll16_v2` | `ssr_vec.cpp:16947` | AVX2 | OpenMP | 16 rows | 1–12 | Same as above with 2 × 8 rows. |

**`ssr.ssr_inference_patched`** (`patched/ssr_inference_patched.h`) is the scalar SSR we benchmark: the upstream
function verbatim except the gather index, `v[row * L + perm[b_idx * L + i]]` → `v[row * K + perm[b_idx * K + i]]`
(a row of X has K values; `ssr_preprocess` stores K permutation entries per block). The header documents the diff.

### SSR functions NOT wrapped, and why

| Upstream | Why |
|---|---|
| `ssr_inference_float_vec_tree` (`ssr_vec.cpp`) | Only a dispatcher to `_L2…_L6`; covered by `ssr_inference_float_vec_L`. |
| `*_int8_*` (all of `ssr_vec.cpp`'s INT8 half) | INT8 activations; FP32 only for now. |
| `binary_ssr_inference`, `vector_ssr_inference` (`ssr.h`) | GEMV on a per-block `vector<vector<int>>` format; the matrix kernels above cover SSR. |
| `ssr_unroll.h` (8.7 k lines), `ssr_tree.h` (154 k lines) | Generated scalar unrolled templates (`ssr_inference_ur{2,4,…}`, `…_tree{L}`). Not called by the paper's benchmark drivers (`SSR/src/main.cpp`, `single-thread.cpp`). Can be added if needed (long compile time). |
| `ssr_preprocess_{float,int8}_vec_{tree,unroll}` | Same preprocessing, but with L picked from a hard-coded table for 6 specific shapes. We sweep `k` ourselves instead. |

---

## RSR / RSR++ (`rsr.cpp`): `SSR/SSR/original_rsr/`

This is the version the SSR authors adapted from the RSR repo (vector → matrix, int → templated
fp32) and used as their RSR / RSR++ baseline. The format comes from `rsr_preprocess(mat, L)`
(`rsr.cpp:96`): per block of L columns and per half, a permutation (K ints) and a segment list
(2^L ints), as `vector<vector<int>>`.

All use scalar C++ on `std::vector`, single-threaded (their `#pragma omp parallel for` is commented out), with row X/Y.

| Kernel (`rsr.` + name) | Where | k range | Notes |
|---|---|---|---|
| `rsr_inference` | `rsr.h:121` | 1–12 | RSR: segmented sum, then `u · Bin_L` (L·2^L adds per block). Takes X as `vector<vector<float>>` **by value**, so each call copies X. That copy is part of their API and is timed, as in their benchmark. |
| `rsr_pp_inference` | `rsrpp.h:128` | 1–12 | RSR++: segmented sum + halving recursion `step_three` (~2·2^L adds). Same API notes. |
| `vector_rsr_inference` | `rsr.h:63` | 1–12 | GEMV (M = 1) version of RSR. |
| `vector_rsr_pp_inference` | `rsrpp.h:70` | 1–12 | GEMV (M = 1) version of RSR++. |

Not wrapped: the original RSR repo's own C++ (`third_party/RSR-repo/native/`). It is an **int**
matrix × vector on `vector<int>`, and the fp32 adaptation above is what the SSR paper compared against.

### RSR-core (`rsrcore.cpp`): the optimized RSR

`third_party/RSR-core-repo` (MIT). Their ternary RSR computes `A·v` blocking k rows of `A`; we pass `A = Wᵀ`.
Format built like their Python wrappers (`multiplier/bit_1_58/cpu/rsr_nonsquare.py`, `rsr_v3_3.py`): counting sort over
4^k ternary codes, uint16 perms / group ends. The all-zero pattern is a group too (no zero skipping). `k ≤ 12`, `K ≤ 65535`.

| Kernel | ISA | Threads | Notes |
|---|---|---|---|
| `rsrcore.rsr_ternary_gemv_v3_1` | scalar + prefetch | OpenMP over blocks | per group a (row, sign) scatter list |
| `rsrcore.rsr_ternary_gemv_v3_3` | NEON / AVX2 for k ≤ 8 (NEON k = 5..8), else scalar | OpenMP over blocks | two uint16 masks per group; their main CPU kernel |
| `rsrcore.rsr_gemv_v4_2` | scalar, unrolled gather + prefetch | OpenMP over blocks | binary RSR (`bit_1/`), run on the +1 plane and on the −1 plane, `y = A₊x − A₋x` (how the RSR papers handle ternary weights). Format as `multiplier/bit_1/cpu/rsr_cpp_nonsquare.py`. `k ≤ 16`. |

All three are GEMV; for M > 1 the wrapper calls them once per row of X (RSR-core has no multi-row kernel). Not used:
`*_batch.c` (batches layers, and fuses int8 activation quantization, which changes the result), `rsr_ternary.c` (v1.4,
superseded), the older binary kernels (`rsr.c`, `rsr_v2_4.c`, superseded by v4.2), CUDA.

There is **no optimized RSR++**: RSR-core's ternary kernels scatter each group sum directly into its outputs.

---

## Dense baselines (`../libs.cpp`, not from the papers' repos)

| Kernel | What | Threads |
|---|---|---|
| `lib.dense_naive` | plain triple loop on fp32 W (scalar dense reference) | 1 |
| `lib.dense_blas` | `cblas_sgemm`: Accelerate on macOS (uses the AMX units on M-series), OpenBLAS/MKL elsewhere | `VECLIB_MAXIMUM_THREADS` / `OPENBLAS_NUM_THREADS` |
| `lib.eigen_dense` | Eigen dense GEMM, W stored as fp32 | OpenMP |
| `lib.eigen_sparse` | Eigen `SparseMatrix` (CSC, fp32 values) × dense. Both papers use it as a baseline. | 1 |

---

## Which kernels to compare SPR against


| SPR level | Same engineering level as | Strongest published at that level |
|---|---|---|
| `spr.naive` … `spr.v1`: scalar | `ssr.ssr_inference_patched`, `rsr.rsr_pp_inference` | `tcsc.GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor{,_4x4_Unroll}` |
| `spr.v2` … `spr.v9`: vectorized, tiled, OpenMP | `ssr.ssr_inference_float_vec_L`, `rsrcore.*` | `tcsc.…colMajor_TCSC_Merged_GroupMin_{MU}xG{G}_{AVX2,AVX512}_OpenMP` (best MU, G per shape) |
| GEMV (M = 1) | `rsr.vector_rsr_pp_inference`, `rsrcore.*` | `tcsc.GEMV_…GroupMin_G{…}_{AVX2,AVX512}_OpenMP` |

**About `ssr.ssr_inference`:** to compare `spr.naive` against SSR at the scalar level, there are
two options:
- a patched copy (`cpu/orig/patched/`) with only the indexing fixed, `L` → `K` in two places,
  documented as a patch;
- skip scalar SSR and compare against RSR++ and naive TCSC only.

The first is what a reviewer would expect.
