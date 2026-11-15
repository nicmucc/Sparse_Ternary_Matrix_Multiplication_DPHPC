# third_party: the papers' code, unmodified

Every file here is a byte-for-byte copy of the upstream repository at the commit below, kept at
its upstream path. Nothing is edited: the Makefile compiles the source files as they are, and
`cpu/orig/` wraps their functions in the common kernel API (`include/spr/kernel.hpp`).
Only the files the wrappers need are vendored (plus each repo's README and LICENSE, and the
drivers or Python wrappers that show how the authors build the weight formats).

| Directory | Upstream | Commit | License |
|---|---|---|---|
| `SSR-ternaryLLM-repo/` | [fpgasystems/ternaryLLM](https://github.com/fpgasystems/ternaryLLM) (SSR, TCSC, and the fp32 RSR / RSR++ the SSR authors benchmarked) | `68d384425261d60bfbde5ff079c33a474c1eb28e` | MIT, `SSR-ternaryLLM-repo/LICENSE` |
| `RSR-core-repo/` | [UIC-InDeXLab/RSR-core](https://github.com/UIC-InDeXLab/RSR-core) (the RSR authors' optimized kernels) | `99bcccded70416839f1ef1fc6d58726c507d1fb9` | MIT, `RSR-core-repo/LICENSE` |

## Files

`SSR-ternaryLLM-repo/`

| File | Used for |
|---|---|
| `ternaryLLM_CPU/src/GEMM_CPU_FP32.{cpp,hpp}` | TCSC kernels (compiled) |
| `ternaryLLM_CPU/src/TCSC.hpp` | TCSC weight formats (included by `cpu/orig/tcsc.cpp`) |
| `ternaryLLM_CPU/src/main.cpp` | reference only: how the authors build formats and call each kernel |
| `SSR/SSR/ssr.{cpp,h}`, `SSR/SSR/ssr_vec.{cpp,h}` | SSR preprocessing and kernels (compiled) |
| `SSR/SSR/original_rsr/*` | fp32 RSR / RSR++ (compiled) |
| `SSR/src/main.cpp`, `SSR/src/single-thread.cpp` | reference only: the SSR paper's benchmark drivers |

`RSR-core-repo/`

| File | Used for |
|---|---|
| `kernels/bit_1_58/cpu/rsr_ternary_prep_nonsquare.c`, `rsr_ternary_v3_1.c`, `rsr_ternary_v3_3.c`, `rsr_mask_coeff_lut.h` | ternary RSR GEMV v3.1 / v3.3 and their preprocessing (compiled) |
| `kernels/bit_1/cpu/rsr_prep_nonsquare.c`, `rsr_v4_2.c` | binary RSR GEMV v4.2 and its preprocessing (compiled; run on the +1 and −1 planes) |
| `multiplier/bit_1_58/cpu/{rsr_nonsquare.py, rsr_v3_3.py, _rsr_v3_common.py}`, `multiplier/bit_1/cpu/{rsr_cpp_nonsquare.py, rsr_cpp_v4_2.py}` | reference only: how the authors build the formats |

## Rules

- Never edit these files. A fix to upstream code goes in `cpu/orig/patched/` as a documented copy
  (today only the scalar `ssr_inference`, whose indexing bug is described in `cpu/orig/README.md`).
- To use another upstream function, vendor its file unchanged (same path), add it to `UPSTREAM_CXX` /
  `UPSTREAM_C` in the Makefile, and register a wrapper in `cpu/orig/`.
- To update a repo, replace its files with the new commit's and change the commit above.
