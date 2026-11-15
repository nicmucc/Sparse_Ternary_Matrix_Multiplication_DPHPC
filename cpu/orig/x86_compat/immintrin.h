// x86 only: the Makefile puts this directory first on the include path of the SSR / TCSC sources
// (third_party, unmodified) and of their wrappers. It includes the real <immintrin.h>, then:
//
// 1. Alignment (every x86 build). The rowMajor GroupMin GEMM and the GEMV kernels of
//    GEMM_CPU_FP32.cpp read row indices with ALIGNED loads (_mm256_load_si256 / _mm512_load_si512)
//    at offsets that are not multiples of the vector width. MSVC, which the authors used, emits
//    unaligned moves for these; GCC / clang emit aligned moves, which fault on such addresses.
//    Both are redirected to their unaligned forms (same speed on aligned data since Nehalem).
//
// 2. AVX-512 (only when the target has no AVX-512). GEMM_CPU_FP32.cpp defines its AVX-512 kernels
//    unconditionally, so it would not compile for an AVX2-only CPU. The seven AVX-512 intrinsics it
//    uses become plain 512-bit generic vectors (lowered to AVX2). Those kernels then give correct
//    results but are not AVX-512 code: cpu/orig/tcsc.cpp reports them unsupported
//    (SPR_AVX512_EMULATED), so they are never benchmarked or reported on such machines.
#pragma once
#include_next <immintrin.h>

// ---- 1. aligned integer loads -> unaligned
#undef _mm256_load_si256
#define _mm256_load_si256(p) _mm256_loadu_si256(p)
#if defined(__AVX512F__)
#undef _mm512_load_si512
#define _mm512_load_si512(p) _mm512_loadu_si512(p)
#endif

// ---- 2. AVX-512 emulation for AVX2-only targets
#if !defined(__AVX512F__)
#include <stdint.h>
#include <string.h>

typedef float spr_compat_v16f __attribute__((vector_size(64), aligned(64)));
typedef int32_t spr_compat_v16i __attribute__((vector_size(64), aligned(64)));
#define SPR_COMPAT_INLINE static inline __attribute__((always_inline))

typedef spr_compat_v16f spr_compat_m512;
typedef spr_compat_v16i spr_compat_m512i;
#define __m512 spr_compat_m512
#define __m512i spr_compat_m512i

SPR_COMPAT_INLINE __m512 spr_compat_mm512_setzero_ps(void) { __m512 r = {0}; return r; }
SPR_COMPAT_INLINE __m512 spr_compat_mm512_load_ps(const void* p) { __m512 r; memcpy(&r, p, 64); return r; }
SPR_COMPAT_INLINE void spr_compat_mm512_store_ps(void* p, __m512 a) { memcpy(p, &a, 64); }
SPR_COMPAT_INLINE __m512i spr_compat_mm512_load_si512(const void* p) { __m512i r; memcpy(&r, p, 64); return r; }
SPR_COMPAT_INLINE __m512 spr_compat_mm512_add_ps(__m512 a, __m512 b) { return a + b; }
SPR_COMPAT_INLINE __m512 spr_compat_mm512_sub_ps(__m512 a, __m512 b) { return a - b; }
SPR_COMPAT_INLINE __m512 spr_compat_mm512_i32gather_ps(__m512i idx, const void* base, int scale) {
    __m512 r;
    for (int i = 0; i < 16; ++i) memcpy((char*)&r + 4 * i, (const char*)base + (int64_t)idx[i] * scale, 4);
    return r;
}
#undef _mm512_setzero_ps
#define _mm512_setzero_ps spr_compat_mm512_setzero_ps
#undef _mm512_load_ps
#define _mm512_load_ps spr_compat_mm512_load_ps
#undef _mm512_store_ps
#define _mm512_store_ps spr_compat_mm512_store_ps
#undef _mm512_load_si512
#define _mm512_load_si512 spr_compat_mm512_load_si512
#undef _mm512_add_ps
#define _mm512_add_ps spr_compat_mm512_add_ps
#undef _mm512_sub_ps
#define _mm512_sub_ps spr_compat_mm512_sub_ps
#undef _mm512_i32gather_ps
#define _mm512_i32gather_ps spr_compat_mm512_i32gather_ps
#endif
