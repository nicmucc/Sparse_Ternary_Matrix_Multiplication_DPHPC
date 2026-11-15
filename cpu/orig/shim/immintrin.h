// Minimal AVX2 / AVX-512 -> portable shim, used ONLY on Apple Silicon (never on x86).
//
// Lets the *unmodified* source files of the SSR and TCSC repositories compile on ARM.
// Every 256-bit (512-bit) operation is expressed with clang generic vectors and lowers
// to 2 (4) 128-bit NEON instructions, so the original algorithms, loop structure and
// data layout are preserved. Only the intrinsics actually used by GEMM_CPU_FP32.cpp,
// ssr_vec.cpp and ssr.cpp are provided. Gathers become scalar loads (as on most x86
// cores, where vgatherdps is also ~1 load per lane).
//
// NOTE: AVX-512 kernels run here with 4x NEON, so their timings on the Mac say nothing
// about real AVX-512 hardware. Benchmark AVX-512 variants on x86 only.
#pragma once
#include <stdint.h>
#include <string.h>

typedef float __m256 __attribute__((vector_size(32), aligned(32)));
typedef long long __m256i __attribute__((vector_size(32), aligned(32)));
typedef int8_t __shim_v32qi __attribute__((vector_size(32), aligned(32)));
typedef int32_t __shim_v8si __attribute__((vector_size(32), aligned(32)));

#define SHIM_INLINE static inline __attribute__((always_inline))

SHIM_INLINE __m256 _mm256_add_ps(__m256 a, __m256 b) { return a + b; }
SHIM_INLINE __m256 _mm256_sub_ps(__m256 a, __m256 b) { return a - b; }
SHIM_INLINE __m256 _mm256_mul_ps(__m256 a, __m256 b) { return a * b; }
SHIM_INLINE __m256 _mm256_setzero_ps(void) { return (__m256){0, 0, 0, 0, 0, 0, 0, 0}; }
SHIM_INLINE __m256 _mm256_set1_ps(float x) { return (__m256){x, x, x, x, x, x, x, x}; }
SHIM_INLINE __m256 _mm256_set_ps(float e7, float e6, float e5, float e4,
                                 float e3, float e2, float e1, float e0) {
    return (__m256){e0, e1, e2, e3, e4, e5, e6, e7};
}
SHIM_INLINE __m256 _mm256_load_ps(const float* p) { return *(const __m256*)p; }
SHIM_INLINE __m256 _mm256_loadu_ps(const float* p) { __m256 r; memcpy(&r, p, 32); return r; }
SHIM_INLINE void _mm256_store_ps(float* p, __m256 a) { *(__m256*)p = a; }
SHIM_INLINE void _mm256_storeu_ps(float* p, __m256 a) { memcpy(p, &a, 32); }
SHIM_INLINE __m256 _mm256_i32gather_ps(const float* base, __m256i vindex, const int scale) {
    __shim_v8si idx = (__shim_v8si)vindex;
    __m256 r;
    for (int i = 0; i < 8; ++i)
        r[i] = *(const float*)((const char*)base + (long)idx[i] * scale);
    return r;
}

SHIM_INLINE __m256i _mm256_setzero_si256(void) { return (__m256i){0, 0, 0, 0}; }
SHIM_INLINE __m256i _mm256_add_epi8(__m256i a, __m256i b) {
    return (__m256i)((__shim_v32qi)a + (__shim_v32qi)b);
}
SHIM_INLINE __m256i _mm256_sub_epi8(__m256i a, __m256i b) {
    return (__m256i)((__shim_v32qi)a - (__shim_v32qi)b);
}
SHIM_INLINE __m256i _mm256_load_si256(const __m256i* p) { __m256i r; memcpy(&r, p, 32); return r; }   // upstream calls it on unaligned addresses (see x86_compat)
SHIM_INLINE __m256i _mm256_loadu_si256(const __m256i* p) { __m256i r; memcpy(&r, p, 32); return r; }
SHIM_INLINE void _mm256_store_si256(__m256i* p, __m256i a) { *p = a; }
SHIM_INLINE __m256i _mm256_set_epi8(
    char e31, char e30, char e29, char e28, char e27, char e26, char e25, char e24,
    char e23, char e22, char e21, char e20, char e19, char e18, char e17, char e16,
    char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8,
    char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0) {
    __shim_v32qi r = {e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
                      e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31};
    return (__m256i)r;
}
SHIM_INLINE __m256i _mm256_set1_epi32(int x) {
    return (__m256i)(__shim_v8si){x, x, x, x, x, x, x, x};
}
SHIM_INLINE __m256i _mm256_setr_epi32(int e0, int e1, int e2, int e3, int e4, int e5, int e6, int e7) {
    return (__m256i)(__shim_v8si){e0, e1, e2, e3, e4, e5, e6, e7};
}
SHIM_INLINE __m256i _mm256_set_epi32(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0) {
    return (__m256i)(__shim_v8si){e0, e1, e2, e3, e4, e5, e6, e7};
}

// ---------------------------------------------------------------- AVX-512 (float only)
typedef float __m512 __attribute__((vector_size(64), aligned(64)));
typedef long long __m512i __attribute__((vector_size(64), aligned(64)));
typedef int32_t __shim_v16si __attribute__((vector_size(64), aligned(64)));

SHIM_INLINE __m512 _mm512_add_ps(__m512 a, __m512 b) { return a + b; }
SHIM_INLINE __m512 _mm512_sub_ps(__m512 a, __m512 b) { return a - b; }
SHIM_INLINE __m512 _mm512_setzero_ps(void) { __m512 r = {0}; return r; }
SHIM_INLINE __m512 _mm512_load_ps(const void* p) { return *(const __m512*)p; }
SHIM_INLINE void _mm512_store_ps(void* p, __m512 a) { *(__m512*)p = a; }
SHIM_INLINE __m512i _mm512_load_si512(const void* p) { __m512i r; memcpy(&r, p, 64); return r; }   // same
SHIM_INLINE __m512 _mm512_i32gather_ps(__m512i vindex, const void* base, const int scale) {
    __shim_v16si idx = (__shim_v16si)vindex;
    __m512 r;
    for (int i = 0; i < 16; ++i)
        r[i] = *(const float*)((const char*)base + (long)idx[i] * scale);
    return r;
}
