/*
 * RSR Ternary Inference Kernel v3.3
 *
 * Idea:
 *   - Build on v3.1's 16-bit direct-gather path.
 *   - Replace variable-length scatter lists with fixed-size positive/negative
 *     row masks, cutting hot-loop metadata traffic to two uint16 loads.
 *   - For suitable k <= 8, decode those masks through a compact coefficient
 *     table and accumulate all output rows with AVX2 or Arm NEON.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if !defined(RSR_DISABLE_SIMD) && \
    (defined(__AVX2__) || defined(__ARM_NEON))
#define RSR_MASK_SIMD 1
#include "rsr_mask_coeff_lut.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

void rsr_ternary_gemv_v3_3(
    const uint16_t *restrict perms,
    const uint16_t *restrict group_ends,
    const uint16_t *restrict pos_masks,
    const uint16_t *restrict neg_masks,
    const int32_t  *restrict block_meta,
    const float    *restrict v,
    float          *restrict out,
    int             n,
    int             k,
    int             num_blocks
)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < num_blocks; b++) {
        const uint16_t *perm = perms + (size_t)b * (size_t)n;
        const int ge_off = block_meta[2 * b];
        const int ge_end = (b + 1 < num_blocks) ? block_meta[2 * (b + 1)] : (ge_off + n);
        float *bout = out + (size_t)b * (size_t)k;
        int start = 0;

#if defined(RSR_MASK_SIMD)
#if defined(__AVX2__)
        if (__builtin_expect(k <= 8, 1)) {
#else
        /*
         * On Apple Silicon, k=4's scalar mask walk is slightly faster at
         * production BitNet-3B dimensions; the table path wins for k=5..8.
         */
        if (__builtin_expect(k >= 5 && k <= 8, 1)) {
#endif
#if defined(__AVX2__)
            __m256 accum = _mm256_setzero_ps();
#else
            float32x4_t accum_lo = vdupq_n_f32(0.0f);
            float32x4_t accum_hi = vdupq_n_f32(0.0f);
#endif

            for (int g = ge_off; g < ge_end && start < n; g++) {
                const int end = (int)group_ends[g];
                const int len = end - start;
                float agg = 0.0f;

                if (__builtin_expect(len <= 4, 1)) {
                    switch (len) {
                    case 4:
                        agg += v[perm[start + 3]];
                        /* fall through */
                    case 3:
                        agg += v[perm[start + 2]];
                        /* fall through */
                    case 2:
                        agg += v[perm[start + 1]];
                        /* fall through */
                    case 1:
                        agg += v[perm[start + 0]];
                        /* fall through */
                    default:
                        break;
                    }
                } else {
                    int i = start;
                    const int pf_dist = 48;
                    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                    for (; i + 7 < end; i += 8) {
                        if (i + pf_dist + 7 < end) {
                            __builtin_prefetch(
                                &v[perm[i + pf_dist + 0]], 0, 3);
                            __builtin_prefetch(
                                &v[perm[i + pf_dist + 2]], 0, 3);
                            __builtin_prefetch(
                                &v[perm[i + pf_dist + 4]], 0, 3);
                            __builtin_prefetch(
                                &v[perm[i + pf_dist + 6]], 0, 3);
                        }
                        s0 += v[perm[i + 0]] + v[perm[i + 1]];
                        s1 += v[perm[i + 2]] + v[perm[i + 3]];
                        s2 += v[perm[i + 4]] + v[perm[i + 5]];
                        s3 += v[perm[i + 6]] + v[perm[i + 7]];
                    }
                    agg += (s0 + s1) + (s2 + s3);
                    for (; i < end; i++) {
                        agg += v[perm[i]];
                    }
                }

                const unsigned pos_mask = pos_masks[g] & 0xffu;
                const unsigned neg_mask = neg_masks[g] & 0xffu;
#if defined(__AVX2__)
                const __m256 pos = _mm256_load_ps(rsr_mask_coeff_lut[pos_mask]);
                const __m256 neg = _mm256_load_ps(rsr_mask_coeff_lut[neg_mask]);
                const __m256 coeff = _mm256_sub_ps(pos, neg);
                const __m256 value = _mm256_set1_ps(agg);
#if defined(__FMA__)
                accum = _mm256_fmadd_ps(value, coeff, accum);
#else
                accum = _mm256_add_ps(
                    accum,
                    _mm256_mul_ps(value, coeff));
#endif
#else
                const float32x4_t pos_lo =
                    vld1q_f32(&rsr_mask_coeff_lut[pos_mask][0]);
                const float32x4_t pos_hi =
                    vld1q_f32(&rsr_mask_coeff_lut[pos_mask][4]);
                const float32x4_t neg_lo =
                    vld1q_f32(&rsr_mask_coeff_lut[neg_mask][0]);
                const float32x4_t neg_hi =
                    vld1q_f32(&rsr_mask_coeff_lut[neg_mask][4]);
                const float32x4_t coeff_lo = vsubq_f32(pos_lo, neg_lo);
                const float32x4_t coeff_hi = vsubq_f32(pos_hi, neg_hi);
#if defined(__aarch64__)
                accum_lo = vfmaq_n_f32(accum_lo, coeff_lo, agg);
                accum_hi = vfmaq_n_f32(accum_hi, coeff_hi, agg);
#else
                accum_lo = vmlaq_n_f32(accum_lo, coeff_lo, agg);
                accum_hi = vmlaq_n_f32(accum_hi, coeff_hi, agg);
#endif
#endif
                start = end;
            }

            /* Store only the active rows; lanes k..7 are padding. */
            float tmp[8] __attribute__((aligned(32)));
#if defined(__AVX2__)
            _mm256_store_ps(tmp, accum);
#else
            vst1q_f32(&tmp[0], accum_lo);
            vst1q_f32(&tmp[4], accum_hi);
#endif
            memcpy(bout, tmp, (size_t)k * sizeof(float));
            continue;
        }
#endif

        /* Original v3.3 path when the SIMD path is unavailable or unsuitable. */
        memset(bout, 0, (size_t)k * sizeof(float));
        for (int g = ge_off; g < ge_end && start < n; g++) {
            const int end = (int)group_ends[g];
            const int len = end - start;
            float agg = 0.0f;

            if (__builtin_expect(len <= 4, 1)) {
                switch (len) {
                    case 4:
                        agg += v[perm[start + 3]];
                    case 3:
                        agg += v[perm[start + 2]];
                    case 2:
                        agg += v[perm[start + 1]];
                    case 1:
                        agg += v[perm[start + 0]];
                    default:
                        break;
                }
            } else {
                int i = start;
#if defined(__AVX2__)
                const int pf_dist = 48;
#endif
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (; i + 7 < end; i += 8) {
#if defined(__AVX2__)
                    if (i + pf_dist + 7 < end) {
                        _mm_prefetch(
                            (const char *)&v[perm[i + pf_dist + 0]],
                            _MM_HINT_T0);
                        _mm_prefetch(
                            (const char *)&v[perm[i + pf_dist + 2]],
                            _MM_HINT_T0);
                        _mm_prefetch(
                            (const char *)&v[perm[i + pf_dist + 4]],
                            _MM_HINT_T0);
                        _mm_prefetch(
                            (const char *)&v[perm[i + pf_dist + 6]],
                            _MM_HINT_T0);
                    }
#endif
                    s0 += v[perm[i + 0]] + v[perm[i + 1]];
                    s1 += v[perm[i + 2]] + v[perm[i + 3]];
                    s2 += v[perm[i + 4]] + v[perm[i + 5]];
                    s3 += v[perm[i + 6]] + v[perm[i + 7]];
                }
                agg += (s0 + s1) + (s2 + s3);
                for (; i < end; i++) {
                    agg += v[perm[i]];
                }
            }

            uint16_t pos_mask = pos_masks[g];
            uint16_t neg_mask = neg_masks[g];

            while (pos_mask != 0) {
                const unsigned row = (unsigned)__builtin_ctz((unsigned)pos_mask);
                bout[row] += agg;
                pos_mask &= (uint16_t)(pos_mask - 1);
            }
            while (neg_mask != 0) {
                const unsigned row = (unsigned)__builtin_ctz((unsigned)neg_mask);
                bout[row] -= agg;
                neg_mask &= (uint16_t)(neg_mask - 1);
            }

            start = end;
        }
    }
}
