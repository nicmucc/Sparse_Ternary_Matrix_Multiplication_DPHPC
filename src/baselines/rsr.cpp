#include "rsr.h"

#include <array>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <limits>
#include "utils.h"
#include "naive.h"

using namespace std;

// Return <permutation, segmentation>
pair<vector<int>, vector<int>> handle_block(const tern_t& mat_block, int K, int block_width) {

    const size_t pattern_count = size_t{1} << block_width;

    // Permutation
    vector<int> values(K);

    for(int i = 0; i < K; i++) {
        vector<int> row(block_width);

        for(int j = 0; j < block_width; j++) {
            row[j] = mat_block[i*block_width + j];

            if(row[j] != 0 && row[j] != 1) {
                throw invalid_argument("RSR requires binary weights (0 or 1)");
            }
        }

        values[i] = binaryVectorToInt(row);
    }

    // Permutation
    vector<int> permutation(K);
    for (int i = 0; i < K; i++) {
        permutation[i] = i;
    }

    sort(permutation.begin(), permutation.end(), [&](int i, int j) {
        return values[i] < values[j];
    });

    // Segmentation
    vector<int> seg(pattern_count, -1);
    seg[0] = 0;

    for (int row = 0; row < K; row++) {
        int value = values[permutation[row]];

        if (seg[value] == -1) {
            seg[value] = row;
        }
    }

    if (seg.back() == -1) {
        seg.back() = K;
    }

    int last_one = seg.back();

    for (int i = static_cast<int>(seg.size()) - 2; i >= 0; i--) {
        if (seg[i] == -1) {
            seg[i] = last_one;
        }
        last_one = seg[i];
    }

    return make_pair(permutation, seg);
}

pair<vector<vector<int>>, vector<vector<int>>> preprocess(
    const tern_t& mat, int block_width, int K) {
    
    if (mat.empty() || K <= 0 || block_width <= 0 ||
        block_width >= numeric_limits<int>::digits) {
        throw invalid_argument("Invalid preprocessing dimensions");
    }

    const int N = mat.size() / K;
    if (mat.size() != static_cast<size_t>(K * N)) {
        throw invalid_argument("Invalid matrix size");
    } 

    // Pad output columns so they can be processed in blocks of block_width. The input
    // row count must stay unchanged because it determines the input vector size.

    const int padding = (block_width - N % block_width) % block_width;
    const int padded_N = N + padding;

    tern_t padded_mat(K * padded_N, 0);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            padded_mat[i * padded_N + j] = mat[i * N + j];
        }
    }

    // Allocate permutations and segments
    const int num_blocks = padded_N / block_width;
    const size_t pattern_count = size_t{1} << block_width;

    vector<vector<int>> permutations(num_blocks, vector<int>(K));
    vector<vector<int>> segs(num_blocks, vector<int>(pattern_count));


    // Blocking
    tern_t block(K * block_width);

    for (int i = 0; i < num_blocks; i++) {

        int start = i * block_width;

        for (int row = 0; row < K; row++) {
            for (int col = 0; col < block_width; col++) {

                block[row * block_width + col] =
                    padded_mat[row * padded_N + start + col];
            }
        }

        auto per_seg = handle_block(block, K, block_width);

        permutations[i] = per_seg.first;
        segs[i] = per_seg.second;
    }

    return make_pair(permutations, segs);
}

vec_t rsr_inference(const vec_t& v, const permutation_t& permutations, const segment_t& segments, const binary_matrix_t& binary_patterns, size_t K, int block_width) {
    
    if (K == 0 || block_width <= 0 ||
        block_width >= numeric_limits<int>::digits) {
        throw invalid_argument("Invalid inference dimensions");
    }

    if (v.size() != K || permutations.size() % K != 0) {
        throw invalid_argument("Invalid input or permutation size");
    }

    // segmented sums
    const size_t pattern_count = size_t{1} << block_width;
    const size_t num_blocks = permutations.size() / K;
    vec_t us(num_blocks * pattern_count, 0.0f);

    if (segments.size() != num_blocks * pattern_count ||
        binary_patterns.size() != pattern_count * block_width) {
        throw invalid_argument("Invalid segments or binary patterns");
    }

    size_t start;
    size_t end;
    for (size_t i = 0; i < num_blocks; i++) {
        // Each block
        for (size_t j = 0; j < pattern_count; j++) {
            start = segments[i * pattern_count + j];
            if (j + 1 < pattern_count) {
                end = segments[i * pattern_count + j + 1];
            } else {
                end = K;
            }
            // Segmented sum
            for (size_t index = start; index < end; index++) {
                us[i * pattern_count + j] +=
                    v[permutations[i * K + index]];
            }
        }
    }

    vec_t result(num_blocks * block_width);

    // Block product to binary_patterns
    for (size_t i = 0; i < num_blocks; i++) {
        vec_t block(us.begin() + i * pattern_count,
                    us.begin() + (i + 1) * pattern_count);
        vec_t partial_result(block_width, 0.0f);
        vectorMatrixMultiply(block, binary_patterns, partial_result);

        for (int j = 0; j < block_width; j++) {
            result[i * block_width + j] = partial_result[j];
        }
    }
    return result;
}

// Original RSR++ reduction.
vec_t step_three(vec_t u, int block_width) {
    vec_t result(block_width);
    float sum;
    for (int i = block_width; i > 0; i--) {
        sum = 0;
        for (int j = 1; j < pow(2, i); j += 2) {
            sum += u[j];
        }
        result[i - 1] = sum;
        for (int j = 0; j < pow(2, i - 1); j++) {
            u[j] = u[j * 2] + u[j * 2 + 1];
        }
    }

    return result;
}

vec_t rsr_pp_inference(const vec_t& v, const permutation_t& permutations,
                       const segment_t& segments, size_t K, int block_width) {
    if (K == 0 || block_width <= 0 ||
        block_width >= numeric_limits<int>::digits) {
        throw invalid_argument("Invalid inference dimensions");
    }

    if (v.size() != K || permutations.size() % K != 0) {
        throw invalid_argument("Invalid input or permutation size");
    }

    // segmented sums
    const size_t pattern_count = size_t{1} << block_width;
    const size_t num_blocks = permutations.size() / K;
    vector<vec_t> us(num_blocks, vec_t(pattern_count, 0.0f));

    if (segments.size() != num_blocks * pattern_count) {
        throw invalid_argument("Invalid segments");
    }

    size_t start;
    size_t end;
    for (size_t i = 0; i < num_blocks; i++) {
        // Each block
        for (size_t j = 0; j < pattern_count; j++) {
            start = segments[i * pattern_count + j];
            if (j + 1 < pattern_count) {
                end = segments[i * pattern_count + j + 1];
            } else {
                end = K;
            }
            // Segmented sum
            for (size_t index = start; index < end; index++) {
                us[i][j] += v[permutations[i * K + index]];
            }
        }
    }

    vec_t result(num_blocks * block_width);

    // Block product to binary_patterns
    vec_t partial_result;
    for (size_t i = 0; i < us.size(); i++) {
        partial_result = step_three(us[i], block_width);
        for (int j = 0; j < block_width; j++) {
            result[i * block_width + j] = partial_result[j];
        }
    }
    return result;
}

// Optimized RSR kernels adapted from UIC-InDeXLab/RSR-core (MIT).
// Source revision and license are recorded in README.md and licenses/rsr-core.LICENSE.
namespace rsr_core_baseline {
namespace {

#if !defined(TERNARY_BASELINE_DISABLE_AVX2) && (defined(__AVX2__) || defined(__ARM_NEON))
#define RSR_MASK_SIMD 1
// Each aligned row expands one byte into eight FP32 coefficients.
alignas(64) constexpr auto rsr_mask_coeff_lut = [] {
    std::array<std::array<float, 8>, 256> table{};
    for (unsigned mask = 0; mask < table.size(); ++mask)
        for (unsigned lane = 0; lane < 8; ++lane)
            table[mask][lane] = float((mask >> lane) & 1u);
    return table;
}();
#endif

void rsr_gemv_v4_2(
    const int32_t * __restrict__ perms,
    const int32_t * __restrict__ group_ends,
    const int32_t * __restrict__ scatter_offsets,
    const uint8_t * __restrict__ scatter_rows,
    const int32_t * __restrict__ block_meta,
    const float * __restrict__ v,
    float * __restrict__ out,
    int n,
    int k,
    int num_blocks
) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < num_blocks; b++) {
        const int32_t *perm = perms + (size_t)b * n;
        const int g_off = block_meta[b * 2];
        const int n_groups = block_meta[b * 2 + 1];
        float *bout = out + (size_t)b * k;

        memset(bout, 0, (size_t)k * sizeof(float));

        int start = 0;
        for (int g = 0; g < n_groups; g++) {
            const int gg = g_off + g;
            const int end = group_ends[gg];
            const int len = end - start;

            float agg = 0.0f;
            int i = start;

            if (__builtin_expect(len <= 4, 1)) {
                switch (len) {
                    case 4:
                        agg += v[perm[start + 3]];
                    [[fallthrough]];
                    case 3:
                        agg += v[perm[start + 2]];
                    [[fallthrough]];
                    case 2:
                        agg += v[perm[start + 1]];
                    [[fallthrough]];
                    case 1:
                        agg += v[perm[start]];
                    [[fallthrough]];
                    default:
                        break;
                }
            } else {
                [[maybe_unused]] const int pf_dist = 64;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (; i + 7 < end; i += 8) {
#if defined(__AVX2__)
                    if (i + pf_dist + 7 < end) {
                        _mm_prefetch((const char *)&v[perm[i + pf_dist + 0]], _MM_HINT_T0);
                        _mm_prefetch((const char *)&v[perm[i + pf_dist + 2]], _MM_HINT_T0);
                        _mm_prefetch((const char *)&v[perm[i + pf_dist + 4]], _MM_HINT_T0);
                        _mm_prefetch((const char *)&v[perm[i + pf_dist + 6]], _MM_HINT_T0);
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

            const int s_begin = scatter_offsets[gg];
            const int s_end = scatter_offsets[gg + 1];
            for (int s = s_begin; s < s_end; s++) {
                bout[(int)scatter_rows[s]] += agg;
            }

            start = end;
        }
    }
}

void rsr_ternary_gemv_v3_1(
    const uint16_t * __restrict__ perms,
    const uint16_t * __restrict__ group_ends,
    const int32_t  * __restrict__ scatter_offsets,
    const int8_t   * __restrict__ scatter_rows,
    const int8_t   * __restrict__ scatter_signs,
    const int32_t  * __restrict__ block_meta,
    const float    * __restrict__ v,
    float          * __restrict__ out,
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

        memset(bout, 0, (size_t)k * sizeof(float));

        int start = 0;
        for (int g = ge_off; g < ge_end && start < n; g++) {
            const int end = (int)group_ends[g];
            const int len = end - start;
            float agg = 0.0f;

            if (__builtin_expect(len <= 4, 1)) {
                switch (len) {
                    case 4:
                        agg += v[perm[start + 3]];
                    [[fallthrough]];
                    case 3:
                        agg += v[perm[start + 2]];
                    [[fallthrough]];
                    case 2:
                        agg += v[perm[start + 1]];
                    [[fallthrough]];
                    case 1:
                        agg += v[perm[start + 0]];
                    [[fallthrough]];
                    default:
                        break;
                }
            } else {
                int i = start;
                [[maybe_unused]] const int pf_dist = 48;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                for (; i + 7 < end; i += 8) {
#if defined(__AVX2__)
                    if (i + pf_dist + 7 < end) {
                        _mm_prefetch((const char *)&v[perm[i + pf_dist + 0]], _MM_HINT_T0);
                        _mm_prefetch((const char *)&v[perm[i + pf_dist + 2]], _MM_HINT_T0);
                        _mm_prefetch((const char *)&v[perm[i + pf_dist + 4]], _MM_HINT_T0);
                        _mm_prefetch((const char *)&v[perm[i + pf_dist + 6]], _MM_HINT_T0);
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

            const int s_begin = scatter_offsets[g];
            const int s_end = scatter_offsets[g + 1];
            for (int s = s_begin; s < s_end; s++) {
                bout[(int)scatter_rows[s]] += (float)scatter_signs[s] * agg;
            }

            start = end;
        }
    }
}

void rsr_ternary_gemv_v3_3(
    const uint16_t * __restrict__ perms,
    const uint16_t * __restrict__ group_ends,
    const uint16_t * __restrict__ pos_masks,
    const uint16_t * __restrict__ neg_masks,
    const int32_t  * __restrict__ block_meta,
    const float    * __restrict__ v,
    float          * __restrict__ out,
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
                    [[maybe_unused]] const int pf_dist = 48;
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
                const __m256 pos = _mm256_load_ps(rsr_mask_coeff_lut[pos_mask].data());
                const __m256 neg = _mm256_load_ps(rsr_mask_coeff_lut[neg_mask].data());
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
                    [[fallthrough]];
                    case 3:
                        agg += v[perm[start + 2]];
                    [[fallthrough]];
                    case 2:
                        agg += v[perm[start + 1]];
                    [[fallthrough]];
                    case 1:
                        agg += v[perm[start + 0]];
                    [[fallthrough]];
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

// Digit 0 is zero, 1 is positive, and 2 is negative. The least significant
// digit represents the first output column of the block.
enum class Plane { positive, negative, ternary };

int pattern(const tern_t& W, int row, int N, int first, int lanes, Plane plane) {
    int code = 0;
    for (int lane = lanes - 1; lane >= 0; --lane) {
        const int value = W[row * N + first + lane];
        if (plane == Plane::ternary)
            code = 3 * code + (value < 0 ? 2 : value);
        else
            code = 2 * code + (value == (plane == Plane::positive ? 1 : -1));
    }
    return code;
}

int pattern_count(int radix, int width) {
    int count = 1;
    while (width-- > 0) count *= radix;
    return count;
}

template<class T>
std::size_t bytes(const std::vector<T>& v) { return v.size() * sizeof(T); }

// Produce the native kernels' ABI directly from row-major K x N weights.
// Each block's permutation is stable within a pattern.
template<class Index, class Row>
std::size_t build_groups(Groups<Index, Row>& g, const tern_t& W, int K, int N,
                         int width, Plane plane, bool masks = false) {
    const int radix = plane == Plane::ternary ? 3 : 2;
    std::vector<int> counts(pattern_count(radix, width)), cursor(counts.size()), codes(K);
    if (!masks) g.offsets.push_back(0);
    for (int first = 0; first < N; first += width) {
        const int lanes = std::min(width, N - first);
        std::fill(counts.begin(), counts.end(), 0);
        for (int i = 0; i < K; ++i) {
            codes[i] = pattern(W, i, N, first, lanes, plane);
            ++counts[codes[i]];
        }
        const int32_t begin = static_cast<int32_t>(g.ends.size());
        int end = 0;
        for (int code = 0; code < static_cast<int>(counts.size()); ++code) {
            if (!counts[code]) continue;
            cursor[code] = end;
            end += counts[code];
            g.ends.push_back(static_cast<Index>(end));
            uint16_t pos = 0, neg = 0;
            int digits = code;
            for (int lane = 0; digits; ++lane, digits /= radix) {
                const int digit = digits % radix;
                if (!digit) continue;
                if (masks) {
                    (digit == 1 ? pos : neg) |= uint16_t(1u << lane);
                } else {
                    g.rows.push_back(static_cast<Row>(lane));
                    if (radix == 3) g.signs.push_back(digit == 1 ? 1 : -1);
                }
            }
            if (masks) {
                g.positive_masks.push_back(pos);
                g.negative_masks.push_back(neg);
            } else {
                g.offsets.push_back(static_cast<int32_t>(g.rows.size()));
            }
        }
        g.blocks.push_back(begin);
        g.blocks.push_back(static_cast<int32_t>(g.ends.size()) - begin);
        const auto base = g.perm.size();
        g.perm.resize(base + K);
        for (int i = 0; i < K; ++i)
            g.perm[base + cursor[codes[i]]++] = static_cast<Index>(i);
    }
    return bytes(g.perm) + bytes(g.ends) + bytes(g.blocks) + bytes(g.offsets) + bytes(g.rows)
        + bytes(g.signs) + bytes(g.positive_masks) + bytes(g.negative_masks);
}

using BinaryGroups = Groups<int32_t, uint8_t>;

void binary_gemv(const BinaryGroups& g, const float* x, float* y, int K, int width) {
    rsr_gemv_v4_2(g.perm.data(), g.ends.data(), g.offsets.data(), g.rows.data(),
                  g.blocks.data(), x, y, K, width, static_cast<int>(g.blocks.size() / 2));
}

} // namespace

Weights preprocess(const tern_t& W, int K, int N, int width, Variant variant) {
    if (K <= 0 || N <= 0 || width < 1 || width > 8 ||
        W.size() != static_cast<std::size_t>(K) * N)
        throw std::invalid_argument("Invalid RSR-core weight shape or block width");
    const int64_t padded = ((int64_t(N) + width - 1) / width) * width;
    if (padded * K > std::numeric_limits<int>::max() ||
        (variant != Variant::binary_v42 && K > std::numeric_limits<uint16_t>::max()))
        throw std::invalid_argument("RSR-core weight dimensions exceed index range");
    if (std::any_of(W.begin(), W.end(), [](int v) { return v < -1 || v > 1; }))
        throw std::invalid_argument("RSR-core requires ternary weights");

    Weights result{K, N, width, static_cast<int>(padded), variant, {}, {}, {}};
    if (variant == Variant::binary_v42) {
        result.storage_bytes = build_groups(result.positive, W, K, N, width, Plane::positive);
        result.storage_bytes += build_groups(result.negative, W, K, N, width, Plane::negative);
    } else {
        result.storage_bytes = build_groups(result.ternary, W, K, N, width, Plane::ternary,
                                            variant == Variant::ternary_v33);
    }
    return result;
}

const char* backend(Variant variant, int width) {
    if (variant != Variant::ternary_v33) return "scalar";
#if !defined(TERNARY_BASELINE_DISABLE_AVX2) && defined(__AVX2__)
    if (width <= 8) return "avx2";
#elif !defined(TERNARY_BASELINE_DISABLE_AVX2) && defined(__ARM_NEON)
    if (width >= 5 && width <= 8) return "neon";
#else
    (void)width;
#endif
    return "scalar";
}

void multiply(const vec_t& X, const Weights& W, vec_t& Y, int M, vec_t& scratch) {
    if (M <= 0 || X.size() != static_cast<std::size_t>(M) * W.K ||
        &X == &Y || &X == &scratch || &Y == &scratch)
        throw std::invalid_argument("Invalid RSR-core input or aliased output");
    Y.resize(static_cast<std::size_t>(M) * W.N);
    const bool split = W.variant == Variant::binary_v42;
    const bool padded = W.N != W.padded_N;
    scratch.resize(static_cast<std::size_t>(W.padded_N) * (int(split) + int(padded)));
    const auto& g = W.ternary;
    for (int r = 0; r < M; ++r) {
        const float* x = X.data() + static_cast<std::size_t>(r) * W.K;
        float* dst = Y.data() + static_cast<std::size_t>(r) * W.N;
        float* y = padded ? scratch.data() : dst;
        if (split) {
            float* neg = scratch.data() + (padded ? W.padded_N : 0);
            binary_gemv(W.positive, x, y, W.K, W.width);
            binary_gemv(W.negative, x, neg, W.K, W.width);
            for (int c = 0; c < W.N; ++c) dst[c] = y[c] - neg[c];
        } else {
            if (W.variant == Variant::ternary_v33)
                rsr_ternary_gemv_v3_3(g.perm.data(), g.ends.data(), g.positive_masks.data(),
                    g.negative_masks.data(), g.blocks.data(), x, y, W.K, W.width, W.padded_N / W.width);
            else
                rsr_ternary_gemv_v3_1(g.perm.data(), g.ends.data(), g.offsets.data(), g.rows.data(),
                    g.signs.data(), g.blocks.data(), x, y, W.K, W.width, W.padded_N / W.width);
            if (padded) std::copy_n(y, W.N, dst);
        }
    }
}

} // namespace rsr_core_baseline
