// Contains code adapted from fpgasystems/ternaryLLM, commit
// 68d384425261d60bfbde5ff079c33a474c1eb28e.
// Copyright (c) 2025 FPGA @ Systems Group, ETH Zurich.
// Imported portions: MIT; see ../licenses/ternaryllm.LICENSE.
// Source mapping and local changes are documented in ../README.md.

#include "ssr_baseline.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include "ternary_baseline_common.h"

#if TERNARY_BASELINE_HAS_AVX2
#include <immintrin.h>
#endif

namespace ssr_baseline {
namespace {

using namespace std;

// Imported preprocessing and scalar implementation.
int binaryVectorToInt(const vector<int>& values) {
    int value = 0;
    for (int bit : values) value = (value << 1) | bit;
    return value;
}
pair<vector<int>, vector<int>> handle_block_ssr(vector<vector<int>> mat_block) {
    int K = mat_block.size();
    int L = mat_block[0].size();

    // Permutation
    vector<int> offsets(pow(2, L), 0);
    vector<int> permutation(K);
    for (int i = 0; i < K; i++) {
        offsets[binaryVectorToInt(mat_block[i])]++;
        permutation[i] = i;
    }

    sort(permutation.begin(), permutation.end(), [&](int i, int j) {
        return binaryVectorToInt(mat_block[i]) < binaryVectorToInt(mat_block[j]);
        });


    // Segmentation
    vector<int> seg(pow(2, L), 0);
    int total = K;
    seg[seg.size() - 1] = total;
    for (int i = pow(2, L) - 1; i > 0; i--) {
        total = total - offsets[i];
        seg[i - 1] = total;
    }

    return make_pair(permutation, seg);
}

pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>> ssr_preprocess(const vector<vector<int8_t>>& mat, int L, int K, int N) {

    int padding = (L - N % L) % L;
    vector<int> permutations_pos((N + padding) / L * K);
    vector<int> segments_pos((N + padding) / L * pow(2, L));
    vector<int> permutations_neg((N + padding) / L * K);
    vector<int> segments_neg((N + padding) / L * pow(2, L));

    // Blocking
    int start;
    int end;
    vector<vector<int>> block_pos(K, vector<int>(L));
    vector<vector<int>> block_neg(K, vector<int>(L));

    for (int i = 0; i < (N - N % L) / L; i++) {

        start = i * L;
        end = start + L;
        for (int col = start; col < end; col++) {
            for (int row = 0; row < K; row++) {
                block_pos[row][col - start] = max(int(mat[row][col]), 0);
                block_neg[row][col - start] = -min(int(mat[row][col]), 0);
            }
        }
        auto per_seg_pos = handle_block_ssr(block_pos);
        auto per_seg_neg = handle_block_ssr(block_neg);
        for (int j = 0; j < K; j++) {
            permutations_pos[i * K + j] = per_seg_pos.first[j];
            permutations_neg[i * K + j] = per_seg_neg.first[j];
        }
        for (int j = 0; j < pow(2, L); j++) {
            segments_pos[i * pow(2, L) + j] = per_seg_pos.second[j];
            segments_neg[i * pow(2, L) + j] = per_seg_neg.second[j];
        }

    }
    if (N % L != 0) {
        start = (N - N % L);
        end = N;
        for (int col = start; col < end; col++) {
            for (int row = 0; row < K; row++) {
                block_pos[row][col - start] = max(int(mat[row][col]), 0);
                block_neg[row][col - start] = -min(int(mat[row][col]), 0);
            }
        }
        for (int col = end; col < start + L; col++) {
            for (int row = 0; row < K; row++) {
                block_pos[row][col - start] = 0;
                block_neg[row][col - start] = 0;
            }
        }
        auto per_seg_pos = handle_block_ssr(block_pos);
        auto per_seg_neg = handle_block_ssr(block_neg);
        for (int j = 0; j < K; j++) {
            permutations_pos[((N + padding) / L - 1) * K + j] = per_seg_pos.first[j];
            permutations_neg[((N + padding) / L - 1) * K + j] = per_seg_neg.first[j];
        }
        for (int j = 0; j < pow(2, L); j++) {
            segments_pos[((N + padding) / L - 1) * pow(2, L) + j] = per_seg_pos.second[j];
            segments_neg[((N + padding) / L - 1) * pow(2, L) + j] = per_seg_neg.second[j];
        }
    }
    auto pos = make_pair(permutations_pos, segments_pos);
    auto neg = make_pair(permutations_neg, segments_neg);

    return make_pair(pos, neg);
}
template <typename T>
void ssr_inference_rows(const vector<T>& v, const vector<int>& permutations_pos, const vector<int>& segments_pos, const vector<int>& permutations_neg, const vector<int>& segments_neg, int L, int M, int N, int K, vector<T>& result, int first_row) {

    N = N + (L - N % L) % L;
    int block_size = L;
    int num_blocks = N / L;
    int num_bins = 1 << block_size;


#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int row = first_row; row < M; ++row) {
        int output_offset = 0;
        vector<T> seg_sum(2 * (num_bins - 1));
        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {

            // Step 1: Apply permutation
            T sum_pos;
            T sum_neg;
            int len = num_bins - 1;
            int start_pos = segments_pos[b_idx * (num_bins)];
            int start_neg = segments_neg[b_idx * (num_bins)];

            // Step 2: Segmented sum
            for (int bin = 0; bin < len; ++bin) {

                int end_pos = segments_pos[b_idx * num_bins + bin + 1];
                int end_neg = segments_neg[b_idx * num_bins + bin + 1];

                sum_pos = 0;
                sum_neg = 0;
                for (int i = start_pos; i < end_pos; ++i) {
                    sum_pos += v[row * K + permutations_pos[b_idx * K + i]];

                }
                for (int i = start_neg; i < end_neg; ++i) {
                    sum_neg += v[row * K + permutations_neg[b_idx * K + i]];

                }
                seg_sum[bin] = sum_pos;
                seg_sum[num_bins - 1 + bin] = sum_neg;
                start_pos = end_pos;
                start_neg = end_neg;
            }

            //Step 3 other option - small speedup compared to above version RSR++:

            for (int j = block_size - 1; j >= 0; --j) {

                T sum_pos = seg_sum[0];
                T sum_neg = seg_sum[num_bins - 1];
                int new_len = len / 2;
                for (int i = 0; i < new_len; ++i) {
                    sum_pos += seg_sum[2 * i + 2];
                    sum_neg += seg_sum[num_bins - 1 + 2 * i + 2];
                    seg_sum[i] = seg_sum[2 * i + 1] + seg_sum[2 * i + 2];
                    seg_sum[num_bins - 1 + i] = seg_sum[num_bins - 1 + 2 * i + 1] + seg_sum[num_bins - 1 + 2 * i + 2];
                }
                result[row * N + output_offset + j] = sum_pos - sum_neg;
                len = new_len;
            }
            output_offset += block_size;
        }
    }


}
template <typename T>
vector<T> ssr_inference(const vector<T>& v, const vector<int>& permutations_pos,
    const vector<int>& segments_pos, const vector<int>& permutations_neg,
    const vector<int>& segments_neg, int L, int M, int N, int K) {
    N += (L - N % L) % L;
    vector<T> result(M * N);
    ssr_inference_rows(v, permutations_pos, segments_pos, permutations_neg,
                       segments_neg, L, M, N, K, result, 0);
    return result;
}
#if TERNARY_BASELINE_HAS_AVX2
// Imported AVX2 implementations; selected at runtime by the public functions.
struct alignas(32) AvxFloat {
    __m256 value;
    TERNARY_BASELINE_AVX2_TARGET operator __m256() const { return value; }
    TERNARY_BASELINE_AVX2_TARGET AvxFloat& operator=(__m256 other) { value = other; return *this; }
};
TERNARY_BASELINE_AVX2_TARGET
vector<float> ssr_inference_float_vec_unroll16_v2(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int L, int M, int N, int K) {

    N = N + (L - N % L) % L;
    int num_blocks = N / L;
    int num_bins = 1 << L;

    vector<float> result(K * N);


    // Main unrolled loop
    int unrolled_limit = K - (K % (2 * 8));
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int row = 0; row < unrolled_limit; row += 2 * 8) {
        // Working memory
        alignas(32) float gather_buffer[8];
        alignas(32) float temp_result[8];

        // Create completely separate seg_sum arrays for each vector group (like scalar)
        vector<AvxFloat> seg_sum_0(2 * (num_bins - 1));
        vector<AvxFloat> seg_sum_1(2 * (num_bins - 1));

        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {

            // Step 1: Declare variables for all 2 vector groups
            __m256 sum_pos_0, sum_neg_0;
            __m256 sum_pos_1, sum_neg_1;

            int len = num_bins - 1;
            int start_pos = segments_pos[b_idx * num_bins];
            int start_neg = segments_neg[b_idx * num_bins];

            // Step 2: Segmented sum for all vector groups simultaneously
            for (int bin = 0; bin < len; ++bin) {
                int end_pos = segments_pos[b_idx * num_bins + bin + 1];
                int end_neg = segments_neg[b_idx * num_bins + bin + 1];

                sum_pos_0 = _mm256_setzero_ps();
                sum_neg_0 = _mm256_setzero_ps();
                sum_pos_1 = _mm256_setzero_ps();
                sum_neg_1 = _mm256_setzero_ps();

                // Compute sums for all vector groups in parallel
                for (int i = start_pos; i < end_pos; ++i) {
                    int col_idx = permutations_pos[b_idx * M + i];

                    // Vector group 0
                    for (int j = 0; j < 8; ++j) {
                        gather_buffer[j] = v[(row + 0 + j) * M + col_idx];
                    }
                    __m256 vec_0 = _mm256_load_ps(gather_buffer);
                    sum_pos_0 = _mm256_add_ps(sum_pos_0, vec_0);
                    // Vector group 1
                    for (int j = 0; j < 8; ++j) {
                        gather_buffer[j] = v[(row + 8 + j) * M + col_idx];
                    }
                    __m256 vec_1 = _mm256_load_ps(gather_buffer);
                    sum_pos_1 = _mm256_add_ps(sum_pos_1, vec_1);
                }

                for (int i = start_neg; i < end_neg; ++i) {
                    int col_idx = permutations_neg[b_idx * M + i];

                    // Vector group 0
                    for (int j = 0; j < 8; ++j) {
                        gather_buffer[j] = v[(row + 0 + j) * M + col_idx];
                    }
                    __m256 vec_0 = _mm256_load_ps(gather_buffer);
                    sum_neg_0 = _mm256_add_ps(sum_neg_0, vec_0);
                    // Vector group 1
                    for (int j = 0; j < 8; ++j) {
                        gather_buffer[j] = v[(row + 8 + j) * M + col_idx];
                    }
                    __m256 vec_1 = _mm256_load_ps(gather_buffer);
                    sum_neg_1 = _mm256_add_ps(sum_neg_1, vec_1);
                }

                // Store results for all vector groups
                seg_sum_0[bin] = sum_pos_0;
                seg_sum_0[num_bins - 1 + bin] = sum_neg_0;
                seg_sum_1[bin] = sum_pos_1;
                seg_sum_1[num_bins - 1 + bin] = sum_neg_1;

                start_pos = end_pos;
                start_neg = end_neg;
            }

            // Step 3: RSR++ reduction
            int len_0 = len;
            int len_1 = len;

            for (int j = L - 1; j >= 0; --j) {
                // Process vector group 0 with independent length len_0
                __m256 sum_pos_0 = seg_sum_0[0];
                __m256 sum_neg_0 = seg_sum_0[num_bins - 1];

                int new_len_0 = len_0 / 2;
                for (int idx = 0; idx < new_len_0; ++idx) {
                    sum_pos_0 = _mm256_add_ps(sum_pos_0, seg_sum_0[2 * idx + 2]);
                    sum_neg_0 = _mm256_add_ps(sum_neg_0, seg_sum_0[num_bins - 1 + 2 * idx + 2]);
                    seg_sum_0[idx] = _mm256_add_ps(seg_sum_0[2 * idx + 1], seg_sum_0[2 * idx + 2]);
                    seg_sum_0[num_bins - 1 + idx] = _mm256_add_ps(seg_sum_0[num_bins - 1 + 2 * idx + 1], seg_sum_0[num_bins - 1 + 2 * idx + 2]);
                }

                __m256 result_0 = _mm256_sub_ps(sum_pos_0, sum_neg_0);
                _mm256_store_ps(temp_result, result_0);

                // Store final result for vector group 0
                for (int elem = 0; elem < 8; ++elem) {
                    result[(row + 0 + elem) * N + b_idx * L + j] = temp_result[elem];
                }

                len_0 = new_len_0;
                // Process vector group 1 with independent length len_1
                __m256 sum_pos_1 = seg_sum_1[0];
                __m256 sum_neg_1 = seg_sum_1[num_bins - 1];

                int new_len_1 = len_1 / 2;
                for (int idx = 0; idx < new_len_1; ++idx) {
                    sum_pos_1 = _mm256_add_ps(sum_pos_1, seg_sum_1[2 * idx + 2]);
                    sum_neg_1 = _mm256_add_ps(sum_neg_1, seg_sum_1[num_bins - 1 + 2 * idx + 2]);
                    seg_sum_1[idx] = _mm256_add_ps(seg_sum_1[2 * idx + 1], seg_sum_1[2 * idx + 2]);
                    seg_sum_1[num_bins - 1 + idx] = _mm256_add_ps(seg_sum_1[num_bins - 1 + 2 * idx + 1], seg_sum_1[num_bins - 1 + 2 * idx + 2]);
                }

                __m256 result_1 = _mm256_sub_ps(sum_pos_1, sum_neg_1);
                _mm256_store_ps(temp_result, result_1);

                // Store final result for vector group 1
                for (int elem = 0; elem < 8; ++elem) {
                    result[(row + 8 + elem) * N + b_idx * L + j] = temp_result[elem];
                }

                len_1 = new_len_1;
            }
        }
    }

    // Cleanup: Process remaining rows one by one (exactly like scalar version)

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int row = unrolled_limit; row < K; ++row) {
        vector<float> seg_sum(2 * (num_bins - 1));
        int output_offset = 0;

        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {

            float sum_pos, sum_neg;
            int len = num_bins - 1;
            int start_pos = segments_pos[b_idx * num_bins];
            int start_neg = segments_neg[b_idx * num_bins];

            // Segmented sum
            for (int bin = 0; bin < len; ++bin) {
                int end_pos = segments_pos[b_idx * num_bins + bin + 1];
                int end_neg = segments_neg[b_idx * num_bins + bin + 1];

                sum_pos = 0.0f;
                sum_neg = 0.0f;
                for (int i = start_pos; i < end_pos; ++i) {
                    sum_pos += v[row * M + permutations_pos[b_idx * M + i]];
                }
                for (int i = start_neg; i < end_neg; ++i) {
                    sum_neg += v[row * M + permutations_neg[b_idx * M + i]];
                }
                seg_sum[bin] = sum_pos;
                seg_sum[num_bins - 1 + bin] = sum_neg;
                start_pos = end_pos;
                start_neg = end_neg;
            }

            // RSR++ reduction - EXACT same pattern as scalar
            for (int j = L - 1; j >= 0; --j) {
                float sum_pos = seg_sum[0];
                float sum_neg = seg_sum[num_bins - 1];
                int new_len = len / 2;
                for (int i = 0; i < new_len; ++i) {
                    sum_pos += seg_sum[2 * i + 2];
                    sum_neg += seg_sum[num_bins - 1 + 2 * i + 2];
                    seg_sum[i] = seg_sum[2 * i + 1] + seg_sum[2 * i + 2];
                    seg_sum[num_bins - 1 + i] = seg_sum[num_bins - 1 + 2 * i + 1] + seg_sum[num_bins - 1 + 2 * i + 2];
                }
                result[row * N + output_offset + j] = sum_pos - sum_neg;
                len = new_len;
            }

            output_offset += L;
        }
    }

    return result;
}

TERNARY_BASELINE_AVX2_TARGET
vector<float> ssr_inference_float_vec_L2(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M) {

    N = N + (2 - N % 2) % 2;
    int num_bins = 1 << 2;
    int num_blocks = N / 2;

    vector<float> result(M * N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int row = 0; row < M - M % 8; row += 8) {
        const float* v0 = &v[(row + 0) * K];
        const float* v1 = &v[(row + 1) * K];
        const float* v2 = &v[(row + 2) * K];
        const float* v3 = &v[(row + 3) * K];
        const float* v4 = &v[(row + 4) * K];
        const float* v5 = &v[(row + 5) * K];
        const float* v6 = &v[(row + 6) * K];
        const float* v7 = &v[(row + 7) * K];

        int output_offset = 0;

        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {


            // Precomute the boudries in the summation s variables
            // Precomute the boudries for subtraction t varaiables

            const int s0 = segments_pos[b_idx * num_bins + 0], s1 = segments_pos[b_idx * num_bins + 1], s2 = segments_pos[b_idx * num_bins + 2], s3 = segments_pos[b_idx * num_bins + 3];
            const int t0 = segments_neg[b_idx * num_bins + 0], t1 = segments_neg[b_idx * num_bins + 1], t2 = segments_neg[b_idx * num_bins + 2], t3 = segments_neg[b_idx * num_bins + 3];

            // Initialize main accumulators - 2 variables for k=2
            __m256 m0 = _mm256_setzero_ps();
            __m256 m1 = _mm256_setzero_ps();

            // Positive permutation loops
            // s0 - s1: m1
            __m256 acc0 = _mm256_setzero_ps();
            for (int i = s0; i < s1; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0 = _mm256_add_ps(acc0, vec);
            }
            m1 = _mm256_add_ps(m1, acc0);

            // s1 - s2: m0
            __m256 acc1 = _mm256_setzero_ps();
            for (int i = s1; i < s2; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1 = _mm256_add_ps(acc1, vec);
            }
            m0 = _mm256_add_ps(m0, acc1);

            // s2 - s3: m0, m1
            __m256 acc2 = _mm256_setzero_ps();
            for (int i = s2; i < s3; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2 = _mm256_add_ps(acc2, vec);
            }
            m0 = _mm256_add_ps(m0, acc2);
            m1 = _mm256_add_ps(m1, acc2);

            // Negative permutation loops
            // t0 - t1: m1
            __m256 acc0_neg = _mm256_setzero_ps();
            for (int i = t0; i < t1; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0_neg = _mm256_add_ps(acc0_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc0_neg);

            // t1 - t2: m0
            __m256 acc1_neg = _mm256_setzero_ps();
            for (int i = t1; i < t2; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1_neg = _mm256_add_ps(acc1_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc1_neg);

            // t2 - t3: m0, m1
            __m256 acc2_neg = _mm256_setzero_ps();
            for (int i = t2; i < t3; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2_neg = _mm256_add_ps(acc2_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc2_neg);
            m1 = _mm256_sub_ps(m1, acc2_neg);

            // Store results directly - works for any L value
            // Extract individual elements and store to correct positions
            alignas(32) float temp_m0[8], temp_m1[8];
            _mm256_store_ps(temp_m0, m0);
            _mm256_store_ps(temp_m1, m1);

            // Store results for each row (no bounds checking)
            result[(row + 0) * N + output_offset + 0] = temp_m0[0];
            result[(row + 0) * N + output_offset + 1] = temp_m1[0];
            result[(row + 1) * N + output_offset + 0] = temp_m0[1];
            result[(row + 1) * N + output_offset + 1] = temp_m1[1];
            result[(row + 2) * N + output_offset + 0] = temp_m0[2];
            result[(row + 2) * N + output_offset + 1] = temp_m1[2];
            result[(row + 3) * N + output_offset + 0] = temp_m0[3];
            result[(row + 3) * N + output_offset + 1] = temp_m1[3];
            result[(row + 4) * N + output_offset + 0] = temp_m0[4];
            result[(row + 4) * N + output_offset + 1] = temp_m1[4];
            result[(row + 5) * N + output_offset + 0] = temp_m0[5];
            result[(row + 5) * N + output_offset + 1] = temp_m1[5];
            result[(row + 6) * N + output_offset + 0] = temp_m0[6];
            result[(row + 6) * N + output_offset + 1] = temp_m1[6];
            result[(row + 7) * N + output_offset + 0] = temp_m0[7];
            result[(row + 7) * N + output_offset + 1] = temp_m1[7];

            output_offset += 2;

        }
    }

    // Upstream assumes complete eight-row batches. Reuse basic SSR for tails.
    if (M % 8 != 0) {
        ssr_inference_rows(v, permutations_pos, segments_pos, permutations_neg,
                           segments_neg, 2, M, N, K, result, M - M % 8);
    }

    return result;
}

TERNARY_BASELINE_AVX2_TARGET
vector<float> ssr_inference_float_vec_L3(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M){

    N = N + (3 - N % 3) % 3;
    int num_bins = 1 << 3;
    int num_blocks = N / 3;

    vector<float> result(M * N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int row = 0; row < M - M % 8; row += 8) {
        const float* v0 = &v[(row + 0) * K];
        const float* v1 = &v[(row + 1) * K];
        const float* v2 = &v[(row + 2) * K];
        const float* v3 = &v[(row + 3) * K];
        const float* v4 = &v[(row + 4) * K];
        const float* v5 = &v[(row + 5) * K];
        const float* v6 = &v[(row + 6) * K];
        const float* v7 = &v[(row + 7) * K];


        int output_offset = 0;

        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {


            // Precomute the boudries in the summation s variables
            // Precomute the boudries for subtraction t varaiables

            const int s0 = segments_pos[b_idx * num_bins + 0], s1 = segments_pos[b_idx * num_bins + 1], s2 = segments_pos[b_idx * num_bins + 2], s3 = segments_pos[b_idx * num_bins + 3], s4 = segments_pos[b_idx * num_bins + 4], s5 = segments_pos[b_idx * num_bins + 5], s6 = segments_pos[b_idx * num_bins + 6], s7 = segments_pos[b_idx * num_bins + 7];
            const int t0 = segments_neg[b_idx * num_bins + 0], t1 = segments_neg[b_idx * num_bins + 1], t2 = segments_neg[b_idx * num_bins + 2], t3 = segments_neg[b_idx * num_bins + 3], t4 = segments_neg[b_idx * num_bins + 4], t5 = segments_neg[b_idx * num_bins + 5], t6 = segments_neg[b_idx * num_bins + 6], t7 = segments_neg[b_idx * num_bins + 7];

            // Initialize main accumulators - 3 variables for k=3
            __m256 m0 = _mm256_setzero_ps();
            __m256 m1 = _mm256_setzero_ps();
            __m256 m2 = _mm256_setzero_ps();

            // Positive permutation loops
            // s0 - s1: m2
            __m256 acc0 = _mm256_setzero_ps();
            for (int i = s0; i < s1; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0 = _mm256_add_ps(acc0, vec);
            }
            m2 = _mm256_add_ps(m2, acc0);

            // s1 - s2: m1
            __m256 acc1 = _mm256_setzero_ps();
            for (int i = s1; i < s2; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1 = _mm256_add_ps(acc1, vec);
            }
            m1 = _mm256_add_ps(m1, acc1);

            // s2 - s3: m1, m2
            __m256 acc2 = _mm256_setzero_ps();
            for (int i = s2; i < s3; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2 = _mm256_add_ps(acc2, vec);
            }
            m1 = _mm256_add_ps(m1, acc2);
            m2 = _mm256_add_ps(m2, acc2);

            // s3 - s4: m0
            __m256 acc3 = _mm256_setzero_ps();
            for (int i = s3; i < s4; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc3 = _mm256_add_ps(acc3, vec);
            }
            m0 = _mm256_add_ps(m0, acc3);

            // s4 - s5: m0, m2
            __m256 acc4 = _mm256_setzero_ps();
            for (int i = s4; i < s5; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc4 = _mm256_add_ps(acc4, vec);
            }
            m0 = _mm256_add_ps(m0, acc4);
            m2 = _mm256_add_ps(m2, acc4);

            // s5 - s6: m0, m1
            __m256 acc5 = _mm256_setzero_ps();
            for (int i = s5; i < s6; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc5 = _mm256_add_ps(acc5, vec);
            }
            m0 = _mm256_add_ps(m0, acc5);
            m1 = _mm256_add_ps(m1, acc5);

            // s6 - s7: m0, m1, m2
            __m256 acc6 = _mm256_setzero_ps();
            for (int i = s6; i < s7; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc6 = _mm256_add_ps(acc6, vec);
            }
            m0 = _mm256_add_ps(m0, acc6);
            m1 = _mm256_add_ps(m1, acc6);
            m2 = _mm256_add_ps(m2, acc6);

            // Negative permutation loops
            // t0 - t1: m2
            __m256 acc0_neg = _mm256_setzero_ps();
            for (int i = t0; i < t1; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0_neg = _mm256_add_ps(acc0_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc0_neg);

            // t1 - t2: m1
            __m256 acc1_neg = _mm256_setzero_ps();
            for (int i = t1; i < t2; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1_neg = _mm256_add_ps(acc1_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc1_neg);

            // t2 - t3: m1, m2
            __m256 acc2_neg = _mm256_setzero_ps();
            for (int i = t2; i < t3; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2_neg = _mm256_add_ps(acc2_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc2_neg);
            m2 = _mm256_sub_ps(m2, acc2_neg);

            // t3 - t4: m0
            __m256 acc3_neg = _mm256_setzero_ps();
            for (int i = t3; i < t4; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc3_neg = _mm256_add_ps(acc3_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc3_neg);

            // t4 - t5: m0, m2
            __m256 acc4_neg = _mm256_setzero_ps();
            for (int i = t4; i < t5; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc4_neg = _mm256_add_ps(acc4_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc4_neg);
            m2 = _mm256_sub_ps(m2, acc4_neg);

            // t5 - t6: m0, m1
            __m256 acc5_neg = _mm256_setzero_ps();
            for (int i = t5; i < t6; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc5_neg = _mm256_add_ps(acc5_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc5_neg);
            m1 = _mm256_sub_ps(m1, acc5_neg);

            // t6 - t7: m0, m1, m2
            __m256 acc6_neg = _mm256_setzero_ps();
            for (int i = t6; i < t7; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc6_neg = _mm256_add_ps(acc6_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc6_neg);
            m1 = _mm256_sub_ps(m1, acc6_neg);
            m2 = _mm256_sub_ps(m2, acc6_neg);

            // Store results directly - works for any L value
            // Extract individual elements and store to correct positions
            alignas(32) float temp_m0[8], temp_m1[8], temp_m2[8];
            _mm256_store_ps(temp_m0, m0);
            _mm256_store_ps(temp_m1, m1);
            _mm256_store_ps(temp_m2, m2);

            // Store results for each row (no bounds checking)
            result[(row + 0) * N + output_offset + 0] = temp_m0[0];
            result[(row + 0) * N + output_offset + 1] = temp_m1[0];
            result[(row + 0) * N + output_offset + 2] = temp_m2[0];
            result[(row + 1) * N + output_offset + 0] = temp_m0[1];
            result[(row + 1) * N + output_offset + 1] = temp_m1[1];
            result[(row + 1) * N + output_offset + 2] = temp_m2[1];
            result[(row + 2) * N + output_offset + 0] = temp_m0[2];
            result[(row + 2) * N + output_offset + 1] = temp_m1[2];
            result[(row + 2) * N + output_offset + 2] = temp_m2[2];
            result[(row + 3) * N + output_offset + 0] = temp_m0[3];
            result[(row + 3) * N + output_offset + 1] = temp_m1[3];
            result[(row + 3) * N + output_offset + 2] = temp_m2[3];
            result[(row + 4) * N + output_offset + 0] = temp_m0[4];
            result[(row + 4) * N + output_offset + 1] = temp_m1[4];
            result[(row + 4) * N + output_offset + 2] = temp_m2[4];
            result[(row + 5) * N + output_offset + 0] = temp_m0[5];
            result[(row + 5) * N + output_offset + 1] = temp_m1[5];
            result[(row + 5) * N + output_offset + 2] = temp_m2[5];
            result[(row + 6) * N + output_offset + 0] = temp_m0[6];
            result[(row + 6) * N + output_offset + 1] = temp_m1[6];
            result[(row + 6) * N + output_offset + 2] = temp_m2[6];
            result[(row + 7) * N + output_offset + 0] = temp_m0[7];
            result[(row + 7) * N + output_offset + 1] = temp_m1[7];
            result[(row + 7) * N + output_offset + 2] = temp_m2[7];

            output_offset += 3;

        }
    }

    // Upstream assumes complete eight-row batches. Reuse basic SSR for tails.
    if (M % 8 != 0) {
        ssr_inference_rows(v, permutations_pos, segments_pos, permutations_neg,
                           segments_neg, 3, M, N, K, result, M - M % 8);
    }

    return result;
}

TERNARY_BASELINE_AVX2_TARGET
vector<float> ssr_inference_float_vec_L4(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M){

    N = N + (4 - N % 4) % 4;
    int num_bins = 1 << 4;
    int num_blocks = N / 4;

    vector<float> result(M * N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int row = 0; row < M - M % 8; row += 8) {
        const float* v0 = &v[(row + 0) * K];
        const float* v1 = &v[(row + 1) * K];
        const float* v2 = &v[(row + 2) * K];
        const float* v3 = &v[(row + 3) * K];
        const float* v4 = &v[(row + 4) * K];
        const float* v5 = &v[(row + 5) * K];
        const float* v6 = &v[(row + 6) * K];
        const float* v7 = &v[(row + 7) * K];


        int output_offset = 0;

        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {


            // Precomute the boudries in the summation s variables
            // Precomute the boudries for subtraction t varaiables

            const int s0 = segments_pos[b_idx * num_bins + 0], s1 = segments_pos[b_idx * num_bins + 1], s2 = segments_pos[b_idx * num_bins + 2], s3 = segments_pos[b_idx * num_bins + 3], s4 = segments_pos[b_idx * num_bins + 4], s5 = segments_pos[b_idx * num_bins + 5], s6 = segments_pos[b_idx * num_bins + 6], s7 = segments_pos[b_idx * num_bins + 7], s8 = segments_pos[b_idx * num_bins + 8], s9 = segments_pos[b_idx * num_bins + 9], s10 = segments_pos[b_idx * num_bins + 10], s11 = segments_pos[b_idx * num_bins + 11], s12 = segments_pos[b_idx * num_bins + 12], s13 = segments_pos[b_idx * num_bins + 13], s14 = segments_pos[b_idx * num_bins + 14], s15 = segments_pos[b_idx * num_bins + 15];
            const int t0 = segments_neg[b_idx * num_bins + 0], t1 = segments_neg[b_idx * num_bins + 1], t2 = segments_neg[b_idx * num_bins + 2], t3 = segments_neg[b_idx * num_bins + 3], t4 = segments_neg[b_idx * num_bins + 4], t5 = segments_neg[b_idx * num_bins + 5], t6 = segments_neg[b_idx * num_bins + 6], t7 = segments_neg[b_idx * num_bins + 7], t8 = segments_neg[b_idx * num_bins + 8], t9 = segments_neg[b_idx * num_bins + 9], t10 = segments_neg[b_idx * num_bins + 10], t11 = segments_neg[b_idx * num_bins + 11], t12 = segments_neg[b_idx * num_bins + 12], t13 = segments_neg[b_idx * num_bins + 13], t14 = segments_neg[b_idx * num_bins + 14], t15 = segments_neg[b_idx * num_bins + 15];

            // Initialize main accumulators - 4 variables for k=4
            __m256 m0 = _mm256_setzero_ps();
            __m256 m1 = _mm256_setzero_ps();
            __m256 m2 = _mm256_setzero_ps();
            __m256 m3 = _mm256_setzero_ps();

            // Positive permutation loops
            // s0 - s1: m3
            __m256 acc0 = _mm256_setzero_ps();
            for (int i = s0; i < s1; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0 = _mm256_add_ps(acc0, vec);
            }
            m3 = _mm256_add_ps(m3, acc0);

            // s1 - s2: m2
            __m256 acc1 = _mm256_setzero_ps();
            for (int i = s1; i < s2; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1 = _mm256_add_ps(acc1, vec);
            }
            m2 = _mm256_add_ps(m2, acc1);

            // s2 - s3: m2, m3
            __m256 acc2 = _mm256_setzero_ps();
            for (int i = s2; i < s3; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2 = _mm256_add_ps(acc2, vec);
            }
            m2 = _mm256_add_ps(m2, acc2);
            m3 = _mm256_add_ps(m3, acc2);

            // s3 - s4: m1
            __m256 acc3 = _mm256_setzero_ps();
            for (int i = s3; i < s4; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc3 = _mm256_add_ps(acc3, vec);
            }
            m1 = _mm256_add_ps(m1, acc3);

            // s4 - s5: m1, m3
            __m256 acc4 = _mm256_setzero_ps();
            for (int i = s4; i < s5; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc4 = _mm256_add_ps(acc4, vec);
            }
            m1 = _mm256_add_ps(m1, acc4);
            m3 = _mm256_add_ps(m3, acc4);

            // s5 - s6: m1, m2
            __m256 acc5 = _mm256_setzero_ps();
            for (int i = s5; i < s6; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc5 = _mm256_add_ps(acc5, vec);
            }
            m1 = _mm256_add_ps(m1, acc5);
            m2 = _mm256_add_ps(m2, acc5);

            // s6 - s7: m1, m2, m3
            __m256 acc6 = _mm256_setzero_ps();
            for (int i = s6; i < s7; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc6 = _mm256_add_ps(acc6, vec);
            }
            m1 = _mm256_add_ps(m1, acc6);
            m2 = _mm256_add_ps(m2, acc6);
            m3 = _mm256_add_ps(m3, acc6);

            // s7 - s8: m0
            __m256 acc7 = _mm256_setzero_ps();
            for (int i = s7; i < s8; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc7 = _mm256_add_ps(acc7, vec);
            }
            m0 = _mm256_add_ps(m0, acc7);

            // s8 - s9: m0, m3
            __m256 acc8 = _mm256_setzero_ps();
            for (int i = s8; i < s9; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc8 = _mm256_add_ps(acc8, vec);
            }
            m0 = _mm256_add_ps(m0, acc8);
            m3 = _mm256_add_ps(m3, acc8);

            // s9 - s10: m0, m2
            __m256 acc9 = _mm256_setzero_ps();
            for (int i = s9; i < s10; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc9 = _mm256_add_ps(acc9, vec);
            }
            m0 = _mm256_add_ps(m0, acc9);
            m2 = _mm256_add_ps(m2, acc9);

            // s10 - s11: m0, m2, m3
            __m256 acc10 = _mm256_setzero_ps();
            for (int i = s10; i < s11; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc10 = _mm256_add_ps(acc10, vec);
            }
            m0 = _mm256_add_ps(m0, acc10);
            m2 = _mm256_add_ps(m2, acc10);
            m3 = _mm256_add_ps(m3, acc10);

            // s11 - s12: m0, m1
            __m256 acc11 = _mm256_setzero_ps();
            for (int i = s11; i < s12; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc11 = _mm256_add_ps(acc11, vec);
            }
            m0 = _mm256_add_ps(m0, acc11);
            m1 = _mm256_add_ps(m1, acc11);

            // s12 - s13: m0, m1, m3
            __m256 acc12 = _mm256_setzero_ps();
            for (int i = s12; i < s13; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc12 = _mm256_add_ps(acc12, vec);
            }
            m0 = _mm256_add_ps(m0, acc12);
            m1 = _mm256_add_ps(m1, acc12);
            m3 = _mm256_add_ps(m3, acc12);

            // s13 - s14: m0, m1, m2
            __m256 acc13 = _mm256_setzero_ps();
            for (int i = s13; i < s14; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc13 = _mm256_add_ps(acc13, vec);
            }
            m0 = _mm256_add_ps(m0, acc13);
            m1 = _mm256_add_ps(m1, acc13);
            m2 = _mm256_add_ps(m2, acc13);

            // s14 - s15: m0, m1, m2, m3
            __m256 acc14 = _mm256_setzero_ps();
            for (int i = s14; i < s15; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc14 = _mm256_add_ps(acc14, vec);
            }
            m0 = _mm256_add_ps(m0, acc14);
            m1 = _mm256_add_ps(m1, acc14);
            m2 = _mm256_add_ps(m2, acc14);
            m3 = _mm256_add_ps(m3, acc14);

            // Negative permutation loops
            // t0 - t1: m3
            __m256 acc0_neg = _mm256_setzero_ps();
            for (int i = t0; i < t1; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0_neg = _mm256_add_ps(acc0_neg, vec);
            }
            m3 = _mm256_sub_ps(m3, acc0_neg);

            // t1 - t2: m2
            __m256 acc1_neg = _mm256_setzero_ps();
            for (int i = t1; i < t2; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1_neg = _mm256_add_ps(acc1_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc1_neg);

            // t2 - t3: m2, m3
            __m256 acc2_neg = _mm256_setzero_ps();
            for (int i = t2; i < t3; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2_neg = _mm256_add_ps(acc2_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc2_neg);
            m3 = _mm256_sub_ps(m3, acc2_neg);

            // t3 - t4: m1
            __m256 acc3_neg = _mm256_setzero_ps();
            for (int i = t3; i < t4; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc3_neg = _mm256_add_ps(acc3_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc3_neg);

            // t4 - t5: m1, m3
            __m256 acc4_neg = _mm256_setzero_ps();
            for (int i = t4; i < t5; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc4_neg = _mm256_add_ps(acc4_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc4_neg);
            m3 = _mm256_sub_ps(m3, acc4_neg);

            // t5 - t6: m1, m2
            __m256 acc5_neg = _mm256_setzero_ps();
            for (int i = t5; i < t6; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc5_neg = _mm256_add_ps(acc5_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc5_neg);
            m2 = _mm256_sub_ps(m2, acc5_neg);

            // t6 - t7: m1, m2, m3
            __m256 acc6_neg = _mm256_setzero_ps();
            for (int i = t6; i < t7; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc6_neg = _mm256_add_ps(acc6_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc6_neg);
            m2 = _mm256_sub_ps(m2, acc6_neg);
            m3 = _mm256_sub_ps(m3, acc6_neg);

            // t7 - t8: m0
            __m256 acc7_neg = _mm256_setzero_ps();
            for (int i = t7; i < t8; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc7_neg = _mm256_add_ps(acc7_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc7_neg);

            // t8 - t9: m0, m3
            __m256 acc8_neg = _mm256_setzero_ps();
            for (int i = t8; i < t9; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc8_neg = _mm256_add_ps(acc8_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc8_neg);
            m3 = _mm256_sub_ps(m3, acc8_neg);

            // t9 - t10: m0, m2
            __m256 acc9_neg = _mm256_setzero_ps();
            for (int i = t9; i < t10; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc9_neg = _mm256_add_ps(acc9_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc9_neg);
            m2 = _mm256_sub_ps(m2, acc9_neg);

            // t10 - t11: m0, m2, m3
            __m256 acc10_neg = _mm256_setzero_ps();
            for (int i = t10; i < t11; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc10_neg = _mm256_add_ps(acc10_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc10_neg);
            m2 = _mm256_sub_ps(m2, acc10_neg);
            m3 = _mm256_sub_ps(m3, acc10_neg);

            // t11 - t12: m0, m1
            __m256 acc11_neg = _mm256_setzero_ps();
            for (int i = t11; i < t12; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc11_neg = _mm256_add_ps(acc11_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc11_neg);
            m1 = _mm256_sub_ps(m1, acc11_neg);

            // t12 - t13: m0, m1, m3
            __m256 acc12_neg = _mm256_setzero_ps();
            for (int i = t12; i < t13; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc12_neg = _mm256_add_ps(acc12_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc12_neg);
            m1 = _mm256_sub_ps(m1, acc12_neg);
            m3 = _mm256_sub_ps(m3, acc12_neg);

            // t13 - t14: m0, m1, m2
            __m256 acc13_neg = _mm256_setzero_ps();
            for (int i = t13; i < t14; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc13_neg = _mm256_add_ps(acc13_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc13_neg);
            m1 = _mm256_sub_ps(m1, acc13_neg);
            m2 = _mm256_sub_ps(m2, acc13_neg);

            // t14 - t15: m0, m1, m2, m3
            __m256 acc14_neg = _mm256_setzero_ps();
            for (int i = t14; i < t15; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc14_neg = _mm256_add_ps(acc14_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc14_neg);
            m1 = _mm256_sub_ps(m1, acc14_neg);
            m2 = _mm256_sub_ps(m2, acc14_neg);
            m3 = _mm256_sub_ps(m3, acc14_neg);

            // Store results directly - works for any L value
            // Extract individual elements and store to correct positions
            alignas(32) float temp_m0[8], temp_m1[8], temp_m2[8], temp_m3[8];
            _mm256_store_ps(temp_m0, m0);
            _mm256_store_ps(temp_m1, m1);
            _mm256_store_ps(temp_m2, m2);
            _mm256_store_ps(temp_m3, m3);

            // Store results for each row (no bounds checking)
            result[(row + 0) * N + output_offset + 0] = temp_m0[0];
            result[(row + 0) * N + output_offset + 1] = temp_m1[0];
            result[(row + 0) * N + output_offset + 2] = temp_m2[0];
            result[(row + 0) * N + output_offset + 3] = temp_m3[0];
            result[(row + 1) * N + output_offset + 0] = temp_m0[1];
            result[(row + 1) * N + output_offset + 1] = temp_m1[1];
            result[(row + 1) * N + output_offset + 2] = temp_m2[1];
            result[(row + 1) * N + output_offset + 3] = temp_m3[1];
            result[(row + 2) * N + output_offset + 0] = temp_m0[2];
            result[(row + 2) * N + output_offset + 1] = temp_m1[2];
            result[(row + 2) * N + output_offset + 2] = temp_m2[2];
            result[(row + 2) * N + output_offset + 3] = temp_m3[2];
            result[(row + 3) * N + output_offset + 0] = temp_m0[3];
            result[(row + 3) * N + output_offset + 1] = temp_m1[3];
            result[(row + 3) * N + output_offset + 2] = temp_m2[3];
            result[(row + 3) * N + output_offset + 3] = temp_m3[3];
            result[(row + 4) * N + output_offset + 0] = temp_m0[4];
            result[(row + 4) * N + output_offset + 1] = temp_m1[4];
            result[(row + 4) * N + output_offset + 2] = temp_m2[4];
            result[(row + 4) * N + output_offset + 3] = temp_m3[4];
            result[(row + 5) * N + output_offset + 0] = temp_m0[5];
            result[(row + 5) * N + output_offset + 1] = temp_m1[5];
            result[(row + 5) * N + output_offset + 2] = temp_m2[5];
            result[(row + 5) * N + output_offset + 3] = temp_m3[5];
            result[(row + 6) * N + output_offset + 0] = temp_m0[6];
            result[(row + 6) * N + output_offset + 1] = temp_m1[6];
            result[(row + 6) * N + output_offset + 2] = temp_m2[6];
            result[(row + 6) * N + output_offset + 3] = temp_m3[6];
            result[(row + 7) * N + output_offset + 0] = temp_m0[7];
            result[(row + 7) * N + output_offset + 1] = temp_m1[7];
            result[(row + 7) * N + output_offset + 2] = temp_m2[7];
            result[(row + 7) * N + output_offset + 3] = temp_m3[7];

            output_offset += 4;

        }
    }

    // Upstream assumes complete eight-row batches. Reuse basic SSR for tails.
    if (M % 8 != 0) {
        ssr_inference_rows(v, permutations_pos, segments_pos, permutations_neg,
                           segments_neg, 4, M, N, K, result, M - M % 8);
    }

    return result;
}

TERNARY_BASELINE_AVX2_TARGET
vector<float> ssr_inference_float_vec_L5(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M){

    N = N + (5 - N % 5) % 5;
    int num_bins = 1 << 5;
    int num_blocks = N / 5;

    vector<float> result(M * N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int row = 0; row < M - M % 8; row += 8) {
        const float* v0 = &v[(row + 0) * K];
        const float* v1 = &v[(row + 1) * K];
        const float* v2 = &v[(row + 2) * K];
        const float* v3 = &v[(row + 3) * K];
        const float* v4 = &v[(row + 4) * K];
        const float* v5 = &v[(row + 5) * K];
        const float* v6 = &v[(row + 6) * K];
        const float* v7 = &v[(row + 7) * K];


        int output_offset = 0;

        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {


            // Precomute the boudries in the summation s variables
            // Precomute the boudries for subtraction t varaiables

            const int s0 = segments_pos[b_idx * num_bins + 0], s1 = segments_pos[b_idx * num_bins + 1], s2 = segments_pos[b_idx * num_bins + 2], s3 = segments_pos[b_idx * num_bins + 3], s4 = segments_pos[b_idx * num_bins + 4], s5 = segments_pos[b_idx * num_bins + 5], s6 = segments_pos[b_idx * num_bins + 6], s7 = segments_pos[b_idx * num_bins + 7], s8 = segments_pos[b_idx * num_bins + 8], s9 = segments_pos[b_idx * num_bins + 9], s10 = segments_pos[b_idx * num_bins + 10], s11 = segments_pos[b_idx * num_bins + 11], s12 = segments_pos[b_idx * num_bins + 12], s13 = segments_pos[b_idx * num_bins + 13], s14 = segments_pos[b_idx * num_bins + 14], s15 = segments_pos[b_idx * num_bins + 15], s16 = segments_pos[b_idx * num_bins + 16], s17 = segments_pos[b_idx * num_bins + 17], s18 = segments_pos[b_idx * num_bins + 18], s19 = segments_pos[b_idx * num_bins + 19], s20 = segments_pos[b_idx * num_bins + 20], s21 = segments_pos[b_idx * num_bins + 21], s22 = segments_pos[b_idx * num_bins + 22], s23 = segments_pos[b_idx * num_bins + 23], s24 = segments_pos[b_idx * num_bins + 24], s25 = segments_pos[b_idx * num_bins + 25], s26 = segments_pos[b_idx * num_bins + 26], s27 = segments_pos[b_idx * num_bins + 27], s28 = segments_pos[b_idx * num_bins + 28], s29 = segments_pos[b_idx * num_bins + 29], s30 = segments_pos[b_idx * num_bins + 30], s31 = segments_pos[b_idx * num_bins + 31];
            const int t0 = segments_neg[b_idx * num_bins + 0], t1 = segments_neg[b_idx * num_bins + 1], t2 = segments_neg[b_idx * num_bins + 2], t3 = segments_neg[b_idx * num_bins + 3], t4 = segments_neg[b_idx * num_bins + 4], t5 = segments_neg[b_idx * num_bins + 5], t6 = segments_neg[b_idx * num_bins + 6], t7 = segments_neg[b_idx * num_bins + 7], t8 = segments_neg[b_idx * num_bins + 8], t9 = segments_neg[b_idx * num_bins + 9], t10 = segments_neg[b_idx * num_bins + 10], t11 = segments_neg[b_idx * num_bins + 11], t12 = segments_neg[b_idx * num_bins + 12], t13 = segments_neg[b_idx * num_bins + 13], t14 = segments_neg[b_idx * num_bins + 14], t15 = segments_neg[b_idx * num_bins + 15], t16 = segments_neg[b_idx * num_bins + 16], t17 = segments_neg[b_idx * num_bins + 17], t18 = segments_neg[b_idx * num_bins + 18], t19 = segments_neg[b_idx * num_bins + 19], t20 = segments_neg[b_idx * num_bins + 20], t21 = segments_neg[b_idx * num_bins + 21], t22 = segments_neg[b_idx * num_bins + 22], t23 = segments_neg[b_idx * num_bins + 23], t24 = segments_neg[b_idx * num_bins + 24], t25 = segments_neg[b_idx * num_bins + 25], t26 = segments_neg[b_idx * num_bins + 26], t27 = segments_neg[b_idx * num_bins + 27], t28 = segments_neg[b_idx * num_bins + 28], t29 = segments_neg[b_idx * num_bins + 29], t30 = segments_neg[b_idx * num_bins + 30], t31 = segments_neg[b_idx * num_bins + 31];

            // Initialize main accumulators - 5 variables for k=5
            __m256 m0 = _mm256_setzero_ps();
            __m256 m1 = _mm256_setzero_ps();
            __m256 m2 = _mm256_setzero_ps();
            __m256 m3 = _mm256_setzero_ps();
            __m256 m4 = _mm256_setzero_ps();

            // Positive permutation loops
            // s0 - s1: m4
            __m256 acc0 = _mm256_setzero_ps();
            for (int i = s0; i < s1; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0 = _mm256_add_ps(acc0, vec);
            }
            m4 = _mm256_add_ps(m4, acc0);

            // s1 - s2: m3
            __m256 acc1 = _mm256_setzero_ps();
            for (int i = s1; i < s2; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1 = _mm256_add_ps(acc1, vec);
            }
            m3 = _mm256_add_ps(m3, acc1);

            // s2 - s3: m3, m4
            __m256 acc2 = _mm256_setzero_ps();
            for (int i = s2; i < s3; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2 = _mm256_add_ps(acc2, vec);
            }
            m3 = _mm256_add_ps(m3, acc2);
            m4 = _mm256_add_ps(m4, acc2);

            // s3 - s4: m2
            __m256 acc3 = _mm256_setzero_ps();
            for (int i = s3; i < s4; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc3 = _mm256_add_ps(acc3, vec);
            }
            m2 = _mm256_add_ps(m2, acc3);

            // s4 - s5: m2, m4
            __m256 acc4 = _mm256_setzero_ps();
            for (int i = s4; i < s5; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc4 = _mm256_add_ps(acc4, vec);
            }
            m2 = _mm256_add_ps(m2, acc4);
            m4 = _mm256_add_ps(m4, acc4);

            // s5 - s6: m2, m3
            __m256 acc5 = _mm256_setzero_ps();
            for (int i = s5; i < s6; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc5 = _mm256_add_ps(acc5, vec);
            }
            m2 = _mm256_add_ps(m2, acc5);
            m3 = _mm256_add_ps(m3, acc5);

            // s6 - s7: m2, m3, m4
            __m256 acc6 = _mm256_setzero_ps();
            for (int i = s6; i < s7; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc6 = _mm256_add_ps(acc6, vec);
            }
            m2 = _mm256_add_ps(m2, acc6);
            m3 = _mm256_add_ps(m3, acc6);
            m4 = _mm256_add_ps(m4, acc6);

            // s7 - s8: m1
            __m256 acc7 = _mm256_setzero_ps();
            for (int i = s7; i < s8; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc7 = _mm256_add_ps(acc7, vec);
            }
            m1 = _mm256_add_ps(m1, acc7);

            // s8 - s9: m1, m4
            __m256 acc8 = _mm256_setzero_ps();
            for (int i = s8; i < s9; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc8 = _mm256_add_ps(acc8, vec);
            }
            m1 = _mm256_add_ps(m1, acc8);
            m4 = _mm256_add_ps(m4, acc8);

            // s9 - s10: m1, m3
            __m256 acc9 = _mm256_setzero_ps();
            for (int i = s9; i < s10; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc9 = _mm256_add_ps(acc9, vec);
            }
            m1 = _mm256_add_ps(m1, acc9);
            m3 = _mm256_add_ps(m3, acc9);

            // s10 - s11: m1, m3, m4
            __m256 acc10 = _mm256_setzero_ps();
            for (int i = s10; i < s11; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc10 = _mm256_add_ps(acc10, vec);
            }
            m1 = _mm256_add_ps(m1, acc10);
            m3 = _mm256_add_ps(m3, acc10);
            m4 = _mm256_add_ps(m4, acc10);

            // s11 - s12: m1, m2
            __m256 acc11 = _mm256_setzero_ps();
            for (int i = s11; i < s12; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc11 = _mm256_add_ps(acc11, vec);
            }
            m1 = _mm256_add_ps(m1, acc11);
            m2 = _mm256_add_ps(m2, acc11);

            // s12 - s13: m1, m2, m4
            __m256 acc12 = _mm256_setzero_ps();
            for (int i = s12; i < s13; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc12 = _mm256_add_ps(acc12, vec);
            }
            m1 = _mm256_add_ps(m1, acc12);
            m2 = _mm256_add_ps(m2, acc12);
            m4 = _mm256_add_ps(m4, acc12);

            // s13 - s14: m1, m2, m3
            __m256 acc13 = _mm256_setzero_ps();
            for (int i = s13; i < s14; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc13 = _mm256_add_ps(acc13, vec);
            }
            m1 = _mm256_add_ps(m1, acc13);
            m2 = _mm256_add_ps(m2, acc13);
            m3 = _mm256_add_ps(m3, acc13);

            // s14 - s15: m1, m2, m3, m4
            __m256 acc14 = _mm256_setzero_ps();
            for (int i = s14; i < s15; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc14 = _mm256_add_ps(acc14, vec);
            }
            m1 = _mm256_add_ps(m1, acc14);
            m2 = _mm256_add_ps(m2, acc14);
            m3 = _mm256_add_ps(m3, acc14);
            m4 = _mm256_add_ps(m4, acc14);

            // s15 - s16: m0
            __m256 acc15 = _mm256_setzero_ps();
            for (int i = s15; i < s16; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc15 = _mm256_add_ps(acc15, vec);
            }
            m0 = _mm256_add_ps(m0, acc15);

            // s16 - s17: m0, m4
            __m256 acc16 = _mm256_setzero_ps();
            for (int i = s16; i < s17; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc16 = _mm256_add_ps(acc16, vec);
            }
            m0 = _mm256_add_ps(m0, acc16);
            m4 = _mm256_add_ps(m4, acc16);

            // s17 - s18: m0, m3
            __m256 acc17 = _mm256_setzero_ps();
            for (int i = s17; i < s18; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc17 = _mm256_add_ps(acc17, vec);
            }
            m0 = _mm256_add_ps(m0, acc17);
            m3 = _mm256_add_ps(m3, acc17);

            // s18 - s19: m0, m3, m4
            __m256 acc18 = _mm256_setzero_ps();
            for (int i = s18; i < s19; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc18 = _mm256_add_ps(acc18, vec);
            }
            m0 = _mm256_add_ps(m0, acc18);
            m3 = _mm256_add_ps(m3, acc18);
            m4 = _mm256_add_ps(m4, acc18);

            // s19 - s20: m0, m2
            __m256 acc19 = _mm256_setzero_ps();
            for (int i = s19; i < s20; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc19 = _mm256_add_ps(acc19, vec);
            }
            m0 = _mm256_add_ps(m0, acc19);
            m2 = _mm256_add_ps(m2, acc19);

            // s20 - s21: m0, m2, m4
            __m256 acc20 = _mm256_setzero_ps();
            for (int i = s20; i < s21; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc20 = _mm256_add_ps(acc20, vec);
            }
            m0 = _mm256_add_ps(m0, acc20);
            m2 = _mm256_add_ps(m2, acc20);
            m4 = _mm256_add_ps(m4, acc20);

            // s21 - s22: m0, m2, m3
            __m256 acc21 = _mm256_setzero_ps();
            for (int i = s21; i < s22; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc21 = _mm256_add_ps(acc21, vec);
            }
            m0 = _mm256_add_ps(m0, acc21);
            m2 = _mm256_add_ps(m2, acc21);
            m3 = _mm256_add_ps(m3, acc21);

            // s22 - s23: m0, m2, m3, m4
            __m256 acc22 = _mm256_setzero_ps();
            for (int i = s22; i < s23; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc22 = _mm256_add_ps(acc22, vec);
            }
            m0 = _mm256_add_ps(m0, acc22);
            m2 = _mm256_add_ps(m2, acc22);
            m3 = _mm256_add_ps(m3, acc22);
            m4 = _mm256_add_ps(m4, acc22);

            // s23 - s24: m0, m1
            __m256 acc23 = _mm256_setzero_ps();
            for (int i = s23; i < s24; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc23 = _mm256_add_ps(acc23, vec);
            }
            m0 = _mm256_add_ps(m0, acc23);
            m1 = _mm256_add_ps(m1, acc23);

            // s24 - s25: m0, m1, m4
            __m256 acc24 = _mm256_setzero_ps();
            for (int i = s24; i < s25; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc24 = _mm256_add_ps(acc24, vec);
            }
            m0 = _mm256_add_ps(m0, acc24);
            m1 = _mm256_add_ps(m1, acc24);
            m4 = _mm256_add_ps(m4, acc24);

            // s25 - s26: m0, m1, m3
            __m256 acc25 = _mm256_setzero_ps();
            for (int i = s25; i < s26; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc25 = _mm256_add_ps(acc25, vec);
            }
            m0 = _mm256_add_ps(m0, acc25);
            m1 = _mm256_add_ps(m1, acc25);
            m3 = _mm256_add_ps(m3, acc25);

            // s26 - s27: m0, m1, m3, m4
            __m256 acc26 = _mm256_setzero_ps();
            for (int i = s26; i < s27; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc26 = _mm256_add_ps(acc26, vec);
            }
            m0 = _mm256_add_ps(m0, acc26);
            m1 = _mm256_add_ps(m1, acc26);
            m3 = _mm256_add_ps(m3, acc26);
            m4 = _mm256_add_ps(m4, acc26);

            // s27 - s28: m0, m1, m2
            __m256 acc27 = _mm256_setzero_ps();
            for (int i = s27; i < s28; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc27 = _mm256_add_ps(acc27, vec);
            }
            m0 = _mm256_add_ps(m0, acc27);
            m1 = _mm256_add_ps(m1, acc27);
            m2 = _mm256_add_ps(m2, acc27);

            // s28 - s29: m0, m1, m2, m4
            __m256 acc28 = _mm256_setzero_ps();
            for (int i = s28; i < s29; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc28 = _mm256_add_ps(acc28, vec);
            }
            m0 = _mm256_add_ps(m0, acc28);
            m1 = _mm256_add_ps(m1, acc28);
            m2 = _mm256_add_ps(m2, acc28);
            m4 = _mm256_add_ps(m4, acc28);

            // s29 - s30: m0, m1, m2, m3
            __m256 acc29 = _mm256_setzero_ps();
            for (int i = s29; i < s30; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc29 = _mm256_add_ps(acc29, vec);
            }
            m0 = _mm256_add_ps(m0, acc29);
            m1 = _mm256_add_ps(m1, acc29);
            m2 = _mm256_add_ps(m2, acc29);
            m3 = _mm256_add_ps(m3, acc29);

            // s30 - s31: m0, m1, m2, m3, m4
            __m256 acc30 = _mm256_setzero_ps();
            for (int i = s30; i < s31; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc30 = _mm256_add_ps(acc30, vec);
            }
            m0 = _mm256_add_ps(m0, acc30);
            m1 = _mm256_add_ps(m1, acc30);
            m2 = _mm256_add_ps(m2, acc30);
            m3 = _mm256_add_ps(m3, acc30);
            m4 = _mm256_add_ps(m4, acc30);

            // Negative permutation loops
            // t0 - t1: m4
            __m256 acc0_neg = _mm256_setzero_ps();
            for (int i = t0; i < t1; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0_neg = _mm256_add_ps(acc0_neg, vec);
            }
            m4 = _mm256_sub_ps(m4, acc0_neg);

            // t1 - t2: m3
            __m256 acc1_neg = _mm256_setzero_ps();
            for (int i = t1; i < t2; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1_neg = _mm256_add_ps(acc1_neg, vec);
            }
            m3 = _mm256_sub_ps(m3, acc1_neg);

            // t2 - t3: m3, m4
            __m256 acc2_neg = _mm256_setzero_ps();
            for (int i = t2; i < t3; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2_neg = _mm256_add_ps(acc2_neg, vec);
            }
            m3 = _mm256_sub_ps(m3, acc2_neg);
            m4 = _mm256_sub_ps(m4, acc2_neg);

            // t3 - t4: m2
            __m256 acc3_neg = _mm256_setzero_ps();
            for (int i = t3; i < t4; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc3_neg = _mm256_add_ps(acc3_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc3_neg);

            // t4 - t5: m2, m4
            __m256 acc4_neg = _mm256_setzero_ps();
            for (int i = t4; i < t5; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc4_neg = _mm256_add_ps(acc4_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc4_neg);
            m4 = _mm256_sub_ps(m4, acc4_neg);

            // t5 - t6: m2, m3
            __m256 acc5_neg = _mm256_setzero_ps();
            for (int i = t5; i < t6; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc5_neg = _mm256_add_ps(acc5_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc5_neg);
            m3 = _mm256_sub_ps(m3, acc5_neg);

            // t6 - t7: m2, m3, m4
            __m256 acc6_neg = _mm256_setzero_ps();
            for (int i = t6; i < t7; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc6_neg = _mm256_add_ps(acc6_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc6_neg);
            m3 = _mm256_sub_ps(m3, acc6_neg);
            m4 = _mm256_sub_ps(m4, acc6_neg);

            // t7 - t8: m1
            __m256 acc7_neg = _mm256_setzero_ps();
            for (int i = t7; i < t8; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc7_neg = _mm256_add_ps(acc7_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc7_neg);

            // t8 - t9: m1, m4
            __m256 acc8_neg = _mm256_setzero_ps();
            for (int i = t8; i < t9; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc8_neg = _mm256_add_ps(acc8_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc8_neg);
            m4 = _mm256_sub_ps(m4, acc8_neg);

            // t9 - t10: m1, m3
            __m256 acc9_neg = _mm256_setzero_ps();
            for (int i = t9; i < t10; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc9_neg = _mm256_add_ps(acc9_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc9_neg);
            m3 = _mm256_sub_ps(m3, acc9_neg);

            // t10 - t11: m1, m3, m4
            __m256 acc10_neg = _mm256_setzero_ps();
            for (int i = t10; i < t11; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc10_neg = _mm256_add_ps(acc10_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc10_neg);
            m3 = _mm256_sub_ps(m3, acc10_neg);
            m4 = _mm256_sub_ps(m4, acc10_neg);

            // t11 - t12: m1, m2
            __m256 acc11_neg = _mm256_setzero_ps();
            for (int i = t11; i < t12; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc11_neg = _mm256_add_ps(acc11_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc11_neg);
            m2 = _mm256_sub_ps(m2, acc11_neg);

            // t12 - t13: m1, m2, m4
            __m256 acc12_neg = _mm256_setzero_ps();
            for (int i = t12; i < t13; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc12_neg = _mm256_add_ps(acc12_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc12_neg);
            m2 = _mm256_sub_ps(m2, acc12_neg);
            m4 = _mm256_sub_ps(m4, acc12_neg);

            // t13 - t14: m1, m2, m3
            __m256 acc13_neg = _mm256_setzero_ps();
            for (int i = t13; i < t14; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc13_neg = _mm256_add_ps(acc13_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc13_neg);
            m2 = _mm256_sub_ps(m2, acc13_neg);
            m3 = _mm256_sub_ps(m3, acc13_neg);

            // t14 - t15: m1, m2, m3, m4
            __m256 acc14_neg = _mm256_setzero_ps();
            for (int i = t14; i < t15; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc14_neg = _mm256_add_ps(acc14_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc14_neg);
            m2 = _mm256_sub_ps(m2, acc14_neg);
            m3 = _mm256_sub_ps(m3, acc14_neg);
            m4 = _mm256_sub_ps(m4, acc14_neg);

            // t15 - t16: m0
            __m256 acc15_neg = _mm256_setzero_ps();
            for (int i = t15; i < t16; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc15_neg = _mm256_add_ps(acc15_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc15_neg);

            // t16 - t17: m0, m4
            __m256 acc16_neg = _mm256_setzero_ps();
            for (int i = t16; i < t17; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc16_neg = _mm256_add_ps(acc16_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc16_neg);
            m4 = _mm256_sub_ps(m4, acc16_neg);

            // t17 - t18: m0, m3
            __m256 acc17_neg = _mm256_setzero_ps();
            for (int i = t17; i < t18; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc17_neg = _mm256_add_ps(acc17_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc17_neg);
            m3 = _mm256_sub_ps(m3, acc17_neg);

            // t18 - t19: m0, m3, m4
            __m256 acc18_neg = _mm256_setzero_ps();
            for (int i = t18; i < t19; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc18_neg = _mm256_add_ps(acc18_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc18_neg);
            m3 = _mm256_sub_ps(m3, acc18_neg);
            m4 = _mm256_sub_ps(m4, acc18_neg);

            // t19 - t20: m0, m2
            __m256 acc19_neg = _mm256_setzero_ps();
            for (int i = t19; i < t20; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc19_neg = _mm256_add_ps(acc19_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc19_neg);
            m2 = _mm256_sub_ps(m2, acc19_neg);

            // t20 - t21: m0, m2, m4
            __m256 acc20_neg = _mm256_setzero_ps();
            for (int i = t20; i < t21; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc20_neg = _mm256_add_ps(acc20_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc20_neg);
            m2 = _mm256_sub_ps(m2, acc20_neg);
            m4 = _mm256_sub_ps(m4, acc20_neg);

            // t21 - t22: m0, m2, m3
            __m256 acc21_neg = _mm256_setzero_ps();
            for (int i = t21; i < t22; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc21_neg = _mm256_add_ps(acc21_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc21_neg);
            m2 = _mm256_sub_ps(m2, acc21_neg);
            m3 = _mm256_sub_ps(m3, acc21_neg);

            // t22 - t23: m0, m2, m3, m4
            __m256 acc22_neg = _mm256_setzero_ps();
            for (int i = t22; i < t23; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc22_neg = _mm256_add_ps(acc22_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc22_neg);
            m2 = _mm256_sub_ps(m2, acc22_neg);
            m3 = _mm256_sub_ps(m3, acc22_neg);
            m4 = _mm256_sub_ps(m4, acc22_neg);

            // t23 - t24: m0, m1
            __m256 acc23_neg = _mm256_setzero_ps();
            for (int i = t23; i < t24; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc23_neg = _mm256_add_ps(acc23_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc23_neg);
            m1 = _mm256_sub_ps(m1, acc23_neg);

            // t24 - t25: m0, m1, m4
            __m256 acc24_neg = _mm256_setzero_ps();
            for (int i = t24; i < t25; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc24_neg = _mm256_add_ps(acc24_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc24_neg);
            m1 = _mm256_sub_ps(m1, acc24_neg);
            m4 = _mm256_sub_ps(m4, acc24_neg);

            // t25 - t26: m0, m1, m3
            __m256 acc25_neg = _mm256_setzero_ps();
            for (int i = t25; i < t26; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc25_neg = _mm256_add_ps(acc25_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc25_neg);
            m1 = _mm256_sub_ps(m1, acc25_neg);
            m3 = _mm256_sub_ps(m3, acc25_neg);

            // t26 - t27: m0, m1, m3, m4
            __m256 acc26_neg = _mm256_setzero_ps();
            for (int i = t26; i < t27; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc26_neg = _mm256_add_ps(acc26_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc26_neg);
            m1 = _mm256_sub_ps(m1, acc26_neg);
            m3 = _mm256_sub_ps(m3, acc26_neg);
            m4 = _mm256_sub_ps(m4, acc26_neg);

            // t27 - t28: m0, m1, m2
            __m256 acc27_neg = _mm256_setzero_ps();
            for (int i = t27; i < t28; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc27_neg = _mm256_add_ps(acc27_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc27_neg);
            m1 = _mm256_sub_ps(m1, acc27_neg);
            m2 = _mm256_sub_ps(m2, acc27_neg);

            // t28 - t29: m0, m1, m2, m4
            __m256 acc28_neg = _mm256_setzero_ps();
            for (int i = t28; i < t29; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc28_neg = _mm256_add_ps(acc28_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc28_neg);
            m1 = _mm256_sub_ps(m1, acc28_neg);
            m2 = _mm256_sub_ps(m2, acc28_neg);
            m4 = _mm256_sub_ps(m4, acc28_neg);

            // t29 - t30: m0, m1, m2, m3
            __m256 acc29_neg = _mm256_setzero_ps();
            for (int i = t29; i < t30; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc29_neg = _mm256_add_ps(acc29_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc29_neg);
            m1 = _mm256_sub_ps(m1, acc29_neg);
            m2 = _mm256_sub_ps(m2, acc29_neg);
            m3 = _mm256_sub_ps(m3, acc29_neg);

            // t30 - t31: m0, m1, m2, m3, m4
            __m256 acc30_neg = _mm256_setzero_ps();
            for (int i = t30; i < t31; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc30_neg = _mm256_add_ps(acc30_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc30_neg);
            m1 = _mm256_sub_ps(m1, acc30_neg);
            m2 = _mm256_sub_ps(m2, acc30_neg);
            m3 = _mm256_sub_ps(m3, acc30_neg);
            m4 = _mm256_sub_ps(m4, acc30_neg);

            // Store results directly - works for any L value
            // Extract individual elements and store to correct positions
            alignas(32) float temp_m0[8], temp_m1[8], temp_m2[8], temp_m3[8], temp_m4[8];
            _mm256_store_ps(temp_m0, m0);
            _mm256_store_ps(temp_m1, m1);
            _mm256_store_ps(temp_m2, m2);
            _mm256_store_ps(temp_m3, m3);
            _mm256_store_ps(temp_m4, m4);

            // Store results for each row (no bounds checking)
            result[(row + 0) * N + output_offset + 0] = temp_m0[0];
            result[(row + 0) * N + output_offset + 1] = temp_m1[0];
            result[(row + 0) * N + output_offset + 2] = temp_m2[0];
            result[(row + 0) * N + output_offset + 3] = temp_m3[0];
            result[(row + 0) * N + output_offset + 4] = temp_m4[0];
            result[(row + 1) * N + output_offset + 0] = temp_m0[1];
            result[(row + 1) * N + output_offset + 1] = temp_m1[1];
            result[(row + 1) * N + output_offset + 2] = temp_m2[1];
            result[(row + 1) * N + output_offset + 3] = temp_m3[1];
            result[(row + 1) * N + output_offset + 4] = temp_m4[1];
            result[(row + 2) * N + output_offset + 0] = temp_m0[2];
            result[(row + 2) * N + output_offset + 1] = temp_m1[2];
            result[(row + 2) * N + output_offset + 2] = temp_m2[2];
            result[(row + 2) * N + output_offset + 3] = temp_m3[2];
            result[(row + 2) * N + output_offset + 4] = temp_m4[2];
            result[(row + 3) * N + output_offset + 0] = temp_m0[3];
            result[(row + 3) * N + output_offset + 1] = temp_m1[3];
            result[(row + 3) * N + output_offset + 2] = temp_m2[3];
            result[(row + 3) * N + output_offset + 3] = temp_m3[3];
            result[(row + 3) * N + output_offset + 4] = temp_m4[3];
            result[(row + 4) * N + output_offset + 0] = temp_m0[4];
            result[(row + 4) * N + output_offset + 1] = temp_m1[4];
            result[(row + 4) * N + output_offset + 2] = temp_m2[4];
            result[(row + 4) * N + output_offset + 3] = temp_m3[4];
            result[(row + 4) * N + output_offset + 4] = temp_m4[4];
            result[(row + 5) * N + output_offset + 0] = temp_m0[5];
            result[(row + 5) * N + output_offset + 1] = temp_m1[5];
            result[(row + 5) * N + output_offset + 2] = temp_m2[5];
            result[(row + 5) * N + output_offset + 3] = temp_m3[5];
            result[(row + 5) * N + output_offset + 4] = temp_m4[5];
            result[(row + 6) * N + output_offset + 0] = temp_m0[6];
            result[(row + 6) * N + output_offset + 1] = temp_m1[6];
            result[(row + 6) * N + output_offset + 2] = temp_m2[6];
            result[(row + 6) * N + output_offset + 3] = temp_m3[6];
            result[(row + 6) * N + output_offset + 4] = temp_m4[6];
            result[(row + 7) * N + output_offset + 0] = temp_m0[7];
            result[(row + 7) * N + output_offset + 1] = temp_m1[7];
            result[(row + 7) * N + output_offset + 2] = temp_m2[7];
            result[(row + 7) * N + output_offset + 3] = temp_m3[7];
            result[(row + 7) * N + output_offset + 4] = temp_m4[7];

            output_offset += 5;

        }
    }

    // Upstream assumes complete eight-row batches. Reuse basic SSR for tails.
    if (M % 8 != 0) {
        ssr_inference_rows(v, permutations_pos, segments_pos, permutations_neg,
                           segments_neg, 5, M, N, K, result, M - M % 8);
    }

    return result;
}

TERNARY_BASELINE_AVX2_TARGET
vector<float> ssr_inference_float_vec_L6(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M){

    N = N + (6 - N % 6) % 6;
    int num_bins = 1 << 6;
    int num_blocks = N / 6;

    vector<float> result(M * N);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int row = 0; row < M - M % 8; row += 8) {
        const float* v0 = &v[(row + 0) * K];
        const float* v1 = &v[(row + 1) * K];
        const float* v2 = &v[(row + 2) * K];
        const float* v3 = &v[(row + 3) * K];
        const float* v4 = &v[(row + 4) * K];
        const float* v5 = &v[(row + 5) * K];
        const float* v6 = &v[(row + 6) * K];
        const float* v7 = &v[(row + 7) * K];


        int output_offset = 0;

        for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {


            // Precomute the boudries in the summation s variables
            // Precomute the boudries for subtraction t varaiables

            const int s0 = segments_pos[b_idx * num_bins + 0], s1 = segments_pos[b_idx * num_bins + 1], s2 = segments_pos[b_idx * num_bins + 2], s3 = segments_pos[b_idx * num_bins + 3], s4 = segments_pos[b_idx * num_bins + 4], s5 = segments_pos[b_idx * num_bins + 5], s6 = segments_pos[b_idx * num_bins + 6], s7 = segments_pos[b_idx * num_bins + 7], s8 = segments_pos[b_idx * num_bins + 8], s9 = segments_pos[b_idx * num_bins + 9], s10 = segments_pos[b_idx * num_bins + 10], s11 = segments_pos[b_idx * num_bins + 11], s12 = segments_pos[b_idx * num_bins + 12], s13 = segments_pos[b_idx * num_bins + 13], s14 = segments_pos[b_idx * num_bins + 14], s15 = segments_pos[b_idx * num_bins + 15], s16 = segments_pos[b_idx * num_bins + 16], s17 = segments_pos[b_idx * num_bins + 17], s18 = segments_pos[b_idx * num_bins + 18], s19 = segments_pos[b_idx * num_bins + 19], s20 = segments_pos[b_idx * num_bins + 20], s21 = segments_pos[b_idx * num_bins + 21], s22 = segments_pos[b_idx * num_bins + 22], s23 = segments_pos[b_idx * num_bins + 23], s24 = segments_pos[b_idx * num_bins + 24], s25 = segments_pos[b_idx * num_bins + 25], s26 = segments_pos[b_idx * num_bins + 26], s27 = segments_pos[b_idx * num_bins + 27], s28 = segments_pos[b_idx * num_bins + 28], s29 = segments_pos[b_idx * num_bins + 29], s30 = segments_pos[b_idx * num_bins + 30], s31 = segments_pos[b_idx * num_bins + 31], s32 = segments_pos[b_idx * num_bins + 32], s33 = segments_pos[b_idx * num_bins + 33], s34 = segments_pos[b_idx * num_bins + 34], s35 = segments_pos[b_idx * num_bins + 35], s36 = segments_pos[b_idx * num_bins + 36], s37 = segments_pos[b_idx * num_bins + 37], s38 = segments_pos[b_idx * num_bins + 38], s39 = segments_pos[b_idx * num_bins + 39], s40 = segments_pos[b_idx * num_bins + 40], s41 = segments_pos[b_idx * num_bins + 41], s42 = segments_pos[b_idx * num_bins + 42], s43 = segments_pos[b_idx * num_bins + 43], s44 = segments_pos[b_idx * num_bins + 44], s45 = segments_pos[b_idx * num_bins + 45], s46 = segments_pos[b_idx * num_bins + 46], s47 = segments_pos[b_idx * num_bins + 47], s48 = segments_pos[b_idx * num_bins + 48], s49 = segments_pos[b_idx * num_bins + 49], s50 = segments_pos[b_idx * num_bins + 50], s51 = segments_pos[b_idx * num_bins + 51], s52 = segments_pos[b_idx * num_bins + 52], s53 = segments_pos[b_idx * num_bins + 53], s54 = segments_pos[b_idx * num_bins + 54], s55 = segments_pos[b_idx * num_bins + 55], s56 = segments_pos[b_idx * num_bins + 56], s57 = segments_pos[b_idx * num_bins + 57], s58 = segments_pos[b_idx * num_bins + 58], s59 = segments_pos[b_idx * num_bins + 59], s60 = segments_pos[b_idx * num_bins + 60], s61 = segments_pos[b_idx * num_bins + 61], s62 = segments_pos[b_idx * num_bins + 62], s63 = segments_pos[b_idx * num_bins + 63];
            const int t0 = segments_neg[b_idx * num_bins + 0], t1 = segments_neg[b_idx * num_bins + 1], t2 = segments_neg[b_idx * num_bins + 2], t3 = segments_neg[b_idx * num_bins + 3], t4 = segments_neg[b_idx * num_bins + 4], t5 = segments_neg[b_idx * num_bins + 5], t6 = segments_neg[b_idx * num_bins + 6], t7 = segments_neg[b_idx * num_bins + 7], t8 = segments_neg[b_idx * num_bins + 8], t9 = segments_neg[b_idx * num_bins + 9], t10 = segments_neg[b_idx * num_bins + 10], t11 = segments_neg[b_idx * num_bins + 11], t12 = segments_neg[b_idx * num_bins + 12], t13 = segments_neg[b_idx * num_bins + 13], t14 = segments_neg[b_idx * num_bins + 14], t15 = segments_neg[b_idx * num_bins + 15], t16 = segments_neg[b_idx * num_bins + 16], t17 = segments_neg[b_idx * num_bins + 17], t18 = segments_neg[b_idx * num_bins + 18], t19 = segments_neg[b_idx * num_bins + 19], t20 = segments_neg[b_idx * num_bins + 20], t21 = segments_neg[b_idx * num_bins + 21], t22 = segments_neg[b_idx * num_bins + 22], t23 = segments_neg[b_idx * num_bins + 23], t24 = segments_neg[b_idx * num_bins + 24], t25 = segments_neg[b_idx * num_bins + 25], t26 = segments_neg[b_idx * num_bins + 26], t27 = segments_neg[b_idx * num_bins + 27], t28 = segments_neg[b_idx * num_bins + 28], t29 = segments_neg[b_idx * num_bins + 29], t30 = segments_neg[b_idx * num_bins + 30], t31 = segments_neg[b_idx * num_bins + 31], t32 = segments_neg[b_idx * num_bins + 32], t33 = segments_neg[b_idx * num_bins + 33], t34 = segments_neg[b_idx * num_bins + 34], t35 = segments_neg[b_idx * num_bins + 35], t36 = segments_neg[b_idx * num_bins + 36], t37 = segments_neg[b_idx * num_bins + 37], t38 = segments_neg[b_idx * num_bins + 38], t39 = segments_neg[b_idx * num_bins + 39], t40 = segments_neg[b_idx * num_bins + 40], t41 = segments_neg[b_idx * num_bins + 41], t42 = segments_neg[b_idx * num_bins + 42], t43 = segments_neg[b_idx * num_bins + 43], t44 = segments_neg[b_idx * num_bins + 44], t45 = segments_neg[b_idx * num_bins + 45], t46 = segments_neg[b_idx * num_bins + 46], t47 = segments_neg[b_idx * num_bins + 47], t48 = segments_neg[b_idx * num_bins + 48], t49 = segments_neg[b_idx * num_bins + 49], t50 = segments_neg[b_idx * num_bins + 50], t51 = segments_neg[b_idx * num_bins + 51], t52 = segments_neg[b_idx * num_bins + 52], t53 = segments_neg[b_idx * num_bins + 53], t54 = segments_neg[b_idx * num_bins + 54], t55 = segments_neg[b_idx * num_bins + 55], t56 = segments_neg[b_idx * num_bins + 56], t57 = segments_neg[b_idx * num_bins + 57], t58 = segments_neg[b_idx * num_bins + 58], t59 = segments_neg[b_idx * num_bins + 59], t60 = segments_neg[b_idx * num_bins + 60], t61 = segments_neg[b_idx * num_bins + 61], t62 = segments_neg[b_idx * num_bins + 62], t63 = segments_neg[b_idx * num_bins + 63];

            // Initialize main accumulators - 6 variables for k=6
            __m256 m0 = _mm256_setzero_ps();
            __m256 m1 = _mm256_setzero_ps();
            __m256 m2 = _mm256_setzero_ps();
            __m256 m3 = _mm256_setzero_ps();
            __m256 m4 = _mm256_setzero_ps();
            __m256 m5 = _mm256_setzero_ps();

            // Positive permutation loops
            // s0 - s1: m5
            __m256 acc0 = _mm256_setzero_ps();
            for (int i = s0; i < s1; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0 = _mm256_add_ps(acc0, vec);
            }
            m5 = _mm256_add_ps(m5, acc0);

            // s1 - s2: m4
            __m256 acc1 = _mm256_setzero_ps();
            for (int i = s1; i < s2; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1 = _mm256_add_ps(acc1, vec);
            }
            m4 = _mm256_add_ps(m4, acc1);

            // s2 - s3: m4, m5
            __m256 acc2 = _mm256_setzero_ps();
            for (int i = s2; i < s3; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2 = _mm256_add_ps(acc2, vec);
            }
            m4 = _mm256_add_ps(m4, acc2);
            m5 = _mm256_add_ps(m5, acc2);

            // s3 - s4: m3
            __m256 acc3 = _mm256_setzero_ps();
            for (int i = s3; i < s4; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc3 = _mm256_add_ps(acc3, vec);
            }
            m3 = _mm256_add_ps(m3, acc3);

            // s4 - s5: m3, m5
            __m256 acc4 = _mm256_setzero_ps();
            for (int i = s4; i < s5; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc4 = _mm256_add_ps(acc4, vec);
            }
            m3 = _mm256_add_ps(m3, acc4);
            m5 = _mm256_add_ps(m5, acc4);

            // s5 - s6: m3, m4
            __m256 acc5 = _mm256_setzero_ps();
            for (int i = s5; i < s6; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc5 = _mm256_add_ps(acc5, vec);
            }
            m3 = _mm256_add_ps(m3, acc5);
            m4 = _mm256_add_ps(m4, acc5);

            // s6 - s7: m3, m4, m5
            __m256 acc6 = _mm256_setzero_ps();
            for (int i = s6; i < s7; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc6 = _mm256_add_ps(acc6, vec);
            }
            m3 = _mm256_add_ps(m3, acc6);
            m4 = _mm256_add_ps(m4, acc6);
            m5 = _mm256_add_ps(m5, acc6);

            // s7 - s8: m2
            __m256 acc7 = _mm256_setzero_ps();
            for (int i = s7; i < s8; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc7 = _mm256_add_ps(acc7, vec);
            }
            m2 = _mm256_add_ps(m2, acc7);

            // s8 - s9: m2, m5
            __m256 acc8 = _mm256_setzero_ps();
            for (int i = s8; i < s9; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc8 = _mm256_add_ps(acc8, vec);
            }
            m2 = _mm256_add_ps(m2, acc8);
            m5 = _mm256_add_ps(m5, acc8);

            // s9 - s10: m2, m4
            __m256 acc9 = _mm256_setzero_ps();
            for (int i = s9; i < s10; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc9 = _mm256_add_ps(acc9, vec);
            }
            m2 = _mm256_add_ps(m2, acc9);
            m4 = _mm256_add_ps(m4, acc9);

            // s10 - s11: m2, m4, m5
            __m256 acc10 = _mm256_setzero_ps();
            for (int i = s10; i < s11; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc10 = _mm256_add_ps(acc10, vec);
            }
            m2 = _mm256_add_ps(m2, acc10);
            m4 = _mm256_add_ps(m4, acc10);
            m5 = _mm256_add_ps(m5, acc10);

            // s11 - s12: m2, m3
            __m256 acc11 = _mm256_setzero_ps();
            for (int i = s11; i < s12; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc11 = _mm256_add_ps(acc11, vec);
            }
            m2 = _mm256_add_ps(m2, acc11);
            m3 = _mm256_add_ps(m3, acc11);

            // s12 - s13: m2, m3, m5
            __m256 acc12 = _mm256_setzero_ps();
            for (int i = s12; i < s13; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc12 = _mm256_add_ps(acc12, vec);
            }
            m2 = _mm256_add_ps(m2, acc12);
            m3 = _mm256_add_ps(m3, acc12);
            m5 = _mm256_add_ps(m5, acc12);

            // s13 - s14: m2, m3, m4
            __m256 acc13 = _mm256_setzero_ps();
            for (int i = s13; i < s14; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc13 = _mm256_add_ps(acc13, vec);
            }
            m2 = _mm256_add_ps(m2, acc13);
            m3 = _mm256_add_ps(m3, acc13);
            m4 = _mm256_add_ps(m4, acc13);

            // s14 - s15: m2, m3, m4, m5
            __m256 acc14 = _mm256_setzero_ps();
            for (int i = s14; i < s15; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc14 = _mm256_add_ps(acc14, vec);
            }
            m2 = _mm256_add_ps(m2, acc14);
            m3 = _mm256_add_ps(m3, acc14);
            m4 = _mm256_add_ps(m4, acc14);
            m5 = _mm256_add_ps(m5, acc14);

            // s15 - s16: m1
            __m256 acc15 = _mm256_setzero_ps();
            for (int i = s15; i < s16; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc15 = _mm256_add_ps(acc15, vec);
            }
            m1 = _mm256_add_ps(m1, acc15);

            // s16 - s17: m1, m5
            __m256 acc16 = _mm256_setzero_ps();
            for (int i = s16; i < s17; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc16 = _mm256_add_ps(acc16, vec);
            }
            m1 = _mm256_add_ps(m1, acc16);
            m5 = _mm256_add_ps(m5, acc16);

            // s17 - s18: m1, m4
            __m256 acc17 = _mm256_setzero_ps();
            for (int i = s17; i < s18; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc17 = _mm256_add_ps(acc17, vec);
            }
            m1 = _mm256_add_ps(m1, acc17);
            m4 = _mm256_add_ps(m4, acc17);

            // s18 - s19: m1, m4, m5
            __m256 acc18 = _mm256_setzero_ps();
            for (int i = s18; i < s19; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc18 = _mm256_add_ps(acc18, vec);
            }
            m1 = _mm256_add_ps(m1, acc18);
            m4 = _mm256_add_ps(m4, acc18);
            m5 = _mm256_add_ps(m5, acc18);

            // s19 - s20: m1, m3
            __m256 acc19 = _mm256_setzero_ps();
            for (int i = s19; i < s20; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc19 = _mm256_add_ps(acc19, vec);
            }
            m1 = _mm256_add_ps(m1, acc19);
            m3 = _mm256_add_ps(m3, acc19);

            // s20 - s21: m1, m3, m5
            __m256 acc20 = _mm256_setzero_ps();
            for (int i = s20; i < s21; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc20 = _mm256_add_ps(acc20, vec);
            }
            m1 = _mm256_add_ps(m1, acc20);
            m3 = _mm256_add_ps(m3, acc20);
            m5 = _mm256_add_ps(m5, acc20);

            // s21 - s22: m1, m3, m4
            __m256 acc21 = _mm256_setzero_ps();
            for (int i = s21; i < s22; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc21 = _mm256_add_ps(acc21, vec);
            }
            m1 = _mm256_add_ps(m1, acc21);
            m3 = _mm256_add_ps(m3, acc21);
            m4 = _mm256_add_ps(m4, acc21);

            // s22 - s23: m1, m3, m4, m5
            __m256 acc22 = _mm256_setzero_ps();
            for (int i = s22; i < s23; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc22 = _mm256_add_ps(acc22, vec);
            }
            m1 = _mm256_add_ps(m1, acc22);
            m3 = _mm256_add_ps(m3, acc22);
            m4 = _mm256_add_ps(m4, acc22);
            m5 = _mm256_add_ps(m5, acc22);

            // s23 - s24: m1, m2
            __m256 acc23 = _mm256_setzero_ps();
            for (int i = s23; i < s24; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc23 = _mm256_add_ps(acc23, vec);
            }
            m1 = _mm256_add_ps(m1, acc23);
            m2 = _mm256_add_ps(m2, acc23);

            // s24 - s25: m1, m2, m5
            __m256 acc24 = _mm256_setzero_ps();
            for (int i = s24; i < s25; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc24 = _mm256_add_ps(acc24, vec);
            }
            m1 = _mm256_add_ps(m1, acc24);
            m2 = _mm256_add_ps(m2, acc24);
            m5 = _mm256_add_ps(m5, acc24);

            // s25 - s26: m1, m2, m4
            __m256 acc25 = _mm256_setzero_ps();
            for (int i = s25; i < s26; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc25 = _mm256_add_ps(acc25, vec);
            }
            m1 = _mm256_add_ps(m1, acc25);
            m2 = _mm256_add_ps(m2, acc25);
            m4 = _mm256_add_ps(m4, acc25);

            // s26 - s27: m1, m2, m4, m5
            __m256 acc26 = _mm256_setzero_ps();
            for (int i = s26; i < s27; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc26 = _mm256_add_ps(acc26, vec);
            }
            m1 = _mm256_add_ps(m1, acc26);
            m2 = _mm256_add_ps(m2, acc26);
            m4 = _mm256_add_ps(m4, acc26);
            m5 = _mm256_add_ps(m5, acc26);

            // s27 - s28: m1, m2, m3
            __m256 acc27 = _mm256_setzero_ps();
            for (int i = s27; i < s28; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc27 = _mm256_add_ps(acc27, vec);
            }
            m1 = _mm256_add_ps(m1, acc27);
            m2 = _mm256_add_ps(m2, acc27);
            m3 = _mm256_add_ps(m3, acc27);

            // s28 - s29: m1, m2, m3, m5
            __m256 acc28 = _mm256_setzero_ps();
            for (int i = s28; i < s29; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc28 = _mm256_add_ps(acc28, vec);
            }
            m1 = _mm256_add_ps(m1, acc28);
            m2 = _mm256_add_ps(m2, acc28);
            m3 = _mm256_add_ps(m3, acc28);
            m5 = _mm256_add_ps(m5, acc28);

            // s29 - s30: m1, m2, m3, m4
            __m256 acc29 = _mm256_setzero_ps();
            for (int i = s29; i < s30; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc29 = _mm256_add_ps(acc29, vec);
            }
            m1 = _mm256_add_ps(m1, acc29);
            m2 = _mm256_add_ps(m2, acc29);
            m3 = _mm256_add_ps(m3, acc29);
            m4 = _mm256_add_ps(m4, acc29);

            // s30 - s31: m1, m2, m3, m4, m5
            __m256 acc30 = _mm256_setzero_ps();
            for (int i = s30; i < s31; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc30 = _mm256_add_ps(acc30, vec);
            }
            m1 = _mm256_add_ps(m1, acc30);
            m2 = _mm256_add_ps(m2, acc30);
            m3 = _mm256_add_ps(m3, acc30);
            m4 = _mm256_add_ps(m4, acc30);
            m5 = _mm256_add_ps(m5, acc30);

            // s31 - s32: m0
            __m256 acc31 = _mm256_setzero_ps();
            for (int i = s31; i < s32; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc31 = _mm256_add_ps(acc31, vec);
            }
            m0 = _mm256_add_ps(m0, acc31);

            // s32 - s33: m0, m5
            __m256 acc32 = _mm256_setzero_ps();
            for (int i = s32; i < s33; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc32 = _mm256_add_ps(acc32, vec);
            }
            m0 = _mm256_add_ps(m0, acc32);
            m5 = _mm256_add_ps(m5, acc32);

            // s33 - s34: m0, m4
            __m256 acc33 = _mm256_setzero_ps();
            for (int i = s33; i < s34; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc33 = _mm256_add_ps(acc33, vec);
            }
            m0 = _mm256_add_ps(m0, acc33);
            m4 = _mm256_add_ps(m4, acc33);

            // s34 - s35: m0, m4, m5
            __m256 acc34 = _mm256_setzero_ps();
            for (int i = s34; i < s35; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc34 = _mm256_add_ps(acc34, vec);
            }
            m0 = _mm256_add_ps(m0, acc34);
            m4 = _mm256_add_ps(m4, acc34);
            m5 = _mm256_add_ps(m5, acc34);

            // s35 - s36: m0, m3
            __m256 acc35 = _mm256_setzero_ps();
            for (int i = s35; i < s36; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc35 = _mm256_add_ps(acc35, vec);
            }
            m0 = _mm256_add_ps(m0, acc35);
            m3 = _mm256_add_ps(m3, acc35);

            // s36 - s37: m0, m3, m5
            __m256 acc36 = _mm256_setzero_ps();
            for (int i = s36; i < s37; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc36 = _mm256_add_ps(acc36, vec);
            }
            m0 = _mm256_add_ps(m0, acc36);
            m3 = _mm256_add_ps(m3, acc36);
            m5 = _mm256_add_ps(m5, acc36);

            // s37 - s38: m0, m3, m4
            __m256 acc37 = _mm256_setzero_ps();
            for (int i = s37; i < s38; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc37 = _mm256_add_ps(acc37, vec);
            }
            m0 = _mm256_add_ps(m0, acc37);
            m3 = _mm256_add_ps(m3, acc37);
            m4 = _mm256_add_ps(m4, acc37);

            // s38 - s39: m0, m3, m4, m5
            __m256 acc38 = _mm256_setzero_ps();
            for (int i = s38; i < s39; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc38 = _mm256_add_ps(acc38, vec);
            }
            m0 = _mm256_add_ps(m0, acc38);
            m3 = _mm256_add_ps(m3, acc38);
            m4 = _mm256_add_ps(m4, acc38);
            m5 = _mm256_add_ps(m5, acc38);

            // s39 - s40: m0, m2
            __m256 acc39 = _mm256_setzero_ps();
            for (int i = s39; i < s40; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc39 = _mm256_add_ps(acc39, vec);
            }
            m0 = _mm256_add_ps(m0, acc39);
            m2 = _mm256_add_ps(m2, acc39);

            // s40 - s41: m0, m2, m5
            __m256 acc40 = _mm256_setzero_ps();
            for (int i = s40; i < s41; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc40 = _mm256_add_ps(acc40, vec);
            }
            m0 = _mm256_add_ps(m0, acc40);
            m2 = _mm256_add_ps(m2, acc40);
            m5 = _mm256_add_ps(m5, acc40);

            // s41 - s42: m0, m2, m4
            __m256 acc41 = _mm256_setzero_ps();
            for (int i = s41; i < s42; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc41 = _mm256_add_ps(acc41, vec);
            }
            m0 = _mm256_add_ps(m0, acc41);
            m2 = _mm256_add_ps(m2, acc41);
            m4 = _mm256_add_ps(m4, acc41);

            // s42 - s43: m0, m2, m4, m5
            __m256 acc42 = _mm256_setzero_ps();
            for (int i = s42; i < s43; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc42 = _mm256_add_ps(acc42, vec);
            }
            m0 = _mm256_add_ps(m0, acc42);
            m2 = _mm256_add_ps(m2, acc42);
            m4 = _mm256_add_ps(m4, acc42);
            m5 = _mm256_add_ps(m5, acc42);

            // s43 - s44: m0, m2, m3
            __m256 acc43 = _mm256_setzero_ps();
            for (int i = s43; i < s44; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc43 = _mm256_add_ps(acc43, vec);
            }
            m0 = _mm256_add_ps(m0, acc43);
            m2 = _mm256_add_ps(m2, acc43);
            m3 = _mm256_add_ps(m3, acc43);

            // s44 - s45: m0, m2, m3, m5
            __m256 acc44 = _mm256_setzero_ps();
            for (int i = s44; i < s45; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc44 = _mm256_add_ps(acc44, vec);
            }
            m0 = _mm256_add_ps(m0, acc44);
            m2 = _mm256_add_ps(m2, acc44);
            m3 = _mm256_add_ps(m3, acc44);
            m5 = _mm256_add_ps(m5, acc44);

            // s45 - s46: m0, m2, m3, m4
            __m256 acc45 = _mm256_setzero_ps();
            for (int i = s45; i < s46; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc45 = _mm256_add_ps(acc45, vec);
            }
            m0 = _mm256_add_ps(m0, acc45);
            m2 = _mm256_add_ps(m2, acc45);
            m3 = _mm256_add_ps(m3, acc45);
            m4 = _mm256_add_ps(m4, acc45);

            // s46 - s47: m0, m2, m3, m4, m5
            __m256 acc46 = _mm256_setzero_ps();
            for (int i = s46; i < s47; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc46 = _mm256_add_ps(acc46, vec);
            }
            m0 = _mm256_add_ps(m0, acc46);
            m2 = _mm256_add_ps(m2, acc46);
            m3 = _mm256_add_ps(m3, acc46);
            m4 = _mm256_add_ps(m4, acc46);
            m5 = _mm256_add_ps(m5, acc46);

            // s47 - s48: m0, m1
            __m256 acc47 = _mm256_setzero_ps();
            for (int i = s47; i < s48; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc47 = _mm256_add_ps(acc47, vec);
            }
            m0 = _mm256_add_ps(m0, acc47);
            m1 = _mm256_add_ps(m1, acc47);

            // s48 - s49: m0, m1, m5
            __m256 acc48 = _mm256_setzero_ps();
            for (int i = s48; i < s49; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc48 = _mm256_add_ps(acc48, vec);
            }
            m0 = _mm256_add_ps(m0, acc48);
            m1 = _mm256_add_ps(m1, acc48);
            m5 = _mm256_add_ps(m5, acc48);

            // s49 - s50: m0, m1, m4
            __m256 acc49 = _mm256_setzero_ps();
            for (int i = s49; i < s50; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc49 = _mm256_add_ps(acc49, vec);
            }
            m0 = _mm256_add_ps(m0, acc49);
            m1 = _mm256_add_ps(m1, acc49);
            m4 = _mm256_add_ps(m4, acc49);

            // s50 - s51: m0, m1, m4, m5
            __m256 acc50 = _mm256_setzero_ps();
            for (int i = s50; i < s51; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc50 = _mm256_add_ps(acc50, vec);
            }
            m0 = _mm256_add_ps(m0, acc50);
            m1 = _mm256_add_ps(m1, acc50);
            m4 = _mm256_add_ps(m4, acc50);
            m5 = _mm256_add_ps(m5, acc50);

            // s51 - s52: m0, m1, m3
            __m256 acc51 = _mm256_setzero_ps();
            for (int i = s51; i < s52; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc51 = _mm256_add_ps(acc51, vec);
            }
            m0 = _mm256_add_ps(m0, acc51);
            m1 = _mm256_add_ps(m1, acc51);
            m3 = _mm256_add_ps(m3, acc51);

            // s52 - s53: m0, m1, m3, m5
            __m256 acc52 = _mm256_setzero_ps();
            for (int i = s52; i < s53; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc52 = _mm256_add_ps(acc52, vec);
            }
            m0 = _mm256_add_ps(m0, acc52);
            m1 = _mm256_add_ps(m1, acc52);
            m3 = _mm256_add_ps(m3, acc52);
            m5 = _mm256_add_ps(m5, acc52);

            // s53 - s54: m0, m1, m3, m4
            __m256 acc53 = _mm256_setzero_ps();
            for (int i = s53; i < s54; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc53 = _mm256_add_ps(acc53, vec);
            }
            m0 = _mm256_add_ps(m0, acc53);
            m1 = _mm256_add_ps(m1, acc53);
            m3 = _mm256_add_ps(m3, acc53);
            m4 = _mm256_add_ps(m4, acc53);

            // s54 - s55: m0, m1, m3, m4, m5
            __m256 acc54 = _mm256_setzero_ps();
            for (int i = s54; i < s55; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc54 = _mm256_add_ps(acc54, vec);
            }
            m0 = _mm256_add_ps(m0, acc54);
            m1 = _mm256_add_ps(m1, acc54);
            m3 = _mm256_add_ps(m3, acc54);
            m4 = _mm256_add_ps(m4, acc54);
            m5 = _mm256_add_ps(m5, acc54);

            // s55 - s56: m0, m1, m2
            __m256 acc55 = _mm256_setzero_ps();
            for (int i = s55; i < s56; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc55 = _mm256_add_ps(acc55, vec);
            }
            m0 = _mm256_add_ps(m0, acc55);
            m1 = _mm256_add_ps(m1, acc55);
            m2 = _mm256_add_ps(m2, acc55);

            // s56 - s57: m0, m1, m2, m5
            __m256 acc56 = _mm256_setzero_ps();
            for (int i = s56; i < s57; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc56 = _mm256_add_ps(acc56, vec);
            }
            m0 = _mm256_add_ps(m0, acc56);
            m1 = _mm256_add_ps(m1, acc56);
            m2 = _mm256_add_ps(m2, acc56);
            m5 = _mm256_add_ps(m5, acc56);

            // s57 - s58: m0, m1, m2, m4
            __m256 acc57 = _mm256_setzero_ps();
            for (int i = s57; i < s58; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc57 = _mm256_add_ps(acc57, vec);
            }
            m0 = _mm256_add_ps(m0, acc57);
            m1 = _mm256_add_ps(m1, acc57);
            m2 = _mm256_add_ps(m2, acc57);
            m4 = _mm256_add_ps(m4, acc57);

            // s58 - s59: m0, m1, m2, m4, m5
            __m256 acc58 = _mm256_setzero_ps();
            for (int i = s58; i < s59; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc58 = _mm256_add_ps(acc58, vec);
            }
            m0 = _mm256_add_ps(m0, acc58);
            m1 = _mm256_add_ps(m1, acc58);
            m2 = _mm256_add_ps(m2, acc58);
            m4 = _mm256_add_ps(m4, acc58);
            m5 = _mm256_add_ps(m5, acc58);

            // s59 - s60: m0, m1, m2, m3
            __m256 acc59 = _mm256_setzero_ps();
            for (int i = s59; i < s60; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc59 = _mm256_add_ps(acc59, vec);
            }
            m0 = _mm256_add_ps(m0, acc59);
            m1 = _mm256_add_ps(m1, acc59);
            m2 = _mm256_add_ps(m2, acc59);
            m3 = _mm256_add_ps(m3, acc59);

            // s60 - s61: m0, m1, m2, m3, m5
            __m256 acc60 = _mm256_setzero_ps();
            for (int i = s60; i < s61; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc60 = _mm256_add_ps(acc60, vec);
            }
            m0 = _mm256_add_ps(m0, acc60);
            m1 = _mm256_add_ps(m1, acc60);
            m2 = _mm256_add_ps(m2, acc60);
            m3 = _mm256_add_ps(m3, acc60);
            m5 = _mm256_add_ps(m5, acc60);

            // s61 - s62: m0, m1, m2, m3, m4
            __m256 acc61 = _mm256_setzero_ps();
            for (int i = s61; i < s62; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc61 = _mm256_add_ps(acc61, vec);
            }
            m0 = _mm256_add_ps(m0, acc61);
            m1 = _mm256_add_ps(m1, acc61);
            m2 = _mm256_add_ps(m2, acc61);
            m3 = _mm256_add_ps(m3, acc61);
            m4 = _mm256_add_ps(m4, acc61);

            // s62 - s63: m0, m1, m2, m3, m4, m5
            __m256 acc62 = _mm256_setzero_ps();
            for (int i = s62; i < s63; ++i) {
                int idx = permutations_pos[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc62 = _mm256_add_ps(acc62, vec);
            }
            m0 = _mm256_add_ps(m0, acc62);
            m1 = _mm256_add_ps(m1, acc62);
            m2 = _mm256_add_ps(m2, acc62);
            m3 = _mm256_add_ps(m3, acc62);
            m4 = _mm256_add_ps(m4, acc62);
            m5 = _mm256_add_ps(m5, acc62);

            // Negative permutation loops
            // t0 - t1: m5
            __m256 acc0_neg = _mm256_setzero_ps();
            for (int i = t0; i < t1; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc0_neg = _mm256_add_ps(acc0_neg, vec);
            }
            m5 = _mm256_sub_ps(m5, acc0_neg);

            // t1 - t2: m4
            __m256 acc1_neg = _mm256_setzero_ps();
            for (int i = t1; i < t2; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc1_neg = _mm256_add_ps(acc1_neg, vec);
            }
            m4 = _mm256_sub_ps(m4, acc1_neg);

            // t2 - t3: m4, m5
            __m256 acc2_neg = _mm256_setzero_ps();
            for (int i = t2; i < t3; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc2_neg = _mm256_add_ps(acc2_neg, vec);
            }
            m4 = _mm256_sub_ps(m4, acc2_neg);
            m5 = _mm256_sub_ps(m5, acc2_neg);

            // t3 - t4: m3
            __m256 acc3_neg = _mm256_setzero_ps();
            for (int i = t3; i < t4; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc3_neg = _mm256_add_ps(acc3_neg, vec);
            }
            m3 = _mm256_sub_ps(m3, acc3_neg);

            // t4 - t5: m3, m5
            __m256 acc4_neg = _mm256_setzero_ps();
            for (int i = t4; i < t5; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc4_neg = _mm256_add_ps(acc4_neg, vec);
            }
            m3 = _mm256_sub_ps(m3, acc4_neg);
            m5 = _mm256_sub_ps(m5, acc4_neg);

            // t5 - t6: m3, m4
            __m256 acc5_neg = _mm256_setzero_ps();
            for (int i = t5; i < t6; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc5_neg = _mm256_add_ps(acc5_neg, vec);
            }
            m3 = _mm256_sub_ps(m3, acc5_neg);
            m4 = _mm256_sub_ps(m4, acc5_neg);

            // t6 - t7: m3, m4, m5
            __m256 acc6_neg = _mm256_setzero_ps();
            for (int i = t6; i < t7; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc6_neg = _mm256_add_ps(acc6_neg, vec);
            }
            m3 = _mm256_sub_ps(m3, acc6_neg);
            m4 = _mm256_sub_ps(m4, acc6_neg);
            m5 = _mm256_sub_ps(m5, acc6_neg);

            // t7 - t8: m2
            __m256 acc7_neg = _mm256_setzero_ps();
            for (int i = t7; i < t8; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc7_neg = _mm256_add_ps(acc7_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc7_neg);

            // t8 - t9: m2, m5
            __m256 acc8_neg = _mm256_setzero_ps();
            for (int i = t8; i < t9; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc8_neg = _mm256_add_ps(acc8_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc8_neg);
            m5 = _mm256_sub_ps(m5, acc8_neg);

            // t9 - t10: m2, m4
            __m256 acc9_neg = _mm256_setzero_ps();
            for (int i = t9; i < t10; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc9_neg = _mm256_add_ps(acc9_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc9_neg);
            m4 = _mm256_sub_ps(m4, acc9_neg);

            // t10 - t11: m2, m4, m5
            __m256 acc10_neg = _mm256_setzero_ps();
            for (int i = t10; i < t11; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc10_neg = _mm256_add_ps(acc10_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc10_neg);
            m4 = _mm256_sub_ps(m4, acc10_neg);
            m5 = _mm256_sub_ps(m5, acc10_neg);

            // t11 - t12: m2, m3
            __m256 acc11_neg = _mm256_setzero_ps();
            for (int i = t11; i < t12; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc11_neg = _mm256_add_ps(acc11_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc11_neg);
            m3 = _mm256_sub_ps(m3, acc11_neg);

            // t12 - t13: m2, m3, m5
            __m256 acc12_neg = _mm256_setzero_ps();
            for (int i = t12; i < t13; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc12_neg = _mm256_add_ps(acc12_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc12_neg);
            m3 = _mm256_sub_ps(m3, acc12_neg);
            m5 = _mm256_sub_ps(m5, acc12_neg);

            // t13 - t14: m2, m3, m4
            __m256 acc13_neg = _mm256_setzero_ps();
            for (int i = t13; i < t14; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc13_neg = _mm256_add_ps(acc13_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc13_neg);
            m3 = _mm256_sub_ps(m3, acc13_neg);
            m4 = _mm256_sub_ps(m4, acc13_neg);

            // t14 - t15: m2, m3, m4, m5
            __m256 acc14_neg = _mm256_setzero_ps();
            for (int i = t14; i < t15; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc14_neg = _mm256_add_ps(acc14_neg, vec);
            }
            m2 = _mm256_sub_ps(m2, acc14_neg);
            m3 = _mm256_sub_ps(m3, acc14_neg);
            m4 = _mm256_sub_ps(m4, acc14_neg);
            m5 = _mm256_sub_ps(m5, acc14_neg);

            // t15 - t16: m1
            __m256 acc15_neg = _mm256_setzero_ps();
            for (int i = t15; i < t16; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc15_neg = _mm256_add_ps(acc15_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc15_neg);

            // t16 - t17: m1, m5
            __m256 acc16_neg = _mm256_setzero_ps();
            for (int i = t16; i < t17; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc16_neg = _mm256_add_ps(acc16_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc16_neg);
            m5 = _mm256_sub_ps(m5, acc16_neg);

            // t17 - t18: m1, m4
            __m256 acc17_neg = _mm256_setzero_ps();
            for (int i = t17; i < t18; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc17_neg = _mm256_add_ps(acc17_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc17_neg);
            m4 = _mm256_sub_ps(m4, acc17_neg);

            // t18 - t19: m1, m4, m5
            __m256 acc18_neg = _mm256_setzero_ps();
            for (int i = t18; i < t19; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc18_neg = _mm256_add_ps(acc18_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc18_neg);
            m4 = _mm256_sub_ps(m4, acc18_neg);
            m5 = _mm256_sub_ps(m5, acc18_neg);

            // t19 - t20: m1, m3
            __m256 acc19_neg = _mm256_setzero_ps();
            for (int i = t19; i < t20; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc19_neg = _mm256_add_ps(acc19_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc19_neg);
            m3 = _mm256_sub_ps(m3, acc19_neg);

            // t20 - t21: m1, m3, m5
            __m256 acc20_neg = _mm256_setzero_ps();
            for (int i = t20; i < t21; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc20_neg = _mm256_add_ps(acc20_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc20_neg);
            m3 = _mm256_sub_ps(m3, acc20_neg);
            m5 = _mm256_sub_ps(m5, acc20_neg);

            // t21 - t22: m1, m3, m4
            __m256 acc21_neg = _mm256_setzero_ps();
            for (int i = t21; i < t22; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc21_neg = _mm256_add_ps(acc21_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc21_neg);
            m3 = _mm256_sub_ps(m3, acc21_neg);
            m4 = _mm256_sub_ps(m4, acc21_neg);

            // t22 - t23: m1, m3, m4, m5
            __m256 acc22_neg = _mm256_setzero_ps();
            for (int i = t22; i < t23; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc22_neg = _mm256_add_ps(acc22_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc22_neg);
            m3 = _mm256_sub_ps(m3, acc22_neg);
            m4 = _mm256_sub_ps(m4, acc22_neg);
            m5 = _mm256_sub_ps(m5, acc22_neg);

            // t23 - t24: m1, m2
            __m256 acc23_neg = _mm256_setzero_ps();
            for (int i = t23; i < t24; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc23_neg = _mm256_add_ps(acc23_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc23_neg);
            m2 = _mm256_sub_ps(m2, acc23_neg);

            // t24 - t25: m1, m2, m5
            __m256 acc24_neg = _mm256_setzero_ps();
            for (int i = t24; i < t25; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc24_neg = _mm256_add_ps(acc24_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc24_neg);
            m2 = _mm256_sub_ps(m2, acc24_neg);
            m5 = _mm256_sub_ps(m5, acc24_neg);

            // t25 - t26: m1, m2, m4
            __m256 acc25_neg = _mm256_setzero_ps();
            for (int i = t25; i < t26; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc25_neg = _mm256_add_ps(acc25_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc25_neg);
            m2 = _mm256_sub_ps(m2, acc25_neg);
            m4 = _mm256_sub_ps(m4, acc25_neg);

            // t26 - t27: m1, m2, m4, m5
            __m256 acc26_neg = _mm256_setzero_ps();
            for (int i = t26; i < t27; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc26_neg = _mm256_add_ps(acc26_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc26_neg);
            m2 = _mm256_sub_ps(m2, acc26_neg);
            m4 = _mm256_sub_ps(m4, acc26_neg);
            m5 = _mm256_sub_ps(m5, acc26_neg);

            // t27 - t28: m1, m2, m3
            __m256 acc27_neg = _mm256_setzero_ps();
            for (int i = t27; i < t28; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc27_neg = _mm256_add_ps(acc27_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc27_neg);
            m2 = _mm256_sub_ps(m2, acc27_neg);
            m3 = _mm256_sub_ps(m3, acc27_neg);

            // t28 - t29: m1, m2, m3, m5
            __m256 acc28_neg = _mm256_setzero_ps();
            for (int i = t28; i < t29; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc28_neg = _mm256_add_ps(acc28_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc28_neg);
            m2 = _mm256_sub_ps(m2, acc28_neg);
            m3 = _mm256_sub_ps(m3, acc28_neg);
            m5 = _mm256_sub_ps(m5, acc28_neg);

            // t29 - t30: m1, m2, m3, m4
            __m256 acc29_neg = _mm256_setzero_ps();
            for (int i = t29; i < t30; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc29_neg = _mm256_add_ps(acc29_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc29_neg);
            m2 = _mm256_sub_ps(m2, acc29_neg);
            m3 = _mm256_sub_ps(m3, acc29_neg);
            m4 = _mm256_sub_ps(m4, acc29_neg);

            // t30 - t31: m1, m2, m3, m4, m5
            __m256 acc30_neg = _mm256_setzero_ps();
            for (int i = t30; i < t31; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc30_neg = _mm256_add_ps(acc30_neg, vec);
            }
            m1 = _mm256_sub_ps(m1, acc30_neg);
            m2 = _mm256_sub_ps(m2, acc30_neg);
            m3 = _mm256_sub_ps(m3, acc30_neg);
            m4 = _mm256_sub_ps(m4, acc30_neg);
            m5 = _mm256_sub_ps(m5, acc30_neg);

            // t31 - t32: m0
            __m256 acc31_neg = _mm256_setzero_ps();
            for (int i = t31; i < t32; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc31_neg = _mm256_add_ps(acc31_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc31_neg);

            // t32 - t33: m0, m5
            __m256 acc32_neg = _mm256_setzero_ps();
            for (int i = t32; i < t33; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc32_neg = _mm256_add_ps(acc32_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc32_neg);
            m5 = _mm256_sub_ps(m5, acc32_neg);

            // t33 - t34: m0, m4
            __m256 acc33_neg = _mm256_setzero_ps();
            for (int i = t33; i < t34; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc33_neg = _mm256_add_ps(acc33_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc33_neg);
            m4 = _mm256_sub_ps(m4, acc33_neg);

            // t34 - t35: m0, m4, m5
            __m256 acc34_neg = _mm256_setzero_ps();
            for (int i = t34; i < t35; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc34_neg = _mm256_add_ps(acc34_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc34_neg);
            m4 = _mm256_sub_ps(m4, acc34_neg);
            m5 = _mm256_sub_ps(m5, acc34_neg);

            // t35 - t36: m0, m3
            __m256 acc35_neg = _mm256_setzero_ps();
            for (int i = t35; i < t36; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc35_neg = _mm256_add_ps(acc35_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc35_neg);
            m3 = _mm256_sub_ps(m3, acc35_neg);

            // t36 - t37: m0, m3, m5
            __m256 acc36_neg = _mm256_setzero_ps();
            for (int i = t36; i < t37; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc36_neg = _mm256_add_ps(acc36_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc36_neg);
            m3 = _mm256_sub_ps(m3, acc36_neg);
            m5 = _mm256_sub_ps(m5, acc36_neg);

            // t37 - t38: m0, m3, m4
            __m256 acc37_neg = _mm256_setzero_ps();
            for (int i = t37; i < t38; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc37_neg = _mm256_add_ps(acc37_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc37_neg);
            m3 = _mm256_sub_ps(m3, acc37_neg);
            m4 = _mm256_sub_ps(m4, acc37_neg);

            // t38 - t39: m0, m3, m4, m5
            __m256 acc38_neg = _mm256_setzero_ps();
            for (int i = t38; i < t39; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc38_neg = _mm256_add_ps(acc38_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc38_neg);
            m3 = _mm256_sub_ps(m3, acc38_neg);
            m4 = _mm256_sub_ps(m4, acc38_neg);
            m5 = _mm256_sub_ps(m5, acc38_neg);

            // t39 - t40: m0, m2
            __m256 acc39_neg = _mm256_setzero_ps();
            for (int i = t39; i < t40; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc39_neg = _mm256_add_ps(acc39_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc39_neg);
            m2 = _mm256_sub_ps(m2, acc39_neg);

            // t40 - t41: m0, m2, m5
            __m256 acc40_neg = _mm256_setzero_ps();
            for (int i = t40; i < t41; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc40_neg = _mm256_add_ps(acc40_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc40_neg);
            m2 = _mm256_sub_ps(m2, acc40_neg);
            m5 = _mm256_sub_ps(m5, acc40_neg);

            // t41 - t42: m0, m2, m4
            __m256 acc41_neg = _mm256_setzero_ps();
            for (int i = t41; i < t42; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc41_neg = _mm256_add_ps(acc41_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc41_neg);
            m2 = _mm256_sub_ps(m2, acc41_neg);
            m4 = _mm256_sub_ps(m4, acc41_neg);

            // t42 - t43: m0, m2, m4, m5
            __m256 acc42_neg = _mm256_setzero_ps();
            for (int i = t42; i < t43; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc42_neg = _mm256_add_ps(acc42_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc42_neg);
            m2 = _mm256_sub_ps(m2, acc42_neg);
            m4 = _mm256_sub_ps(m4, acc42_neg);
            m5 = _mm256_sub_ps(m5, acc42_neg);

            // t43 - t44: m0, m2, m3
            __m256 acc43_neg = _mm256_setzero_ps();
            for (int i = t43; i < t44; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc43_neg = _mm256_add_ps(acc43_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc43_neg);
            m2 = _mm256_sub_ps(m2, acc43_neg);
            m3 = _mm256_sub_ps(m3, acc43_neg);

            // t44 - t45: m0, m2, m3, m5
            __m256 acc44_neg = _mm256_setzero_ps();
            for (int i = t44; i < t45; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc44_neg = _mm256_add_ps(acc44_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc44_neg);
            m2 = _mm256_sub_ps(m2, acc44_neg);
            m3 = _mm256_sub_ps(m3, acc44_neg);
            m5 = _mm256_sub_ps(m5, acc44_neg);

            // t45 - t46: m0, m2, m3, m4
            __m256 acc45_neg = _mm256_setzero_ps();
            for (int i = t45; i < t46; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc45_neg = _mm256_add_ps(acc45_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc45_neg);
            m2 = _mm256_sub_ps(m2, acc45_neg);
            m3 = _mm256_sub_ps(m3, acc45_neg);
            m4 = _mm256_sub_ps(m4, acc45_neg);

            // t46 - t47: m0, m2, m3, m4, m5
            __m256 acc46_neg = _mm256_setzero_ps();
            for (int i = t46; i < t47; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc46_neg = _mm256_add_ps(acc46_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc46_neg);
            m2 = _mm256_sub_ps(m2, acc46_neg);
            m3 = _mm256_sub_ps(m3, acc46_neg);
            m4 = _mm256_sub_ps(m4, acc46_neg);
            m5 = _mm256_sub_ps(m5, acc46_neg);

            // t47 - t48: m0, m1
            __m256 acc47_neg = _mm256_setzero_ps();
            for (int i = t47; i < t48; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc47_neg = _mm256_add_ps(acc47_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc47_neg);
            m1 = _mm256_sub_ps(m1, acc47_neg);

            // t48 - t49: m0, m1, m5
            __m256 acc48_neg = _mm256_setzero_ps();
            for (int i = t48; i < t49; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc48_neg = _mm256_add_ps(acc48_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc48_neg);
            m1 = _mm256_sub_ps(m1, acc48_neg);
            m5 = _mm256_sub_ps(m5, acc48_neg);

            // t49 - t50: m0, m1, m4
            __m256 acc49_neg = _mm256_setzero_ps();
            for (int i = t49; i < t50; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc49_neg = _mm256_add_ps(acc49_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc49_neg);
            m1 = _mm256_sub_ps(m1, acc49_neg);
            m4 = _mm256_sub_ps(m4, acc49_neg);

            // t50 - t51: m0, m1, m4, m5
            __m256 acc50_neg = _mm256_setzero_ps();
            for (int i = t50; i < t51; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc50_neg = _mm256_add_ps(acc50_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc50_neg);
            m1 = _mm256_sub_ps(m1, acc50_neg);
            m4 = _mm256_sub_ps(m4, acc50_neg);
            m5 = _mm256_sub_ps(m5, acc50_neg);

            // t51 - t52: m0, m1, m3
            __m256 acc51_neg = _mm256_setzero_ps();
            for (int i = t51; i < t52; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc51_neg = _mm256_add_ps(acc51_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc51_neg);
            m1 = _mm256_sub_ps(m1, acc51_neg);
            m3 = _mm256_sub_ps(m3, acc51_neg);

            // t52 - t53: m0, m1, m3, m5
            __m256 acc52_neg = _mm256_setzero_ps();
            for (int i = t52; i < t53; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc52_neg = _mm256_add_ps(acc52_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc52_neg);
            m1 = _mm256_sub_ps(m1, acc52_neg);
            m3 = _mm256_sub_ps(m3, acc52_neg);
            m5 = _mm256_sub_ps(m5, acc52_neg);

            // t53 - t54: m0, m1, m3, m4
            __m256 acc53_neg = _mm256_setzero_ps();
            for (int i = t53; i < t54; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc53_neg = _mm256_add_ps(acc53_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc53_neg);
            m1 = _mm256_sub_ps(m1, acc53_neg);
            m3 = _mm256_sub_ps(m3, acc53_neg);
            m4 = _mm256_sub_ps(m4, acc53_neg);

            // t54 - t55: m0, m1, m3, m4, m5
            __m256 acc54_neg = _mm256_setzero_ps();
            for (int i = t54; i < t55; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc54_neg = _mm256_add_ps(acc54_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc54_neg);
            m1 = _mm256_sub_ps(m1, acc54_neg);
            m3 = _mm256_sub_ps(m3, acc54_neg);
            m4 = _mm256_sub_ps(m4, acc54_neg);
            m5 = _mm256_sub_ps(m5, acc54_neg);

            // t55 - t56: m0, m1, m2
            __m256 acc55_neg = _mm256_setzero_ps();
            for (int i = t55; i < t56; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc55_neg = _mm256_add_ps(acc55_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc55_neg);
            m1 = _mm256_sub_ps(m1, acc55_neg);
            m2 = _mm256_sub_ps(m2, acc55_neg);

            // t56 - t57: m0, m1, m2, m5
            __m256 acc56_neg = _mm256_setzero_ps();
            for (int i = t56; i < t57; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc56_neg = _mm256_add_ps(acc56_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc56_neg);
            m1 = _mm256_sub_ps(m1, acc56_neg);
            m2 = _mm256_sub_ps(m2, acc56_neg);
            m5 = _mm256_sub_ps(m5, acc56_neg);

            // t57 - t58: m0, m1, m2, m4
            __m256 acc57_neg = _mm256_setzero_ps();
            for (int i = t57; i < t58; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc57_neg = _mm256_add_ps(acc57_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc57_neg);
            m1 = _mm256_sub_ps(m1, acc57_neg);
            m2 = _mm256_sub_ps(m2, acc57_neg);
            m4 = _mm256_sub_ps(m4, acc57_neg);

            // t58 - t59: m0, m1, m2, m4, m5
            __m256 acc58_neg = _mm256_setzero_ps();
            for (int i = t58; i < t59; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc58_neg = _mm256_add_ps(acc58_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc58_neg);
            m1 = _mm256_sub_ps(m1, acc58_neg);
            m2 = _mm256_sub_ps(m2, acc58_neg);
            m4 = _mm256_sub_ps(m4, acc58_neg);
            m5 = _mm256_sub_ps(m5, acc58_neg);

            // t59 - t60: m0, m1, m2, m3
            __m256 acc59_neg = _mm256_setzero_ps();
            for (int i = t59; i < t60; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc59_neg = _mm256_add_ps(acc59_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc59_neg);
            m1 = _mm256_sub_ps(m1, acc59_neg);
            m2 = _mm256_sub_ps(m2, acc59_neg);
            m3 = _mm256_sub_ps(m3, acc59_neg);

            // t60 - t61: m0, m1, m2, m3, m5
            __m256 acc60_neg = _mm256_setzero_ps();
            for (int i = t60; i < t61; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc60_neg = _mm256_add_ps(acc60_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc60_neg);
            m1 = _mm256_sub_ps(m1, acc60_neg);
            m2 = _mm256_sub_ps(m2, acc60_neg);
            m3 = _mm256_sub_ps(m3, acc60_neg);
            m5 = _mm256_sub_ps(m5, acc60_neg);

            // t61 - t62: m0, m1, m2, m3, m4
            __m256 acc61_neg = _mm256_setzero_ps();
            for (int i = t61; i < t62; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc61_neg = _mm256_add_ps(acc61_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc61_neg);
            m1 = _mm256_sub_ps(m1, acc61_neg);
            m2 = _mm256_sub_ps(m2, acc61_neg);
            m3 = _mm256_sub_ps(m3, acc61_neg);
            m4 = _mm256_sub_ps(m4, acc61_neg);

            // t62 - t63: m0, m1, m2, m3, m4, m5
            __m256 acc62_neg = _mm256_setzero_ps();
            for (int i = t62; i < t63; ++i) {
                int idx = permutations_neg[b_idx * K + i];
                __m256 vec = _mm256_set_ps(v7[idx], v6[idx], v5[idx], v4[idx], v3[idx], v2[idx], v1[idx], v0[idx]);
                acc62_neg = _mm256_add_ps(acc62_neg, vec);
            }
            m0 = _mm256_sub_ps(m0, acc62_neg);
            m1 = _mm256_sub_ps(m1, acc62_neg);
            m2 = _mm256_sub_ps(m2, acc62_neg);
            m3 = _mm256_sub_ps(m3, acc62_neg);
            m4 = _mm256_sub_ps(m4, acc62_neg);
            m5 = _mm256_sub_ps(m5, acc62_neg);

            // Store results directly - works for any L value
            // Extract individual elements and store to correct positions
            alignas(32) float temp_m0[8], temp_m1[8], temp_m2[8], temp_m3[8], temp_m4[8], temp_m5[8];
            _mm256_store_ps(temp_m0, m0);
            _mm256_store_ps(temp_m1, m1);
            _mm256_store_ps(temp_m2, m2);
            _mm256_store_ps(temp_m3, m3);
            _mm256_store_ps(temp_m4, m4);
            _mm256_store_ps(temp_m5, m5);

            // Store results for each row (no bounds checking)
            result[(row + 0) * N + output_offset + 0] = temp_m0[0];
            result[(row + 0) * N + output_offset + 1] = temp_m1[0];
            result[(row + 0) * N + output_offset + 2] = temp_m2[0];
            result[(row + 0) * N + output_offset + 3] = temp_m3[0];
            result[(row + 0) * N + output_offset + 4] = temp_m4[0];
            result[(row + 0) * N + output_offset + 5] = temp_m5[0];
            result[(row + 1) * N + output_offset + 0] = temp_m0[1];
            result[(row + 1) * N + output_offset + 1] = temp_m1[1];
            result[(row + 1) * N + output_offset + 2] = temp_m2[1];
            result[(row + 1) * N + output_offset + 3] = temp_m3[1];
            result[(row + 1) * N + output_offset + 4] = temp_m4[1];
            result[(row + 1) * N + output_offset + 5] = temp_m5[1];
            result[(row + 2) * N + output_offset + 0] = temp_m0[2];
            result[(row + 2) * N + output_offset + 1] = temp_m1[2];
            result[(row + 2) * N + output_offset + 2] = temp_m2[2];
            result[(row + 2) * N + output_offset + 3] = temp_m3[2];
            result[(row + 2) * N + output_offset + 4] = temp_m4[2];
            result[(row + 2) * N + output_offset + 5] = temp_m5[2];
            result[(row + 3) * N + output_offset + 0] = temp_m0[3];
            result[(row + 3) * N + output_offset + 1] = temp_m1[3];
            result[(row + 3) * N + output_offset + 2] = temp_m2[3];
            result[(row + 3) * N + output_offset + 3] = temp_m3[3];
            result[(row + 3) * N + output_offset + 4] = temp_m4[3];
            result[(row + 3) * N + output_offset + 5] = temp_m5[3];
            result[(row + 4) * N + output_offset + 0] = temp_m0[4];
            result[(row + 4) * N + output_offset + 1] = temp_m1[4];
            result[(row + 4) * N + output_offset + 2] = temp_m2[4];
            result[(row + 4) * N + output_offset + 3] = temp_m3[4];
            result[(row + 4) * N + output_offset + 4] = temp_m4[4];
            result[(row + 4) * N + output_offset + 5] = temp_m5[4];
            result[(row + 5) * N + output_offset + 0] = temp_m0[5];
            result[(row + 5) * N + output_offset + 1] = temp_m1[5];
            result[(row + 5) * N + output_offset + 2] = temp_m2[5];
            result[(row + 5) * N + output_offset + 3] = temp_m3[5];
            result[(row + 5) * N + output_offset + 4] = temp_m4[5];
            result[(row + 5) * N + output_offset + 5] = temp_m5[5];
            result[(row + 6) * N + output_offset + 0] = temp_m0[6];
            result[(row + 6) * N + output_offset + 1] = temp_m1[6];
            result[(row + 6) * N + output_offset + 2] = temp_m2[6];
            result[(row + 6) * N + output_offset + 3] = temp_m3[6];
            result[(row + 6) * N + output_offset + 4] = temp_m4[6];
            result[(row + 6) * N + output_offset + 5] = temp_m5[6];
            result[(row + 7) * N + output_offset + 0] = temp_m0[7];
            result[(row + 7) * N + output_offset + 1] = temp_m1[7];
            result[(row + 7) * N + output_offset + 2] = temp_m2[7];
            result[(row + 7) * N + output_offset + 3] = temp_m3[7];
            result[(row + 7) * N + output_offset + 4] = temp_m4[7];
            result[(row + 7) * N + output_offset + 5] = temp_m5[7];

            output_offset += 6;

        }
    }

    // Upstream assumes complete eight-row batches. Reuse basic SSR for tails.
    if (M % 8 != 0) {
        ssr_inference_rows(v, permutations_pos, segments_pos, permutations_neg,
                           segments_neg, 6, M, N, K, result, M - M % 8);
    }

    return result;
}
#endif


void copy_output(const vec_t& padded, vec_t& Y, int M, int N, int L) {
    const int padded_N = N + (L - N % L) % L;
    Y.resize(static_cast<std::size_t>(M) * N);
    for (int row = 0; row < M; ++row) {
        std::copy_n(padded.data() + static_cast<std::size_t>(row) * padded_N, N,
                    Y.data() + static_cast<std::size_t>(row) * N);
    }
}

void validate(const vec_t& X, const Weights& W, const vec_t& Y, int M) {
    ternary_baseline_detail::validate_product(X, Y, M, W.K, W.N);
    const int padded_N = W.N + (W.block_width - W.N % W.block_width) % W.block_width;
    ternary_baseline_detail::checked_size(M, padded_N);
}

} // namespace

Weights preprocess(const tern_t& W, int K, int N, int block_width) {
    ternary_baseline_detail::validate_weights(W, K, N);
    if (block_width < 1 || block_width > 8 || N > INT_MAX - block_width) {
        throw std::invalid_argument("SSR segment width must be 1..8 with room for padding");
    }
    const int blocks = (N + block_width - 1) / block_width;
    ternary_baseline_detail::checked_size(blocks, K);
    ternary_baseline_detail::checked_size(blocks, 1 << block_width);

    std::vector<std::vector<int8_t>> matrix(K, std::vector<int8_t>(N));
    for (int row = 0; row < K; ++row) {
        for (int col = 0; col < N; ++col) {
            matrix[row][col] = static_cast<int8_t>(W[static_cast<std::size_t>(row) * N + col]);
        }
    }
    auto data = ssr_preprocess(matrix, block_width, K, N);
    return {K, N, block_width, std::move(data.first.first),
            std::move(data.second.first), std::move(data.first.second),
            std::move(data.second.second)};
}

void multiply_basic(const vec_t& X, const Weights& W, vec_t& Y, int M) {
    validate(X, W, Y, M);
    const auto result = ssr_inference(
        X, W.permutations_pos, W.segments_pos, W.permutations_neg, W.segments_neg,
        W.block_width, M, W.N, W.K);
    copy_output(result, Y, M, W.N, W.block_width);
}

void multiply_unrolled(const vec_t& X, const Weights& W, vec_t& Y, int M) {
    validate(X, W, Y, M);
#if TERNARY_BASELINE_HAS_AVX2
    if (avx2_available()) {
        // Upstream calls the inner dimension M and the batch dimension K.
        const auto result = ssr_inference_float_vec_unroll16_v2(
            X, W.permutations_pos, W.segments_pos, W.permutations_neg,
            W.segments_neg, W.block_width, W.K, W.N, M);
        copy_output(result, Y, M, W.N, W.block_width);
        return;
    }
#endif
    multiply_basic(X, W, Y, M);
}

void multiply_tree(const vec_t& X, const Weights& W, vec_t& Y, int M) {
    validate(X, W, Y, M);
#if TERNARY_BASELINE_HAS_AVX2
    if (avx2_available() && W.block_width >= 2 && W.block_width <= 6) {
        using Kernel = vec_t (*)(const vec_t&, const permutation_t&, const segment_t&,
                                 const permutation_t&, const segment_t&, int, int, int);
        static const Kernel kernels[] = {
            ssr_inference_float_vec_L2,
            ssr_inference_float_vec_L3,
            ssr_inference_float_vec_L4,
            ssr_inference_float_vec_L5,
            ssr_inference_float_vec_L6,
        };
        const auto result = kernels[W.block_width - 2](
            X, W.permutations_pos, W.segments_pos, W.permutations_neg,
            W.segments_neg, W.K, W.N, M);
        copy_output(result, Y, M, W.N, W.block_width);
        return;
    }
#endif
    multiply_unrolled(X, W, Y, M);
}

bool avx2_available() { return ternary_baseline_detail::avx2_available(); }

} // namespace ssr_baseline
