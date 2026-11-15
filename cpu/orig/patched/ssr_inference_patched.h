#pragma once
// PATCHED copy of ssr_inference ("SSR Naive", the scalar SSR kernel of Pittet et al., DATE 2026)
// from third_party/SSR-ternaryLLM-repo/SSR/SSR/ssr.h (commit 68d3844).
//
// The ONLY change is an indexing bug fix in the segmented-sum gather (4 occurrences):
//
//   upstream:  v[row * L + permutations_pos[b_idx * L + i]]
//   patched:   v[row * K + permutations_pos[b_idx * K + i]]          (same for _neg)
//
// - a row of X (v, row-major M x K) has K values, not L
// - ssr_preprocess stores K permutation entries per block (ssr.cpp:48), not L
//
// With the upstream indexing, 150/180 test cases fail (they only pass when nothing is gathered,
// e.g. 100% sparsity). Everything else -- the format, the loops, the tree, the OpenMP pragma --
// is byte-for-byte the authors' code, only the function name differs.
#include <vector>

template <typename T>
std::vector<T> ssr_inference_patched(std::vector<T> v, const std::vector<int>& permutations_pos, const std::vector<int>& segments_pos, const std::vector<int>& permutations_neg, const std::vector<int>& segments_neg, int L, int M, int N, int K) {
    using std::vector;

    N = N + (L - N % L) % L;
    int block_size = L;
    int num_blocks = N / L;
    int num_bins = 1 << block_size;

    vector<T> result(M * N);

#pragma omp parallel for
    for (int row = 0; row < M; ++row) {
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
                    sum_pos += v[row * K + permutations_pos[b_idx * K + i]];   // PATCHED (was row * L, b_idx * L)

                }
                for (int i = start_neg; i < end_neg; ++i) {
                    sum_neg += v[row * K + permutations_neg[b_idx * K + i]];   // PATCHED (was row * L, b_idx * L)

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

    return result;
}
