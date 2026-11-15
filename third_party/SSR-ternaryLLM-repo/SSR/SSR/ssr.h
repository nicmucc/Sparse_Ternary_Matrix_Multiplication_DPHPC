#pragma once
#include <vector>
#include <utility>

using namespace std;

pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>> ssr_preprocess(vector<vector<int8_t>>& mat, int L, int K, int N);

pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_int8(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity);
pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_float(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity);
pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_int8_vec_tree(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity);
pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_int8_vec_unroll(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity);
pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_float_vec_tree(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity);
pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_float_vec_unroll(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity);
// SSR Inference - Base implementation
template <typename T>
vector<T> binary_ssr_inference(vector<T> v, const vector<vector<int>>& permutations, const vector<vector<int>>& segments, int L) {
    int K = v.size();
    int N = permutations.size() * L;

    int block_size = L;
    int num_blocks = N/L;
    int num_bins = 1 << block_size;

    vector<T> result(N);
    vector<T> seg_sum(num_bins-1);

    int output_offset = 0;

    for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {

        vector<int> perm = permutations[b_idx];
        vector<int> seg = segments[b_idx];

        // Step 1: Apply permutation
        T sum; 
        int len = num_bins-1;
        int start = seg[0];

        // Step 2: Segmented sum
        for (int bin = 0; bin < len; ++bin) {
            int end = seg[bin + 1];
            sum = 0;
            for (int i = start; i < end; ++i){
                sum += v[perm[i]];
            }
            seg_sum[bin] = sum;
            start = end;
        }

        //Step 3 tree combination

        for (int j = block_size - 1; j >= 0; --j) {
            T sum = seg_sum[0];
            int new_len = len/2;
            for (int i = 0; i < new_len; ++i) {
                sum += seg_sum[2 * i + 2];
                seg_sum[i] = seg_sum[2 * i + 1] + seg_sum[2 * i + 2];
            }
            result[output_offset + j] = sum;
            len = new_len;
        }

        output_offset += block_size;
    }

    return result;
}

template <typename T>
vector<T> vector_ssr_inference(vector<T> v, const vector<vector<int>>& permutations_pos, const vector<vector<int>>& segments_pos, const vector<vector<int>>& permutations_neg, const vector<vector<int>>& segments_neg, int L) {
    int K = v.size();
    int N = permutations_pos.size() * L;

    int block_size = L;
    int num_blocks = N/L;
    int num_bins = 1 << block_size;

    vector<T> result(N);
    vector<vector<T>> seg_sum(2, vector<T>(num_bins-1));

    int output_offset = 0;

    for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {

        vector<int> perm_pos = permutations_pos[b_idx];
        vector<int> seg_pos = segments_pos[b_idx];
        vector<int> perm_neg = permutations_neg[b_idx];
        vector<int> seg_neg = segments_neg[b_idx];

        // Step 1: Apply permutation
        // first index stores the length not used as permuatation
        T sum_pos;
        T sum_neg; 
        int len = num_bins-1;
        int start_pos = seg_pos[0];
        int start_neg = seg_neg[0];

        // Step 2: Segmented sum
        for (int bin = 0; bin < len; ++bin) {

            int end_pos = seg_pos[bin + 1];
            int end_neg = seg_neg[bin + 1];
            
            sum_pos = 0;
            sum_neg = 0;
            for (int i = start_pos; i < end_pos; ++i){
                sum_pos += v[perm_pos[i]];
                
            }
            for (int i = start_neg; i < end_neg; ++i){
                sum_neg += v[perm_neg[i]];
                
            }
            seg_sum[0][bin] = sum_pos;
            seg_sum[1][bin] = sum_neg;
            start_pos = end_pos;
            start_neg = end_neg;
        }

        //Step 3 other option - small speedup compared to above version RSR++:

        for (int j = block_size - 1; j >= 0; --j) {

            // Reduce vector: x_tmp[i] = x_tmp[2i] + x_tmp[2i + 1]

            T sum_pos = seg_sum[0][0];
            T sum_neg = seg_sum[1][0];
            int new_len = len/2;
            for (int i = 0; i < new_len; ++i) {
                sum_pos += seg_sum[0][2 * i + 2];
                sum_neg += seg_sum[1][2 * i + 2];
                seg_sum[0][i] = seg_sum[0][2 * i + 1] + seg_sum[0][2 * i + 2];
                seg_sum[1][i] = seg_sum[1][2 * i + 1] + seg_sum[1][2 * i + 2];
            }
            result[output_offset + j] = sum_pos - sum_neg;
            len = new_len;
        }

        output_offset += block_size;
    }

    return result;
}

template <typename T>
vector<T> ssr_inference(vector<T> v, const vector<int>& permutations_pos, const vector<int>& segments_pos, const vector<int>& permutations_neg, const vector<int>& segments_neg, int L, int M, int N, int K) {

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
                    sum_pos += v[row * L + permutations_pos[b_idx * L + i]];

                }
                for (int i = start_neg; i < end_neg; ++i) {
                    sum_neg += v[row * L + permutations_neg[b_idx * L + i]];

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