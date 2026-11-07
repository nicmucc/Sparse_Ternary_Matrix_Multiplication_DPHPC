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
