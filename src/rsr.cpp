#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include "utils.h"
#include "naive.h"

using namespace std;

// Return <permutation, segmentation>
pair<vector<int>, vector<int>> handle_block(const vector<vector<int>>& mat_block) {
    int K = mat_block.size();
    int block_width = mat_block[0].size();

    // Permutation
    vector<int> permutation(K);
    for (int i = 0; i < K; i++) {
        permutation[i] = i;
    }

    sort(permutation.begin(), permutation.end(), [&](int i, int j) {
        // return vec[i] < vec[j]
        return binaryVectorToInt(mat_block[i]) < binaryVectorToInt(mat_block[j]);
    });

    
    // Segmentation
    vector<int> seg(pow(2, block_width), -1);
    seg[0] = 0;
    for (int row = 0; row < K; row++) {
        int value = binaryVectorToInt(mat_block[permutation[row]]);
        if (seg[value] == -1) {
            seg[value] = row;
        }
    }
    if (seg[seg.size() - 1] == -1) {
        seg[seg.size() - 1] = K;
    }
    int last_one = seg[seg.size() - 1];
    for (int i = seg.size() - 2; i >= 0; i--) {
        if (seg[i] == -1) {
            seg[i] = last_one;
        }
        last_one = seg[i];
    }

    return make_pair(permutation, seg);
}

pair<vector<vector<int>>, vector<vector<int>>> preprocess(vector<vector<int>>& mat, int block_width) {
    if (mat.empty() || block_width <= 0) {
        throw invalid_argument("preprocess requires a non-empty matrix and positive block width");
    }

    const int K = mat.size();
    const int N = mat.front().size();

    // Pad output columns so they can be processed in blocks of block_width. The input
    // row count must stay unchanged because it determines the input vector size.
    const int padding = (block_width - N % block_width) % block_width;
    for (auto& row : mat) {
        if (row.size() != static_cast<size_t>(N)) {
            throw invalid_argument("matrix rows must have equal lengths");
        }
        row.resize(N + padding, 0);
    }

    const int padded_N = N + padding;

    vector<vector<int>> permutations(padded_N / block_width, vector<int>(K));
    vector<vector<int>> segs(padded_N / block_width, vector<int>(pow(2, block_width)));


    // Blocking
    int start;
    int end;
    vector<vector<int>> block(K, vector<int>(block_width));
    for (int i = 0; i < padded_N / block_width; i++) {
        // cout << "block " << i + 1 << " out of " << padded_N / block_width << " blocks" << endl;
        start = i * block_width;
        end = start + block_width;
        for (int col = start; col < end; col++) {
            for (int row = 0; row < K; row++) {
                block[row][col - start] = mat[row][col];
            }
        }
        auto per_seg = handle_block(block);
        permutations[i] = per_seg.first;
        segs[i] = per_seg.second;
    }

    return make_pair(permutations, segs);
}

vec_t rsr_inference(const vec_t& v, const permutation_t& permutations, const segment_t& segments, const binary_matrix_t& binary_patterns, size_t K, int block_width) {
     // segmented sums
    const size_t pattern_count = size_t{1} << block_width;
    const size_t num_blocks = permutations.size() / K;
    vec_t us(num_blocks * pattern_count, 0.0f);

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
