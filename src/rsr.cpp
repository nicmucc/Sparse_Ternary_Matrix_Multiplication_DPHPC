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
    int n = mat_block.size();
    int k = mat_block[0].size();

    // Permutation
    vector<int> permutation(n);
    for (int i = 0; i < n; i++) {
        permutation[i] = i;
    }

    sort(permutation.begin(), permutation.end(), [&](int i, int j) {
        // return vec[i] < vec[j]
        return binaryVectorToInt(mat_block[i]) < binaryVectorToInt(mat_block[j]);
    });

    
    // Segmentation
    vector<int> seg(pow(2, k), -1);
    seg[0] = 0;
    for (int row = 0; row < n; row++) {
        int value = binaryVectorToInt(mat_block[permutation[row]]);
        if (seg[value] == -1) {
            seg[value] = row;
        }
    }
    if (seg[seg.size() - 1] == -1) {
        seg[seg.size() - 1] = n;
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

pair<vector<vector<int>>, vector<vector<int>>> preprocess(vector<vector<int>>& mat, int k) {
    if (mat.empty() || k <= 0) {
        throw invalid_argument("preprocess requires a non-empty matrix and positive block width");
    }

    const int row_count = mat.size();
    const int column_count = mat.front().size();

    // Pad output columns so they can be processed in blocks of k. The input
    // row count must stay unchanged because it determines the input vector size.
    const int padding = (k - column_count % k) % k;
    for (auto& row : mat) {
        if (row.size() != static_cast<size_t>(column_count)) {
            throw invalid_argument("matrix rows must have equal lengths");
        }
        row.resize(column_count + padding, 0);
    }

    const int padded_column_count = column_count + padding;

    vector<vector<int>> permutations(padded_column_count / k, vector<int>(row_count));
    vector<vector<int>> segs(padded_column_count / k, vector<int>(pow(2, k)));


    // Blocking
    int start;
    int end;
    vector<vector<int>> block(row_count, vector<int>(k));
    for (int i = 0; i < padded_column_count / k; i++) {
        // cout << "block " << i + 1 << " out of " << n / k << " blocks" << endl;
        start = i * k;
        end = start + k;
        for (int col = start; col < end; col++) {
            for (int row = 0; row < row_count; row++) {
                block[row][col - start] = mat[row][col];
            }
        }
        auto per_seg = handle_block(block);
        permutations[i] = per_seg.first;
        segs[i] = per_seg.second;
    }

    return make_pair(permutations, segs);
}

vec_t rsr_inference(const vec_t& v, const permutation_t& permutations, const segment_t& segments, const binary_matrix_t& bin_k, size_t N, int k) {
     // segmented sums
    const size_t block_size = size_t{1} << k;
    const size_t num_blocks = permutations.size() / N;
    vec_t us(num_blocks * block_size, 0.0f);

    size_t start;
    size_t end;
    for (size_t i = 0; i < num_blocks; i++) {
        // Each block
        for (size_t j = 0; j < block_size; j++) {
            start = segments[i * block_size + j];
            if (j + 1 < block_size) {
                end = segments[i * block_size + j + 1];
            } else {
                end = N;
            }
            // Segmented sum
            for (size_t index = start; index < end; index++) {
                us[i * block_size + j] +=
                    v[permutations[i * N + index]];
            }
        }
    }

    vec_t result(num_blocks * k);

    // Block product to Bin_k
    for (size_t i = 0; i < num_blocks; i++) {
        vec_t block(us.begin() + i * block_size,
                    us.begin() + (i + 1) * block_size);
        vec_t partial_result(k, 0.0f);
        vectorMatrixMultiply(block, bin_k, partial_result);

        for (int j = 0; j < k; j++) {
            result[i * k + j] = partial_result[j];
        }
    }
    return result;
}
