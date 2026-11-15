#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "utility"
#include "utils.h"
#include "naive.h"

using namespace std;

// Return <permutation, segmentation>
pair<vector<int>, vector<int>> handle_block(vector<vector<int>> mat_block) {
    int K = mat_block.size();
    int L = mat_block[0].size();

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
    vector<int> seg(pow(2, L), -1);
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

pair<vector<vector<int>>, vector<vector<int>>> binary_rsr_preprocess(vector<vector<int8_t>>& mat, int L) {
    int K = mat.size();
    int N = mat[0].size();

    // Padding
    int padding = (L - N % L) % L;
    for (auto& row : mat) {
        row.resize(row.size() + padding, 0);
    }
    /*
    for (int i = 0; i < padding; i++) {
        mat.push_back(vector<int>(n + padding, 0));
    }
    */
    N = N + padding;

    vector<vector<int>> permutations(N/L);
    vector<vector<int>> segments(N/L);
    
    // Blocking
    int start;
    int end;
    vector<vector<int>> block(K, vector<int>(L));

    //int seg_size =  (1 << L) - 1;

    for(int i = 0; i < N / L; i++) {
        
        start = i * L;
        end = start + L;
        for (int col = start; col < end; col++) {
            for (int row = 0; row < K; row++) {
                block[row][col - start] = int(mat[row][col]);
            }
        }
        auto per_seg = handle_block(block);
        permutations[i] = per_seg.first;
        segments[i] = per_seg.second;

        
    }

    return make_pair(permutations, segments);
}

pair<pair<vector<vector<int>>, vector<vector<int>>>, pair<vector<vector<int>>, vector<vector<int>>>> rsr_preprocess(vector<vector<int8_t>>& mat, int L) {
    int K = mat.size();
    int N = mat[0].size();

    // Padding

    int padding = (L - N % L) % L;

    vector<vector<int>> permutations_pos((N + padding) / L);
    vector<vector<int>> segments_pos((N + padding) / L);
    vector<vector<int>> permutations_neg((N + padding) / L);
    vector<vector<int>> segments_neg((N + padding) / L);

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
        auto per_seg_pos = handle_block(block_pos);
        auto per_seg_neg = handle_block(block_neg);
        permutations_pos[i] = per_seg_pos.first;
        segments_pos[i] = per_seg_pos.second;
        permutations_neg[i] = per_seg_neg.first;
        segments_neg[i] = per_seg_neg.second;


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
        auto per_seg_pos = handle_block(block_pos);
        auto per_seg_neg = handle_block(block_neg);
        permutations_pos[((N + padding) / L) - 1] = per_seg_pos.first;
        segments_pos[((N + padding) / L) - 1] = per_seg_pos.second;
        permutations_neg[((N + padding) / L) - 1] = per_seg_neg.first;
        segments_neg[((N + padding) / L) - 1] = per_seg_neg.second;
    }
    auto pos = make_pair(permutations_pos, segments_pos);
    auto neg = make_pair(permutations_neg, segments_neg);

    return make_pair(pos, neg);
}

pair<pair<pair<vector<vector<int>>, vector<vector<int>>>, pair<vector<vector<int>>, vector<vector<int>>>>, int> rsr_preprocess_float(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity)
{
std:vector<int> ssr_L_list = {
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7 , // 1024x4096, 4096x1024
    6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7 , // 2048x8192
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 };// 4096x16384
int spIndex = int((sparsity - 0.45) / 0.05);
int LRow = 1;
if (K == 1024) {
    LRow = 0;
}
else if (K == 2048) {
    LRow = 1;
}
else if (K == 4096) {
    LRow = 2;
}
int selectedL = 6;
if ((-1 < spIndex) && (spIndex < 11))
    selectedL = ssr_L_list[LRow * 11 + spIndex];

auto preprocessed_mat = rsr_preprocess(mat, selectedL);
return make_pair(preprocessed_mat, selectedL);


}

pair<pair<pair<vector<vector<int>>, vector<vector<int>>>, pair<vector<vector<int>>, vector<vector<int>>>>, int> rsr_preprocess_int8(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity)
{
std:vector<int> ssr_L_list = {
    7, 6, 7, 7, 7, 7, 6, 7, 7, 7, 7 , // 1024x4096, 4096x1024
    7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, // 2048x8192
    8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9 };// 4096x16384
int spIndex = int((sparsity - 0.45) / 0.05);
int LRow = 1;
if (K == 1024) {
    LRow = 0;
}
else if (K == 2048) {
    LRow = 1;
}
else if (K == 4096) {
    LRow = 2;
}
int selectedL = 6;
if ((-1 < spIndex) && (spIndex < 11))
    selectedL = ssr_L_list[LRow * 11 + spIndex];

auto preprocessed_mat = rsr_preprocess(mat, selectedL);
return make_pair(preprocessed_mat, selectedL);


}