#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <cstdint>
#include "utility"
#include "original_rsr/utils.h"

using namespace std;

//
// SSR Preprocessing
//

// Return <permutation, segmentation>
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

pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>> ssr_preprocess(vector<vector<int8_t>>& mat, int L, int K, int N) {

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

//
// Wrapper functions to select L based on sparsity and matrix dimensions
// Optimal L values might be different if run on different hardware
//

//Base implementations
pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_int8(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity)
{
vector<int> ssr_L_list = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3 , // 1024x4096
    5, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4 , // 2048x8192
    6, 6, 6, 6, 6, 5, 5, 5, 4, 4, 4 , // 4096x16384
    6, 5, 5, 5, 5, 4, 4, 4, 5, 4, 4 , // 4096x1024
    7, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4 , // 8192x2048
    8, 8, 7, 8, 7, 7, 6, 6, 6, 6, 5 };// 16384x4096
int spIndex = int((sparsity - 0.45) / 0.05);
int LRow = 1;
if (K == 1024 && N == 4096) {
    LRow = 0;
}
else if (K == 2048 && N == 8192) {
    LRow = 1;
}
else if (K == 4096 && N == 16384) {
    LRow = 2;
}
else if (K == 4096 && N == 1024) {
    LRow = 3;
}
else if (K == 8192 && N == 2048) {
    LRow = 4;
}
else if (K == 16384 && N == 4096) {
    LRow = 5;
}
int selectedL = 6;
if ((-1 < spIndex) && (spIndex < 11))
    selectedL = ssr_L_list[LRow * 11 + spIndex];

auto preprocessed_mat = ssr_preprocess(mat, selectedL, K, N);
return make_pair(preprocessed_mat, selectedL);


}

pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_float(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity)
{
vector<int> ssr_L_list = {
    5, 4, 4, 4, 4, 4, 3, 4, 4, 3, 3 , // 1024x4096
    6, 6, 5, 5, 5, 5, 5, 4, 4, 5, 4 , // 2048x8192
    7, 6, 6, 6, 6, 6, 6, 5, 5, 4, 4 , // 4096x16384
    6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 3 , // 4096x1024
    7, 7, 7, 7, 7, 7, 6, 7, 5, 5, 4 , // 8192x2048
    8, 9, 9, 9, 8, 7, 8, 8, 6, 6, 6 };// 16384x4096
int spIndex = int((sparsity - 0.45) / 0.05);
int LRow = 1;
if (K == 1024 && N == 4096) {
    LRow = 0;
}
else if (K == 2048 && N == 8192) {
    LRow = 1;
}
else if (K == 4096 && N == 16384) {
    LRow = 2;
}
else if (K == 4096 && N == 1024) {
    LRow = 3;
}
else if (K == 8192 && N == 2048) {
    LRow = 4;
}
else if (K == 16384 && N == 4096) {
    LRow = 5;
}
int selectedL = 6;
if ((-1 < spIndex) && (spIndex < 11))
    selectedL = ssr_L_list[LRow * 11 + spIndex];

auto preprocessed_mat = ssr_preprocess(mat, selectedL, K, N);
return make_pair(preprocessed_mat, selectedL);


}

// Vectorized implementations
pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_int8_vec_tree(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity)
{
std:vector<int> ssr_L_list = {
    5, 6, 4, 6, 5, 6, 4, 5, 6, 6, 5 ,  // 1024x4096
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 ,  // 2048x8192
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5 ,  // 4096x16384
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 ,  // 4096x1024
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 ,  // 8192x2048
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 }; // 16384x4096
int spIndex = int((sparsity - 0.45) / 0.05);
int LRow = 1;
if (K == 1024 && N == 4096) {
    LRow = 0;
}
else if (K == 2048 && N == 8192) {
    LRow = 1;
}
else if (K == 4096 && N == 16384) {
    LRow = 2;
}
else if (K == 4096 && N == 1024) {
    LRow = 3;
}
else if (K == 8192 && N == 2048) {
    LRow = 4;
}
else if (K == 16384 && N == 4096) {
    LRow = 5;
}
int selectedL = 6;
if ((-1 < spIndex) && (spIndex < 11)) {
    // L capped at 6 for faster compile time while testing
    // ssr_inference_float_vec_tree in ssr_vec.cpp needs to be modified accordingly if higher L values are to be used
    selectedL = min(ssr_L_list[LRow * 11 + spIndex], 6);
    // selectedL = ssr_L_list[LRow * 11 + spIndex];
}

auto preprocessed_mat = ssr_preprocess(mat, selectedL, K, N);
return make_pair(preprocessed_mat, selectedL);


}

pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_int8_vec_unroll(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity)
{
std:vector<int> ssr_L_list = {
    8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 5 ,           // 1024x4096
    11, 11, 11, 11, 11, 11, 11, 10, 10, 9, 6 ,  // 2048x8192
    13, 13, 13, 13, 13, 13, 13, 13, 11, 10, 9 , // 4096x16384
    11, 11, 12, 12, 11, 12, 12, 11, 11, 10, 8 , // 4096x1024
    13, 12, 12, 13, 13, 12, 13, 12, 12, 11, 9 , // 8192x2048
    14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 12};// 16384x4096
int spIndex = int((sparsity - 0.45) / 0.05);
int LRow = 1;
if (K == 1024 && N == 4096) {
    LRow = 0;
}
else if (K == 2048 && N == 8192) {
    LRow = 1;
}
else if (K == 4096 && N == 16384) {
    LRow = 2;
}
else if (K == 4096 && N == 1024) {
    LRow = 3;
}
else if (K == 8192 && N == 2048) {
    LRow = 4;
}
else if (K == 16384 && N == 4096) {
    LRow = 5;
}
int selectedL = 6;
if ((-1 < spIndex) && (spIndex < 11))

    selectedL = ssr_L_list[LRow * 11 + spIndex];

auto preprocessed_mat = ssr_preprocess(mat, selectedL, K, N);
return make_pair(preprocessed_mat, selectedL);
}

pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_float_vec_tree(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity)
{
std:vector<int> ssr_L_list = {
    6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 4 , // 1024x4096
    8, 8, 8, 8, 8, 8, 8, 7, 7, 5, 2 , // 2048x8192
    8, 8, 8, 8, 8, 8, 8, 8, 7, 4, 5 , // 4096x16384
    8, 8, 8, 8, 6, 6, 6, 5, 4, 4, 4 , // 4096x1024
    8, 8, 7, 7, 8, 7, 3, 3, 3, 3, 3 , // 8192x2048
    8, 8, 8, 8, 7, 8, 8, 7, 5, 4, 4 };// 16384x4096
int spIndex = int((sparsity - 0.45) / 0.05);
int LRow = 1;
if (K == 1024 && N == 4096) {
    LRow = 0;
}
else if (K == 2048 && N == 8192) {
    LRow = 1;
}
else if (K == 4096 && N == 16384) {
    LRow = 2;
}
else if (K == 4096 && N == 1024) {
    LRow = 3;
}
else if (K == 8192 && N == 2048) {
    LRow = 4;
}
else if (K == 16384 && N == 4096) {
    LRow = 5;
}
int selectedL = 6;
if ((-1 < spIndex) && (spIndex < 11)) {
    // L capped at 6 for faster compile time while testing
    // ssr_inference_float_vec_tree in ssr_vec.cpp needs to be modified accordingly if higher L values are to be used
    selectedL = min(ssr_L_list[LRow * 11 + spIndex], 6);
    // selectedL = ssr_L_list[LRow * 11 + spIndex];
}

auto preprocessed_mat = ssr_preprocess(mat, selectedL, K, N);
return make_pair(preprocessed_mat, selectedL);


}

pair<pair<pair<vector<int>, vector<int>>, pair<vector<int>, vector<int>>>, int> ssr_preprocess_float_vec_unroll(
    vector<vector<int8_t>>& mat,
    const int K, const int N, const float sparsity)
{
std:vector<int> ssr_L_list = {
    7, 8, 7, 7, 7, 7, 6, 6, 5, 5, 4 ,               // 1024x4096
    8, 8, 8, 9, 8, 8, 8, 7, 7, 7, 5 ,               // 2048x8192
    9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 6 ,               // 4096x16384
    9, 9, 8, 9, 9, 9, 8, 8, 8, 7, 6 ,               // 4096x1024
    10, 10, 10, 10, 10, 10, 10, 10, 8, 7, 6 ,       // 8192x2048
    11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 8 };    // 16384x4096
int spIndex = int((sparsity - 0.45) / 0.05);
int LRow = 1;
if (K == 1024 && N == 4096) {
    LRow = 0;
}
else if (K == 2048 && N == 8192) {
    LRow = 1;
}
else if (K == 4096 && N == 16384) {
    LRow = 2;
}
else if (K == 4096 && N == 1024) {
    LRow = 3;
}
else if (K == 8192 && N == 2048) {
    LRow = 4;
}
else if (K == 16384 && N == 4096) {
    LRow = 5;
}
int selectedL = 6;
if ((-1 < spIndex) && (spIndex < 11))
    selectedL = ssr_L_list[LRow * 11 + spIndex];

auto preprocessed_mat = ssr_preprocess(mat, selectedL, K, N);
return make_pair(preprocessed_mat, selectedL);


}