#pragma once
#include <vector>
#include <utility>

using namespace std;

//
// INT8 VERSIONS
//

// UNROLLING VERSION
vector<int8_t> ssr_inference_int8_vec_unroll8_v2(
    const vector<int8_t>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int L, int M, int N, int K);
// unrolling 32 (to fill vector length)
// version 1:
vector<vector<int8_t>> ssr_inference_int8_vec_unroll32(
    const vector<vector<int8_t>>& v, 
    const vector<vector<int>>& permutations_pos, 
    const vector<vector<int>>& segments_pos, 
    const vector<vector<int>>& permutations_neg, 
    const vector<vector<int>>& segments_neg,
    int L);

// version 2:
vector<vector<int8_t>> ssr_inference_int8_vec_unroll32_v2(
    const vector<vector<int8_t>>& v, 
    const vector<vector<int>>& permutations_pos, 
    const vector<vector<int>>& segments_pos, 
    const vector<vector<int>>& permutations_neg, 
    const vector<vector<int>>& segments_neg,
    int L);

// TREE ACCUMULATOR VERSION

vector<int8_t> ssr_inference_int8_vec_tree(
    const vector<int8_t>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    const int M, const int K, const int N, const int selectedK);
// L = 2
vector<int8_t> ssr_inference_int8_vec_L2(
    const vector<int8_t>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 3
vector<int8_t> ssr_inference_int8_vec_L3(
    const vector<int8_t>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 4
vector<int8_t> ssr_inference_int8_vec_L4(
    const vector<int8_t>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 5
vector<int8_t> ssr_inference_int8_vec_L5(
    const vector<int8_t>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 6
vector<int8_t> ssr_inference_int8_vec_L6(
    const vector<int8_t>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 7
vector<int8_t> ssr_inference_int8_vec_L7(
    const vector<int8_t>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 8
vector<int8_t> ssr_inference_int8_vec_L8(
    const vector<int8_t>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

//
// FLOAT VERSIONS
//

// UNROLLING VERSIONS
// unrolling 8 (to fill vector length)

// version 2: 
vector<float> ssr_inference_float_vec_unroll8_v2(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int L, int M, int N, int K);

// unrolling factor 16 

// version 2:
vector<float> ssr_inference_float_vec_unroll16_v2(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int L, int M, int N, int K);

// TREE ACCUMULATOR VERSIONS

vector<float> ssr_inference_float_vec_tree(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    const int M, const int K, const int N, const int selectedK);

// L = 2
vector<float> ssr_inference_float_vec_L2(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 3
vector<float> ssr_inference_float_vec_L3(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 4
vector<float> ssr_inference_float_vec_L4(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 5
vector<float> ssr_inference_float_vec_L5(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 6
vector<float> ssr_inference_float_vec_L6(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 7
vector<float> ssr_inference_float_vec_L7(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);

// L = 8
vector<float> ssr_inference_float_vec_L8(
    const vector<float>& v,
    const vector<int>& permutations_pos,
    const vector<int>& segments_pos,
    const vector<int>& permutations_neg,
    const vector<int>& segments_neg,
    int K, int N, int M);