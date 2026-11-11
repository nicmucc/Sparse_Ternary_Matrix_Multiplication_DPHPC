#pragma once

#include <cstdint>
#include <vector>
#include <utility>
#include "types.h"

using namespace std;

pair<vector<vector<int>>, vector<vector<int>>> preprocess(
    const tern_t& mat, int block_width, int K);

vec_t rsr_inference(const vec_t& v, const permutation_t& permutations, const segment_t& segments, const binary_matrix_t& binary_patterns, size_t K, int block_width);

vec_t rsr_pp_inference(const vec_t& v, const permutation_t& permutations, const segment_t& segments, size_t K, int block_width);

namespace rsr_core_baseline {

enum class Variant { binary_v42, ternary_v31, ternary_v33 };

template<class Index, class Row>
struct Groups {
    std::vector<Index> perm, ends;
    std::vector<int32_t> blocks, offsets;
    std::vector<Row> rows;
    std::vector<int8_t> signs;
    std::vector<uint16_t> positive_masks, negative_masks;
};

struct Weights {
    int K, N, width, padded_N;
    Variant variant;
    Groups<int32_t, uint8_t> positive, negative;
    Groups<uint16_t, int8_t> ternary;
    std::size_t storage_bytes = 0;
};

Weights preprocess(const tern_t& W, int K, int N, int width, Variant variant);
// Row-major Y = XW. Scratch is reusable across inputs and variants.
void multiply(const vec_t& X, const Weights& W, vec_t& Y, int M, vec_t& scratch);
const char* backend(Variant variant, int width);

} // namespace rsr_core_baseline
