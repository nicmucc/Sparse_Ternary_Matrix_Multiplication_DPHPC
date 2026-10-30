#pragma once

#include <cstdint>
#include <vector>
#include "types.h"

namespace tcsc_baseline {

struct Weights {
    int K;
    int N;
    std::vector<std::int32_t> col_start_pos;
    std::vector<std::int32_t> col_start_neg;
    std::vector<std::int32_t> row_index_pos;
    std::vector<std::int32_t> row_index_neg;
    // GroupMin representation for complete groups of eight output columns.
    std::vector<std::int32_t> metadata;
    std::vector<std::int32_t> row_index;
};

Weights preprocess(const tern_t& W, int K, int N);

// Row-major Y = XW, overwriting Y. X and Y must not alias. The optimized
// implementation uses AVX2 GroupMin plus a scalar column tail, or basic TCSC
// without AVX2. Indices are widened to 32 bits from upstream.
void multiply_basic(const vec_t& X, const Weights& W, vec_t& Y, int M);
void multiply_merged(const vec_t& X, const Weights& W, vec_t& Y, int M);
bool avx2_available();

} // namespace tcsc_baseline
