#pragma once

#include "types.h"

namespace ssr_baseline {

// Preprocess once and reuse for all FP32 variants. Segment width L (1..8) is
// independent of input dimension K. The default matches upstream's L=6 case.
struct Weights {
    int K;
    int N;
    int block_width;
    permutation_t permutations_pos;
    permutation_t permutations_neg;
    segment_t segments_pos;
    segment_t segments_neg;
};

Weights preprocess(const tern_t& W, int K, int N, int block_width = 6);

// Row-major Y = XW, overwriting Y. X and Y must not alias. Bias/activation are
// applied by the caller. Upstream inference allocates a padded temporary result.
void multiply_basic(const vec_t& X, const Weights& W, vec_t& Y, int M);
void multiply_unrolled(const vec_t& X, const Weights& W, vec_t& Y, int M);
void multiply_tree(const vec_t& X, const Weights& W, vec_t& Y, int M);

// Both optimized variants fall back to basic SSR without AVX2. The tree kernels
// specialize L=2..6; other widths use the AVX2 unrolled kernel.
bool avx2_available();

} // namespace ssr_baseline
