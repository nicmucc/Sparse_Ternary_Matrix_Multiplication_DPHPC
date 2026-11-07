#pragma once

#include <climits>
#include <cstddef>
#include <stdexcept>
#include "types.h"

// Compile AVX2 functions individually; keep the executable's default ISA.
#if !defined(TERNARY_BASELINE_DISABLE_AVX2) && \
    (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define TERNARY_BASELINE_HAS_AVX2 1
#define TERNARY_BASELINE_AVX2_TARGET __attribute__((target("avx2")))
#else
#define TERNARY_BASELINE_HAS_AVX2 0
#endif

namespace ternary_baseline_detail {

inline bool avx2_available() {
#if TERNARY_BASELINE_HAS_AVX2
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}


// The imported kernels use signed int arithmetic for dimensions and offsets.
inline std::size_t checked_size(int rows, int columns) {
    if (rows <= 0 || columns <= 0 ||
        static_cast<std::size_t>(rows) * columns > INT_MAX) {
        throw std::invalid_argument("ternary baseline dimensions exceed supported range");
    }
    return static_cast<std::size_t>(rows) * columns;
}

inline void validate_weights(const tern_t& W, int K, int N) {
    if (W.size() != checked_size(K, N)) {
        throw std::invalid_argument("ternary weight size must equal K*N");
    }
    for (int value : W) {
        if (value < -1 || value > 1) {
            throw std::invalid_argument("weights must be in {-1, 0, +1}");
        }
    }
}

inline void validate_product(const vec_t& X, const vec_t& Y, int M, int K, int N) {
    if (&X == &Y) {
        throw std::invalid_argument("output must not alias the input");
    }
    if (X.size() != checked_size(M, K)) {
        throw std::invalid_argument("input size must equal M*K");
    }
    checked_size(M, N);
}

} // namespace ternary_baseline_detail
