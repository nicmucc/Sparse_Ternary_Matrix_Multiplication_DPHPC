#include "blas_baseline.h"

#include <cblas.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

// Added it because with large N we risk overflow in the conversion from int to size::t
std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
    if (lhs != 0 &&
        rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error("Matrix size overflow");
    }
    return lhs * rhs;
}

int to_blas_int(std::size_t value) {
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "Matrix dimension exceeds the BLAS integer range");
    }
    return static_cast<int>(value);
}

} // namespace

namespace blas_baseline {

vec_t int2float(const tern_t& matrix) {
    vec_t out(matrix.size());
    std::transform(matrix.begin(), matrix.end(), out.begin(),
                   [](int value) { return static_cast<float>(value); });
    return out;
}

void multiply(const vec_t& X, const vec_t& W, vec_t& Y,
              std::size_t M, std::size_t K, std::size_t N) {
    if (X.size() != checked_product(M, K)) {
        throw std::invalid_argument("X size does not match M x K");
    }

    if (W.size() != checked_product(K, N)) {
        throw std::invalid_argument("W size does not match K x N");
    }

    if (Y.size() != checked_product(M, N)) {
        throw std::invalid_argument("Y size does not match M x N");
    }

    if (&Y == &X || &Y == &W) {
        throw std::invalid_argument("Output must not alias an input matrix");
    }

    multiply(X.data(), W.data(), Y.data(), M, K, N);
}

void multiply(const float* X, const float* W, float* Y,
              std::size_t M, std::size_t K, std::size_t N) {
    if ((M && K && !X) || (K && N && !W) || (M && N && !Y)) {
        throw std::invalid_argument("Matrix pointer must not be null");
    }

    if (M == 0 || N == 0) {
        return;
    }

    if (K == 0) {
        std::fill_n(Y, checked_product(M, N), 0.0f);
        return;
    }

    const int blas_M = to_blas_int(M);
    const int blas_K = to_blas_int(K);
    const int blas_N = to_blas_int(N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                blas_M, blas_N, blas_K,
                1.0f, X, blas_K,
                W, blas_N,
                0.0f, Y, blas_N);
}

} // namespace blas_baseline
