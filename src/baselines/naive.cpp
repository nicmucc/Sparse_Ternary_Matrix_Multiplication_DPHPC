#include <stdexcept>
#include "naive.h"

void vectorMatrixMultiply(const vec_t& vec, const tern_t& mat, vec_t& res) {
    
    if (vec.size() <= 0) {
        throw std::invalid_argument("vector lenght must be > 0");
    }
    size_t K = vec.size();
  
    
    if (res.size() <= 0) {
        throw std::invalid_argument("result lenght must be > 0");
    }
    size_t N = res.size();

    if (vec.size() * res.size() != mat.size()) {
        throw std::invalid_argument("vector length must match the matrix row count");
    }

    vectorMatrixMultiply(vec.data(), mat.data(), res.data(), K, N);
}

void vectorMatrixMultiply(const float* vec, const int* mat, float* res,
                          std::size_t K, std::size_t N) {
    if (!vec || !mat || !res || K == 0 || N == 0) {
        throw std::invalid_argument("Invalid vector-matrix input");
    }
    for (std::size_t col = 0; col < N; ++col) {
        for (std::size_t row = 0; row < K; ++row) {
            res[col] += vec[row] * mat[row * N + col];
        }
    }
}
