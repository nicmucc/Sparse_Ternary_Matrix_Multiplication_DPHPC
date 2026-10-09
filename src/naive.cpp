#include <stdexcept>
#include "types.h"

void vectorMatrixMultiply(const vec_t& vec, const tern_t& mat, vec_t& res, size_t K) {
    if (vec.size() * K != mat.size()) {
        throw std::invalid_argument("vector length must match the matrix row count");
    }
    
    if (vec.size() <= 0) {
        throw std::invalid_argument("vector lenght must be > 0");
    }
    size_t N = vec.size();
  

    if (mat.empty()) {
        return;
    }

    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            res[i*N + j] += vec[i] * mat[i*N + j];
        }
    }
    
}
