#pragma once

#include <cstddef>

#include "types.h"

namespace blas_baseline {

vec_t int2float(const tern_t& matrix);
void multiply(const vec_t& X, const vec_t& W, vec_t& Y,
              std::size_t M, std::size_t K, std::size_t N);

} // blas_baseline namespace
