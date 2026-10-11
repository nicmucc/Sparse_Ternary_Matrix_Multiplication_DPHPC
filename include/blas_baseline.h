#pragma once
#include<cblas.h>
#include <vector>

namespace blas_baseline{

    using IntMatrix = std::vector<int>;
    using FloatMatrix = std::vector<float>;

    FloatMatrix int2float(const IntMatrix &M);
    void blas_GEMM(const FloatMatrix& X, const FloatMatrix& W, FloatMatrix& Y, const int M, const int K, const int N);
    IntMatrix float2int(const FloatMatrix &M);

} // blas_baseline namespace
