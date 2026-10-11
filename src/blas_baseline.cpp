#include "blas_baseline.h"
#include <algorithm>
#include <stdexcept>

namespace blas_baseline
{

    FloatMatrix int2float(const IntMatrix &M)
    {
        FloatMatrix out(M.size());
        std::transform(M.begin(), M.end(), out.begin(),
                       [] (int value) {return static_cast<float>(value);});

        return out;
    }

    void blas_GEMM(const FloatMatrix& X, const FloatMatrix& W, FloatMatrix& Y, const int M, const int K, const int N){

        if (M < 0 || K < 0 || N < 0)
            throw std::invalid_argument("Matrix dimensions must be non-negative");

        const auto m = static_cast<std::size_t>(M);
        const auto k = static_cast<std::size_t>(K);
        const auto n = static_cast<std::size_t>(N);

        if (X.size() != m * k || W.size() != k*n || Y.size() != m*n)
            throw std::invalid_argument("Matrix sizes mismatch");
        
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, X.data(), K, W.data(), N, 0.0f, Y.data(), N);
    }

    IntMatrix float2int(const FloatMatrix &M){
        IntMatrix out(M.size());
    
        std::transform(M.begin(), M.end(), out.begin(),
                       [] (float value) {return static_cast<int>(value);});

        return out;
    }

}
