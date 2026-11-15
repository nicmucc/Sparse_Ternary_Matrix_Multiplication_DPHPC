// Library baselines used in both the SSR and the TCSC papers:
//   dense_naive  : plain triple loop, W as fp32 (the scalar dense reference; -O3 may auto-vectorize the inner loop)
//   dense_blas   : vendor sgemm (Accelerate on macOS, OpenBLAS/MKL elsewhere)
//   eigen_dense  : Eigen dense GEMM, W stored as fp32
//   eigen_sparse : Eigen sparse (CSC) x dense, W stored as fp32 non-zeros
// Thread count: BLAS reads VECLIB_MAXIMUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS,
// Eigen dense uses OpenMP (omp_set_num_threads), Eigen sparse x dense is single-threaded.
#include "spr/kernel.hpp"

#ifdef SPR_HAVE_BLAS
#  ifdef __APPLE__
#    define ACCELERATE_NEW_LAPACK
#    include <Accelerate/Accelerate.h>
#  else
#    include <cblas.h>
#  endif
#endif
#ifdef SPR_HAVE_EIGEN
#  include <Eigen/Dense>
#  include <Eigen/Sparse>
#endif

namespace spr {
namespace {

class DenseNaive final : public Kernel {
    std::vector<float> W_;
    int K_ = 0, N_ = 0;

public:
    std::string name() const override { return "lib.dense_naive"; }
    void prepare(const int8_t* W, int K, int N, const Params&) override {
        K_ = K; N_ = N;
        W_.assign(W, W + (size_t)K * N);
    }
    void run(IO& io) override {
        for (int m = 0; m < io.M; ++m) {
            float* y = io.Y + (size_t)m * N_;
            for (int n = 0; n < N_; ++n) y[n] = 0.f;
            for (int j = 0; j < K_; ++j) {
                const float x = io.X[(size_t)m * K_ + j];
                const float* w = W_.data() + (size_t)j * N_;
                for (int n = 0; n < N_; ++n) y[n] += x * w[n];
            }
        }
    }
    size_t bytes() const override { return W_.size() * sizeof(float); }
    double adds_per_row() const override { return (double)K_ * N_; }
};
SPR_REGISTER_KERNEL("lib.dense_naive", DenseNaive);

#ifdef SPR_HAVE_BLAS
class DenseBlas final : public Kernel {
    std::vector<float> W_;
    int K_ = 0, N_ = 0;

public:
    std::string name() const override { return "lib.dense_blas"; }
    void prepare(const int8_t* W, int K, int N, const Params&) override {
        K_ = K; N_ = N;
        W_.assign(W, W + (size_t)K * N);
    }
    void run(IO& io) override {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, io.M, N_, K_, 1.f, io.X, K_, W_.data(), N_, 0.f, io.Y, N_);
    }
    size_t bytes() const override { return W_.size() * sizeof(float); }
};
SPR_REGISTER_KERNEL("lib.dense_blas", DenseBlas);
#endif

#ifdef SPR_HAVE_EIGEN
using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class EigenDense final : public Kernel {
    RowMat W_;

public:
    std::string name() const override { return "lib.eigen_dense"; }
    void prepare(const int8_t* W, int K, int N, const Params&) override {
        W_.resize(K, N);
        for (int r = 0; r < K; ++r)
            for (int c = 0; c < N; ++c) W_(r, c) = W[(size_t)r * N + c];
    }
    void run(IO& io) override {
        Eigen::Map<const RowMat> X(io.X, io.M, io.K);
        Eigen::Map<RowMat> Y(io.Y, io.M, io.N);
        Y.noalias() = X * W_;
    }
    size_t bytes() const override { return (size_t)W_.size() * sizeof(float); }
};
SPR_REGISTER_KERNEL("lib.eigen_dense", EigenDense);

class EigenSparse final : public Kernel {
    Eigen::SparseMatrix<float, Eigen::ColMajor> W_;

public:
    std::string name() const override { return "lib.eigen_sparse"; }
    void prepare(const int8_t* W, int K, int N, const Params&) override {
        std::vector<Eigen::Triplet<float>> t;
        for (int r = 0; r < K; ++r)
            for (int c = 0; c < N; ++c)
                if (W[(size_t)r * N + c]) t.emplace_back(r, c, (float)W[(size_t)r * N + c]);
        W_.resize(K, N);
        W_.setFromTriplets(t.begin(), t.end());
        W_.makeCompressed();
    }
    void run(IO& io) override {
        Eigen::Map<const RowMat> X(io.X, io.M, io.K);
        Eigen::Map<RowMat> Y(io.Y, io.M, io.N);
        Y.noalias() = X * W_;
    }
    size_t bytes() const override { return (size_t)W_.nonZeros() * (sizeof(float) + sizeof(int)) + (W_.cols() + 1) * sizeof(int); }
};
SPR_REGISTER_KERNEL("lib.eigen_sparse", EigenSparse);
#endif

}  // namespace
}  // namespace spr
