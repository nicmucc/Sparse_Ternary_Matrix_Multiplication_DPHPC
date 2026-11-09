#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "blas_baseline.h"
#include "eigen_baseline.h"
#include "naive.h"
#include "rsr.h"
#include "rsrpp.h"
#include "ssr_baseline.h"
#include "tcsc_baseline.h"
#include "utils.h"

namespace py = pybind11;

namespace {

int checked_int(std::int64_t value, const char* name) {
    TORCH_CHECK(value > 0, name, " must be positive");
    TORCH_CHECK(value <= std::numeric_limits<int>::max(),
                name, " exceeds the C++ baseline range");
    return static_cast<int>(value);
}

tern_t copy_ternary_weights(const torch::Tensor& W) {
    TORCH_CHECK(W.device().is_cpu(), "W must be a CPU tensor");
    TORCH_CHECK(W.scalar_type() == torch::kInt8, "W must use torch.int8");
    TORCH_CHECK(W.dim() == 2, "W must have shape K x N");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");

    const int K = checked_int(W.size(0), "K");
    const int N = checked_int(W.size(1), "N");
    TORCH_CHECK(static_cast<std::int64_t>(K) * N <=
                    std::numeric_limits<int>::max(),
                "K*N exceeds the C++ baseline range");

    const auto* source = W.data_ptr<std::int8_t>();
    tern_t weights(static_cast<std::size_t>(K) * N);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        TORCH_CHECK(source[i] >= -1 && source[i] <= 1,
                    "W must contain only -1, 0, and 1");
        weights[i] = source[i];
    }
    return weights;
}

int check_forward(const torch::Tensor& X, const torch::Tensor& Y,
                  int K, int N) {
    TORCH_CHECK(X.device().is_cpu() && Y.device().is_cpu(),
                "X and Y must be CPU tensors");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 &&
                    Y.scalar_type() == torch::kFloat32,
                "X and Y must use torch.float32");
    TORCH_CHECK(X.dim() == 2 && Y.dim() == 2,
                "X and Y must be matrices");
    TORCH_CHECK(X.is_contiguous() && Y.is_contiguous(),
                "X and Y must be contiguous");

    const int M = checked_int(X.size(0), "M");
    TORCH_CHECK(X.size(1) == K, "X must have shape M x K");
    TORCH_CHECK(Y.size(0) == M && Y.size(1) == N,
                "Y must have shape M x N");
    TORCH_CHECK(X.data_ptr<float>() != Y.data_ptr<float>(),
                "Y must not alias X");
    return M;
}

template <class Operator>
torch::Tensor forward(Operator& op, const torch::Tensor& X) {
    TORCH_CHECK(X.dim() == 2, "X must have shape M x K");
    auto Y = torch::empty({X.size(0), op.N()}, X.options());
    op.forward_out(X, Y);
    return Y;
}

template <class Blocks>
std::vector<int> flatten(const Blocks& blocks) {
    std::vector<int> result;
    for (const auto& block : blocks) {
        result.insert(result.end(), block.begin(), block.end());
    }
    return result;
}

class NaiveOperator {
public:
    explicit NaiveOperator(const torch::Tensor& W)
        : K_(checked_int(W.size(0), "K")),
          N_(checked_int(W.size(1), "N")),
          weights_(copy_ternary_weights(W)) {}

    std::string name() const { return "Naive"; }
    std::string backend() const { return "portable"; }
    int K() const { return K_; }
    int N() const { return N_; }
    std::size_t weight_bytes() const {
        return weights_.size() * sizeof(int);
    }

    void forward_out(const torch::Tensor& X, torch::Tensor Y) const {
        const int M = check_forward(X, Y, K_, N_);
        const float* input = X.data_ptr<float>();
        float* output = Y.data_ptr<float>();
        std::fill_n(output, static_cast<std::size_t>(M) * N_, 0.0f);
        for (int row = 0; row < M; ++row) {
            vectorMatrixMultiply(
                input + static_cast<std::size_t>(row) * K_, weights_.data(),
                output + static_cast<std::size_t>(row) * N_, K_, N_);
        }
    }

private:
    int K_;
    int N_;
    tern_t weights_;
};

class BlasOperator {
public:
    explicit BlasOperator(const torch::Tensor& W)
        : K_(checked_int(W.size(0), "K")),
          N_(checked_int(W.size(1), "N")),
          weights_(blas_baseline::int2float(copy_ternary_weights(W))) {}

    std::string name() const { return "BLAS"; }
    std::string backend() const { return "openblas"; }
    int K() const { return K_; }
    int N() const { return N_; }
    std::size_t weight_bytes() const {
        return weights_.size() * sizeof(float);
    }

    void forward_out(const torch::Tensor& X, torch::Tensor Y) const {
        const int M = check_forward(X, Y, K_, N_);
        blas_baseline::multiply(
            X.data_ptr<float>(), weights_.data(), Y.data_ptr<float>(),
            M, K_, N_);
    }

private:
    int K_;
    int N_;
    vec_t weights_;
};

class EigenCsrOperator {
public:
    explicit EigenCsrOperator(const torch::Tensor& W)
        : K_(checked_int(W.size(0), "K")),
          N_(checked_int(W.size(1), "N")) {
        const vec_t dense_weights =
            blas_baseline::int2float(copy_ternary_weights(W));
        const Eigen::Map<const eigen_baseline::DenseMatrix> dense(
            dense_weights.data(), K_, N_);
        weights_ = eigen_baseline::preprocess_csr(dense);
    }

    std::string name() const { return "Eigen CSR"; }
    std::string backend() const { return "eigen"; }
    int K() const { return K_; }
    int N() const { return N_; }
    std::size_t weight_bytes() const {
        return static_cast<std::size_t>(weights_.nonZeros()) *
                   (sizeof(float) + sizeof(int)) +
               static_cast<std::size_t>(weights_.outerSize() + 1) * sizeof(int);
    }

    void forward_out(const torch::Tensor& X, torch::Tensor Y) const {
        const int M = check_forward(X, Y, K_, N_);
        eigen_baseline::multiply(
            X.data_ptr<float>(), weights_, Y.data_ptr<float>(), M, K_, N_);
    }

private:
    int K_;
    int N_;
    eigen_baseline::CSRMatrix weights_;
};

class EigenCscOperator {
public:
    explicit EigenCscOperator(const torch::Tensor& W)
        : K_(checked_int(W.size(0), "K")),
          N_(checked_int(W.size(1), "N")) {
        const vec_t dense_weights =
            blas_baseline::int2float(copy_ternary_weights(W));
        const Eigen::Map<const eigen_baseline::DenseMatrix> dense(
            dense_weights.data(), K_, N_);
        weights_ = eigen_baseline::preprocess_csc(dense);
    }

    std::string name() const { return "Eigen CSC"; }
    std::string backend() const { return "eigen"; }
    int K() const { return K_; }
    int N() const { return N_; }
    std::size_t weight_bytes() const {
        return static_cast<std::size_t>(weights_.nonZeros()) *
                   (sizeof(float) + sizeof(int)) +
               static_cast<std::size_t>(weights_.outerSize() + 1) * sizeof(int);
    }

    void forward_out(const torch::Tensor& X, torch::Tensor Y) const {
        const int M = check_forward(X, Y, K_, N_);
        eigen_baseline::multiply(
            X.data_ptr<float>(), weights_, Y.data_ptr<float>(), M, K_, N_);
    }

private:
    int K_;
    int N_;
    eigen_baseline::CSCMatrix weights_;
};

class RsrOperator {
public:
    RsrOperator(const torch::Tensor& W, int width, bool plus)
        : K_(checked_int(W.size(0), "K")),
          N_(checked_int(W.size(1), "N")),
          width_(width),
          plus_(plus) {
        TORCH_CHECK(width >= 1 && width <= 8,
                    "RSR block width must be between 1 and 8");
        const tern_t weights = copy_ternary_weights(W);
        tern_t positive(weights.size());
        tern_t negative(weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            positive[i] = weights[i] == 1;
            negative[i] = weights[i] == -1;
        }

        const auto pos = preprocess(positive, width_, K_);
        const auto neg = preprocess(negative, width_, K_);
        pos_permutations_ = flatten(pos.first);
        pos_segments_ = flatten(pos.second);
        neg_permutations_ = flatten(neg.first);
        neg_segments_ = flatten(neg.second);
        if (!plus_) {
            patterns_ = generateBinaryMatrix(width_);
        }
    }

    std::string name() const { return plus_ ? "RSR++" : "RSR"; }
    std::string backend() const { return "portable"; }
    int K() const { return K_; }
    int N() const { return N_; }
    std::size_t weight_bytes() const {
        return (pos_permutations_.size() + pos_segments_.size() +
                neg_permutations_.size() + neg_segments_.size() +
                patterns_.size()) * sizeof(int);
    }

    void forward_out(const torch::Tensor& X, torch::Tensor Y) const {
        const int M = check_forward(X, Y, K_, N_);
        const float* input = X.data_ptr<float>();
        float* output = Y.data_ptr<float>();
        for (int row = 0; row < M; ++row) {
            const float* input_row =
                input + static_cast<std::size_t>(row) * K_;
            const vec_t positive = plus_
                ? rsr_pp_inference(input_row, pos_permutations_, pos_segments_,
                                   K_, width_)
                : rsr_inference(input_row, pos_permutations_, pos_segments_,
                                patterns_, K_, width_);
            const vec_t negative = plus_
                ? rsr_pp_inference(input_row, neg_permutations_, neg_segments_,
                                   K_, width_)
                : rsr_inference(input_row, neg_permutations_, neg_segments_,
                                patterns_, K_, width_);
            TORCH_CHECK(positive.size() >= static_cast<std::size_t>(N_) &&
                            negative.size() >= static_cast<std::size_t>(N_),
                        "RSR returned too few output columns");
            for (int col = 0; col < N_; ++col) {
                output[static_cast<std::size_t>(row) * N_ + col] =
                    positive[col] - negative[col];
            }
        }
    }

private:
    int K_;
    int N_;
    int width_;
    bool plus_;
    permutation_t pos_permutations_;
    permutation_t neg_permutations_;
    segment_t pos_segments_;
    segment_t neg_segments_;
    binary_matrix_t patterns_;
};

enum class SsrVariant { Basic, Unrolled, Tree };

class SsrOperator {
public:
    SsrOperator(const torch::Tensor& W, int width, SsrVariant variant)
        : K_(checked_int(W.size(0), "K")),
          N_(checked_int(W.size(1), "N")),
          variant_(variant),
          weights_(ssr_baseline::preprocess(
              copy_ternary_weights(W), K_, N_, width)) {}

    std::string name() const {
        if (variant_ == SsrVariant::Basic) return "SSR basic";
        if (variant_ == SsrVariant::Unrolled) return "SSR unrolled";
        return "SSR tree";
    }
    std::string backend() const {
        if (variant_ == SsrVariant::Basic) return "scalar";
        return ssr_baseline::avx2_available() ? "avx2_dispatch" :
                                               "scalar_fallback";
    }
    int K() const { return K_; }
    int N() const { return N_; }
    std::size_t weight_bytes() const {
        return (weights_.permutations_pos.size() +
                weights_.permutations_neg.size() +
                weights_.segments_pos.size() +
                weights_.segments_neg.size()) * sizeof(int);
    }

    void forward_out(const torch::Tensor& X, torch::Tensor Y) const {
        const int M = check_forward(X, Y, K_, N_);
        if (variant_ == SsrVariant::Basic) {
            ssr_baseline::multiply_basic_raw(
                X.data_ptr<float>(), weights_, Y.data_ptr<float>(), M);
        } else if (variant_ == SsrVariant::Unrolled) {
            ssr_baseline::multiply_unrolled_raw(
                X.data_ptr<float>(), weights_, Y.data_ptr<float>(), M);
        } else {
            ssr_baseline::multiply_tree_raw(
                X.data_ptr<float>(), weights_, Y.data_ptr<float>(), M);
        }
    }

private:
    int K_;
    int N_;
    SsrVariant variant_;
    ssr_baseline::Weights weights_;
};

enum class TcscVariant { Basic, Merged };

class TcscOperator {
public:
    TcscOperator(const torch::Tensor& W, TcscVariant variant)
        : K_(checked_int(W.size(0), "K")),
          N_(checked_int(W.size(1), "N")),
          variant_(variant),
          weights_(tcsc_baseline::preprocess(
              copy_ternary_weights(W), K_, N_)) {}

    std::string name() const {
        return variant_ == TcscVariant::Basic ? "TCSC basic" : "TCSC merged";
    }
    std::string backend() const {
        if (variant_ == TcscVariant::Basic) return "scalar";
        return tcsc_baseline::avx2_available() ? "avx2_dispatch" :
                                                "scalar_fallback";
    }
    int K() const { return K_; }
    int N() const { return N_; }
    std::size_t weight_bytes() const {
        return (weights_.col_start_pos.size() +
                weights_.col_start_neg.size() +
                weights_.row_index_pos.size() +
                weights_.row_index_neg.size() +
                weights_.metadata.size() +
                weights_.row_index.size()) * sizeof(std::int32_t);
    }

    void forward_out(const torch::Tensor& X, torch::Tensor Y) const {
        const int M = check_forward(X, Y, K_, N_);
        if (variant_ == TcscVariant::Basic) {
            tcsc_baseline::multiply_basic_raw(
                X.data_ptr<float>(), weights_, Y.data_ptr<float>(), M);
        } else {
            tcsc_baseline::multiply_merged_raw(
                X.data_ptr<float>(), weights_, Y.data_ptr<float>(), M);
        }
    }

private:
    int K_;
    int N_;
    TcscVariant variant_;
    tcsc_baseline::Weights weights_;
};

template <class Operator>
void bind_operator(py::module_& module, const char* class_name) {
    py::class_<Operator>(module, class_name)
        .def_property_readonly("name", &Operator::name)
        .def_property_readonly("backend", &Operator::backend)
        .def_property_readonly("weight_bytes", &Operator::weight_bytes)
        .def_property_readonly("K", &Operator::K)
        .def_property_readonly("N", &Operator::N)
        .def("forward", [](Operator& op, const torch::Tensor& X) {
            return forward(op, X);
        })
        .def("forward_out", &Operator::forward_out);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "PyTorch bindings for the C++ ternary matmul baselines";

    bind_operator<NaiveOperator>(module, "NaiveOperator");
    bind_operator<BlasOperator>(module, "BlasOperator");
    bind_operator<EigenCsrOperator>(module, "EigenCsrOperator");
    bind_operator<EigenCscOperator>(module, "EigenCscOperator");
    bind_operator<RsrOperator>(module, "RsrOperator");
    bind_operator<SsrOperator>(module, "SsrOperator");
    bind_operator<TcscOperator>(module, "TcscOperator");

    module.def("prepare_naive", [](const torch::Tensor& W) {
        return std::make_unique<NaiveOperator>(W);
    });
    module.def("prepare_blas", [](const torch::Tensor& W) {
        return std::make_unique<BlasOperator>(W);
    });
    module.def("prepare_eigen_csr", [](const torch::Tensor& W) {
        return std::make_unique<EigenCsrOperator>(W);
    });
    module.def("prepare_eigen_csc", [](const torch::Tensor& W) {
        return std::make_unique<EigenCscOperator>(W);
    });
    module.def("prepare_rsr", [](const torch::Tensor& W, int width) {
        return std::make_unique<RsrOperator>(W, width, false);
    });
    module.def("prepare_rsrpp", [](const torch::Tensor& W, int width) {
        return std::make_unique<RsrOperator>(W, width, true);
    });
    module.def("prepare_ssr_basic", [](const torch::Tensor& W, int width) {
        return std::make_unique<SsrOperator>(W, width, SsrVariant::Basic);
    });
    module.def("prepare_ssr_unrolled", [](const torch::Tensor& W, int width) {
        return std::make_unique<SsrOperator>(W, width, SsrVariant::Unrolled);
    });
    module.def("prepare_ssr_tree", [](const torch::Tensor& W, int width) {
        return std::make_unique<SsrOperator>(W, width, SsrVariant::Tree);
    });
    module.def("prepare_tcsc_basic", [](const torch::Tensor& W) {
        return std::make_unique<TcscOperator>(W, TcscVariant::Basic);
    });
    module.def("prepare_tcsc_merged", [](const torch::Tensor& W) {
        return std::make_unique<TcscOperator>(W, TcscVariant::Merged);
    });
}
