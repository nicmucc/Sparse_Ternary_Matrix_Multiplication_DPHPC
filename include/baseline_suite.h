#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
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

namespace baseline_suite {

using Clock = std::chrono::steady_clock;
inline double microseconds(Clock::duration elapsed) {
    return std::chrono::duration<double, std::micro>(elapsed).count();
}

struct Kernel {
    std::string name;
    std::function<void(std::size_t)> run;
    std::function<float*()> output;
    double preprocess_us;
    std::size_t weight_bytes;
};

class Suite {
    const std::vector<vec_t>& inputs;
    const tern_t& weights;
    int M, K, N, width;
    vec_t output, row_output, float_weights;
    std::vector<std::vector<vec_t>> rows;
    std::vector<eigen_baseline::DenseMatrix> eigen_inputs;
    eigen_baseline::DenseMatrix eigen_output;
    eigen_baseline::CSRMatrix csr;
    eigen_baseline::CSCMatrix csc;
    ssr_baseline::Weights ssr;
    tcsc_baseline::Weights tcsc;
    permutation_t pos_p, neg_p;
    segment_t pos_s, neg_s;
    tern_t patterns;

    template<class Blocks>
    static std::vector<int> flatten(const Blocks& blocks) {
        std::vector<int> flat;
        for (const auto& block : blocks) flat.insert(flat.end(), block.begin(), block.end());
        return flat;
    }

public:
    std::vector<Kernel> kernels;

    Suite(const std::vector<vec_t>& X, const tern_t& W, int m, int k, int n, int l)
        : inputs(X), weights(W), M(m), K(k), N(n), width(l),
          output(static_cast<std::size_t>(m) * n), row_output(n), eigen_output(m, n) {
        if (m <= 0 || k <= 0 || n <= 0 || l < 1 || l > 8)
            throw std::invalid_argument("Invalid suite dimensions or block width");
        for (const auto& x : inputs) {
            std::vector<vec_t> input_rows;
            for (int row = 0; row < M; ++row)
                input_rows.emplace_back(x.begin() + row * K, x.begin() + (row + 1) * K);
            rows.push_back(std::move(input_rows));
            eigen_inputs.emplace_back(Eigen::Map<const eigen_baseline::DenseMatrix>(x.data(), M, K));
        }
        auto data = [this] { return output.data(); };
        kernels.push_back({"Naive", [this](std::size_t input) {
            for (int row = 0; row < M; ++row) {
                std::fill(row_output.begin(), row_output.end(), 0.0f);
                vectorMatrixMultiply(rows[input][row], weights, row_output);
                std::copy(row_output.begin(), row_output.end(), output.begin() + row * N);
            }
        }, data, 0, W.size() * sizeof(int)});

        auto start = Clock::now();
        tern_t positive(W.size()), negative(W.size());
        for (std::size_t i = 0; i < W.size(); ++i) {
            positive[i] = W[i] == 1;
            negative[i] = W[i] == -1;
        }
        const auto pos = preprocess(positive, width, K);
        const auto neg = preprocess(negative, width, K);
        pos_p = flatten(pos.first); pos_s = flatten(pos.second);
        neg_p = flatten(neg.first); neg_s = flatten(neg.second);
        const double rsr_setup = microseconds(Clock::now() - start);
        const auto rsr_bytes = (pos_p.size() + pos_s.size() + neg_p.size() + neg_s.size()) * sizeof(int);
        start = Clock::now();
        patterns = generateBinaryMatrix(width);
        const double pattern_setup = microseconds(Clock::now() - start);
        for (bool plus : {false, true}) {
            kernels.push_back({plus ? "RSR++" : "RSR", [this, plus](std::size_t input) {
                for (int row = 0; row < M; ++row) {
                    const auto& x = rows[input][row];
                    const auto p = plus ? rsr_pp_inference(x, pos_p, pos_s, K, width)
                                        : rsr_inference(x, pos_p, pos_s, patterns, K, width);
                    const auto n = plus ? rsr_pp_inference(x, neg_p, neg_s, K, width)
                                        : rsr_inference(x, neg_p, neg_s, patterns, K, width);
                    for (int col = 0; col < N; ++col) output[row * N + col] = p[col] - n[col];
                }
            }, data, rsr_setup + (plus ? 0 : pattern_setup),
                rsr_bytes + (plus ? 0 : patterns.size() * sizeof(int))});
        }

        start = Clock::now();
        ssr = ssr_baseline::preprocess(W, K, N, width);
        const double ssr_setup = microseconds(Clock::now() - start);
        const auto ssr_bytes = (ssr.permutations_pos.size() + ssr.permutations_neg.size()
                             + ssr.segments_pos.size() + ssr.segments_neg.size()) * sizeof(int);
        using Ssr = void (*)(const vec_t&, const ssr_baseline::Weights&, vec_t&, int);
        const std::pair<const char*, Ssr> ssr_kernels[] = {
            {"SSR basic", ssr_baseline::multiply_basic},
            {"SSR unrolled", ssr_baseline::multiply_unrolled},
            {"SSR tree", ssr_baseline::multiply_tree}};
        for (const auto& item : ssr_kernels)
            kernels.push_back({item.first, [this, f=item.second](std::size_t i) {
                f(inputs[i], ssr, output, M);
            }, data, ssr_setup, ssr_bytes});

        start = Clock::now();
        tcsc = tcsc_baseline::preprocess(W, K, N);
        const double tcsc_setup = microseconds(Clock::now() - start);
        const auto tcsc_bytes = (tcsc.col_start_pos.size() + tcsc.col_start_neg.size()
            + tcsc.row_index_pos.size() + tcsc.row_index_neg.size()
            + tcsc.metadata.size() + tcsc.row_index.size()) * sizeof(std::int32_t);
        kernels.push_back({"TCSC basic", [this](std::size_t i) {
            tcsc_baseline::multiply_basic(inputs[i], tcsc, output, M);
        }, data, tcsc_setup, tcsc_bytes});
        kernels.push_back({"TCSC merged", [this](std::size_t i) {
            tcsc_baseline::multiply_merged(inputs[i], tcsc, output, M);
        }, data, tcsc_setup, tcsc_bytes});

        start = Clock::now();
        float_weights = blas_baseline::int2float(W);
        const double conversion = microseconds(Clock::now() - start);
        kernels.push_back({"BLAS", [this](std::size_t i) {
            blas_baseline::multiply(inputs[i], float_weights, output, M, K, N);
        }, data, conversion, float_weights.size() * sizeof(float)});
        const Eigen::Map<const eigen_baseline::DenseMatrix> dense(float_weights.data(), K, N);
        start = Clock::now();
        csr = eigen_baseline::preprocess_csr(dense);
        const double csr_setup = conversion + microseconds(Clock::now() - start);
        start = Clock::now();
        csc = eigen_baseline::preprocess_csc(dense);
        const double csc_setup = conversion + microseconds(Clock::now() - start);
        kernels.push_back({"Eigen CSR", [this](std::size_t i) {
            eigen_baseline::multiply(eigen_inputs[i], csr, eigen_output);
        }, [this] { return eigen_output.data(); }, csr_setup,
            static_cast<std::size_t>(csr.nonZeros()) * (sizeof(float) + sizeof(int))
                + static_cast<std::size_t>(csr.outerSize() + 1) * sizeof(int)});
        kernels.push_back({"Eigen CSC", [this](std::size_t i) {
            eigen_baseline::multiply(eigen_inputs[i], csc, eigen_output);
        }, [this] { return eigen_output.data(); }, csc_setup,
            static_cast<std::size_t>(csc.nonZeros()) * (sizeof(float) + sizeof(int))
                + static_cast<std::size_t>(csc.outerSize() + 1) * sizeof(int)});
    }
};

// gamma bounds sequential FP32 accumulation, including cancellation.
// K also covers rows that cancel inside a pattern before its sum is reused.
inline double tolerance(double sum_abs, double expected, int K,
                        double atol = 1e-6, double rtol = 1e-5) {
    const double u = std::numeric_limits<float>::epsilon() / 2;
    const double ku = (K + 4.0) * u;
    return atol + rtol * std::abs(expected) + 2 * ku / (1 - ku) * sum_abs;
}

inline std::vector<double> error_bounds(const vec_t& X, const tern_t& W,
                                       int M, int K, int N) {
    std::vector<double> sums(static_cast<std::size_t>(M) * N);
    for (int row = 0; row < M; ++row)
        for (int k = 0; k < K; ++k)
            for (int col = 0; col < N; ++col)
                if (W[k * N + col]) sums[row * N + col] += std::abs(X[row * K + k]);
    return sums;
}
} // namespace baseline_suite
