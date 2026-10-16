#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "blas_baseline.h"
#include "data_reader.h"
#include "eigen_baseline.h"
#include "naive.h"
#include "rsr.h"
#include "rsrpp.h"
#include "timing.h"
#include "utils.h"

struct implementation_t {
    const char* name;
    std::function<void()> function;
    float* output;
};

int main(int argc, char* argv[]) {
    const auto data = read_sample(
        argc > 1 ? argv[1] : "data/sample_001");

    const int M = data.M;
    const int K = data.K;
    const int N = data.N;

    const auto& X = data.X;
    const auto& W = data.W;
    const auto& bias = data.bias;

    const int measured_runs = argc > 2 ? std::stoi(argv[2]) : 501;
    const int warmup_runs = argc > 3 ? std::stoi(argv[3]) : 200;

    if (measured_runs < 1 || warmup_runs < 0) {
        std::cerr << "Runs must be positive; warmup runs must be non-negative.\n";
        return 1;
    }

    std::cout << "Measured runs: " << measured_runs
              << "\nWarmup runs:   " << warmup_runs << std::endl;

    int block_width = 1;
    if (K > 1) {
        block_width = std::max(
            1, static_cast<int>(
                std::ceil(std::log2(K) - std::log2(std::log2(K)))));
    }

    // Decompose ternary W into two binary matrices (not timed)
    tern_t W_pos(W.size(), 0);
    tern_t W_neg(W.size(), 0);

    for (size_t i = 0; i < W.size(); ++i) {
        W_pos[i] = W[i] == 1;
        W_neg[i] = W[i] == -1;
    }

    const auto pos = preprocess(W_pos, block_width, K);
    const auto neg = preprocess(W_neg, block_width, K);
    const auto binary_patterns = generateBinaryMatrix(block_width);

    const auto flatten = [](const std::vector<std::vector<int>>& blocks) {
        std::vector<int> result;
        for (const auto& block : blocks) {
            result.insert(result.end(), block.begin(), block.end());
        }
        return result;
    };

    const permutation_t permutations_pos = flatten(pos.first);
    const permutation_t permutations_neg = flatten(neg.first);
    const segment_t segments_pos = flatten(pos.second);
    const segment_t segments_neg = flatten(neg.second);

    const auto blas_weights = blas_baseline::int2float(W);

    using eigen_baseline::DenseMatrix;

    const DenseMatrix eigen_x =
        Eigen::Map<const DenseMatrix>(X.data(), M, K);
    const DenseMatrix eigen_w =
        Eigen::Map<const DenseMatrix>(blas_weights.data(), K, N);

    const auto csr_w = eigen_baseline::preprocess_csr(eigen_w);
    const auto csc_w = eigen_baseline::preprocess_csc(eigen_w);

    vec_t Y(data.M * data.N);
    DenseMatrix eigen_y(M, N);

    const implementation_t implementations[] = {
        {"Naive", [&]() {
            for (int j = 0; j < M; ++j) {
                vec_t input(X.begin() + j * K, X.begin() + (j + 1) * K);
                vec_t result(N, 0.0f);

                vectorMatrixMultiply(input, W, result);

                std::copy(result.begin(), result.end(), Y.begin() + j * N);
            }
        }, Y.data()},

        {"RSR", [&]() {
            for (int j = 0; j < M; ++j) {
                vec_t input(X.begin() + j * K, X.begin() + (j + 1) * K);

                const auto rsr_pos = rsr_inference(
                    input, permutations_pos, segments_pos,
                    binary_patterns, K, block_width);
                const auto rsr_neg = rsr_inference(
                    input, permutations_neg, segments_neg,
                    binary_patterns, K, block_width);

                for (int col = 0; col < N; ++col) {
                    Y[j * N + col] = rsr_pos[col] - rsr_neg[col];
                }
            }
        }, Y.data()},

        {"RSR++", [&]() {
            for (int j = 0; j < M; ++j) {
                vec_t input(X.begin() + j * K, X.begin() + (j + 1) * K);

                const auto rsrpp_pos = rsr_pp_inference(
                    input, permutations_pos, segments_pos, K, block_width);
                const auto rsrpp_neg = rsr_pp_inference(
                    input, permutations_neg, segments_neg, K, block_width);

                for (int col = 0; col < N; ++col) {
                    Y[j * N + col] = rsrpp_pos[col] - rsrpp_neg[col];
                }
            }
        }, Y.data()},

        {"BLAS", [&]() {
            blas_baseline::blas_GEMM(X, blas_weights, Y, M, K, N);
        }, Y.data()},

        {"Eigen CSR", [&]() {
            eigen_baseline::multiply(eigen_x, csr_w, eigen_y);
        }, eigen_y.data()},

        {"Eigen CSC", [&]() {
            eigen_baseline::multiply(eigen_x, csc_w, eigen_y);
        }, eigen_y.data()},
    };

    std::cout << std::left << std::setw(20) << "Implementation"
              << "Median (ms)\n";

    for (const auto& impl : implementations) {
        const auto run = [&]() {
            impl.function();

            for (int j = 0; j < M; ++j) {
                for (int col = 0; col < N; ++col) {
                    impl.output[j * N + col] += bias[col];
                }
            }
        };

        for (int run_index = 0; run_index < warmup_runs; ++run_index) {
            run();
        }

        std::vector<double> times(measured_runs);
        timer_context_t timer_ctx;

        for (int run_index = 0; run_index < measured_runs; ++run_index) {
            start_timer(&timer_ctx);
            run();
            stop_timer(&timer_ctx);

            times[run_index] = get_elapsed_os_sec(&timer_ctx) * 1000.0;
        }

        for (std::size_t i = 0; i < data.expected_linear.size(); ++i) {
            const float expected = data.expected_linear[i];
            const double tolerance =
                data.atol + data.rtol * std::abs(expected);

            if (!std::isfinite(impl.output[i]) ||
                std::abs(impl.output[i] - expected) > tolerance) {
                std::cerr << impl.name << " failed at index " << i << '\n';
                return 1;
            }
        }

        std::sort(times.begin(), times.end());

        const int middle = measured_runs / 2;
        const double median = measured_runs % 2
            ? times[middle]
            : (times[middle - 1] + times[middle]) / 2.0;

        std::cout << std::left << std::setw(20) << impl.name
                  << std::fixed << std::setprecision(6)
                  << median << '\n';
    }
}
