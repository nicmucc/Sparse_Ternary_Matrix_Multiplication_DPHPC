#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "data_reader.h"
#include "ssr_baseline.h"
#include "tcsc_baseline.h"

namespace {

void check_case(const vec_t& X, const tern_t& W, const vec_t& bias,
                const vec_t& expected, int M, int K, int N, int width,
                double atol = 1e-6, double rtol = 1e-5) {
    const auto ssr = ssr_baseline::preprocess(W, K, N, width);
    const auto tcsc = tcsc_baseline::preprocess(W, K, N);
    vec_t output(static_cast<std::size_t>(M) * N);
    const std::pair<const char*, std::function<void()>> implementations[] = {
        {"SSR basic", [&] { ssr_baseline::multiply_basic(X, ssr, output, M); }},
        {"SSR unrolled", [&] { ssr_baseline::multiply_unrolled(X, ssr, output, M); }},
        {"SSR tree", [&] { ssr_baseline::multiply_tree(X, ssr, output, M); }},
        {"TCSC basic", [&] { tcsc_baseline::multiply_basic(X, tcsc, output, M); }},
        {"TCSC merged", [&] { tcsc_baseline::multiply_merged(X, tcsc, output, M); }},
    };
    for (const auto& impl : implementations) {
        // Exercise repeated calls and assignment into an existing dirty output.
        for (int repeat = 0; repeat < 2; ++repeat) {
            std::fill(output.begin(), output.end(),
                      std::numeric_limits<float>::quiet_NaN());
            impl.second();
            if (output.size() != expected.size()) {
                throw std::runtime_error(std::string(impl.first) + ": wrong output shape");
            }
            for (std::size_t i = 0; i < output.size(); ++i) {
                const float actual = output[i] + bias[i % N];
                if (!std::isfinite(actual) ||
                    std::abs(actual - expected[i]) > atol + rtol * std::abs(expected[i])) {
                    throw std::runtime_error(
                        std::string(impl.first) + ": reference mismatch at index " +
                        std::to_string(i) + " (M,K,N,L=" + std::to_string(M) + "," +
                        std::to_string(K) + "," + std::to_string(N) + "," +
                        std::to_string(width) + ")");
                }
            }
        }
    }
}

void generated_case(int M, int K, int N, int width, int mode, std::mt19937& rng) {
    vec_t X(static_cast<std::size_t>(M) * K);
    tern_t W(static_cast<std::size_t>(K) * N);
    vec_t bias(N);
    for (float& x : X) x = (static_cast<int>(rng() % 33) - 16) * 0.125f;
    for (int& w : W) {
        w = mode == 0 ? 0 : mode == 1 ? 1 : mode == 2 ? -1 :
            static_cast<int>(rng() % 3) - 1;
    }
    for (int col = 0; col < N; ++col) bias[col] = (col % 5 - 2) * 0.25f;
    if (mode == 4) {
        // Empty rows/columns alongside nonempty ones.
        for (int row = 0; row < K; ++row) W[static_cast<std::size_t>(row) * N] = 0;
        for (int col = 0; col < N; ++col) W[(K - 1) * N + col] = 0;
    }
    if (mode == 5) {
        // Every complete group has a paired SIMD prefix and unequal remainders.
        for (int row = 0; row < K; ++row)
            for (int col = 0; col < N; ++col)
                W[static_cast<std::size_t>(row) * N + col] = (row + col) % 3 - 1;
    }
    if (mode == 6) {
        std::fill(W.begin(), W.end(), 0);
        // Row 32768 must retain its full 32-bit index.
        for (int col = 0; col < N; ++col) W[(K - 1) * N + col] = col % 2 ? -1 : 1;
    }

    vec_t expected(static_cast<std::size_t>(M) * N);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = bias[col];
            for (int inner = 0; inner < K; ++inner) {
                sum += static_cast<double>(X[static_cast<std::size_t>(row) * K + inner]) *
                       W[static_cast<std::size_t>(inner) * N + col];
            }
            expected[static_cast<std::size_t>(row) * N + col] = static_cast<float>(sum);
        }
    }
    check_case(X, W, bias, expected, M, K, N, width);
}

void check_invalid_inputs() {
    const auto must_reject = [](const std::function<void()>& call) {
        try { call(); }
        catch (const std::invalid_argument&) { return; }
        throw std::runtime_error("SSR/TCSC accepted an invalid input");
    };
    for (int width : {0, 9}) {
        must_reject([&] { ssr_baseline::preprocess({1}, 1, 1, width); });
    }
    for (const tern_t& weights : {tern_t{2}, tern_t{-2}, tern_t{}}) {
        must_reject([&] { ssr_baseline::preprocess(weights, 1, 1); });
        must_reject([&] { tcsc_baseline::preprocess(weights, 1, 1); });
    }
    must_reject([] { ssr_baseline::preprocess({1}, 0, 1); });
    must_reject([] { tcsc_baseline::preprocess({1}, 1, -1); });

    const auto ssr = ssr_baseline::preprocess({1, -1}, 2, 1);
    const auto tcsc = tcsc_baseline::preprocess({1, -1}, 2, 1);
    using SsrKernel = void (*)(const vec_t&, const ssr_baseline::Weights&, vec_t&, int);
    for (SsrKernel kernel : {ssr_baseline::multiply_basic,
                             ssr_baseline::multiply_unrolled,
                             ssr_baseline::multiply_tree}) {
        vec_t output;
        must_reject([&] { kernel(vec_t{1}, ssr, output, 1); });
        must_reject([&] { kernel(vec_t{1, 2}, ssr, output, 0); });
        vec_t alias{1, 2};
        must_reject([&] { kernel(alias, ssr, alias, 1); });
    }
    using TcscKernel = void (*)(const vec_t&, const tcsc_baseline::Weights&, vec_t&, int);
    for (TcscKernel kernel : {tcsc_baseline::multiply_basic,
                              tcsc_baseline::multiply_merged}) {
        vec_t output;
        must_reject([&] { kernel(vec_t{1}, tcsc, output, 1); });
        must_reject([&] { kernel(vec_t{1, 2}, tcsc, output, 0); });
        vec_t alias{1, 2};
        must_reject([&] { kernel(alias, tcsc, alias, 1); });
    }
}

} // namespace

void run_ternary_baseline_checks(const SampleData& data) {
    check_case(data.X, data.W, data.bias, data.expected_linear,
               static_cast<int>(data.M), static_cast<int>(data.K),
               static_cast<int>(data.N), 6, data.atol, data.rtol);

    std::mt19937 rng(20261030);
    // Both sides of SIMD batch/group boundaries, rectangular and padded shapes.
    const std::tuple<int, int, int> shapes[] = {
        {1, 1, 1}, {7, 13, 7}, {8, 13, 8}, {9, 5, 9}, {15, 7, 15},
        {16, 11, 16}, {17, 257, 37}, {33, 31, 19},
    };
    for (const auto& [M, K, N] : shapes) {
        for (int mode = 0; mode <= 4; ++mode) generated_case(M, K, N, 6, mode, rng);
    }
    // Every imported tree specialization and the L=1/7/8 unrolled fallback.
    for (int width = 1; width <= 8; ++width) {
        generated_case(17, 19, 19, width, 5, rng);
    }
    generated_case(2, 32769, 9, 6, 6, rng);
    check_invalid_inputs();
}
