#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "baseline_suite.h"
#include "blas_baseline.h"
#include "data_reader.h"
#include "eigen_baseline.h"
#include "rsr.h"
#include "rsrpp.h"
#include "ssr_baseline.h"
#include "tcsc_baseline.h"
#include "utils.h"

using namespace std;
using eigen_baseline::DenseMatrix;

namespace {

void check_eigen_case(const DenseMatrix& X, const DenseMatrix& W,
                      const DenseMatrix& expected) {
    const auto csr = eigen_baseline::preprocess_csr(W);
    const auto csc = eigen_baseline::preprocess_csc(W);
    const auto nonzeros = (W.array() != 0.0f).count();

    if (!csr.isCompressed() || !csc.isCompressed() ||
        csr.nonZeros() != nonzeros || csc.nonZeros() != nonzeros) {
        throw runtime_error("weights must contain only compressed nonzeros");
    }

    DenseMatrix csr_output(X.rows(), W.cols());
    DenseMatrix csc_output(X.rows(), W.cols());

    // Repeat with dirty outputs to verify assignment does not accumulate.
    for (int repeat = 0; repeat < 2; ++repeat) {
        csr_output.setConstant(123.0f);
        csc_output.setConstant(-456.0f);

        eigen_baseline::multiply(X, csr, csr_output);
        eigen_baseline::multiply(X, csc, csc_output);

        for (Eigen::Index row = 0; row < expected.rows(); ++row) {
            for (Eigen::Index col = 0; col < expected.cols(); ++col) {
                const float tolerance =
                    1e-6f + 1e-5f * abs(expected(row, col));

                if (!isfinite(csr_output(row, col)) ||
                    !isfinite(csc_output(row, col)) ||
                    abs(csr_output(row, col) - expected(row, col)) > tolerance ||
                    abs(csc_output(row, col) - expected(row, col)) > tolerance) {
                    throw runtime_error("CSR/CSC output differs from reference");
                }
            }
        }
    }
}

void run_eigen_api_checks() {
    DenseMatrix X(2, 3);
    X << 0.5f, -2.0f, 3.0f,
         -1.5f, 4.0f, -0.25f;

    DenseMatrix W(3, 4);
    // Rectangular ternary weights, with an empty row and an empty column.
    W << -1, 0, 1, 0,
          0, 0, 0, 0,
          1, 0, -1, 1;

    DenseMatrix expected(2, 4);
    expected << 2.5f, 0, -2.5f, 3.0f,
                1.25f, 0, -1.25f, -0.25f;

    check_eigen_case(X, W, expected);
    check_eigen_case(X, DenseMatrix::Zero(3, 4),
                     DenseMatrix::Zero(2, 4));
    check_eigen_case(DenseMatrix::Constant(1, 1, -0.5f),
                     DenseMatrix::Constant(1, 1, -1.0f),
                     DenseMatrix::Constant(1, 1, 0.5f));

    const auto csr = eigen_baseline::preprocess_csr(W);
    const auto csc = eigen_baseline::preprocess_csc(W);
    DenseMatrix output;
    int rejected = 0;

    try { eigen_baseline::multiply(DenseMatrix::Zero(1, 2), csr, output); }
    catch (const invalid_argument&) { ++rejected; }
    try { eigen_baseline::multiply(DenseMatrix::Zero(1, 2), csc, output); }
    catch (const invalid_argument&) { ++rejected; }
    try { eigen_baseline::to_dense({{1, 2}, {3}}); }
    catch (const invalid_argument&) { ++rejected; }
    try { eigen_baseline::multiply(X, csr, X); }
    catch (const invalid_argument&) { ++rejected; }

    if (rejected != 4) {
        throw runtime_error("invalid Eigen inputs were not rejected");
    }
}

void run_rsr_api_checks() {
    // N is not a multiple of block_width, so preprocessing must use padding
    // internally without changing the caller's weights.
    const tern_t weights = {
        1, 0, 1,
        0, 1, 0,
    };
    const auto preprocessed = preprocess(weights, 2, 2);

    if (preprocessed.first.size() != 2 ||
        preprocessed.second.size() != 2) {
        throw runtime_error("RSR preprocessing produced an invalid block count");
    }

    const vec_t valid_input(2, 1.0f);
    const permutation_t valid_permutations = {0, 1};
    const segment_t valid_segments = {0, 1};
    int rejected = 0;

    try { rsr_pp_inference({}, {}, {}, 0, 1); }
    catch (const invalid_argument&) { ++rejected; }
    try {
        rsr_pp_inference(valid_input, valid_permutations,
                         valid_segments, 2, 0);
    } catch (const invalid_argument&) { ++rejected; }
    try {
        rsr_pp_inference(valid_input, valid_permutations, valid_segments, 2,
                         numeric_limits<int>::digits);
    } catch (const invalid_argument&) { ++rejected; }
    try {
        rsr_pp_inference(vec_t(1, 1.0f), valid_permutations,
                         valid_segments, 2, 1);
    } catch (const invalid_argument&) { ++rejected; }
    try {
        rsr_pp_inference(valid_input, permutation_t{0},
                         valid_segments, 2, 1);
    } catch (const invalid_argument&) { ++rejected; }
    try {
        rsr_pp_inference(valid_input, valid_permutations,
                         segment_t{0}, 2, 1);
    } catch (const invalid_argument&) { ++rejected; }

    if (rejected != 6) {
        throw runtime_error("invalid RSR++ inputs were not rejected");
    }
}

void check_case(const vec_t& X, const tern_t& W, const vec_t& bias,
                const vec_t& expected, int M, int K, int N, int width,
                double atol = 1e-6, double rtol = 1e-5, bool exact = false) {
    const std::vector<vec_t> inputs{X};
    baseline_suite::Suite suite(inputs, W, M, K, N, width);
    const auto sums = baseline_suite::error_bounds(X, W, M, K, N);
    std::vector<double> reference(expected.size()), tolerances(expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        reference[i] = static_cast<double>(expected[i]) - bias[i % N];
        tolerances[i] = exact ? 0.0 : baseline_suite::tolerance(
            sums[i] + std::abs(bias[i % N]), expected[i], K, atol, rtol);
    }
    for (auto& impl : suite.kernels)
        for (int repeat = 0; repeat < 2; ++repeat)
            baseline_suite::validate(impl, 0, reference, tolerances);
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

    if (mode == 7) {
        std::uniform_real_distribution<float> values(-3.0f, 3.0f);
        for (float& x : X) x = values(rng);
    }
    if (mode == 8) {
        std::fill(W.begin(), W.end(), 1);
        for (int row = 0; row < M; ++row)
            for (int k = 0; k < K; ++k)
                X[row * K + k] = (k % 2 ? -1.0f : 1.0f) + (k % 7) * 0.0001f;
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
    check_case(X, W, bias, expected, M, K, N, width, 1e-6, 1e-5, mode <= 6);
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

void run_generated_checks() {
    std::mt19937 rng(20261030);
    // Both sides of SIMD batch/group boundaries, rectangular and padded shapes.
    const std::tuple<int, int, int> shapes[] = {
        {1, 1, 1}, {7, 13, 7}, {8, 13, 8}, {9, 5, 9}, {15, 7, 15},
        {16, 11, 16}, {17, 257, 37}, {33, 31, 19},
    };
    for (const auto& [M, K, N] : shapes) {
        for (int mode : {0, 1, 2, 3, 4, 7, 8}) generated_case(M, K, N, 6, mode, rng);
    }
    // Every imported tree specialization and the L=1/7/8 unrolled fallback.
    for (int width = 1; width <= 8; ++width) {
        generated_case(17, 19, 19, width, 5, rng);
    }
    generated_case(2, 32769, 9, 6, 6, rng);
    for (int seed : {17, 42, 123}) {
        std::mt19937 extra(seed);
        generated_case(17, 4096, 9, 6, 7, extra);
        generated_case(17, 4096, 9, 6, 8, extra);
    }
    check_invalid_inputs();
}

void run_harness_checks() {
    using namespace baseline_suite;
    static_assert(!std::is_copy_constructible<Suite>::value);
    static_assert(!std::is_move_constructible<Suite>::value);
    const std::vector<vec_t> inputs{{1.0f}, {-2.0f}};
    const tern_t weights{1};
    Suite suite(inputs, weights, 1, 1, 1, 6);
    const std::set<std::string> required{
        "Naive", "RSR", "RSR++", "SSR basic", "SSR unrolled", "SSR tree",
        "TCSC basic", "TCSC merged", "BLAS", "Eigen CSR", "Eigen CSC"};
    std::set<std::string> registered;
    for (auto& kernel : suite.kernels) {
        registered.insert(kernel.name);
        validate(kernel, 0, {1.0}, {0.0});
        validate(kernel, 1, {-2.0}, {0.0});
    }
    if (!std::includes(registered.begin(), registered.end(), required.begin(), required.end()))
        throw runtime_error("Missing baseline registration");
    const auto must_reject = [](const std::function<void()>& call) {
        try { call(); }
        catch (const std::exception&) { return; }
        throw runtime_error("Harness accepted an invalid kernel");
    };
    suite.kernels.push_back(suite.kernels.front());
    must_reject([&] { validate_registry(suite.kernels); });
    vec_t output{1.0f, 2.0f};
    Kernel broken{"broken", [](std::size_t) {},
        [&] { return OutputView{output.data(), output.size()}; }, 0, 0, "test"};
    must_reject([&] { validate(broken, 0, {1.0, 2.0}, {0.0, 0.0}); });
    broken.run = [&](std::size_t) { output[0] = 1.0f; };
    must_reject([&] { validate(broken, 0, {1.0, 2.0}, {0.0, 0.0}); });
    broken.output = [&] { return OutputView{output.data(), 1}; };
    must_reject([&] { validate(broken, 0, {1.0, 2.0}, {0.0, 0.0}); });
    broken.output = [&] { return OutputView{output.data(), output.size()}; };
    broken.run = [&](std::size_t) { output.resize(1); };
    must_reject([&] { validate(broken, 0, {1.0, 2.0}, {0.0, 0.0}); });
    must_reject([&] { Suite invalid({}, weights, 1, 1, 1, 6); });
    must_reject([&] { Suite invalid(inputs, weights, -1, 1, 1, 6); });
    must_reject([&] { Suite invalid(inputs, weights, 1, 2, 1, 6); });
}

} // namespace

int main(int argc, char* argv[]) try {
    run_harness_checks();
    run_rsr_api_checks();
    run_eigen_api_checks();
    run_generated_checks();
    if (argc > 1) {
        const auto data = read_sample(argv[1]);
        check_case(data.X, data.W, data.bias, data.expected_linear,
                   data.M, data.K, data.N, 6, data.atol, data.rtol);
    }
    std::cout << "PASS: all baselines, FP32 cancellation, edge cases, and API checks\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
