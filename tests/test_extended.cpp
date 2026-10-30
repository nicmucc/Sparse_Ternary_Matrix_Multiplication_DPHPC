
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

#include "naive.h"
#include "rsr.h"
#include "rsrpp.h"
#include "blas_baseline.h"
#include "utils.h"
#include "eigen_baseline.h"
#include "data_reader.h"

void run_ternary_baseline_checks(const SampleData& data);

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

} // namespace

int main(int argc, char* argv[]) {

    const string directory = (argc > 1) ? argv[1] : "data/sample_001";

    // Read data
    SampleData data = read_sample(directory);

    int M = data.M;
    int K = data.K;
    int N = data.N;

    vec_t X = data.X;
    tern_t W = data.W;

    vec_t bias = data.bias;
    vec_t alpha = data.alpha;

    vec_t expected_linear = data.expected_linear;
    vec_t expected_prelu = data.expected_prelu;

    double atol = data.atol;
    double rtol = data.rtol;

    int block_width = 1;
    if (K > 1) {
        block_width = max(1, static_cast<int>( ceil(log2(K) - log2(log2(K))) ));

    }

    cout << "\n==================================================\n"
         << "  Sparse Ternary Matrix Multiplication - Test\n"
         << "==================================================\n\n"
         << "  Dataset       " << directory << '\n'
         << "  Operation     Y = XW + b\n"
         << "  X             " << M << " x " << K << '\n'
         << "  W             " << K << " x " << N << '\n'
         << "  Y             " << M << " x " << N << '\n'
         << "  Block width   " << block_width << '\n'
         << "  Tolerances    atol=" << atol << "  rtol=" << rtol << "\n\n";

    // Decompose ternary W into two binary matrices
    tern_t W_pos(W.size(), 0);
    tern_t W_neg(W.size(), 0);

    for (size_t i = 0; i < W.size(); i++) {
        if (W[i] == 1) {
            W_pos[i] = 1;
        } else if (W[i] == -1) {
            W_neg[i] = 1;
        } else if (W[i] != 0) {
            throw invalid_argument("Invalid ternary weight");
        }
    }

    // Generate binary patterns
    tern_t binary_patterns = generateBinaryMatrix(block_width);

    // Initialize results
    vec_t postinference_rsr(M * N, 0.0f);
    vec_t postinference_rsrpp(M * N, 0.0f);
    vec_t postinference_blas(
        static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);

    // Preprocessing
    cout << "[1/4] Preprocessing weights..." << endl;

    const auto blas_weights = blas_baseline::int2float(W);
    const DenseMatrix eigen_x = Eigen::Map<const DenseMatrix>(X.data(), M, K);
    const DenseMatrix eigen_w = Eigen::Map<const DenseMatrix>(blas_weights.data(), K, N);

    const auto csr_w = eigen_baseline::preprocess_csr(eigen_w);
    const auto csc_w = eigen_baseline::preprocess_csc(eigen_w);

    const auto eigen_nonzeros = (eigen_w.array() != 0.0f).count();
    if (!csr_w.isCompressed() || !csc_w.isCompressed() ||
        csr_w.nonZeros() != eigen_nonzeros ||
        csc_w.nonZeros() != eigen_nonzeros) {
        cerr << "\n  [FAIL] Eigen preprocessing did not produce compressed "
             << "matrices containing only nonzero weights.\n";
        return 1;
    }

    DenseMatrix eigen_csr(M, N);
    DenseMatrix eigen_csc(M, N);

    auto pos = preprocess(W_pos, block_width, K);
    auto neg = preprocess(W_neg, block_width, K);

    permutation_t permutations_pos, permutations_neg;
    segment_t segments_pos, segments_neg;

    for (const auto& block : pos.first)
        permutations_pos.insert( permutations_pos.end(), block.begin(), block.end());

    for (const auto& block : pos.second)
        segments_pos.insert( segments_pos.end(), block.begin(), block.end());

    for (const auto& block : neg.first)
        permutations_neg.insert( permutations_neg.end(), block.begin(), block.end());

    for (const auto& block : neg.second)
        segments_neg.insert( segments_neg.end(), block.begin(), block.end());

    // Inference
    cout << "[2/4] Running inference..." << endl;

    blas_baseline::multiply(X, blas_weights, postinference_blas, M, K, N);
    eigen_baseline::multiply(eigen_x, csr_w, eigen_csr);
    eigen_baseline::multiply(eigen_x, csc_w, eigen_csc);
    
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            postinference_blas[row * N + col] += bias[col];
            eigen_csr(row, col) += bias[col];
            eigen_csc(row, col) += bias[col];
        }
    }

    for (int j = 0; j < M; j++) {

        // Extract row j of X
        vec_t input( X.begin() + j * K, X.begin() + (j + 1) * K);

        // RSR
        vec_t rsr_pos = rsr_inference( input, permutations_pos, segments_pos, binary_patterns, K, block_width);

        vec_t rsr_neg = rsr_inference( input, permutations_neg, segments_neg, binary_patterns, K, block_width);

        // RSR++
        vec_t rsrpp_pos = rsr_pp_inference( input, permutations_pos, segments_pos, K, block_width );

        vec_t rsrpp_neg = rsr_pp_inference( input, permutations_neg, segments_neg, K, block_width);

        // Check result sizes (padding may add columns)
        if (rsr_pos.size() < static_cast<size_t>(N) ||
            rsr_neg.size() < static_cast<size_t>(N) ||
            rsrpp_pos.size() < static_cast<size_t>(N) ||
            rsrpp_neg.size() < static_cast<size_t>(N)) {
            cerr << "\n  [FAIL] Inference returned too few columns at row "
                 << j << " (zero-based)\n\n"
                 << "         Required        " << N << '\n'
                 << "         RSR positive    " << rsr_pos.size() << '\n'
                 << "         RSR negative    " << rsr_neg.size() << '\n'
                 << "         RSR++ positive  " << rsrpp_pos.size() << '\n'
                 << "         RSR++ negative  " << rsrpp_neg.size() << '\n';
            return 1;
        }

        // Combine positive and negative results, add bias
        for (int col = 0; col < N; col++) {
            postinference_rsr[j * N + col] = rsr_pos[col] - rsr_neg[col] + bias[col];

            postinference_rsrpp[j * N + col] = rsrpp_pos[col] - rsrpp_neg[col] + bias[col];
        }
    }

    // Compare with precomputed results
    cout << "[3/4] Comparing against reference..." << endl;

    const auto print_failure = [N](const char* algorithm, size_t index,
                                    float expected, float actual, float tolerance) {
        cerr << "\n  [FAIL] " << algorithm << " at row " << index / N
             << ", column " << index % N << " (zero-based)\n\n"
             << setprecision(9)
             << "         Expected        " << expected << '\n'
             << "         Actual          " << actual << '\n'
             << "         Absolute error  " << abs(actual - expected) << '\n'
             << "         Tolerance       " << tolerance << '\n';
    };

    for (size_t i = 0; i < expected_linear.size(); i++) {

        float expected = expected_linear[i];
        float tolerance = atol + rtol * abs(expected);

        if (!isfinite(postinference_rsr[i]) ||
            abs(postinference_rsr[i] - expected) > tolerance) {
            print_failure("RSR", i, expected, postinference_rsr[i], tolerance);
            return 1;
        }

        if (!isfinite(postinference_rsrpp[i]) ||
            abs(postinference_rsrpp[i] - expected) > tolerance) {
            print_failure("RSR++", i, expected, postinference_rsrpp[i], tolerance);
            return 1;
        }

        if (!isfinite(postinference_blas[i]) ||
            abs(postinference_blas[i] - expected) > tolerance) {
            print_failure("BLAS", i, expected, postinference_blas[i], tolerance);
            return 1;
        }

        const float csr_actual = eigen_csr.data()[i];
        const float csc_actual = eigen_csc.data()[i];

        if (!isfinite(csr_actual) || abs(csr_actual - expected) > tolerance) {
            print_failure("Eigen CSR", i, expected, csr_actual, tolerance);
            return 1;
        }

        if (!isfinite(csc_actual) || abs(csc_actual - expected) > tolerance) {
            print_failure("Eigen CSC", i, expected, csc_actual, tolerance);
            return 1;
        }
    }

    // Run focused API and edge-case checks
    cout << "[4/4] Running API checks..." << endl;

    try {
        run_rsr_api_checks();
        run_eigen_api_checks();
        run_ternary_baseline_checks(data);
    } catch (const exception& error) {
        cerr << "\n  [FAIL] API checks: " << error.what() << '\n';
        return 1;
    }

    cout << "\n  [PASS] RSR\n"
         << "  [PASS] RSR++\n"
         << "  [PASS] BLAS\n"
         << "  [PASS] Eigen CSR\n"
         << "  [PASS] Eigen CSC\n"
         << "  [PASS] SSR basic / unrolled / tree\n"
         << "  [PASS] TCSC basic / merged\n"
         << "  [PASS] SSR/TCSC edge cases and API checks\n"
         << "  [PASS] RSR API checks\n"
         << "  [PASS] Eigen API checks\n\n"
         << "--------------------------------------------------\n"
         << "  All implementations and checks passed.\n"
         << "==================================================\n";

    return 0;
}
