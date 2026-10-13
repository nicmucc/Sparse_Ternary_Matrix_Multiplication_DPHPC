
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "naive.h"
#include "rsr.h"
#include "rsrpp.h"
#include "blas_baseline.h"
#include "utils.h"
#include "eigen_baseline.h"
#include "data_reader.h"

using namespace std;
using eigen_baseline::DenseMatrix;

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
    blas_baseline::FloatMatrix postinference_blas(
        static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);

    // Preprocessing
    cout << "[1/3] Preprocessing weights..." << endl;

    const auto blas_weights = blas_baseline::int2float(W);
    const DenseMatrix eigen_x = Eigen::Map<const DenseMatrix>(X.data(), M, K);
    const DenseMatrix eigen_w = Eigen::Map<const DenseMatrix>(blas_weights.data(), K, N);

    const auto csr_w = eigen_baseline::preprocess_csr(eigen_w);
    const auto csc_w = eigen_baseline::preprocess_csc(eigen_w);

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
    cout << "[2/3] Running inference..." << endl;

    blas_baseline::blas_GEMM(X, blas_weights, postinference_blas, M, K, N);
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
    cout << "[3/3] Comparing against reference..." << endl;

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

    cout << "\n  [PASS] RSR\n"
         << "  [PASS] RSR++\n"
         << "  [PASS] BLAS\n\n"
         << "--------------------------------------------------\n"
         << "  All 3 implementations passed.\n"
         << "==================================================\n";

    return 0;
}
