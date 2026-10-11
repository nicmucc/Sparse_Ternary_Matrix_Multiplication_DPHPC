
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "naive.h"
#include "rsr.h"
#include "rsrpp.h"
#include "utils.h"
#include "data_reader.h"

using namespace std;

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

    cout << "X = " << M << "x" << K
         << ", W = " << K << "x" << N << endl;

    int block_width = 1;
    if (K > 1) {
        block_width = max(1, static_cast<int>( ceil(log2(K) - log2(log2(K))) ));
    }

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
    tern_t binary_patterns;

    for (const auto& row : generateBinaryMatrix(block_width)) {
        binary_patterns.insert( binary_patterns.end(), row.begin(), row.end());
    }

    // Initialize results
    vec_t postinference_rsr(M * N, 0.0f);
    vec_t postinference_rsrpp(M * N, 0.0f);

    // Preprocessing
    cout << "preprocessing..." << endl;

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
    cout << "inference..." << endl;

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
            cerr << "ERROR: inference returned too few columns\n";
            return 1;
        }

        // Combine positive and negative results, add bias
        for (int col = 0; col < N; col++) {
            postinference_rsr[j * N + col] = rsr_pos[col] - rsr_neg[col] + bias[col];

            postinference_rsrpp[j * N + col] = rsrpp_pos[col] - rsrpp_neg[col] + bias[col];
        }
    }

    // Compare with precomputed results
    for (size_t i = 0; i < expected_linear.size(); i++) {

        float expected = expected_linear[i];
        float tolerance = atol + rtol * abs(expected);

        if (!isfinite(postinference_rsr[i]) ||
            abs(postinference_rsr[i] - expected) > tolerance) {
            cout << "ERROR RSR at index " << i << endl;
            return 1;
        }

        if (!isfinite(postinference_rsrpp[i]) ||
            abs(postinference_rsrpp[i] - expected) > tolerance) {
            cout << "ERROR RSR++ at index " << i << endl;
            return 1;
        }
    }

    cout << "PASSED" << endl;

    return 0;
}
