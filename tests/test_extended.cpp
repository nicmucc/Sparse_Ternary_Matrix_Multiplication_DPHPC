#include <iostream>
#include <vector>
#include <cmath>
#include "naive.h"
#include "rsr.h"
#include "rsrpp.h"
#include "utils.h"

using namespace std;

int main() {
    const int M = 16;

    for (int i = 2; i <= 12; i++) {
        int K = pow(2, i);
        int N = K;
        cout << "X = " << M << "x" << K << ", W = " << K << "x" << N << endl;
        int block_width = static_cast<int>(ceil(log2(K) - log2(log2(K))));

        vector<vector<int>> X = generateRandomMatrix(M, K);
        vector<vector<int>> W = generateBinaryRandomMatrix(K, N);
        tern_t reference_W;
        for (const auto& row : W)
            reference_W.insert(reference_W.end(), row.begin(), row.end());

        tern_t binary_patterns;
        for (const auto& row : generateBinaryMatrix(block_width))
            binary_patterns.insert(binary_patterns.end(), row.begin(), row.end());

        vector<vec_t> postinference_rsr(M, vec_t(N, 0.0f));
        vector<vec_t> postinference_rsrpp(M, vec_t(N, 0.0f));
        vector<vec_t> expected(M, vec_t(N, 0.0f));

        cout << "preprocessing..." << endl;
        auto per_segs = preprocess(W, block_width);
        permutation_t permutations;
        segment_t segments;
        for (const auto& block : per_segs.first)
            permutations.insert(permutations.end(), block.begin(), block.end());
        for (const auto& block : per_segs.second)
            segments.insert(segments.end(), block.begin(), block.end());

        cout << "inference..." << endl;

        // perform inference for each row of X
        for (int j = 0; j < M; ++j)
        {
            vec_t input(X[j].begin(), X[j].end());
            postinference_rsr[j] = rsr_inference(input, permutations, segments, binary_patterns, K, block_width);
            postinference_rsrpp[j] = rsr_pp_inference(input, permutations, segments, K, block_width);
            if (postinference_rsr[j].size() < static_cast<size_t>(N) || postinference_rsrpp[j].size() < static_cast<size_t>(N)) 
            {
                cout << "ERROR: inference returned too few columns" << endl;
                return 1;
            }
            postinference_rsr[j].resize(N);
            postinference_rsrpp[j].resize(N);
            vectorMatrixMultiply(input, reference_W, expected[j]);
        }

        for (size_t r = 0; r < expected.size(); ++r) {
            for (size_t j = 0; j < expected[r].size(); ++j) {
                if (postinference_rsr[r][j] != expected[r][j]) {
                    cout << "ERROR" << endl;
                    return 1;
                }
                if (postinference_rsrpp[r][j] != expected[r][j]) {
                    cout << "ERROR" << endl;
                    return 1;
                }
            }
        }
        
    }
    cout << "PASSED" << endl;

    return 0;
}
