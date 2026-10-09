#include <iostream>
#include <vector>
#include <cmath>
#include "naive.h"
#include "rsr.h"
#include "rsrpp.h"
#include "utils.h"

using namespace std;

int main() {
    const int input_rows = 16;

    for (int i = 2; i <= 12; i++) {
        int n = pow(2, i);
        cout << "X = " << input_rows << "x" << n << ", W = " << n << "x" << n << endl;
        int k = static_cast<int>(ceil(log2(n) - log2(log2(n))));

        vector<vector<int>> mat1 = generateRandomMatrix(input_rows, n);
        vector<vector<int>> mat2 = generateBinaryRandomMatrix(n, n);
        tern_t reference_mat2;
        for (const auto& row : mat2)
            reference_mat2.insert(reference_mat2.end(), row.begin(), row.end());

        tern_t bin_k;
        for (const auto& row : generateBinaryMatrix(k))
            bin_k.insert(bin_k.end(), row.begin(), row.end());

        vector<vec_t> postinference_rsr(input_rows, vec_t(n, 0.0f));
        vector<vec_t> postinference_rsrpp(input_rows, vec_t(n, 0.0f));
        vector<vec_t> expected(input_rows, vec_t(n, 0.0f));

        cout << "preprocessing..." << endl;
        auto per_segs = preprocess(mat2, k);
        permutation_t permutations;
        segment_t segments;
        for (const auto& block : per_segs.first)
            permutations.insert(permutations.end(), block.begin(), block.end());
        for (const auto& block : per_segs.second)
            segments.insert(segments.end(), block.begin(), block.end());

        cout << "inference..." << endl;

        // perform inference for each row of mat1
        for (int j = 0; j < input_rows; ++j)
        {
            vec_t input(mat1[j].begin(), mat1[j].end());
            postinference_rsr[j] = rsr_inference(input, permutations, segments, bin_k, input.size(), k);
            postinference_rsrpp[j] = rsr_pp_inference(input, permutations, segments, input.size(), k);
            if (postinference_rsr[j].size() < static_cast<size_t>(n) || postinference_rsrpp[j].size() < static_cast<size_t>(n)) 
            {
                cout << "ERROR: inference returned too few columns" << endl;
                return 1;
            }
            postinference_rsr[j].resize(n);
            postinference_rsrpp[j].resize(n);
            vectorMatrixMultiply(input, reference_mat2, expected[j]);
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
