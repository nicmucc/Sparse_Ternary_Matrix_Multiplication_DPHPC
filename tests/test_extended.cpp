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
        const vector<vector<int>> reference_mat2 = mat2; //preprocessing of mat2 can change it, so we need to keep a copy of the original mat2 for comparison
        vector<vector<int>> bin_k = generateBinaryMatrix(k);
        vector<vector<int>> postinference_rsr(input_rows, vector<int>(n, 0));
        vector<vector<int>> postinference_rsrpp(input_rows, vector<int>(n, 0));
        vector<vector<int>> expected(input_rows, vector<int>(n, 0));

        cout << "preprocessing..." << endl;
        auto per_segs = preprocess(mat2, k);

        cout << "inference..." << endl;

        // perform inference for each row of mat1
        for (int j = 0; j < input_rows; ++j)
        {
            postinference_rsr[j] = rsr_inference(mat1[j], per_segs.first, per_segs.second, bin_k, k);
            postinference_rsrpp[j] = rsr_pp_inference(mat1[j], per_segs.first, per_segs.second, k);
            if (postinference_rsr[j].size() < static_cast<size_t>(n) || postinference_rsrpp[j].size() < static_cast<size_t>(n)) 
            {
                cout << "ERROR: inference returned too few columns" << endl;
                return 1;
            }
            postinference_rsr[j].resize(n);
            postinference_rsrpp[j].resize(n);
            vectorMatrixMultiply(mat1[j], reference_mat2, expected[j], input_rows);
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
