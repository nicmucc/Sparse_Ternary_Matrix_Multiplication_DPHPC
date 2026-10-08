#include <iostream>
#include <vector>
#include <cmath>
#include "naive.h"
#include "rsr.h"
#include "rsrpp.h"
#include "utils.h"

using namespace std;

int main() {
    for (int i = 2; i <= 12; i++) {
        int n = pow(2, i);
        cout << "n = " << n << endl;
        int k = static_cast<int>(ceil(log2(n) - log2(log2(n))));

        vector<vector<int>> mat1 = generateRandomMatrix(n);
        vector<vector<int>> mat2 = generateBinaryRandomMatrix(n);
        vector<vector<int>> bin_k = generateBinaryMatrix(k);
        vector<vector<int>> postinference_rsr(n, vector<int>(n, 0));
        vector<vector<int>> postinference_rsrpp(n, vector<int>(n, 0));
        vector<vector<int>> expected(n, vector<int>(n, 0));

        cout << "preprocessing..." << endl;
        auto per_segs = preprocess(mat2, k);

        cout << "inference..." << endl;

        // perform inference for each row of mat1
        for (int j=0; j<n; ++j)
        {
            postinference_rsr[j] = rsr_inference(mat1[j], per_segs.first, per_segs.second, bin_k, k);
            postinference_rsrpp[j] = rsr_pp_inference(mat1[j], per_segs.first, per_segs.second, k);
            expected[j] = vectorMatrixMultiply(mat1[j], mat2);
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
