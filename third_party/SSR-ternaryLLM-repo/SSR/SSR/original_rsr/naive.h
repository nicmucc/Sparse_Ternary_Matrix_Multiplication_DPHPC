#include <iostream>
#include <vector>

using namespace std;

#ifndef VMM
#define VMM
//vector<int> vectorMatrixMultiply(const vector<int>& vec, const vector<vector<int>>& mat);
template <typename T>
vector<T> vectorMatrixMultiply(const vector<T>& vec, const vector<vector<int>>& mat) {
    int n = vec.size();
    int m = mat[0].size();

    // Initialize the result vector with zeros
    vector<T> result(m, 0);

    // Perform vector-matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[i] += vec[j] * mat[j][i];
        }
    }

    return result;
}

template <typename T>
vector<vector<T>> matrixMatrixMultiply(const vector<vector<T>>& in, const vector<vector<int8_t>>& mat) {
    int n = in[0].size();
    int m = mat[0].size();
    int k = in.size();

    // Initialize the result vector with zeros
    vector<vector<T>> result(k, vector<T>(m, 0));

    // Perform vector-matrix multiplication
    for (int l = 0; l < k; l++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[l][i] += in[l][j] * mat[j][i];
            }
        }
    }
    return result;
}
#endif