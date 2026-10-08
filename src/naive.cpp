#include <vector>
#include <stdexcept>
#include "utils.h"

using namespace std;

vector<int> vectorMatrixMultiply(const vector<int>& vec, const vector<vector<int>>& mat) {
    if (vec.size() != mat.size()) {
        throw invalid_argument("vector length must match the matrix row count");
    }

    if (mat.empty()) {
        return {};
    }

    const size_t column_count = mat.front().size();
    vector<int> result(column_count, 0);

    for (size_t row = 0; row < mat.size(); ++row) {
        if (mat[row].size() != column_count) {
            throw invalid_argument("matrix rows must have equal lengths");
        }
        for (size_t column = 0; column < column_count; ++column) {
            result[column] += vec[row] * mat[row][column];
        }
    }

    return result;
}
