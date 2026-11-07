#include <vector>
#include "utils.h"

using namespace std;

// Convert a binary vector to an integer
int binaryVectorToInt(const vector<int>& binaryVec) {
    int result = 0;

    for (int bit : binaryVec) {
        result = (result << 1) | bit;
    }

    return result;
}


tern_t generateBinaryMatrix(int block_width) {
    int rows = 1 << block_width;

    tern_t matrix(rows * block_width, 0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < block_width; j++) {
            matrix[i * block_width + block_width - j - 1] = (i >> j) & 1;
        }
    }

    return matrix;
}