#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>
#include <random>

using namespace std;

int binaryVectorToInt(const vector<int>& binaryVec) {
    int result = 0;
    int n = binaryVec.size();
    
    for (int i = 0; i < n; ++i) {
        result = (result << 1) | binaryVec[i];  // Left-shift result and add the next bit
    }
    
    return result;
}

vector<vector<int>> generateBinaryMatrix(int k) {
    int rows = pow(2, k);  // 2^k rows
    vector<vector<int>> matrix(rows, vector<int>(k, 0));  // Initialize matrix with 0s

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < k; ++j) {
            // Generate the binary value for each position
            matrix[i][k - j - 1] = (i >> j) & 1;  // Extract the j-th bit from i
        }
    }

    return matrix;
}

vector<vector<int8_t>> generateBinaryRandomMatrix(int n, int m, double p) {
    // Initialize the random number generator
    random_device rd; // Seed
    mt19937 gen(rd()); // Mersenne Twister engine
    uniform_real_distribution<> dis(0.0, 1.0); // Distribution that produces values between 0 and 1
    
    // Initialize matrix
    vector<vector<int8_t>> matrix(n, vector<int8_t>(m));
    
    // Populate the matrix with random 0s and 1s based on percentage p
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = (dis(gen) < p) ? 0 : 1; // Generate 0 with probability p, 1 with probability (1-p)
        }
    }
    
    return matrix;
}

vector<vector<int8_t>> generateTernaryRandomMatrix(int n, int m, double p) {
    // Initialize the random number generator
    random_device rd; // Seed
    mt19937 gen(rd()); // Mersenne Twister engine
    uniform_real_distribution<> dis(0.0, 1.0); // Distribution that produces values between 0 and 1
    
    // Initialize matrix
    vector<vector<int8_t>> matrix(n, vector<int8_t>(m));
    
    // Populate the matrix with random 0s and 1s based on percentage p
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (dis(gen) < (1-p)/2) {
                matrix[i][j] = -1;
            } else if (dis(gen) < 1-p) {
                matrix[i][j] = 1;
            } else {
                matrix[i][j] = 0;
            }
             // Generate 0 with probability p, 1, -1 with probability (1-p)/2
        }
    }
    
    return matrix;
}
