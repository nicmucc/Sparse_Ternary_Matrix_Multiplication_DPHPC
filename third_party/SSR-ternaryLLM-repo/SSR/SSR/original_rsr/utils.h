#pragma once
#include <vector>
#include <random>

using namespace std;


int binaryVectorToInt(const vector<int>& binaryVec);

vector<vector<int>> generateBinaryMatrix(int k);

vector<vector<int8_t>> generateBinaryRandomMatrix(int n, int m, double p);

vector<vector<int8_t>> generateTernaryRandomMatrix(int n, int m, double p);

/*
vector<int> generateRandomVector(int n);

vector<int> copy(const vector<int>& v);

vector<vector<int>> copy(const vector<vector<int>>& mat);
*/
#ifndef U
#define U

template <typename T>
vector<T> generateRandomVector(int n) {
    // Initialize the random number generator
    random_device rd;  // Seed
    mt19937 gen(rd()); // Mersenne Twister engine

    // Initialize matrix
    vector<T> matrix(n);

    uniform_int_distribution<> dis(0, 100);
    // Populate the matrix with random integers
    for (int i = 0; i < n; ++i) {
        matrix[i] = dis(gen); // Generate random integer in range [minValue, maxValue]
    }

    return matrix;
}

template <typename T>
vector<vector<T>> generateRandomMatrix(int n, int m) {
    // Initialize the random number generator
    random_device rd;  // Seed
    mt19937 gen(rd()); // Mersenne Twister engine

    // Initialize matrix
    vector<vector<T>> matrix(m, vector<T>(n));

    if (is_same<T,float>()) {
        uniform_int_distribution<> dis(-1000, 1000);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                matrix[j][i] = dis(gen); // Generate random integer in range [minValue, maxValue]
            }
        }
    } else {
        uniform_int_distribution<> dis(-10, 10);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                matrix[j][i] = dis(gen); // Generate random integer in range [minValue, maxValue]
            }
        }
    }
    // Populate the matrix with random integers
    

    return matrix;
}


template <typename T>
vector<T> generateRandomMatrix2(int n, int m) {
    // Initialize the random number generator
    random_device rd;  // Seed
    mt19937 gen(rd()); // Mersenne Twister engine

    // Initialize matrix
    vector<T> matrix(m * n);

    if (is_same<T, float>()) {
        uniform_int_distribution<> dis(-1000, 1000);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                matrix[j * n + i] = dis(gen);
            }
        }
    }
    else {
        uniform_int_distribution<> dis(-10, 10);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                matrix[j * n + i] = dis(gen); // Generate random integer in range [minValue, maxValue]
            }
        }
    }
    // Populate the matrix with random integers


    return matrix;
}

template <typename T>
vector<vector<T>> copy(const vector<vector<T>>& mat) {
    vector<vector<T>> copied(mat.size(), vector<T>(mat[0].size()));
    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[0].size(); j++) {
            copied[i][j] = mat[i][j];
        }
    }
    return copied;
}

template <typename T>
vector<T> copy(const vector<T>& v) {
    vector<T> copied(v.size());

    for (int i = 0; i < v.size(); i++) {
        copied[i] = v[i];
    }
    
    return copied;
}
#endif