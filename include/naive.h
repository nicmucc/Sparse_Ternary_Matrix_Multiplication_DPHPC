#ifndef NAIVE_H
#define NAIVE_H

/*
 * Vector-Matrix multiply
 */

// vector<int> vectorMatrixMultiply(const vector<int>& vec, const vector<vector<int>>& mat);
// vec : N x 1
// mat : K x N
// res : K x 1
//
// Preconditions:
//     - res a std::vector of size K x 1 initialized with zeros
//     - vec a std::vector of size N x 1
//     - mat a std::vector of size K x N
//
// Postconditions:
//     - retult stored in res

void vectorMatrixMultiply(const vec_t& vec, 
                          const tern_t& mat, 
                          vec_t& res,
                          size_t K 
                          );

#endif
