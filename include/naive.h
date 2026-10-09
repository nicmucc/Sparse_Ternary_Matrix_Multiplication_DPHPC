#ifndef NAIVE_H
#define NAIVE_H

/*
 * Vector-Matrix multiply
 */

// vector<int> vectorMatrixMultiply(const vector<int>& vec, const vector<vector<int>>& mat);
// vec : 1 x K
// mat : K x N
// res : 1 x N
//
// Preconditions:
//     - res a std::vector of size 1 x N initialized with zeros
//     - vec a std::vector of size 1 x K
//     - mat a std::vector of size K x N
//
// Postconditions:
//     - retult stored in res

void vectorMatrixMultiply(const vec_t& vec, 
                          const tern_t& mat, 
                          vec_t& res
                          );

#endif
