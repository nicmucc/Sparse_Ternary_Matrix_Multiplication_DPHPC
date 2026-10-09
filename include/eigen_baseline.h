#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>

namespace eigen_baseline {

// dense inputs and outputs are row-major.
using DenseMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using CSRMatrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;
using CSCMatrix = Eigen::SparseMatrix<float, Eigen::ColMajor>;

// Adapter for the existing integer tests
DenseMatrix to_dense(const std::vector<std::vector<int>>& matrix);

// Build compressed weights once, outside the inference/timing loop. Zeros are omitted.
CSRMatrix preprocess_csr(const DenseMatrix& weights);
CSCMatrix preprocess_csc(const DenseMatrix& weights);

// Compute Y = XW for the entire batch. Y must be distinct from X (no alias!). Pre-size Y to
// X.rows() x W.cols() before timing to avoid output allocation in the timed call.
void multiply(const DenseMatrix& X, const CSRMatrix& W, DenseMatrix& Y);
void multiply(const DenseMatrix& X, const CSCMatrix& W, DenseMatrix& Y);

}
