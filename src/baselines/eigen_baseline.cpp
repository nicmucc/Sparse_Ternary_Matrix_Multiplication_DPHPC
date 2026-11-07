#include "eigen_baseline.h"

#include <stdexcept>

namespace eigen_baseline {
namespace {

template <typename SparseMatrix>
SparseMatrix compress(const DenseMatrix& weights) {
    std::vector<Eigen::Triplet<float>> entries;
    for (Eigen::Index row = 0; row < weights.rows(); ++row) {
        for (Eigen::Index column = 0; column < weights.cols(); ++column) {
            const float value = weights(row, column);
            if (value != 0.0f) { // only take non zero elements
                entries.emplace_back(row, column, value);
            }
        }
    }
    SparseMatrix result(weights.rows(), weights.cols());
    result.setFromTriplets(entries.begin(), entries.end());
    result.makeCompressed();
    return result;
}

template <typename SparseMatrix>
void product(const DenseMatrix& X, const SparseMatrix& W, DenseMatrix& Y) {
    if (X.cols() != W.rows()) {
        throw std::invalid_argument("X column count must match W row count");
    }
    if (&X == &Y) {
        throw std::invalid_argument("output must not alias the input matrix");
    }
    // choose correct output shape
    Y.resize(X.rows(), W.cols());
    // tells Eigen that the output does not overlap with the inputs
    Y.noalias() = X * W; 
}

}

DenseMatrix to_dense(const std::vector<std::vector<int>>& matrix) {
    const std::size_t columns = matrix.empty() ? 0 : matrix.front().size();
    DenseMatrix result(matrix.size(), columns);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        if (matrix[row].size() != columns) {
            throw std::invalid_argument("matrix rows must have equal lengths");
        }
        for (std::size_t column = 0; column < columns; ++column) {
            result(row, column) = static_cast<float>(matrix[row][column]);
        }
    }
    return result;
}

CSRMatrix preprocess_csr(const DenseMatrix& weights) {
    return compress<CSRMatrix>(weights);
}

CSCMatrix preprocess_csc(const DenseMatrix& weights) {
    return compress<CSCMatrix>(weights);
}

void multiply(const DenseMatrix& X, const CSRMatrix& W, DenseMatrix& Y) {
    product(X, W, Y);
}

void multiply(const DenseMatrix& X, const CSCMatrix& W, DenseMatrix& Y) {
    product(X, W, Y);
}

}
