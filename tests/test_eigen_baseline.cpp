#include "eigen_baseline.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using eigen_baseline::DenseMatrix;

namespace {

void check_case(const DenseMatrix& X, const DenseMatrix& W, const DenseMatrix& expected) {
    const auto csr = eigen_baseline::preprocess_csr(W);
    const auto csc = eigen_baseline::preprocess_csc(W);
    const auto nonzeros = (W.array() != 0.0f).count();
    if (!csr.isCompressed() || !csc.isCompressed() ||
        csr.nonZeros() != nonzeros || csc.nonZeros() != nonzeros) {
        throw std::runtime_error("weights must contain only compressed nonzeros");
    }

    DenseMatrix csr_output(X.rows(), W.cols());
    DenseMatrix csc_output(X.rows(), W.cols());
    // Repeat with dirty outputs to verify assignment does not accumulate.
    for (int repeat = 0; repeat < 2; ++repeat) {
        // fill the output with incorrect values
        csr_output.setConstant(123.0f); 
        csc_output.setConstant(-456.0f);
        // perform both mulitplications
        eigen_baseline::multiply(X, csr, csr_output);
        eigen_baseline::multiply(X, csc, csc_output);
        // check whether the outputs are corresponding to the expected values
        for (Eigen::Index row = 0; row < expected.rows(); ++row) {
            for (Eigen::Index column = 0; column < expected.cols(); ++column) {
                const float tolerance = 1e-6f + 1e-5f * std::abs(expected(row, column));

                if (!std::isfinite(csr_output(row, column)) ||
                    !std::isfinite(csc_output(row, column))) {
                        throw std::runtime_error("Infinite values in the outputs");
                }

                if (std::abs(csr_output(row, column) - expected(row, column)) > tolerance ||
                    std::abs(csc_output(row, column) - expected(row, column)) > tolerance) {
                        throw std::runtime_error("CSR/CSC output differs from reference");
                }
            }
        }
    }
}

} // namespace

int main() {
    try {
        DenseMatrix X(2, 3);
        X << 0.5f, -2.0f, 3.0f,
             -1.5f, 4.0f, -0.25f;
        DenseMatrix W(3, 4);
        // Rectangular ternary weights, with an empty row and an empty column.
        W << -1, 0, 1, 0,
              0, 0, 0, 0,
              1, 0, -1, 1;
        DenseMatrix expected(2, 4);
        expected << 2.5f, 0, -2.5f, 3.0f,
                    1.25f, 0, -1.25f, -0.25f;
        check_case(X, W, expected);
        check_case(X, DenseMatrix::Zero(3, 4), DenseMatrix::Zero(2, 4));
        check_case(DenseMatrix::Constant(1, 1, -0.5f),
                   DenseMatrix::Constant(1, 1, -1.0f),
                   DenseMatrix::Constant(1, 1, 0.5f));

        const auto csr = eigen_baseline::preprocess_csr(W);
        const auto csc = eigen_baseline::preprocess_csc(W);
        DenseMatrix output;
        int rejected = 0;
        // sanity checks
        try { eigen_baseline::multiply(DenseMatrix::Zero(1, 2), csr, output); }
        catch (const std::invalid_argument&) { ++rejected; }
        try { eigen_baseline::multiply(DenseMatrix::Zero(1, 2), csc, output); }
        catch (const std::invalid_argument&) { ++rejected; }
        try { eigen_baseline::to_dense({{1, 2}, {3}}); }
        catch (const std::invalid_argument&) { ++rejected; }
        try { eigen_baseline::multiply(X, csr, X); }
        catch (const std::invalid_argument&) { ++rejected; }
        if (rejected != 4) {
            throw std::runtime_error("invalid inputs were not rejected");
        }
        std::cout << "Eigen CSR/CSC ternary correctness PASSED\n";
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
