// Additional CUDA edge cases for every registered GPU kernel.
// Built alongside the common correctness suite by make build-gpu.
// Run both suites with make test-gpu; TEST_ARGS supplies kernel-name substrings.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "spr/kernel.hpp"

int main(int argc, char** argv) {
    try {
        int selected = 0;
        for (const auto& name : spr::kernel_names()) {
            bool matches = argc == 1;
            for (int arg = 1; arg < argc; ++arg)
                matches = matches || name.find(argv[arg]) != std::string::npos;
            if (!matches) continue;
            ++selected;
            auto kernel = spr::make_kernel(name);
            std::mt19937 rng(42);
            struct Shape { int M, K, N; };
            const Shape shapes[] = {
                {1, 7, 3}, {2, 3, 7}, {35, 129, 65}, {4, 1024, 257},
                {129, 17, 19}, {2, 0, 7}, {0, 7, 5}, {2, 7, 0}, {1, 65535, 3}
            };
            int cases = 0;
            for (const auto shape : shapes) {
                const int M = shape.M, K = shape.K, N = shape.N;
                for (int k : {1, 3, 8, 16}) {
                    for (int pattern = 0; pattern < 3; ++pattern) {
                        spr::Params params;
                        params.k = k;
                        if (!kernel->supports(M, K, N, params)) continue;
                        std::vector<int8_t> W(static_cast<size_t>(K) * N);
                        for (auto& w : W)
                            w = pattern == 0 ? 0 : static_cast<int>(rng() % 3) - 1;
                        // Empty first block followed by nonempty blocks.
                        if (pattern == 2)
                            for (int r = 0; r < K; ++r)
                                for (int c = 0; c < std::min(k, N); ++c)
                                    W[static_cast<size_t>(r) * N + c] = 0;

                        kernel->prepare(W.data(), K, N, params);
                        std::vector<float> X(static_cast<size_t>(M) * K);
                        std::vector<float> Y(static_cast<size_t>(M) * N);
                        std::vector<double> reference(Y.size());

                        for (int binding = 0; binding < 2; ++binding) {
                            for (auto& x : X)
                                x = (static_cast<int>(rng() % 2001) - 1000) / 1000.f;
                            std::fill(reference.begin(), reference.end(), 0.0);
                            for (int m = 0; m < M; ++m)
                                for (int r = 0; r < K; ++r)
                                    for (int n = 0; n < N; ++n)
                                        reference[static_cast<size_t>(m) * N + n] +=
                                            static_cast<double>(X[static_cast<size_t>(m) * K + r]) *
                                            W[static_cast<size_t>(r) * N + n];

                            // Hand X to the kernel in its layout(), like spr::Bound does.
                            const bool col_major = kernel->layout() == spr::Layout::ColMajor;
                            std::vector<float> X_kernel(X);
                            if (col_major)
                                for (int m = 0; m < M; ++m)
                                    for (int r = 0; r < K; ++r)
                                        X_kernel[static_cast<size_t>(r) * M + m] =
                                            X[static_cast<size_t>(m) * K + r];
                            spr::IO io{M, K, N, X_kernel.data(), Y.data()};
                            kernel->bind(io);
                            for (int repeat = 0; repeat < 3; ++repeat) {
                                std::fill(Y.begin(), Y.end(), 123.f);
                                kernel->run(io);
                                kernel->finish(io);
                                const double tolerance = 2e-3 * std::sqrt(std::max(1, K));
                                for (size_t i = 0; i < Y.size(); ++i) {
                                    // reference is row-major: i = m * N + n
                                    const size_t m = N ? i / N : 0, n = N ? i % N : 0;
                                    const size_t y_index = col_major ? n * M + m : i;
                                    if (!std::isfinite(Y[y_index]) ||
                                        std::abs(Y[y_index] - reference[i]) > tolerance) {
                                        std::fprintf(stderr,
                                            "FAIL %s M=%d K=%d N=%d k=%d pattern=%d "
                                            "binding=%d repeat=%d index=%zu: got %.9g expected %.9g\n",
                                            name.c_str(), M, K, N, k, pattern, binding, repeat,
                                            i, static_cast<double>(Y[y_index]), reference[i]);
                                        return 1;
                                    }
                                }
                            }
                        }
                        ++cases;
                    }
                }
            }
            std::printf("PASS %s: %d cases against double-precision reference, "
                        "each with two bindings and three runs per binding.\n", name.c_str(), cases);
        }
        if (!selected) {
            std::fprintf(stderr, "ERROR: no GPU kernels matched the filters\n");
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
