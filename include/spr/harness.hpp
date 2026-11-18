#pragma once
// Runs one Kernel on one problem. Shared by tests/test_correctness.cpp and
// bench/bench.cpp so both exercise the kernels in exactly the same way.
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "spr/kernel.hpp"

namespace spr {

// ---------------------------------------------------------------- utilities
inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// i.i.d. ternary W, row-major K x N: P(0)=s, P(+1)=P(-1)=(1-s)/2 (same as the SSR/TCSC repos).
inline std::vector<int8_t> random_ternary(int K, int N, double s, uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<int8_t> W((size_t)K * N);
    const double h = (1.0 - s) / 2.0;
    for (auto& w : W) { double u = U(g); w = u < h ? 1 : (u < 2 * h ? -1 : 0); }
    return W;
}

inline std::vector<float> random_dense(int M, int K, uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> X((size_t)M * K);
    for (auto& x : X) x = U(g);
    return X;
}

std::string machine_name();
std::string compiler_name();

// 64-byte aligned float buffer (AVX-512 aligned loads in the TCSC kernels need it).
struct AlignedFloats {
    float* p = nullptr;
    size_t n = 0;
    explicit AlignedFloats(size_t n_ = 0) : n(n_) {
        if (n && posix_memalign((void**)&p, 64, std::max<size_t>(n * sizeof(float), 64))) abort();
        if (p) std::memset(p, 0, n * sizeof(float));
    }
    AlignedFloats(const AlignedFloats&) = delete;
    AlignedFloats& operator=(const AlignedFloats&) = delete;
    ~AlignedFloats() { free(p); }
};

// ---------------------------------------------------------------- problem
struct Problem {
    int M, K, N;
    std::vector<float> X;      // row-major M x K
    std::vector<int8_t> W;     // row-major K x N
    std::vector<float> Yref;   // row-major ref_rows x N: the reference for the first ref_rows rows of Y
    int ref_rows;
};

// ref_rows < M: only the first ref_rows rows of Y get a reference (and are verified). The
// benchmark uses this to keep the O(MKN) double-precision reference cheap; tests use all rows.
inline Problem make_problem(int M, int K, int N, double s, uint64_t seed, int ref_rows = -1,
                            const std::string& sparsity_pattern = "iid") {
    ref_rows = ref_rows < 0 ? M : std::min(ref_rows, M);
    if (sparsity_pattern != "iid" && sparsity_pattern != "column_clustered")
        throw std::invalid_argument("unknown sparsity pattern: " + sparsity_pattern);
    if (s < 0 || s > 1) throw std::invalid_argument("sparsity must be in [0, 1]");
    auto W = random_ternary(K, N, sparsity_pattern == "iid" ? s : 0.0, seed + 1);
    if (sparsity_pattern == "column_clustered") {
        // Deliberately imbalanced diagnostic: contiguous zero output columns,
        // followed by dense random +/-1 columns. Zero fraction = floor(s*N)/N.
        const int zero_columns = (int)(s * N);
        for (int r = 0; r < K; ++r)
            std::fill_n(W.data() + (size_t)r * N, zero_columns, int8_t{0});
    }
    Problem P{M, K, N, random_dense(M, K, seed), std::move(W), {}, ref_rows};
    P.Yref.resize((size_t)ref_rows * N);
    reference_gemm(P.X.data(), P.W.data(), P.Yref.data(), ref_rows, K, N);
    return P;
}

// A saved dataset of scripts/generate_data.py (make data): metadata.json, X.bin (float32, M x K,
// row-major) and W.bin (int8, K x N, row-major). Bias / PReLU files are ignored (we measure Y = XW).
// *sparsity receives the generator's target zero fraction.
inline Problem load_dataset(const std::string& dir, int ref_rows = -1, double* sparsity = nullptr) {
    std::ifstream meta(dir + "/metadata.json");
    if (!meta) throw std::runtime_error("cannot open " + dir + "/metadata.json");
    std::stringstream ss;
    ss << meta.rdbuf();
    const std::string json = ss.str();
    const size_t config = json.find("\"config\"");
    if (config == std::string::npos) throw std::runtime_error(dir + ": metadata.json has no \"config\"");
    auto number = [&](const char* key) {   // first "key": <number> after "config"
        const size_t at = json.find(std::string("\"") + key + "\"", config);
        if (at == std::string::npos) throw std::runtime_error(dir + ": metadata.json config has no " + key);
        return std::stod(json.substr(json.find(':', at) + 1));
    };
    const int M = (int)number("M"), K = (int)number("K"), N = (int)number("N");
    if (sparsity) *sparsity = number("target_zero_fraction");

    auto read = [&](const char* name, void* data, size_t bytes) {
        std::ifstream f(dir + "/" + name, std::ios::binary);
        if (!f.read((char*)data, (std::streamsize)bytes)) throw std::runtime_error("cannot read " + dir + "/" + name);
    };
    ref_rows = ref_rows < 0 ? M : std::min(ref_rows, M);
    Problem P{M, K, N, std::vector<float>((size_t)M * K), std::vector<int8_t>((size_t)K * N), {}, ref_rows};
    read("X.bin", P.X.data(), P.X.size() * sizeof(float));
    read("W.bin", P.W.data(), P.W.size());
    P.Yref.resize((size_t)ref_rows * N);
    reference_gemm(P.X.data(), P.W.data(), P.Yref.data(), ref_rows, K, N);
    return P;
}

inline double tolerance(int K) { return 2e-3 * std::sqrt((double)K); }

// One prepared kernel bound to one problem: owns X / Y in the kernel's layout.
struct Bound {
    Kernel* kern;
    AlignedFloats X, Y;
    IO io;

    Bound(Kernel* k, const Problem& P) : kern(k), X((size_t)P.M * P.K), Y((size_t)P.M * P.N) {
        io.M = P.M; io.K = P.K; io.N = P.N;
        if (k->layout() == Layout::RowMajor) {
            std::copy(P.X.begin(), P.X.end(), X.p);
        } else {
            for (int m = 0; m < P.M; ++m)
                for (int j = 0; j < P.K; ++j) X.p[(size_t)j * P.M + m] = P.X[(size_t)m * P.K + j];
        }
        io.X = X.p; io.Y = Y.p;
        kern->bind(io);
    }
    void run() { kern->run(io); }

    // Zeroes Y, runs once, collects the result, returns the max abs error vs the reference.
    double verify(const Problem& P) {
        std::memset(Y.p, 0, Y.n * sizeof(float));
        kern->run(io);
        kern->finish(io);
        double e = 0;
        for (int m = 0; m < P.ref_rows; ++m)
            for (int n = 0; n < P.N; ++n) {
                float y = kern->layout() == Layout::RowMajor ? Y.p[(size_t)m * P.N + n] : Y.p[(size_t)n * P.M + m];
                double d = std::fabs((double)y - P.Yref[(size_t)m * P.N + n]);
                e = std::isnan(d) ? INFINITY : std::max(e, d);
            }
        return e;
    }
};

inline void set_threads(int t) { omp_set_num_threads(t); }

}  // namespace spr
