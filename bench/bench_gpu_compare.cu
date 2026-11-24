// Built by run_gpu_compare.py, once per implementation. The included kernel
// is a generated copy with optional CUDA-event hooks; original files are untouched.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "spr/kernel.hpp"

namespace spr_bench {
cudaEvent_t start, stop;
bool events_enabled = false;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void mark_start() { if (events_enabled) check(cudaEventRecord(start)); }
inline void mark_stop() { if (events_enabled) check(cudaEventRecord(stop)); }
}

#include SPR_BENCH_SOURCE

namespace spr {
std::map<std::string, Factory>& registry() {
    static std::map<std::string, Factory> factories;
    return factories;
}
}

using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Full reference for small problems; evenly spaced output samples for large ones.
// Always checks every output for NaN/Inf. Validation is outside all timed regions.
// X is row-major; Y is in the kernel's layout() (row-major or column-major).
double verify(const std::vector<float>& X, const std::vector<int8_t>& W,
              const std::vector<float>& Y, int M, int K, int N, bool y_col_major) {
    for (float y : Y)
        if (!std::isfinite(y)) throw std::runtime_error("non-finite output");
    const bool full = double(M) * K * N <= 1e7;
    const int nm = full ? M : std::min(M, 8);
    const int nn = full ? N : std::min(N, 64);
    double error = 0;
    for (int mi = 0; mi < nm; ++mi) {
        const int m = nm == 1 ? 0 : mi * (M - 1) / (nm - 1);
        std::vector<double> sums(nn, 0.0);
        for (int r = 0; r < K; ++r)
            for (int ni = 0; ni < nn; ++ni) {
                const int n = nn == 1 ? 0 : ni * (N - 1) / (nn - 1);
                sums[ni] += double(X[size_t(m) * K + r]) * W[size_t(r) * N + n];
            }
        for (int ni = 0; ni < nn; ++ni) {
            const int n = nn == 1 ? 0 : ni * (N - 1) / (nn - 1);
            const size_t i = y_col_major ? size_t(n) * M + m : size_t(m) * N + n;
            error = std::max(error, std::abs(double(Y[i]) - sums[ni]));
        }
    }
    if (error > 2e-3 * std::sqrt(K))
        throw std::runtime_error("reference mismatch: max absolute error=" + std::to_string(error));
    return error;
}

int main(int argc, char** argv) {
    try {
        if (argc != 10) {
            std::fprintf(stderr, "Usage: bench VERSION M K N k ZERO_FRACTION SEED SAMPLES PROFILE_ONLY\n");
            return 2;
        }
        const std::string version = argv[1];
        const int M = std::stoi(argv[2]), K = std::stoi(argv[3]), N = std::stoi(argv[4]);
        spr::Params p; p.k = std::stoi(argv[5]);
        const double zeros = std::stod(argv[6]);
        const unsigned seed = std::stoul(argv[7]);
        const int samples = std::stoi(argv[8]);
        const bool profile = std::stoi(argv[9]) != 0;
        if (M <= 0 || K <= 0 || N <= 0 || samples <= 0 || zeros < 0 || zeros > 1)
            throw std::invalid_argument("invalid benchmark parameters");
        auto kernel = spr::registry().at("spr." + version + ".cuda")();
        if (!kernel->supports(M, K, N, p)) throw std::invalid_argument("unsupported shape");

        std::mt19937 rng(seed);
        std::vector<int8_t> W(size_t(K) * N);
        for (auto& w : W) {
            const bool zero = rng() % 10000 < unsigned(zeros * 10000);
            const int sign = rng() & 1 ? 1 : -1;
            w = zero ? 0 : sign;
        }
        std::vector<float> X(size_t(M) * K), Y(size_t(M) * N);
        for (auto& x : X) x = (int(rng() % 2001) - 1000) / 1000.f;
        // Give the kernel X in the layout it asks for, converted outside every timed region
        // (same contract as spr::Bound in include/spr/harness.hpp).
        const bool col_major = kernel->layout() == spr::Layout::ColMajor;
        std::vector<float> X_kernel(X);
        if (col_major)
            for (int m = 0; m < M; ++m)
                for (int r = 0; r < K; ++r) X_kernel[size_t(r) * M + m] = X[size_t(m) * K + r];
        spr::IO io{M, K, N, X_kernel.data(), Y.data()};
        spr_bench::check(cudaFree(nullptr)); // Initialize CUDA before measuring preparation.
        const auto prep_start = Clock::now();
        kernel->prepare(W.data(), K, N, p);
        const double prep_ms = milliseconds(prep_start);
        kernel->bind(io);

        // This path has exactly five warmup launches before the measured launch,
        // so Nsight Compute can use --launch-skip 5 --launch-count 1.
        for (int i = 0; i < 5; ++i) kernel->run(io);
        if (profile) {
            kernel->run(io);
            kernel->finish(io);
            verify(X, W, Y, M, K, N, col_major);
            return 0;
        }
        kernel->finish(io);
        double error = verify(X, W, Y, M, K, N, col_major);
        const size_t metadata_bytes = kernel->bytes();
        const double adds = kernel->adds_per_row();
        auto emit = [&](const char* metric, int sample, double ms) {
            std::printf("%s,%d,%d,%d,%d,%.4f,%u,%s,%d,%.9f,%zu,%.0f,%.9g\n",
                version.c_str(), M, K, N, p.k, zeros, seed, metric, sample, ms,
                metadata_bytes, adds, error);
        };
        std::puts("version,M,K,N,k,zero_fraction,seed,metric,sample,ms,metadata_bytes,adds_per_row,max_abs_error");
        emit("prepare_wall", 0, prep_ms);
        spr_bench::check(cudaEventCreate(&spr_bench::start));
        spr_bench::check(cudaEventCreate(&spr_bench::stop));

        for (int sample = 0; sample < samples; ++sample) {
            // Alternate phase order. Event hooks are disabled for the wall timings.
            for (int phase_index = 0; phase_index < 3; ++phase_index) {
                const int phase = (sample + phase_index) % 3;
                if (phase == 0) {
                    spr_bench::events_enabled = true;
                    kernel->run(io); // Existing device synchronization completes both events.
                    spr_bench::events_enabled = false;
                    float gpu_ms = 0;
                    spr_bench::check(cudaEventElapsedTime(&gpu_ms, spr_bench::start, spr_bench::stop));
                    emit("gpu_event", sample, gpu_ms);
                } else if (phase == 1) {
                    const auto start = Clock::now();
                    kernel->run(io);
                    emit("run_wall", sample, milliseconds(start));
                } else {
                    const auto start = Clock::now();
                    kernel->bind(io);
                    kernel->run(io);
                    kernel->finish(io);
                    emit("end_to_end_wall", sample, milliseconds(start));
                }
            }
        }
        kernel->finish(io);
        error = verify(X, W, Y, M, K, N, col_major);
        std::fprintf(stderr, "%s M=%d K=%d N=%d k=%d zeros=%.2f: PASS, final max error %.6g\n",
                     version.c_str(), M, K, N, p.k, zeros, error);
        spr_bench::check(cudaEventDestroy(spr_bench::start));
        spr_bench::check(cudaEventDestroy(spr_bench::stop));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
