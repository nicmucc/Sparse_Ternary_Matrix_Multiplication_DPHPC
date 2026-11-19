// Registry, reference GEMM, environment info. Shared by every executable.
#include <unistd.h>

#include "spr/harness.hpp"

namespace spr {

std::map<std::string, Factory>& registry() {
    static std::map<std::string, Factory> r;
    return r;
}

std::unique_ptr<Kernel> make_kernel(const std::string& name) {
    auto it = registry().find(name);
    if (it == registry().end()) throw std::invalid_argument("unknown kernel: " + name);
    return it->second();
}

std::vector<std::string> kernel_names() {
    std::vector<std::string> v;
    for (auto& kv : registry()) v.push_back(kv.first);
    return v;
}

void reference_gemm(const float* X, const int8_t* W, float* Y, int M, int K, int N) {
    std::vector<double> acc(N);
    for (int m = 0; m < M; ++m) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int j = 0; j < K; ++j) {
            const double x = X[(size_t)m * K + j];
            const int8_t* w = W + (size_t)j * N;
            for (int n = 0; n < N; ++n) acc[n] += x * w[n];
        }
        for (int n = 0; n < N; ++n) Y[(size_t)m * N + n] = (float)acc[n];
    }
}

std::string machine_name() {
    char h[256] = {0};
    gethostname(h, sizeof(h) - 1);
    return h;
}

std::string compiler_name() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#else
    return "unknown";
#endif
}

}  // namespace spr
