#pragma once
// THE common interface. Every method -- the authors' published kernels (cpu/orig/),
// the dense BLAS reference and our SPR kernels -- implements Kernel, so the benchmark
// driver, the correctness test, timing and memory accounting are shared and therefore
// identical for all of them.
//
//   Y (M x N, fp32) = X (M x K, fp32) * W (K x N, ternary int8 in {-1,0,1})
//
// Call sequence (harness.hpp):
//   prepare(W)           preprocessing / weight compression        timed separately (prep_ms)
//   bind(io)             copy X into the kernel's own containers   NOT timed
//   run(io)  x R         inference                                 TIMED
//   finish(io)           move the result into io.Y                 NOT timed
//
// bind() and finish() exist because some published kernels take std::vector arguments
// and return a freshly allocated std::vector (SSR, RSR). Their own benchmarks also build
// these vectors outside the timed region, so we do the same.
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace spr {

struct Params {
    int k = 4;            // block width (columns of W per block): L in SSR / RSR, unused otherwise
    int threads = 1;      // OpenMP threads (set by the harness with omp_set_num_threads)
    std::string str() const { return "k=" + std::to_string(k); }
};

// Layout of X and Y as the kernel expects them. The harness converts outside timing.
//   RowMajor : X[m*K + j],  Y[m*N + n]
//   ColMajor : X[j*M + m],  Y[n*M + m]     (TCSC "colMajor" kernels)
enum class Layout { RowMajor, ColMajor };

struct IO {
    int M = 0, K = 0, N = 0;
    const float* X = nullptr;   // 64-byte aligned, in layout()
    float* Y = nullptr;         // 64-byte aligned, in layout(), zeroed before the verified run
};

struct NotImplemented : std::runtime_error { using std::runtime_error::runtime_error; };

struct Kernel {
    virtual ~Kernel() = default;
    virtual std::string name() const = 0;
    virtual Layout layout() const { return Layout::RowMajor; }
    virtual bool uses_k() const { return false; }   // true: the driver sweeps Params::k

    // Shape restrictions of the kernel (e.g. M multiple of the unroll factor, N multiple
    // of the group size, M == 1 for GEMV). Unsupported configs are skipped, not failed.
    virtual bool supports(int M, int K, int N, const Params& p) const { (void)M; (void)K; (void)N; (void)p; return true; }

    virtual void prepare(const int8_t* W, int K, int N, const Params& p) = 0;
    virtual void bind(const IO& io) { (void)io; }
    virtual void run(IO& io) = 0;
    virtual void finish(IO& io) { (void)io; }

    // Bytes of the compressed weights + metadata that run() reads (X and Y excluded).
    virtual size_t bytes() const = 0;

    // Optional: scalar additions run() performs per row of X (0 = unknown). Lets the bench
    // separate "less work" (algorithm) from "faster work" (engineering): see ns_per_add.
    virtual double adds_per_row() const { return 0; }
};

// ---------------------------------------------------------------------------
// Registry: each .cpp or .cu registers its kernels with
//     SPR_REGISTER_KERNEL("name", Class);                 // default-constructible class
//     SPR_REGISTER_FACTORY("name", [] { return ...; });   // any factory (table-driven wrappers)
// Kernel objects are linked directly into each executable so the registrars are never dropped.
// ---------------------------------------------------------------------------
using Factory = std::function<std::unique_ptr<Kernel>()>;
std::map<std::string, Factory>& registry();
std::unique_ptr<Kernel> make_kernel(const std::string& name);
std::vector<std::string> kernel_names();

struct Registrar {
    Registrar(const std::string& name, Factory f) { registry()[name] = std::move(f); }
};
#define SPR_CAT2(a, b) a##b
#define SPR_CAT(a, b) SPR_CAT2(a, b)
#define SPR_REGISTER_FACTORY(NAME, FACTORY) static ::spr::Registrar SPR_CAT(spr_reg_, __COUNTER__)(NAME, FACTORY)
#define SPR_REGISTER_KERNEL(NAME, CLASS) \
    SPR_REGISTER_FACTORY(NAME, [] { return std::unique_ptr<::spr::Kernel>(new CLASS()); })

// Plain O(MKN) reference in double precision (row-major), used by tests and the bench.
void reference_gemm(const float* X, const int8_t* W, float* Y, int M, int K, int N);

}  // namespace spr
