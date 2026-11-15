// Wrappers around the published SSR kernels (Pittet et al., DATE 2026):
//   third_party/SSR-ternaryLLM-repo/SSR/SSR/ssr.cpp       ssr_preprocess            (compiled unmodified)
//   third_party/SSR-ternaryLLM-repo/SSR/SSR/ssr.h         ssr_inference (scalar)    (template, unmodified)
//   third_party/SSR-ternaryLLM-repo/SSR/SSR/ssr_vec.cpp   AVX2 kernels              (compiled unmodified)
//
// All of them use the flat format returned by ssr_preprocess(mat, L, K, N):
//   permutations_{pos,neg} : ceil(N/L) * K ints, segments_{pos,neg} : ceil(N/L) * 2^L ints
// and return a new std::vector<float> of M * N_pad floats (row-major, N padded to a
// multiple of L). The copy of X into a std::vector happens in bind(), the copy of the
// result into io.Y in finish(), both outside timing (as in the authors' benchmarks).
#include "spr/kernel.hpp"
#include "ssr.h"
#include "ssr_vec.h"
#include "patched/ssr_inference_patched.h"   // scalar SSR with the upstream indexing bug fixed

namespace spr {
namespace {

class SSRBase : public Kernel {
public:
    bool uses_k() const override { return true; }

    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        K_ = K; N_ = N; L_ = p.k;
        std::vector<std::vector<int8_t>> mat(K, std::vector<int8_t>(N));
        for (int r = 0; r < K; ++r)
            for (int c = 0; c < N; ++c) mat[r][c] = W[(size_t)r * N + c];
        auto pre = ssr_preprocess(mat, L_, K, N);
        pp_ = std::move(pre.first.first);  sp_ = std::move(pre.first.second);
        pn_ = std::move(pre.second.first); sn_ = std::move(pre.second.second);
    }
    void bind(const IO& io) override { v_.assign(io.X, io.X + (size_t)io.M * io.K); }
    void finish(IO& io) override {
        const int Npad = N_ + (L_ - N_ % L_) % L_;
        for (int m = 0; m < io.M; ++m)
            for (int n = 0; n < N_; ++n) io.Y[(size_t)m * N_ + n] = out_[(size_t)m * Npad + n];
    }
    size_t bytes() const override { return (pp_.size() + sp_.size() + pn_.size() + sn_.size()) * sizeof(int); }

protected:
    int K_ = 0, N_ = 0, L_ = 0;
    std::vector<int> pp_, sp_, pn_, sn_;
    std::vector<float> v_, out_;
};

// ssr.h: plain C++ ("SSR Naive" in the paper), one row of X at a time, tree as loops.
// Has `#pragma omp parallel for` over rows, so it is multi-threaded when OpenMP is on.
class SSRScalar final : public SSRBase {
public:
    std::string name() const override { return "ssr.ssr_inference"; }
    bool supports(int, int, int, const Params& p) const override { return p.k >= 1 && p.k <= 12; }
    void run(IO& io) override { out_ = ssr_inference<float>(v_, pp_, sp_, pn_, sn_, L_, io.M, N_, K_); }
};
SPR_REGISTER_KERNEL("ssr.ssr_inference", SSRScalar);

// cpu/orig/patched/ssr_inference_patched.h: the same function with ONLY the gather index fixed
// (row * L -> row * K, b_idx * L -> b_idx * K). This is the scalar SSR we benchmark.
class SSRScalarPatched final : public SSRBase {
public:
    std::string name() const override { return "ssr.ssr_inference_patched"; }
    bool supports(int, int, int, const Params& p) const override { return p.k >= 1 && p.k <= 12; }
    void run(IO& io) override { out_ = ssr_inference_patched<float>(v_, pp_, sp_, pn_, sn_, L_, io.M, N_, K_); }
};
SPR_REGISTER_KERNEL("ssr.ssr_inference_patched", SSRScalarPatched);

// ssr_vec.cpp: ssr_inference_float_vec_L2 ... _L8 -- one fully unrolled AVX2 kernel per L,
// 8 rows of X per __m256, OpenMP over rows. (ssr_inference_float_vec_tree only dispatches
// to these for L <= 6.)
class SSRVecL final : public SSRBase {
public:
    std::string name() const override { return "ssr.ssr_inference_float_vec_L"; }
    bool supports(int M, int, int, const Params& p) const override { return M % 8 == 0 && p.k >= 2 && p.k <= 8; }
    void run(IO& io) override {
        switch (L_) {
            case 2: out_ = ssr_inference_float_vec_L2(v_, pp_, sp_, pn_, sn_, K_, N_, io.M); break;
            case 3: out_ = ssr_inference_float_vec_L3(v_, pp_, sp_, pn_, sn_, K_, N_, io.M); break;
            case 4: out_ = ssr_inference_float_vec_L4(v_, pp_, sp_, pn_, sn_, K_, N_, io.M); break;
            case 5: out_ = ssr_inference_float_vec_L5(v_, pp_, sp_, pn_, sn_, K_, N_, io.M); break;
            case 6: out_ = ssr_inference_float_vec_L6(v_, pp_, sp_, pn_, sn_, K_, N_, io.M); break;
            case 7: out_ = ssr_inference_float_vec_L7(v_, pp_, sp_, pn_, sn_, K_, N_, io.M); break;
            default: out_ = ssr_inference_float_vec_L8(v_, pp_, sp_, pn_, sn_, K_, N_, io.M); break;
        }
    }
};
SPR_REGISTER_KERNEL("ssr.ssr_inference_float_vec_L", SSRVecL);

// ssr_vec.cpp: generic-L AVX2 kernels, rows of X unrolled by 8 (unroll8_v2) or 16 (unroll16_v2).
// NB the parameter names are (L, M, N, K) but the authors' main.cpp passes (L, K_LEN, N_COL, M_ROW)
// -- inside these functions "M" is the inner dimension and "K" the number of rows. We call them
// exactly like main.cpp does.
class SSRUnroll final : public SSRBase {
public:
    explicit SSRUnroll(int u) : u_(u) {}
    std::string name() const override { return u_ == 8 ? "ssr.ssr_inference_float_vec_unroll8_v2" : "ssr.ssr_inference_float_vec_unroll16_v2"; }
    bool supports(int, int, int, const Params& p) const override { return p.k >= 1 && p.k <= 12; }
    void run(IO& io) override {
        if (u_ == 8) out_ = ssr_inference_float_vec_unroll8_v2(v_, pp_, sp_, pn_, sn_, L_, K_, N_, io.M);
        else out_ = ssr_inference_float_vec_unroll16_v2(v_, pp_, sp_, pn_, sn_, L_, K_, N_, io.M);
    }

private:
    int u_;
};
SPR_REGISTER_FACTORY("ssr.ssr_inference_float_vec_unroll8_v2", [] { return std::unique_ptr<Kernel>(new SSRUnroll(8)); });
SPR_REGISTER_FACTORY("ssr.ssr_inference_float_vec_unroll16_v2", [] { return std::unique_ptr<Kernel>(new SSRUnroll(16)); });

}  // namespace
}  // namespace spr
