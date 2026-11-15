// Wrappers around RSR / RSR++ (Dehghankar et al. 2025) in the version the SSR authors
// adapted to fp32 matrices and benchmarked against:
//   third_party/SSR-ternaryLLM-repo/SSR/SSR/original_rsr/rsr.cpp   rsr_preprocess          (compiled unmodified)
//   third_party/SSR-ternaryLLM-repo/SSR/SSR/original_rsr/rsr.h     rsr_inference, vector_rsr_inference
//   third_party/SSR-ternaryLLM-repo/SSR/SSR/original_rsr/rsrpp.h   rsr_pp_inference, vector_rsr_pp_inference
//
// Format: rsr_preprocess(mat, L) -> per block of L columns and per half (+1 / -1):
//   permutation (K ints) and segment list (2^L ints), as vector<vector<int>>.
// The matrix versions take X as vector<vector<float>> BY VALUE (so every call copies X;
// that copy is part of their API and is timed, exactly as in the authors' benchmark) and
// return vector<vector<float>> (M x N_pad).
#include "rsr.h"
#include "rsrpp.h"
#include "spr/kernel.hpp"
#include "utils.h"

namespace spr {
namespace {

class RSRBase : public Kernel {
public:
    explicit RSRBase(int variant) : variant_(variant) {}
    bool uses_k() const override { return true; }
    bool supports(int M, int, int, const Params& p) const override {
        return p.k >= 1 && p.k <= 12 && (variant_ < 2 || M == 1);
    }
    std::string name() const override {
        static const char* names[] = {"rsr.rsr_inference", "rsr.rsr_pp_inference",
                                      "rsr.vector_rsr_inference", "rsr.vector_rsr_pp_inference"};
        return names[variant_];
    }

    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        K_ = K; N_ = N; L_ = p.k;
        std::vector<std::vector<int8_t>> mat(K, std::vector<int8_t>(N));
        for (int r = 0; r < K; ++r)
            for (int c = 0; c < N; ++c) mat[r][c] = W[(size_t)r * N + c];
        auto pre = rsr_preprocess(mat, L_);
        pp_ = std::move(pre.first.first);  sp_ = std::move(pre.first.second);
        pn_ = std::move(pre.second.first); sn_ = std::move(pre.second.second);
        bin_ = generateBinaryMatrix(L_);
    }
    void bind(const IO& io) override {
        X_.assign(io.M, std::vector<float>(io.K));
        for (int m = 0; m < io.M; ++m) std::copy(io.X + (size_t)m * io.K, io.X + (size_t)(m + 1) * io.K, X_[m].begin());
    }
    void run(IO&) override {
        switch (variant_) {
            case 0: out_ = rsr_inference<float>(X_, pp_, sp_, pn_, sn_, bin_, L_); break;
            case 1: out_ = rsr_pp_inference<float>(X_, pp_, sp_, pn_, sn_, L_); break;
            case 2: out_ = {vector_rsr_inference<float>(X_[0], pp_, sp_, pn_, sn_, bin_, L_)}; break;
            default: out_ = {vector_rsr_pp_inference<float>(X_[0], pp_, sp_, pn_, sn_, L_)}; break;
        }
    }
    void finish(IO& io) override {
        for (int m = 0; m < io.M; ++m)
            for (int n = 0; n < N_; ++n) io.Y[(size_t)m * N_ + n] = out_[m][n];
    }
    size_t bytes() const override {
        size_t s = 0;
        for (auto* v : {&pp_, &sp_, &pn_, &sn_})
            for (auto& b : *v) s += b.size() * sizeof(int);
        return s;
    }

private:
    int variant_, K_ = 0, N_ = 0, L_ = 0;
    std::vector<std::vector<int>> pp_, sp_, pn_, sn_, bin_;
    std::vector<std::vector<float>> X_, out_;
};

SPR_REGISTER_FACTORY("rsr.rsr_inference",           [] { return std::unique_ptr<Kernel>(new RSRBase(0)); });
SPR_REGISTER_FACTORY("rsr.rsr_pp_inference",        [] { return std::unique_ptr<Kernel>(new RSRBase(1)); });
SPR_REGISTER_FACTORY("rsr.vector_rsr_inference",    [] { return std::unique_ptr<Kernel>(new RSRBase(2)); });
SPR_REGISTER_FACTORY("rsr.vector_rsr_pp_inference", [] { return std::unique_ptr<Kernel>(new RSRBase(3)); });

}  // namespace
}  // namespace spr
