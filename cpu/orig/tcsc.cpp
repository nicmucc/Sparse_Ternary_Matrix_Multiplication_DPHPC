// Wrappers around the published TCSC kernels (Zhu et al., ACM TECS 2026):
//   third_party/SSR-ternaryLLM-repo/ternaryLLM_CPU/src/GEMM_CPU_FP32.cpp   (compiled unmodified)
//   third_party/SSR-ternaryLLM-repo/ternaryLLM_CPU/src/TCSC.hpp            (weight formats)
//
// One table entry per upstream function. Each entry says which weight format the function
// expects (built exactly like the authors' main.cpp does), which X/Y layout, and the shape
// constraints implied by its unrolling (M % mu == 0, N % nu == 0). See README.md here.
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

#include "GEMM_CPU_FP32.hpp"
#include "TCSC.hpp"
#include "spr/kernel.hpp"

namespace spr {
namespace {

// 64-byte aligned storage. The kernels read row indices with ALIGNED vector loads
// (_mm256_load_si256 / _mm512_load_si512), which fault on x86 if the address is not 32 / 64-byte
// aligned; std::vector only guarantees 16. The format is built by TCSC.hpp unchanged and copied here.
template <class T>
struct Aligned64 {
    using value_type = T;
    Aligned64() = default;
    template <class U> Aligned64(const Aligned64<U>&) {}
    T* allocate(size_t n) {
        void* p = nullptr;
        if (posix_memalign(&p, 64, std::max<size_t>(n * sizeof(T), 64))) throw std::bad_alloc();
        return (T*)p;
    }
    void deallocate(T* p, size_t) { free(p); }
    template <class U> bool operator==(const Aligned64<U>&) const { return true; }
    template <class U> bool operator!=(const Aligned64<U>&) const { return false; }
};
template <class T> using AlignedVec = std::vector<T, Aligned64<T>>;

enum class Fmt {
    Direct,     // dense int8 W (their dense baseline)
    Naive,      // SparseFormat: separate +1 / -1 CSC lists, int16 row indices
    Merged,     // MergedTCSC: +1/-1 interleaved per column, 4 ints of metadata per column
    Group,      // MergedTCSC_Group(G, method, interleaved): columns grouped by G, int16 indices
    RowGroup,   // MergedTCSC_Group(G, "min", false), row indices widened to int32 (rowMajor GEMM/GEMV)
    RowGroupCS  // MergedTCSC_Group(G, "min", false, SIMD_SIZE=cs), int32 indices (GEMV "CSx")
};

struct Spec {
    const char* fn;          // upstream function name
    Fmt fmt;
    Layout layout;
    int mu, nu;              // M % mu == 0 and N % nu == 0 required (unrolling / grouping)
    int G = 1;               // group size
    const char* method = "min";
    bool interleaved = true;
    int cs = 1;              // SIMD_SIZE argument of the CS constructor
    bool gemv = false;       // GEMV: only M == 1
};

class TCSC;
using RunFn = void (*)(TCSC&, IO&);

class TCSC final : public Kernel {
public:
    TCSC(std::string name, Spec s, RunFn f) : name_(std::move(name)), s_(s), f_(f) {}
    std::string name() const override { return name_; }
    Layout layout() const override { return s_.layout; }

    bool supports(int M, int K, int N, const Params&) const override {
#ifdef SPR_AVX512_EMULATED
        // x86 without AVX-512: these compiled through x86_compat/ and are not AVX-512 code
        if (name_.find("AVX512") != std::string::npos) return false;
#endif
        if (s_.gemv && M != 1) return false;
        if (s_.fmt != Fmt::Direct && K > 32767) return false;   // int16 row indices
        return M % s_.mu == 0 && N % s_.nu == 0;
    }

    void prepare(const int8_t* W, int K, int N, const Params&) override {
        K_ = K; N_ = N;
        if (s_.fmt == Fmt::Direct) { w_.assign(W, W + (size_t)K * N); return; }
        naive_ = std::make_unique<SparseFormat>(W, K, N);
        switch (s_.fmt) {
            case Fmt::Merged: merged_ = std::make_unique<MergedTCSC>(*naive_, K, N); break;
            case Fmt::Group: group_ = std::make_unique<MergedTCSC_Group>(*naive_, K, N, s_.G, s_.method, s_.interleaved); break;
            case Fmt::RowGroup: group_ = std::make_unique<MergedTCSC_Group>(*naive_, K, N, s_.G, "min", false); break;
            case Fmt::RowGroupCS: group_ = std::make_unique<MergedTCSC_Group>(*naive_, K, N, s_.G, "min", false, s_.cs); break;
            default: break;
        }
        if (s_.fmt == Fmt::RowGroup || s_.fmt == Fmt::RowGroupCS)   // as in main.cpp: vector<int> idx(row_index.begin(), end())
            ri32_.assign(group_->row_index.begin(), group_->row_index.end());
        else if (s_.fmt == Fmt::Merged)
            ri16_.assign(merged_->row_index.begin(), merged_->row_index.end());
        else if (s_.fmt == Fmt::Group)
            ri16_.assign(group_->row_index.begin(), group_->row_index.end());
    }

    void run(IO& io) override { f_(*this, io); }

    size_t bytes() const override {
        switch (s_.fmt) {
            case Fmt::Direct: return w_.size();
            case Fmt::Naive:
                return (naive_->col_start_pos.size() + naive_->col_start_neg.size()) * 4 +
                       (naive_->row_index_pos.size() + naive_->row_index_neg.size()) * 2;
            case Fmt::Merged: return merged_->metadata.size() * 4 + merged_->row_index.size() * 2;
            case Fmt::Group: return group_->metadata.size() * 4 + group_->row_index.size() * 2;
            default: return group_->metadata.size() * 4 + ri32_.size() * 4;
        }
    }

    // state used by the call adapters below
    int K_ = 0, N_ = 0;
    Spec s_;
    std::vector<int8_t> w_;
    std::unique_ptr<SparseFormat> naive_;
    std::unique_ptr<MergedTCSC> merged_;
    std::unique_ptr<MergedTCSC_Group> group_;
    AlignedVec<int16_t> ri16_;   // aligned copy of the Merged / Group row indices
    AlignedVec<int> ri32_;       // int32 row indices of the rowMajor formats (aligned)

private:
    std::string name_;
    RunFn f_;
};

// ---- call adapters, one per upstream signature (casts only: upstream takes non-const pointers)
#define X_ (const_cast<float*>(io.X))
#define CALL_DIRECT(F)   [](TCSC& t, IO& io) { F(X_, t.w_.data(), io.Y, io.M, t.N_, t.K_); }
#define CALL_NAIVE(F)    [](TCSC& t, IO& io) { auto& n = *t.naive_; \
        F(X_, n.row_index_neg.data(), n.col_start_neg.data(), n.row_index_pos.data(), n.col_start_pos.data(), io.Y, io.M, t.N_, t.K_); }
#define CALL_MERGED(F)   [](TCSC& t, IO& io) { F(io.X, t.merged_->metadata.data(), t.ri16_.data(), io.Y, io.M, t.N_, t.K_); }
#define CALL_GROUP(F)    [](TCSC& t, IO& io) { F(io.X, t.group_->metadata.data(), t.ri16_.data(), io.Y, io.M, t.N_, t.K_); }
#define CALL_ROWGROUP(F) [](TCSC& t, IO& io) { F(X_, t.group_->metadata.data(), t.ri32_.data(), io.Y, io.M, t.N_, t.K_); }
#define CALL_GEMV(F)     [](TCSC& t, IO& io) { F(io.X, t.group_->metadata.data(), t.ri32_.data(), io.Y, t.N_, t.K_); }

#define REG(F, CALL, ...) SPR_REGISTER_FACTORY("tcsc." #F, [] { \
        return std::unique_ptr<Kernel>(new TCSC("tcsc." #F, Spec{#F, __VA_ARGS__}, CALL(F))); })

constexpr Layout C = Layout::ColMajor, R = Layout::RowMajor;

// ---------------------------------------------------------------- dense baselines (theirs)
REG(GEMM_CPU_FP32_colMajor_Direct_OpenMP, CALL_DIRECT, Fmt::Direct, C, 1, 1);
REG(GEMM_CPU_FP32_rowMajor_Direct_OpenMP, CALL_DIRECT, Fmt::Direct, R, 1, 1);

// ---------------------------------------------------------------- naive TCSC (optimization ladder of the paper, Fig. 4)
REG(GEMM_CPU_FP32_colMajor_TCSC_Naive,                     CALL_NAIVE, Fmt::Naive, C, 1, 1);
REG(GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor,              CALL_NAIVE, Fmt::Naive, C, 1, 1);
REG(GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_4x4_Unroll,   CALL_NAIVE, Fmt::Naive, C, 4, 4);
REG(GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2,     CALL_NAIVE, Fmt::Naive, C, 8, 1);
REG(GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x1_AVX2_OpenMP, CALL_NAIVE, Fmt::Naive, C, 8, 1);
REG(GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2,     CALL_NAIVE, Fmt::Naive, C, 8, 4);
REG(GEMM_CPU_FP32_colMajor_TCSC_Naive_oneFor_8x4_AVX2_OpenMP, CALL_NAIVE, Fmt::Naive, C, 8, 4);
REG(GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_OpenMP,       CALL_NAIVE, Fmt::Naive, R, 1, 1);
REG(GEMM_CPU_FP32_rowMajor_TCSC_Naive_oneFor_4x4_Unroll_OpenMP, CALL_NAIVE, Fmt::Naive, R, 4, 4);

// ---------------------------------------------------------------- merged TCSC, one column at a time
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_8x1_AVX2_OpenMP,     CALL_MERGED, Fmt::Merged, C, 8, 1);
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_OpenMP,    CALL_MERGED, Fmt::Merged, C, 16, 1);
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_16x1_AVX2_if_OpenMP, CALL_MERGED, Fmt::Merged, C, 16, 1);
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMP,     CALL_MERGED, Fmt::Merged, C, 8, 4);
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_OpenMPi,    CALL_MERGED, Fmt::Merged, C, 8, 4);
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMP,  CALL_MERGED, Fmt::Merged, C, 8, 4);
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_8x4_AVX2_ij_OpenMPj, CALL_MERGED, Fmt::Merged, C, 8, 4);
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_16xG1_AVX512_OpenMP, CALL_MERGED, Fmt::Merged, C, 16, 1);
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_32xG1_AVX512_OpenMP, CALL_MERGED, Fmt::Merged, C, 32, 1);

// ---------------------------------------------------------------- merged TCSC, column groups ("GroupMin"), colMajor
#define GMIN(MU, G, ISA) REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMin_##MU##xG##G##_##ISA##_OpenMP, CALL_GROUP, Fmt::Group, C, MU, G, G, "min", true)
GMIN(8, 4, AVX2);  GMIN(8, 8, AVX2);  GMIN(8, 16, AVX2);  GMIN(8, 32, AVX2);
GMIN(16, 4, AVX2); GMIN(16, 8, AVX2); GMIN(16, 16, AVX2); GMIN(16, 32, AVX2);
GMIN(32, 4, AVX2); GMIN(32, 8, AVX2); GMIN(32, 16, AVX2); GMIN(32, 32, AVX2);
GMIN(64, 4, AVX2);
GMIN(16, 4, AVX512); GMIN(16, 8, AVX512); GMIN(16, 16, AVX512); GMIN(16, 32, AVX512);
GMIN(32, 4, AVX512); GMIN(32, 8, AVX512); GMIN(32, 16, AVX512); GMIN(32, 32, AVX512);
REG(GEMM_CPU_FP32_colMajor_TCSC_Merged_GroupMid_8xG4_AVX2_OpenMP, CALL_GROUP, Fmt::Group, C, 8, 4, 4, "mid", true);

// ---------------------------------------------------------------- merged TCSC, column groups, rowMajor GEMM
#define RGMIN(MU, G, SUFFIX) REG(GEMM_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_##MU##xG##G##_AVX2_##SUFFIX, CALL_ROWGROUP, Fmt::RowGroup, R, MU, G, G)
RGMIN(1, 8, OpenMP);  RGMIN(1, 8, OpenMPij); RGMIN(1, 8, OpenMPijj); RGMIN(1, 8, OpenMPji); RGMIN(1, 8, OpenMPjii);
RGMIN(2, 8, OpenMP);  RGMIN(4, 8, OpenMP);   RGMIN(8, 8, OpenMP);   RGMIN(16, 8, OpenMP);
RGMIN(1, 16, OpenMP); RGMIN(2, 16, OpenMP);  RGMIN(4, 16, OpenMP);  RGMIN(8, 16, OpenMP);  RGMIN(16, 16, OpenMP);
RGMIN(1, 32, OpenMP); RGMIN(2, 32, OpenMP);  RGMIN(4, 32, OpenMP);  RGMIN(8, 32, OpenMP);  RGMIN(16, 32, OpenMP);
RGMIN(1, 64, OpenMP); RGMIN(2, 64, OpenMP);  RGMIN(4, 64, OpenMP);  RGMIN(8, 64, OpenMP);  RGMIN(16, 64, OpenMP);

// ---------------------------------------------------------------- GEMV (M == 1, token generation), rowMajor
#define GEMV(NAME, G) REG(GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_##NAME##_OpenMP, CALL_GEMV, Fmt::RowGroup, R, 1, G, G, "min", false, 1, true)
#define GEMVCS(NAME, G, CS) REG(GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_##NAME##_OpenMP, CALL_GEMV, Fmt::RowGroupCS, R, 1, G, G, "min", false, CS, true)
GEMV(G8_AVX2, 8); GEMV(G16_AVX2, 16); GEMV(G32_AVX2, 32); GEMV(G64_AVX2, 64);
GEMVCS(G16_CS8_AVX2, 16, 8); GEMVCS(G32_CS8_AVX2, 32, 8); GEMVCS(G64_CS8_AVX2, 64, 8); GEMVCS(G128_CS8_AVX2, 128, 8);
GEMVCS(G64_CS8_SIMD1_AVX2, 64, 8); GEMVCS(G64_CS8_SIMD2_AVX2, 64, 8); GEMVCS(G64_CS8_SIMD3_AVX2, 64, 8);
GEMV(G16_AVX512, 16); GEMV(G32_AVX512, 32); GEMV(G32_SIMD3_AVX512, 32); GEMV(G64_AVX512, 64); GEMV(G128_AVX512, 128);
GEMVCS(G32_CS16_AVX512, 32, 16); GEMVCS(G64_CS16_AVX512, 64, 16); GEMVCS(G128_CS16_AVX512, 128, 16);
GEMVCS(G128_CS16_SIMD1_AVX512, 128, 16); GEMVCS(G128_CS16_SIMD2_AVX512, 128, 16); GEMVCS(G128_CS16_SIMD3_AVX512, 128, 16);

}  // namespace
}  // namespace spr
