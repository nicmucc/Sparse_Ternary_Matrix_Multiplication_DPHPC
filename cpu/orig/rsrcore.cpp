// Wrappers around RSR-core, the authors' optimized RSR kernels (Dehghankar et al.):
//   third_party/RSR-core-repo/kernels/bit_1_58/cpu/rsr_ternary_prep_nonsquare.c   preprocessing   (compiled unmodified)
//   third_party/RSR-core-repo/kernels/bit_1_58/cpu/rsr_ternary_v3_1.c             GEMV v3.1       (compiled unmodified)
//   third_party/RSR-core-repo/kernels/bit_1_58/cpu/rsr_ternary_v3_3.c             GEMV v3.3       (compiled unmodified)
//   third_party/RSR-core-repo/kernels/bit_1/cpu/rsr_prep_nonsquare.c              binary prep     (compiled unmodified)
//   third_party/RSR-core-repo/kernels/bit_1/cpu/rsr_v4_2.c                        binary GEMV v4.2 (compiled unmodified)
//
// Their ternary RSR computes out = A * v with A (n_rows x n_cols) ternary, blocking k ROWS of A.
// Our problem y = x * W is out = W^T x, so A = W^T: n_rows = N (outputs), n_cols = K (inputs).
//
// Format, built exactly like their Python wrappers (multiplier/bit_1_58/cpu/rsr_nonsquare.py, rsr_v3_3.py):
//   per block, a counting sort of the K inputs by their 2k-bit ternary code; uint16 perms and group ends;
//   v3.1: per group a list of (output row, sign) scatter entries;  v3.3: per group two uint16 bit masks.
// Note: the all-zero pattern is a group too, so RSR still gathers every input in every block (no zero skipping).
//
// They are GEMV kernels (M = 1, LLM decoding). For M > 1 we call them once per row of X; RSR-core has
// no multi-row kernel (its "_batch" files batch layers, and also quantize the activations, so they are not used).
// v3.3 has a NEON / AVX2 path for k <= 8 (NEON: k = 5..8); other k use their scalar path.
//
// Binary v4.2 (the original RSR, binary matrices): run once on the +1 plane and once on the -1
// plane of W, y = A+ x - A- x (the RSR / RSR++ papers' way of handling ternary weights).
// Format as their multiplier/bit_1/cpu/rsr_cpp_nonsquare.py builds it (rows padded to a multiple of k).
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

#include "spr/kernel.hpp"

extern "C" {
void rsr_ternary_prep_nonsquare(const int8_t* M, int n_rows, int n_cols, int k, int32_t* perms, int32_t* group_ends,
                                int32_t* scatter_offsets, int8_t* scatter_rows, int8_t* scatter_signs, int32_t* block_meta,
                                int32_t* out_sizes);
void rsr_ternary_gemv_v3_1(const uint16_t* perms, const uint16_t* group_ends, const int32_t* scatter_offsets,
                           const int8_t* scatter_rows, const int8_t* scatter_signs, const int32_t* block_meta, const float* v,
                           float* out, int n, int k, int num_blocks);
void rsr_prep_nonsquare_omp(const uint8_t* M, int n_rows, int n_cols, int k, int32_t* perms, int32_t* group_ends,
                            int32_t* scatter_offsets, uint8_t* scatter_rows, int32_t* block_meta, int32_t* out_sizes);
void rsr_gemv_v4_2(const int32_t* perms, const int32_t* group_ends, const int32_t* scatter_offsets,
                   const uint8_t* scatter_rows, const int32_t* block_meta, const float* v, float* out, int n, int k,
                   int num_blocks);
void rsr_ternary_gemv_v3_3(const uint16_t* perms, const uint16_t* group_ends, const uint16_t* pos_masks,
                           const uint16_t* neg_masks, const int32_t* block_meta, const float* v, float* out, int n, int k,
                           int num_blocks);
}

namespace spr {
namespace {

class RSRCore final : public Kernel {
public:
    explicit RSRCore(int version) : v33_(version == 33) {}
    std::string name() const override { return v33_ ? "rsrcore.rsr_ternary_gemv_v3_3" : "rsrcore.rsr_ternary_gemv_v3_1"; }
    bool uses_k() const override { return true; }
    // k <= 12: their counting sort allocates 4^k buckets (rsr_nonsquare.py asserts the same); uint16 indices need K <= 65535
    bool supports(int, int K, int, const Params& p) const override { return p.k >= 1 && p.k <= 12 && K <= 65535; }

    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        K_ = K; N_ = N; k_ = p.k;
        Npad_ = (N + k_ - 1) / k_ * k_;
        nb_ = Npad_ / k_;
        std::vector<int8_t> A((size_t)Npad_ * K, 0);                    // A = W^T, rows padded to a multiple of k
        for (int j = 0; j < K; ++j)
            for (int n = 0; n < N; ++n) A[(size_t)n * K + j] = W[(size_t)j * N + n];

        const size_t max_groups = (size_t)nb_ * std::min<size_t>((size_t)1 << (2 * k_), K);
        std::vector<int32_t> perms((size_t)nb_ * K), ge(max_groups), so(max_groups + 1), out_sizes(2);
        std::vector<int8_t> rows(max_groups * 2 * k_), signs(max_groups * 2 * k_);
        meta_.resize(2 * nb_);
        rsr_ternary_prep_nonsquare(A.data(), Npad_, K, k_, perms.data(), ge.data(), so.data(), rows.data(), signs.data(),
                                   meta_.data(), out_sizes.data());
        const int groups = out_sizes[0], scatter = out_sizes[1];

        perms_.assign(perms.begin(), perms.end());                       // astype(np.uint16)
        ge_.assign(ge.begin(), ge.begin() + groups);
        so_.assign(so.begin(), so.begin() + groups + 1);
        rows_.assign(rows.begin(), rows.begin() + scatter);
        signs_.assign(signs.begin(), signs.begin() + scatter);
        pos_.assign(groups, 0); neg_.assign(groups, 0);                  // rsr_v3_3.py: masks from the scatter lists
        for (int g = 0; g < groups; ++g)
            for (int s = so_[g]; s < so_[g + 1]; ++s) (signs_[s] > 0 ? pos_ : neg_)[g] |= (uint16_t)(1u << rows_[s]);
        if (v33_) { so_.clear(); rows_.clear(); signs_.clear(); }        // v3.3 only reads the masks
        out_.assign(Npad_, 0.f);
        adds_ = (double)nb_ * K + scatter;                               // every input gathered once per block + one add per scatter entry
    }

    void run(IO& io) override {
        const bool direct = Npad_ == N_;                                 // write straight into Y when no padding
        for (int m = 0; m < io.M; ++m) {
            const float* x = io.X + (size_t)m * K_;
            float* y = direct ? io.Y + (size_t)m * N_ : out_.data();
            if (v33_) rsr_ternary_gemv_v3_3(perms_.data(), ge_.data(), pos_.data(), neg_.data(), meta_.data(), x, y, K_, k_, nb_);
            else rsr_ternary_gemv_v3_1(perms_.data(), ge_.data(), so_.data(), rows_.data(), signs_.data(), meta_.data(), x, y, K_, k_, nb_);
            if (!direct) std::memcpy(io.Y + (size_t)m * N_, y, N_ * sizeof(float));
        }
    }

    size_t bytes() const override {
        return perms_.size() * 2 + ge_.size() * 2 + meta_.size() * 4 +
               (v33_ ? (pos_.size() + neg_.size()) * 2 : so_.size() * 4 + rows_.size() + signs_.size());
    }
    double adds_per_row() const override { return adds_; }

private:
    bool v33_;
    int K_ = 0, N_ = 0, k_ = 0, Npad_ = 0, nb_ = 0;
    double adds_ = 0;
    std::vector<uint16_t> perms_, ge_, pos_, neg_;
    std::vector<int32_t> so_, meta_;
    std::vector<int8_t> rows_, signs_;
    std::vector<float> out_;
};

// One binary plane (W == +1 or W == -1), transposed: A (Npad x K), A[n][j] = 1.
struct BinaryPlane {
    std::vector<int32_t> perms, ge, so, meta;
    std::vector<uint8_t> rows;
    int scatter = 0;

    void prepare(const int8_t* W, int K, int N, int Npad, int k, int8_t sign) {
        std::vector<uint8_t> A((size_t)Npad * K, 0);
        for (int j = 0; j < K; ++j)
            for (int n = 0; n < N; ++n) A[(size_t)n * K + j] = W[(size_t)j * N + n] == sign;
        const int nb = Npad / k;
        const size_t max_groups = (size_t)nb * std::min<size_t>((size_t)1 << k, K);   // rsr_cpp_nonsquare.py
        perms.resize((size_t)nb * K); ge.resize(max_groups); so.resize(max_groups + 1);
        rows.resize(max_groups * k); meta.resize(2 * nb);
        std::vector<int32_t> out_sizes(2);
        rsr_prep_nonsquare_omp(A.data(), Npad, K, k, perms.data(), ge.data(), so.data(), rows.data(), meta.data(),
                               out_sizes.data());
        ge.resize(out_sizes[0]); so.resize(out_sizes[0] + 1); rows.resize(out_sizes[1]);
        scatter = out_sizes[1];
    }
    void gemv(const float* x, float* out, int K, int k, int nb) const {
        rsr_gemv_v4_2(perms.data(), ge.data(), so.data(), rows.data(), meta.data(), x, out, K, k, nb);
    }
    size_t bytes() const { return perms.size() * 4 + ge.size() * 4 + so.size() * 4 + rows.size() + meta.size() * 4; }
};

class RSRCoreBinary final : public Kernel {
public:
    std::string name() const override { return "rsrcore.rsr_gemv_v4_2"; }
    bool uses_k() const override { return true; }
    // k <= 20 in their counting sort (rsr_cpp_nonsquare.py); uint8 scatter rows need k <= 256
    bool supports(int, int, int, const Params& p) const override { return p.k >= 1 && p.k <= 16; }

    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        K_ = K; N_ = N; k_ = p.k;
        Npad_ = (N + k_ - 1) / k_ * k_;
        nb_ = Npad_ / k_;
        pos_.prepare(W, K, N, Npad_, k_, 1);
        neg_.prepare(W, K, N, Npad_, k_, -1);
        out_pos_.assign(Npad_, 0.f); out_neg_.assign(Npad_, 0.f);
        adds_ = 2.0 * nb_ * K + pos_.scatter + neg_.scatter + N;          // gathers + scatters per plane + the subtraction
    }

    void run(IO& io) override {
        for (int m = 0; m < io.M; ++m) {
            const float* x = io.X + (size_t)m * K_;
            pos_.gemv(x, out_pos_.data(), K_, k_, nb_);
            neg_.gemv(x, out_neg_.data(), K_, k_, nb_);
            float* y = io.Y + (size_t)m * N_;
            for (int n = 0; n < N_; ++n) y[n] = out_pos_[n] - out_neg_[n];
        }
    }

    size_t bytes() const override { return pos_.bytes() + neg_.bytes(); }
    double adds_per_row() const override { return adds_; }

private:
    int K_ = 0, N_ = 0, k_ = 0, Npad_ = 0, nb_ = 0;
    double adds_ = 0;
    BinaryPlane pos_, neg_;
    std::vector<float> out_pos_, out_neg_;
};

SPR_REGISTER_FACTORY("rsrcore.rsr_gemv_v4_2", [] { return std::unique_ptr<Kernel>(new RSRCoreBinary()); });
SPR_REGISTER_FACTORY("rsrcore.rsr_ternary_gemv_v3_1", [] { return std::unique_ptr<Kernel>(new RSRCore(31)); });
SPR_REGISTER_FACTORY("rsrcore.rsr_ternary_gemv_v3_3", [] { return std::unique_ptr<Kernel>(new RSRCore(33)); });

}  // namespace
}  // namespace spr
