// spr.v2

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "spr/kernel.hpp"

namespace spr {
namespace {
template <int B>
class SPRV2 final : public Kernel {
// =====================================
// Result: faster runtime and compiler 
// able to perfectly vectorize the code 
// and we read 25% of the flop ceiling
// =====================================

// =====================================
//    -SPR.V2 (cache, SIMD optimization)
// !! started from v1
// Benchmarks show that more than 90% of
// the work done in inference is on the 
// this single instruction: 
// t += x[(int) *(p_rows++)];
// this access over x is bad, we should 
// get better performance if we block 
// over rows of x and make x in column 
// major format
// -> we will call <B> the input block 
//    size
// =====================================

private:
    // =============================================
    // Matrix Shape and State Bounds 
    // =============================================
    int K_ = 0, N_ = 0;             // W shape
    int k_ = 0;                     // block width
    size_t max_D_ = 0;              // track max nbr of unique row patterns excluding zero
                                    // which we need to determine scratchpad size
    
    // =============================================
    // Segmented Sums
    // =============================================
    // How many unique segments (D) exist in each block.
    // Size = num_blocks. (Max segments is 3^k, easily fits in uint16_t).
    std::vector<uint16_t> blk_num_seg_;
    
    // How many rows belong to each segment.
    // Size = total_segments per block. (A segment length cannot exceed K <= 65535).
    std::vector<uint16_t> seg_len_;

    // The permutation of row indices in X, with all zeros completely removed.
    // Size = total_non_zero_rows. (Row index < K <= 65535).
    std::vector<uint16_t> rows_;


    // =============================================
    // Tree Control Metadata
    // =============================================
    struct LvlMetadata {
        uint16_t p_len;
        uint16_t n_len;
        uint16_t m_len;
    };
    std::vector<LvlMetadata> lvl_meta_;


    // =============================================
    // Optimized Online Tree Reduciton
    // =============================================
    // Slot indices to add and subtract. 
    // Slots are bounded by max_D_ <= K, which fits perfectly in uint16_t.
    std::vector<uint16_t> pos_;     
    std::vector<uint16_t> neg_;
    struct MergeOp {
        uint16_t dst;
        uint16_t src;
    };
    std::vector<MergeOp> merges_;

    
    // =============================================
    // Scratchpad used to do computation (inference)
    // =============================================
    std::vector<float> buf_;        // use just one scratch pad for all blocks: of size max_D_

public:
    std::string name() const override { return "spr.v2.b" + std::to_string(B); }
    bool uses_k() const override { return true; }
    // 3^16 fits in uint32; K <= 65535 because rows / segment counts / slots are uint16
    bool supports(int, int K, int, const Params& p) const override { return p.k >= 1 && p.k <= 16 && K <= 65535; }

    // force harness to transpose X and Y
    Layout layout() const override { return Layout::ColMajor; }

    // ------------------------------------------------------------------ preprocessing (offline)
    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        K_ = K; N_ = N;
        k_ = p.k;
        max_D_ = 0;

        // clear the vectors
        blk_num_seg_.clear(); seg_len_.clear(); rows_.clear();
        lvl_meta_.clear();
        pos_.clear(); neg_.clear(); merges_.clear();
        buf_.clear();

        for (int c0 = 0; c0 < N; c0 += k_) {
            int width = std::min(k_, N - c0);

            // generate base 3 keys
            struct keyed_t {
                uint32_t key; 
                uint16_t row; // acts as row or slot

                bool operator<(const keyed_t& other) const {
                    return key < other.key;
                }
            };
            std::vector<keyed_t>keyed; //(key, row)
            for (int r = 0; r < K; ++r) {
                int c_bound = c0 + width;
                uint32_t key = 0;
                uint32_t pow3 = 1;
                for (int c = c0; c < c_bound; ++c) {
                    const int8_t w = W[r * N + c];
                    key += (w == 1 ? 1u : (w == -1 ? 2u : 0u)) * pow3;
                    pow3 *= 3;
                }
                if (key) keyed.push_back({key, (uint16_t)r});
            }

            // sort and group into segments
            std::sort(keyed.begin(), keyed.end());
            std::vector<keyed_t> live; //(key, slot)
            uint16_t num_segs = 0; // also means slot number
            uint16_t current_seg_len = 0;
            for (size_t t = 0; t < keyed.size(); ++t) {
                if (t == 0 || keyed[t].key != keyed[t - 1].key) {
                    if (t > 0) seg_len_.push_back(current_seg_len);
                    current_seg_len = 0;
                    live.push_back({keyed[t].key, num_segs});
                    num_segs += 1;
                } 
                rows_.push_back(keyed[t].row);
                current_seg_len += 1;
            }
            if (!keyed.empty()) {
                seg_len_.push_back(current_seg_len);
            }

            blk_num_seg_.push_back(num_segs);
            max_D_ = std::max(max_D_, (size_t) num_segs);

            // precompute instructions of the tree
            for (int l = 0; l < width; ++l) {
                std::vector<keyed_t> live_next;
                uint16_t p_len = 0, n_len = 0, m_len = 0;
                for (auto [key, slot]: live) {
                    const int node_curr = key % 3;
                    const uint32_t node_parent = (uint32_t) key / 3;

                    // track pos and negs
                    if (node_curr == 1) {
                        p_len += 1;
                        pos_.push_back(slot);
                    } else if (node_curr == 2) {
                        n_len += 1;
                        neg_.push_back(slot);
                    } 

                    if (!node_parent) continue;
                    // compute merges
                    if (!live_next.empty() && live_next.back().key == node_parent) {
                        merges_.push_back({live_next.back().row, slot});
                        ++m_len;
                    } else {
                        live_next.push_back({node_parent, slot});
                    }
                }
                lvl_meta_.push_back({p_len, n_len, m_len});
                live = std::move(live_next);
            }

        }

        buf_.assign(max_D_ * B, 0.f);
    }

    // ------------------------------------------------------------------ inference (TIMED)
    // do matrix matrix mult over the row input tile from [m, m + mt)
    template <bool FastPath> 
    void tile(IO& io, int m, int mt) {
        const int n = FastPath ? B : mt; // fast path n in static and compiler will be able 
                                         // hold t in register for low B values and even vectorize

        float *buf = buf_.data();
        int num_blocks = blk_num_seg_.size();

        const uint16_t *p_blk_num_seg   = blk_num_seg_.data();
        const uint16_t *p_seg_len       = seg_len_.data();
        const uint16_t *p_rows          = rows_.data();
        const LvlMetadata *p_lvl_meta   = lvl_meta_.data();
        const uint16_t *p_pos           = pos_.data();
        const uint16_t *p_neg           = neg_.data();
        const MergeOp *p_merges         = merges_.data();

        for (int b = 0; b < num_blocks; ++b) {
            int c0 = b * k_;
            int width = std::min(k_, N_ - c0);

            // Segmented Sum
            int num_seg = (int) *(p_blk_num_seg++);
            for (int i = 0; i < num_seg; ++i) {
                float t[B] = {0.f}; // do computation in registers for low B
                int len = (int) *(p_seg_len++);
                for (int j = 0; j < len; ++j) {
                    uint16_t row_idx = *(p_rows++);
                    const float *x_blocked_col = io.X + io.M * (size_t) row_idx + (size_t) m;
                    for (int ib = 0; ib < n; ++ib) t[ib] += x_blocked_col[ib];
                }

                // register to mem
                for (int ib = 0; ib < n; ++ib) buf[i * n + ib] = t[ib];
            }

            // compute the result using the Dynamic Tree Merges
            for (int l = 0; l < width; ++l) {
                float t[B] = {0.f};
                auto [p_len, n_len, m_len] = *(p_lvl_meta++);
                for (int i = 0; i < p_len; ++i) {
                    uint16_t slot = *(p_pos++);
                    for (int ib = 0; ib < n; ++ib) t[ib] += buf[slot * n + ib];
                }
                for (int i = 0; i < n_len; ++i) {
                    uint16_t slot = *(p_neg++);
                    for (int ib = 0; ib < n; ++ib) t[ib] -= buf[slot * n + ib];
                }
                float *y = io.Y + (size_t) (c0 + l) * io.M + m;
                for (int ib = 0; ib < n; ++ib) y[ib] = t[ib];

                // merge
                for (int i = 0; i < m_len; ++i) {
                    auto [dst, src] = *(p_merges++);
                    for (int ib = 0; ib < n; ++ib) buf[dst * n + ib] += buf[src * n + ib];
                }
            }
        }
    }

    void run(IO& io) override {
        const int m_main_end = io.M - (io.M % B);
        for (int m = 0; m < m_main_end; m += B) tile<true>(io, m, B);
        if (m_main_end < io.M) tile<false>(io, m_main_end, io.M - m_main_end);
    }

    // ------------------------------------------------------------------ accounting
    // Bytes run() reads: every array of the compact format (uint16 indices, 6-byte level records, 4-byte merges).
    size_t bytes() const override {
        return blk_num_seg_.size() * sizeof(uint16_t) + seg_len_.size() * sizeof(uint16_t) +
               rows_.size() * sizeof(uint16_t) + lvl_meta_.size() * sizeof(LvlMetadata) +
               pos_.size() * sizeof(uint16_t) + neg_.size() * sizeof(uint16_t) + merges_.size() * sizeof(MergeOp);
    }

    // Additions per row of X: segmented sums (one per non-zero row) + output terms + merges.
    double adds_per_row() const override {
        return (double)rows_.size() + pos_.size() + neg_.size() + merges_.size();
    }
};

SPR_REGISTER_KERNEL("spr.v2.b8", SPRV2<8>);
SPR_REGISTER_KERNEL("spr.v2.b16", SPRV2<16>);
SPR_REGISTER_KERNEL("spr.v2.b32", SPRV2<32>);
SPR_REGISTER_KERNEL("spr.v2.b64", SPRV2<64>);

}  // namespace
}  // namespace spr


