// spr.v6

#include <algorithm>
#include <memory>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "spr/cache.hpp"
#include "spr/kernel.hpp"

namespace spr {
namespace {
template <int B>
class SPRV6 final : public Kernel {
// V5's cache-blocked computation with one OpenMP region for inference.
// collapse(2) distributes (input tile, output group) pairs across threads;
// each thread allocates its scratch buffer once and reuses it across tasks.

private:
    // =============================================
    // Matrix Shape and State Bounds 
    // =============================================
    int K_ = 0, N_ = 0;             // W shape
    int k_ = 0;                     // block width
    size_t max_D_ = 0;              // track max nbr of unique row patterns excluding zero
                                    // which we need to determine scratchpad size
    size_t total_segs_ = 0;

    // =============================================
    // Tiling Params
    // =============================================
    int req_kc_ = 0, req_g_ = 0;    // 0 = automatic tuning
    int Kc_ = 0, G_ = 0;            // Tiling params: X col block size and nbr of weight blocks
    int Nc_ = 0;                    // number of col blocks over X

    // Model used by the automatic choice of Kc_ and G_. No machine numbers here: the cache
    // size comes from cache_info() (detected at runtime, or SPR_L1_BYTES / SPR_L2_BYTES /
    // SPR_LLC_BYTES), everything else is a share of it.
    struct TileModel {
        CacheLevel level = CacheLevel::L1;          // cache the X chunk + G bufs must fit in
        double budget_frac = 0.75;                  // share of that cache we allow ourselves
                                                    // (rest: metadata stream, Y, stack)
        std::vector<double> chunk_fracs{0.5, 0.25}; // X chunk sizes to try, largest first, as a share of the cache
        int min_piece_len = 8;                      // avg rows per piece; shorter: per-piece overhead beats the cache win
        int max_group = 64;                         // upper bound on G
    };
    TileModel model_;
    size_t cache_bytes_ = 0;        // size of model_.level on this machine (set in prepare)

    
    // =============================================
    // Segmented Sums
    // =============================================
    
    // Given that we introduced this 3D blocking to fit in the caches, we must
    // change the preprocessing to be done by input 2D block
    struct Piece {
        uint16_t slot; 
        uint16_t len;
    };
    std::vector<Piece> pieces_;      // pieces themselves
    std::vector<uint16_t> pieces_cnt;   // nbr of pieces per (group, chunk, block), in run order
    std::vector<uint16_t> blk_num_seg_; // nbr of segments (= slots) per block, used by the tree

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
    // Starting positions in each instruction stream, in elements (not bytes).
    // =============================================
    struct GroupOffsets {
        size_t pieces_cnt, pieces, rows;
        size_t lvl_meta, pos, neg, merges;
    };
    std::vector<GroupOffsets> group_offsets_;

    template <class T>
    static const T* group_cursor(const std::vector<T>& stream, size_t offset) {
        // Empty streams occur for all-zero blocks; avoid arithmetic on nullptr.
        return offset ? stream.data() + offset : stream.data();
    }

public:
    // kc / g = 0: chosen automatically in prepare() for the cache `level`
    explicit SPRV6(int kc = 0, int g = 0, CacheLevel level = CacheLevel::L1) : req_kc_(kc), req_g_(g) {
        model_.level = level;
    }

    std::string name() const override { 
        std::string s = "spr.v6.b" + std::to_string(B);
        if (req_kc_) s += ".kc" + std::to_string(req_kc_);
        if (req_g_)  s += ".g" + std::to_string(req_g_);
        if (model_.level != CacheLevel::L1) s += std::string(".") + cache_level_name(model_.level);
        return s;
    }
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
        total_segs_ = 0;                // was never reset: grew on every prepare()

        // clear the vectors
        blk_num_seg_.clear(); pieces_cnt.clear(); pieces_.clear(); rows_.clear();
        lvl_meta_.clear();
        pos_.clear(); neg_.clear(); merges_.clear();
        group_offsets_.clear();

        struct SlotRow { uint16_t slot; uint16_t row; };
        std::vector<SlotRow> sr;          // All blocks concatenated
        std::vector<size_t> blk_off{0};   // Block boundaries in sr
        struct TreeOffsets { size_t levels, pos, neg, merges; };
        std::vector<TreeOffsets> tree_offsets;

        for (int c0 = 0; c0 < N; c0 += k_) {
            tree_offsets.push_back({lvl_meta_.size(), pos_.size(), neg_.size(), merges_.size()});
            int width = std::min(k_, N - c0);

            // generate base 3 keys
            struct keyed_t {
                uint32_t key; 
                uint16_t row; // acts as row or slot

                bool operator<(const keyed_t& other) const {
                    return key != other.key ? key < other.key : row < other.row;
                }
            };
            std::vector<keyed_t>keyed; //(key, row)
            for (int r = 0; r < K; ++r) {
                int c_bound = c0 + width;
                uint32_t key = 0;
                uint32_t pow3 = 1;
                for (int c = c0; c < c_bound; ++c) {
                    const int8_t w = W[(size_t)r * N + c];   // size_t: r * N overflows int for big K*N
                    key += (w == 1 ? 1u : (w == -1 ? 2u : 0u)) * pow3;
                    pow3 *= 3;
                }
                if (key) keyed.push_back({key, (uint16_t)r});
            }

            // sort and group into segments
            std::sort(keyed.begin(), keyed.end());
            std::vector<keyed_t> live; //(key, slot)
            uint16_t num_segs = 0; // also means slot number
            for (size_t t = 0; t < keyed.size(); ++t) {
                if (t == 0 || keyed[t].key != keyed[t - 1].key)
                    live.push_back({keyed[t].key, num_segs++});
                sr.push_back({(uint16_t) (num_segs - 1), keyed[t].row});
            }

            blk_off.push_back(sr.size());
            blk_num_seg_.push_back(num_segs);
            max_D_ = std::max(max_D_, (size_t) num_segs);
            total_segs_ += num_segs;

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

        // =============================================
        // The Cache Math: choose Kc_ and G_ for the target cache level
        // =============================================
        const int nb = (int) blk_num_seg_.size();
        const CacheInfo& cache = cache_info();
        cache_bytes_ = cache.size(model_.level);
        const size_t budget = (size_t) (model_.budget_frac * (double) cache_bytes_);

        // Footprint of one X row of the tile: B floats, but X is read in place (col-major,
        // rows M floats apart), so each row occupies whole cache lines
        const size_t line = cache.line;
        const size_t row_bytes = (B * sizeof(float) + line - 1) / line * line;
        const size_t buf_bytes = std::max<size_t>(max_D_, 1) * B * sizeof(float);   // one block's buf (contiguous)

        // nbr of pieces if K is cut into chunks of kc rows
        // (a piece = one (chunk, slot) pair that has at least one row in a block)
        auto count_pieces = [&](int kc) {
            size_t n = 0;
            std::vector<int> last(std::max<size_t>(max_D_, 1), -1);   // last chunk seen per slot
            for (int b = 0; b < nb; ++b) {
                std::fill(last.begin(), last.begin() + blk_num_seg_[b], -1);
                for (size_t e = blk_off[b]; e < blk_off[b + 1]; ++e) {
                    const int c = sr[e].row / kc;
                    if (last[sr[e].slot] != c) { last[sr[e].slot] = c; ++n; }   // rows of a slot are sorted
                }
            }
            return n;
        };
        // how many bufs fit in the budget next to an X chunk of kc rows
        auto group_for = [&](int kc) {
            const long left = (long) budget - (long) ((size_t) kc * row_bytes);
            return (int) std::clamp<long>(left / (long) buf_bytes, 1, model_.max_group);
        };

        Kc_ = K; G_ = 1;                // default: no tiling (= v2)
        if (req_kc_ > 0) {              // fixed config
            Kc_ = std::min(req_kc_, K);
            G_ = req_g_ > 0 ? req_g_ : group_for(Kc_);
        } else {                        // automatic
            // largest X chunk that leaves room for >= 2 bufs
            // and keeps the pieces long enough to amortize their overhead
            for (double frac : model_.chunk_fracs) {
                const int kc = (int) std::min<size_t>((size_t) (frac * (double) cache_bytes_) / row_bytes, (size_t) K);
                if (kc < 1) continue;
                if (kc >= K) break;     // the whole X tile already fits: nothing to gain
                const int g = group_for(kc);
                if (g < 2) continue;
                const size_t n_pieces = count_pieces(kc);
                if (n_pieces && sr.size() >= (size_t) model_.min_piece_len * n_pieces) { Kc_ = kc; G_ = g; break; }
            }
            if (req_g_ > 0) G_ = req_g_;
        }
        G_ = std::max(1, std::min(G_, std::max(nb, 1)));
        Nc_ = (K + Kc_ - 1) / Kc_;

        // =============================================
        // Rows of each block sorted by (chunk, slot, row)
        // =============================================
        for (int b = 0; b < nb; ++b)
            std::sort(sr.begin() + blk_off[b], sr.begin() + blk_off[b + 1], [&](const SlotRow& x, const SlotRow& y) {
                const int cx = x.row / Kc_, cy = y.row / Kc_;
                if (cx != cy) return cx < cy;
                return x.slot != y.slot ? x.slot < y.slot : x.row < y.row;
            });

        // =============================================
        // Emit the stream in run order: group -> chunk -> block -> piece -> rows
        //   pieces_cnt : one entry per (group, chunk, block)
        //   pieces_    : (slot, len) of each piece
        //   rows_      : the len rows of each piece, back to back
        // =============================================
        rows_.reserve(sr.size());
        std::vector<size_t> cur(blk_off.begin(), blk_off.end() - 1);   // read cursor per block
        for (int g0 = 0; g0 < nb; g0 += G_) {
            const auto& tree = tree_offsets[g0];
            group_offsets_.push_back({pieces_cnt.size(), pieces_.size(), rows_.size(),
                                      tree.levels, tree.pos, tree.neg, tree.merges});
            const int g1 = std::min(g0 + G_, nb);
            for (int c = 0; c < Nc_; ++c) {
                for (int b = g0; b < g1; ++b) {
                    const size_t cnt_at = pieces_cnt.size();
                    pieces_cnt.push_back(0);
                    size_t e = cur[b];
                    while (e < blk_off[b + 1] && sr[e].row / Kc_ == c) {   // rows of chunk c
                        const uint16_t slot = sr[e].slot;
                        uint16_t len = 0;
                        while (e < blk_off[b + 1] && sr[e].row / Kc_ == c && sr[e].slot == slot) {
                            rows_.push_back(sr[e].row);
                            ++len; ++e;
                        }
                        pieces_.push_back({slot, len});
                        ++pieces_cnt[cnt_at];
                    }
                    cur[b] = e;
                }
            }
        }

    }

    // ------------------------------------------------------------------ inference (TIMED)
    // One independent (input tile, output group) task. Metadata is read-only;
    // all intermediate sums live in the caller's thread-private scratch buffer.
    template <bool FastPath, bool Chunked>
    void process_group(IO& io, int m, int mt, int g, float* local_buf) const {
        const int n = FastPath ? B : mt;
        const size_t M = io.M;
        const float* __restrict X = io.X + m;
        float* __restrict Y = io.Y + m;
        const size_t buf_stride = std::max<size_t>(max_D_, 1) * B;
        const int g0 = g * G_;
        const int g1 = std::min(g0 + G_, (int)blk_num_seg_.size());

        const GroupOffsets& offsets = group_offsets_[g];
        const uint16_t *p_pieces_cnt = group_cursor(pieces_cnt, offsets.pieces_cnt);
        const Piece *p_pieces = group_cursor(pieces_, offsets.pieces);
        const uint16_t *p_rows = group_cursor(rows_, offsets.rows);
        const LvlMetadata *p_lvl_meta = group_cursor(lvl_meta_, offsets.lvl_meta);
        const uint16_t *p_pos = group_cursor(pos_, offsets.pos);
        const uint16_t *p_neg = group_cursor(neg_, offsets.neg);
        const MergeOp *p_merges = group_cursor(merges_, offsets.merges);

        // a slot gets pieces from several chunks: start from 0
        if constexpr (Chunked) {
            for (int b = g0; b < g1; ++b) {
                float *bb = local_buf + (size_t) (b - g0) * buf_stride;
                const size_t used = (size_t) blk_num_seg_[b] * B;
                for (size_t i = 0; i < used; ++i) bb[i] = 0.f;
            }
        }

        // Segmented Sum: chunk outer, so the Kc rows of X stay in L1 for all G blocks
        for (int c = 0; c < Nc_; ++c) {
            for (int b = g0; b < g1; ++b) {
                float *bb = local_buf + (size_t) (b - g0) * buf_stride;
                const int num_pieces = (int) *(p_pieces_cnt++);
                for (int i = 0; i < num_pieces; ++i) {
                    const auto [slot, len] = *(p_pieces++);
                    float t[B] = {0.f}; // do computation in registers for low B
                    for (int j = 0; j < len; ++j) {
                        const float *x_blocked_col = X + M * (size_t) *(p_rows++);
                        for (int ib = 0; ib < n; ++ib) t[ib] += x_blocked_col[ib];
                    }

                    // register to mem
                    float *d = bb + (size_t) slot * B;
                    if constexpr (Chunked) { for (int ib = 0; ib < n; ++ib) d[ib] += t[ib]; }
                    else                   { for (int ib = 0; ib < n; ++ib) d[ib] = t[ib]; }
                }
            }
        }

        // compute the result using the Dynamic Tree Merges (block order, as in v2)
        for (int b = g0; b < g1; ++b) {
            float *bb = local_buf + (size_t) (b - g0) * buf_stride;
            const int c0 = b * k_;
            const int width = std::min(k_, N_ - c0);
            for (int l = 0; l < width; ++l) {
                float t[B] = {0.f};
                auto [p_len, n_len, m_len] = *(p_lvl_meta++);
                for (int i = 0; i < p_len; ++i) {
                    const float *s = bb + (size_t) *(p_pos++) * B;
                    for (int ib = 0; ib < n; ++ib) t[ib] += s[ib];
                }
                for (int i = 0; i < n_len; ++i) {
                    const float *s = bb + (size_t) *(p_neg++) * B;
                    for (int ib = 0; ib < n; ++ib) t[ib] -= s[ib];
                }
                float *y = Y + (size_t) (c0 + l) * M;
                for (int ib = 0; ib < n; ++ib) y[ib] = t[ib];

                // merge
                for (int i = 0; i < m_len; ++i) {
                    auto [dst, src] = *(p_merges++);
                    float *d = bb + (size_t) dst * B;
                    const float *s = bb + (size_t) src * B;
                    for (int ib = 0; ib < n; ++ib) d[ib] += s[ib];
                }
            }
        }
    }

    template <bool Chunked>
    void run_all(IO& io) {
        const int num_main_tiles = io.M / B;
        const int num_tiles = num_main_tiles + (io.M % B != 0);
        const int num_groups = (int)group_offsets_.size();
        const size_t buf_stride = std::max<size_t>(max_D_, 1) * B;
        if (num_tiles == 0 || num_groups == 0) return;

#ifdef _OPENMP
        #pragma omp parallel if(num_tiles > 1 || num_groups > 1)
#endif
        {
            std::vector<float> local_buf((size_t)G_ * buf_stride);

#ifdef _OPENMP
            #pragma omp for collapse(2) schedule(static)
#endif
            for (int t = 0; t < num_tiles; ++t) {
                for (int g = 0; g < num_groups; ++g) {
                    const int m = t * B;
                    if (t < num_main_tiles)
                        process_group<true, Chunked>(io, m, B, g, local_buf.data());
                    else
                        process_group<false, Chunked>(io, m, io.M - m, g, local_buf.data());
                }
            }
        }
    }

    void run(IO& io) override {
        if (Nc_ > 1) run_all<true>(io);
        else         run_all<false>(io);
    }

    // ------------------------------------------------------------------ accounting
    // Bytes run() reads: segments per block, piece counts, pieces, rows, and the tree lists.
    size_t bytes() const override {
        return blk_num_seg_.size() * sizeof(uint16_t) + pieces_cnt.size() * sizeof(uint16_t) +
               pieces_.size() * sizeof(Piece) + rows_.size() * sizeof(uint16_t) +
               lvl_meta_.size() * sizeof(LvlMetadata) + pos_.size() * sizeof(uint16_t) +
               neg_.size() * sizeof(uint16_t) + merges_.size() * sizeof(MergeOp) +
               group_offsets_.size() * sizeof(GroupOffsets);
    }

    // Additions per row of X: one per non-zero row, one per extra piece of a slot (adding the
    // partial sum into buf), plus the tree (output terms + merges).
    double adds_per_row() const override {
        return (double)rows_.size() + (double)(pieces_.size() - total_segs_) + pos_.size() + neg_.size() +
               merges_.size();
    }
};

// automatic Kc and G, sized for L1
SPR_REGISTER_KERNEL("spr.v6.b16", SPRV6<16>);
SPR_REGISTER_KERNEL("spr.v6.b32", SPRV6<32>);

// automatic Kc and G, sized for another cache level (name() must match)
#define SPR_V6_LEVEL(B, LVL, NAME) \
    SPR_REGISTER_FACTORY("spr.v6.b" #B "." NAME, [] { return std::unique_ptr<::spr::Kernel>(new SPRV6<B>(0, 0, CacheLevel::LVL)); })
SPR_V6_LEVEL(32, L2, "l2");
#undef SPR_V6_LEVEL

// fixed configurations (name() must match): SPR==spr.v6.b32.kc512.g4 bench/spr_bench.sh
#define SPR_V6_FIXED(B, KC, G) \
    SPR_REGISTER_FACTORY("spr.v6.b" #B ".kc" #KC ".g" #G, [] { return std::unique_ptr<::spr::Kernel>(new SPRV6<B>(KC, G)); })
SPR_V6_FIXED(32, 512, 4);
SPR_V6_FIXED(32, 256, 8);
#undef SPR_V6_FIXED

}  // namespace
}  // namespace spr



