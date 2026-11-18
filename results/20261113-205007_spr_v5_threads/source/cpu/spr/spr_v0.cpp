// spr.v0 -- the first SPR-ternary kernel (formerly spr.naive). Scalar C++, single thread, no intrinsics.
// Unlike spr.naive (which decides the tree at inference), the reduction tree is compiled in prepare()
// into per-level POS / NEG / MERGE index lists. spr.v1 starts from this version.
//
//   Y (M x N) = X (M x K, fp32) * W (K x N, ternary {-1,0,+1})
//
// Algorithm (old-code-expermient/METHODS.md, section 5):
//   W is cut into blocks of k consecutive columns. For each block, done ONCE in prepare():
//     1. key[r] = sum_j digit(W[r][c0+j]) * 3^j     with digit(0)=0, digit(+1)=1, digit(-1)=2
//     2. drop rows with key 0, sort the rest by (key, row), group equal keys into D segments
//     3. build the sparse 3-ary reduction tree over the D occurring keys, lowest digit first,
//        and compile it into index lists per level:  POS, NEG (output terms) and MERGE (dst, src)
//   Inference, per row x of X and per block (run()):
//     buf[d] = sum of x[r] over the rows r of segment d                    (segmented sums)
//     for l in 0..k-1:
//         y[c0+l] = sum(buf[POS[l]]) - sum(buf[NEG[l]])                    (output terms)
//         for (dst, src) in MERGE[l]:  buf[dst] += buf[src]                (tree merges)
//   No data-dependent branch at inference: every loop bound is precomputed.
#include <algorithm>
#include <cstdint>
#include <utility>

#include "spr/kernel.hpp"

namespace spr {
namespace {

// The compiled program of one level of the tree.
struct Level {
    std::vector<int> pos;                      // slots added to output column c0 + l
    std::vector<int> neg;                      // slots subtracted from it
    std::vector<std::pair<int, int>> merge;    // buf[dst] += buf[src], after the output
};

// One block of `width` columns starting at column c0.
struct Block {
    int c0 = 0, width = 0;
    std::vector<int> rows;      // row ids of W grouped by key (segment d = rows[seg_end[d-1] .. seg_end[d]-1])
    std::vector<int> seg_end;   // D segment ends
    std::vector<Level> levels;  // `width` levels
};

class SPRV0 final : public Kernel {
public:
    std::string name() const override { return "spr.v0"; }
    bool uses_k() const override { return true; }
    bool supports(int, int, int, const Params& p) const override { return p.k >= 1 && p.k <= 16; }  // 3^16 fits in uint32

    // ------------------------------------------------------------------ preprocessing (offline)
    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        K_ = K; N_ = N;
        blocks_.clear();
        max_D_ = 0;
        // preparing each block
        for (int c0 = 0; c0 < N; c0 += p.k) {
            Block b;
            b.c0 = c0;
            b.width = std::min(p.k, N - c0);

            // base-3 key of every row, keep only the non-zero ones
            std::vector<std::pair<uint32_t, int>> keyed;   // (key, row)
            for (int r = 0; r < K; ++r) {
                uint32_t key = 0, pow3 = 1;
                for (int j = 0; j < b.width; ++j) {
                    const int8_t w = W[(size_t)r * N + c0 + j];
                    key += (w == 1 ? 1u : w == -1 ? 2u : 0u) * pow3;
                    pow3 *= 3;
                }
                if (key) keyed.emplace_back(key, r);
            }

            // 2. sort by (key, row) and group equal keys into segments
            std::sort(keyed.begin(), keyed.end());
            std::vector<std::pair<uint32_t, int>> live;    // (key, slot) of the tree nodes, sorted by key
            for (size_t t = 0; t < keyed.size(); ++t) {
                if (t == 0 || keyed[t].first != keyed[t - 1].first) { // start idx of a new pattern
                    if (t) b.seg_end.push_back((int)t);
                    live.emplace_back(keyed[t].first, (int)live.size());
                }
                b.rows.push_back(keyed[t].second);
            }
            if (!keyed.empty()) b.seg_end.push_back((int)keyed.size());
            max_D_ = std::max(max_D_, live.size());

            // 3. sparse tree, lowest digit first, compiled into index lists
            b.levels.resize(b.width);
            for (int l = 0; l < b.width; ++l) {
                Level& L = b.levels[l];
                for (auto& [key, slot] : live) {
                    if (key % 3 == 1) L.pos.push_back(slot);
                    if (key % 3 == 2) L.neg.push_back(slot);
                }
                // shift right by one digit; nodes with equal parents are adjacent -> merge
                std::vector<std::pair<uint32_t, int>> next;
                for (auto& [key, slot] : live) {
                    const uint32_t parent = key / 3;
                    if (parent == 0) continue;                                   // node dies
                    if (!next.empty() && next.back().first == parent)
                        L.merge.emplace_back(next.back().second, slot);          // merge into sibling
                    else
                        next.emplace_back(parent, slot);                         // relabel (free)
                }
                live = std::move(next);
            }
            blocks_.push_back(std::move(b));
        }
        buf_.assign(max_D_, 0.f);
    }

    // ------------------------------------------------------------------ inference (TIMED)
    void run(IO& io) override {
        float* buf = buf_.data();
        for (int m = 0; m < io.M; ++m) {
            const float* x = io.X + (size_t)m * K_;
            float* y = io.Y + (size_t)m * N_;
            for (const Block& b : blocks_) {
                // segmented sums: one read of x per non-zero row of the block
                int t = 0;
                for (size_t d = 0; d < b.seg_end.size(); ++d) {
                    float s = 0.f;
                    for (; t < b.seg_end[d]; ++t) s += x[b.rows[t]];
                    buf[d] = s;
                }
                // compiled sparse tree
                for (int l = 0; l < b.width; ++l) {
                    const Level& L = b.levels[l];
                    float acc = 0.f;
                    for (int i : L.pos) acc += buf[i];
                    for (int i : L.neg) acc -= buf[i];
                    y[b.c0 + l] = acc;
                    for (auto [dst, src] : L.merge) buf[dst] += buf[src];
                }
            }
        }
    }

    // ------------------------------------------------------------------ accounting
    // Format as stored here: every index is a 4-byte int (an obvious place to shrink: uint16).
    size_t bytes() const override {
        size_t n = 0;
        for (const Block& b : blocks_) {
            n += b.rows.size() + b.seg_end.size() + 2;
            for (const Level& L : b.levels) n += L.pos.size() + L.neg.size() + 2 * L.merge.size() + 3;
        }
        return n * sizeof(int);
    }

    // Additions per row of X: segmented sums + output terms + merges.
    double adds_per_row() const override {
        double a = 0;
        for (const Block& b : blocks_) {
            a += b.rows.size();
            for (const Level& L : b.levels) a += L.pos.size() + L.neg.size() + L.merge.size();
        }
        return a;
    }

private:
    int K_ = 0, N_ = 0;         // W shape
    size_t max_D_ = 0;          // track max nbr of unique row patterns
    std::vector<Block> blocks_; // actual column blocks
    std::vector<float> buf_;    // D tree nodes (one per occurring pattern)
                                // scratchpad during inference
};

SPR_REGISTER_KERNEL("spr.v0", SPRV0);

}  // namespace
}  // namespace spr
