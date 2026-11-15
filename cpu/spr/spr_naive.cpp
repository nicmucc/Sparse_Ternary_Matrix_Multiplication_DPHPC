// spr.naive -- the reference SPR-ternary kernel. Scalar C++, single thread, no intrinsics.
// This is the readable baseline that later versions (spr.v2, spr.avx2, ...) are optimized from.
//
//   Y (M x N) = X (M x K, fp32) * W (K x N, ternary {-1,0,+1})
//
// prepare() only builds the sparse pattern format for each block of k columns:
//   patterns[d]                  sorted, distinct, non-zero row patterns
//   rowids[offsets[d] .. offsets[d+1])   rows having pattern d
// Patterns are base-3 numbers, with digit(0)=0, digit(+1)=1, digit(-1)=2;
// the rightmost column is the least significant digit, as in the binary example.
//
// run() directly traverses the sparse tree for each activation row and block:
//   sums[d] = sum(x[rowids[t]], t = offsets[d] .. offsets[d+1]-1)
//   for column from right to left:
//       output = sum(sums[d] for low digit 1) - sum(sums[d] for low digit 2)
//       shift every pattern right by one digit (pattern /= 3)
//       discard zero patterns and merge adjacent equal patterns and their sums
// The tree decisions happen at inference; no reduction schedule is precomputed.
#include <algorithm>
#include <cstdint>
#include <utility>

#include "spr/kernel.hpp"

namespace spr {
namespace {

// One block of `width` columns starting at column c0.
struct Block {
    int c0 = 0, width = 0;
    std::vector<uint32_t> patterns;  // D sorted, distinct, non-zero patterns
    std::vector<int> offsets;        // D + 1 segment boundaries into rowids
    std::vector<int> rowids;         // only non-zero rows, grouped by pattern
};

class SPRNaive final : public Kernel {
public:
    std::string name() const override { return "spr.naive"; }
    bool uses_k() const override { return true; }
    bool supports(int, int, int, const Params& p) const override { return p.k >= 1 && p.k <= 16; }  // 3^16 fits in uint32

    // ------------------------------------------------------------------ sparse format (offline)
    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        K_ = K; N_ = N;
        blocks_.clear();
        size_t max_patterns = 0;
        for (int c0 = 0; c0 < N; c0 += p.k) {
            Block b;
            b.c0 = c0;
            b.width = std::min(p.k, N - c0);

            // 1. base-3 key of every row, keep only the non-zero ones
            std::vector<std::pair<uint32_t, int>> keyed;   // (pattern key, original row index)
            for (int r = 0; r < K; ++r) {
                uint32_t key = 0;
                for (int j = 0; j < b.width; ++j) {
                    const int8_t w = W[(size_t)r * N + c0 + j];
                    key = key * 3 + (w == 1 ? 1u : w == -1 ? 2u : 0u);
                }
                if (key) keyed.emplace_back(key, r);
            }

            // 2. sort by (key, row) and group equal keys into segments
            std::sort(keyed.begin(), keyed.end());
            for (size_t t = 0; t < keyed.size(); ++t) {
                if (t == 0 || keyed[t].first != keyed[t - 1].first) {
                    b.patterns.push_back(keyed[t].first);
                    b.offsets.push_back((int)t);
                }
                b.rowids.push_back(keyed[t].second);
            }
            b.offsets.push_back((int)keyed.size());  // also gives [0] for an empty block
            max_patterns = std::max(max_patterns, b.patterns.size());
            blocks_.push_back(std::move(b));
        }
        live_patterns_.resize(max_patterns);
        sums_.resize(max_patterns);
    }

    // ------------------------------------------------------------------ inference (TIMED)
    void run(IO& io) override {
        for (int m = 0; m < io.M; ++m) {
            const float* x = io.X + (size_t)m * K_;
            float* y = io.Y + (size_t)m * N_;
            for (const Block& b : blocks_) {
                // Work on a fresh copy: shifting must not change the stored format.
                std::copy(b.patterns.begin(), b.patterns.end(), live_patterns_.begin());
                size_t live = b.patterns.size();

                // segmented sums: one read of x per non-zero row of the block
                for (size_t d = 0; d < live; ++d) {
                    float s = 0.f;
                    for (int t = b.offsets[d]; t < b.offsets[d + 1]; ++t)
                        s += x[b.rowids[t]];
                    sums_[d] = s;
                }

                // Sparse tree: emit the low digit, then shift, discard and merge.
                for (int col = b.width - 1; col >= 0; --col) {
                    float acc = 0.f;
                    for (size_t d = 0; d < live; ++d) {
                        const uint32_t digit = live_patterns_[d] % 3;
                        if (digit == 1) acc += sums_[d];
                        else if (digit == 2) acc -= sums_[d];
                    }
                    y[b.c0 + col] = acc;

                    // Sorted patterns have adjacent equal parents. Compact in place;
                    // next <= d, so this never overwrites an unread node.
                    size_t next = 0;
                    for (size_t d = 0; d < live; ++d) {
                        const uint32_t parent = live_patterns_[d] / 3;
                        if (parent == 0) continue;
                        if (next > 0 && live_patterns_[next - 1] == parent) {
                            sums_[next - 1] += sums_[d];
                        } else {
                            live_patterns_[next] = parent;
                            sums_[next] = sums_[d];
                            ++next;
                        }
                    }
                    live = next;
                }
            }
        }
    }

    // ------------------------------------------------------------------ accounting
    // Compressed weights and metadata; mutable inference scratch is excluded.
    size_t bytes() const override {
        size_t n = 0;
        for (const Block& b : blocks_) {
            n += b.patterns.size() * sizeof(uint32_t);
            n += (b.rowids.size() + b.offsets.size() + 2) * sizeof(int);
        }
        return n;
    }

    // Count segmented sums, output additions/subtractions and merges on demand.
    // This simulates the topology for reporting only; no schedule is retained.
    double adds_per_row() const override {
        double a = 0;
        for (const Block& b : blocks_) {
            a += b.rowids.size();
            std::vector<uint32_t> patterns = b.patterns;
            size_t live = patterns.size();
            for (int col = b.width - 1; col >= 0; --col) {
                for (size_t d = 0; d < live; ++d)
                    if (patterns[d] % 3 != 0) ++a;
                size_t next = 0;
                for (size_t d = 0; d < live; ++d) {
                    const uint32_t parent = patterns[d] / 3;
                    if (parent == 0) continue;
                    if (next > 0 && patterns[next - 1] == parent) ++a;
                    else patterns[next++] = parent;
                }
                live = next;
            }
        }
        return a;
    }

private:
    int K_ = 0, N_ = 0;
    std::vector<Block> blocks_;
    std::vector<uint32_t> live_patterns_;  // mutable keys, reset for every row and block
    std::vector<float> sums_;              // one sum per live pattern
};

SPR_REGISTER_KERNEL("spr.naive", SPRNaive);

}  // namespace
}  // namespace spr
