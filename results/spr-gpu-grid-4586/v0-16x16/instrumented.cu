#include <vector>
#include <utility>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cuda_runtime.h>

#include "spr/kernel.hpp"

// Macro to check CUDA error status
#define CUDA_CHECK(expr) do {                                        \
    cudaError_t result = (expr);                                     \
    if (result != cudaSuccess) {                                     \
        throw std::runtime_error(                                    \
            std::string(__FILE__) + ":" + std::to_string(__LINE__)   \
            + ": " + cudaGetErrorString(result));                    \
    }                                                                \
} while (0)


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

// Level flattened arrays offsets and sizes to use on device
struct DeviceLevel {
    int pos_offset, pos_count;
    int neg_offset, neg_count;
    int merge_offset, merge_count;
};

// Block flattened arrays offsets and sizes to move to device
struct DeviceBlock {
    int c0, width;
    int rows_offset;
    int segments_offset, segments_count;
    int levels_offset;
};

// SPR CUDA kernel
__global__ void spr_v0_kernel(
    const DeviceBlock* blocks,
    const DeviceLevel* levels,
    const int* rows,
    const int* segment_ends,
    const int* pos,
    const int* neg,
    const int* merges,
    const float* X,
    float* Y,
    float* scratch,
    int M, int K, int N,
    size_t num_blocks,
    size_t max_D)
{
    // Device computation
    
    // Block spans row: i, col: [j...j+blocks.size()-1]
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;      // j is the block id

    // Extract block information
    if (i < M && j < num_blocks) {
        const DeviceBlock& block = blocks[j];

        if (block.segments_count == 0) {
            for (int l = 0; l < block.width; ++l)
                Y[i * N + block.c0 + l] = 0.f;
            return;
        }

        float* buf = scratch + max_D * (i * num_blocks + j);

        const int* seg_end = segment_ends + block.segments_offset;
        const int* row = rows + block.rows_offset;

        const int segment_size = block.segments_count;
        const int width = block.width;

        int t = 0;
        for (size_t d = 0; d < segment_size; ++d) {
            float s = 0.f;
            for (; t < seg_end[d]; ++t) s += X[i * K + row[t]];
            buf[d] = s;
        }

        // compiled sparse tree
        for (int l = 0; l < width; ++l) {
            const DeviceLevel& L = levels[block.levels_offset + l]; 

            float acc = 0.f;
            for (int p = 0; p < L.pos_count; ++p) acc += buf[(pos + L.pos_offset)[p]];
            for (int p = 0; p < L.neg_count; ++p) acc -= buf[(neg + L.neg_offset)[p]];

            Y[i * N + block.c0 + l] = acc;
            for (int p = 0; p < L.merge_count; ++p) {
                int dst = (merges + L.merge_offset)[2 * p];
                int src = (merges + L.merge_offset)[2 * p + 1];

                buf[dst] += buf[src];
            }
        }
    }
}


class SPRV0CUDA final : public Kernel {
public:
    std::string name() const override { return "spr.v0.cuda"; }
    bool uses_k() const override { return true; }
    bool supports(int, int, int, const Params &p) const override {
        return p.k >= 1 && p.k <= 16;
    }

    // ------------------------------------------------------------------ preprocessing (CPU, offline)
    void prepare(const int8_t *W, int K, int N, const Params &p) override {
        free_weights();

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


        // Flatten array to prepare copy
        std::vector<DeviceBlock> flat_blocks;
        std::vector<DeviceLevel> flat_levels;
        std::vector<int> flat_rows;
        std::vector<int> flat_segment_ends;
        std::vector<int> flat_pos;
        std::vector<int> flat_neg;
        std::vector<int> flat_merges;

        for (const Block& b : blocks_) {
            DeviceBlock db{};
            db.c0 = b.c0;
            db.width = b.width;
            db.rows_offset = static_cast<int>(flat_rows.size());
            db.segments_offset = static_cast<int>(flat_segment_ends.size());
            db.segments_count = static_cast<int>(b.seg_end.size());
            db.levels_offset = static_cast<int>(flat_levels.size());

            flat_rows.insert(flat_rows.end(), b.rows.begin(), b.rows.end());
            flat_segment_ends.insert(flat_segment_ends.end(), b.seg_end.begin(), b.seg_end.end());

            for (const Level& l : b.levels) {
                DeviceLevel dl{};
                dl.pos_offset = static_cast<int>(flat_pos.size());
                dl.pos_count = static_cast<int>(l.pos.size());
                dl.neg_offset = static_cast<int>(flat_neg.size());
                dl.neg_count = static_cast<int>(l.neg.size());
                dl.merge_offset = static_cast<int>(flat_merges.size());
                dl.merge_count = static_cast<int>(l.merge.size());

                flat_pos.insert(flat_pos.end(), l.pos.begin(), l.pos.end());
                flat_neg.insert(flat_neg.end(), l.neg.begin(), l.neg.end());

                for (auto [dst, src] : l.merge) {
                    flat_merges.push_back(dst);
                    flat_merges.push_back(src);
                }

                flat_levels.push_back(dl);
            }

            flat_blocks.push_back(db);
        }

        // Send to device
        if (!flat_blocks.empty()) {
            size_t bytes = flat_blocks.size() * sizeof(DeviceBlock);
            CUDA_CHECK(cudaMalloc(&d_blocks, bytes));
            CUDA_CHECK(cudaMemcpy(d_blocks, flat_blocks.data(), bytes, cudaMemcpyHostToDevice));
        }

        if (!flat_levels.empty()) {
            size_t bytes = flat_levels.size() * sizeof(DeviceLevel);
            CUDA_CHECK(cudaMalloc(&d_levels, bytes));
            CUDA_CHECK(cudaMemcpy(d_levels, flat_levels.data(), bytes, cudaMemcpyHostToDevice));
        }

        if (!flat_rows.empty()) {
            size_t bytes = flat_rows.size() * sizeof(int);
            CUDA_CHECK(cudaMalloc(&d_rows, bytes));
            CUDA_CHECK(cudaMemcpy(d_rows, flat_rows.data(), bytes, cudaMemcpyHostToDevice));
        }

        if (!flat_segment_ends.empty()) {
            size_t bytes = flat_segment_ends.size() * sizeof(int);
            CUDA_CHECK(cudaMalloc(&d_segment_ends, bytes));
            CUDA_CHECK(cudaMemcpy(d_segment_ends, flat_segment_ends.data(), bytes, cudaMemcpyHostToDevice));
        }

        if (!flat_pos.empty()) {
            size_t bytes = flat_pos.size() * sizeof(int);
            CUDA_CHECK(cudaMalloc(&d_pos, bytes));
            CUDA_CHECK(cudaMemcpy(d_pos, flat_pos.data(), bytes, cudaMemcpyHostToDevice));
        }

        if (!flat_neg.empty()) {
            size_t bytes = flat_neg.size() * sizeof(int);
            CUDA_CHECK(cudaMalloc(&d_neg, bytes));
            CUDA_CHECK(cudaMemcpy(d_neg, flat_neg.data(), bytes, cudaMemcpyHostToDevice));
        }

        if (!flat_merges.empty()) {
            size_t bytes = flat_merges.size() * sizeof(int);
            CUDA_CHECK(cudaMalloc(&d_merges, bytes));
            CUDA_CHECK(cudaMemcpy(d_merges, flat_merges.data(), bytes, cudaMemcpyHostToDevice));
        }

    }

    void bind(const IO& io) override {
        free_buffers();

        size_t x_bytes = size_t(io.M) * K_ * sizeof(float);
        size_t y_bytes = size_t(io.M) * N_ * sizeof(float);
        size_t scratch_bytes = size_t(io.M) * blocks_.size() * max_D_ * sizeof(float);

        if (x_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&d_X, x_bytes));
            CUDA_CHECK(cudaMemcpy(d_X, io.X, x_bytes, cudaMemcpyHostToDevice));
        }

        if (y_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&d_Y, y_bytes));
        }

        if (scratch_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
        }
    }

    // ------------------------------------------------------------------ inference (GPU, TIMED)
    void run(IO& io) override {
        int num_blocks = blocks_.size();

        if (io.M == 0 || num_blocks == 0)
            return;

        int task_x = io.M;
        int task_y = num_blocks;
        
        // TODO: double check kernel launch parameters
        dim3 threads(16, 16);
        dim3 grid(
            (task_x + threads.x - 1) / threads.x,
            (task_y + threads.y - 1) / threads.y
        );

        spr_bench::mark_start();
        spr_v0_kernel<<<grid, threads>>>(
            d_blocks, d_levels,
            d_rows, d_segment_ends,
            d_pos, d_neg, d_merges,
            d_X, d_Y, d_scratch,
            io.M, K_, N_, num_blocks, max_D_);

        spr_bench::mark_stop();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------------ results (CPU)
    void finish(IO& io) override {
        size_t bytes = size_t(io.M) * N_ * sizeof(float);

        if (bytes > 0) {
            CUDA_CHECK(cudaMemcpy(io.Y, d_Y, bytes, cudaMemcpyDeviceToHost));
        }
    }
    
    // ------------------------------------------------------------------ accounting (CPU)
    size_t bytes() const override {
        size_t n = blocks_.size() * sizeof(DeviceBlock);
        for (const Block& b : blocks_) {
            n += (b.rows.size() + b.seg_end.size()) * sizeof(int);
            n += b.levels.size() * sizeof(DeviceLevel);
            for (const Level& l : b.levels)
                n += (l.pos.size() + l.neg.size() + 2 * l.merge.size()) * sizeof(int);
        }
        return n;
    }

    // Additions per row of X: segmented sums + output terms + merges.
    // TODO: double check
    double adds_per_row() const override {
        double a = 0;
        for (const Block &b : blocks_) {
            a += b.rows.size();
            for (const Level &L : b.levels)
                a += L.pos.size() + L.neg.size() + L.merge.size();
        }
        return a;
    }

    // Destructor
    ~SPRV0CUDA() override {
        free_buffers();
        free_weights();
    }

private:
    int K_ = 0, N_ = 0;         // W shape
    size_t max_D_ = 0;          // track max nbr of unique row patterns
    std::vector<Block> blocks_; // actual column blocks
    
    DeviceBlock* d_blocks = nullptr;
    DeviceLevel* d_levels = nullptr;
    int* d_rows = nullptr;
    int* d_segment_ends = nullptr;
    int* d_pos = nullptr;
    int* d_neg = nullptr;
    int* d_merges = nullptr;

    float* d_X = nullptr;
    float* d_Y = nullptr;
    float* d_scratch = nullptr;

    // Weight-freeing helper
    void free_weights() noexcept {
        cudaFree(d_blocks);
        cudaFree(d_levels);
        cudaFree(d_rows);
        cudaFree(d_segment_ends);
        cudaFree(d_pos);
        cudaFree(d_neg);
        cudaFree(d_merges);

        d_blocks = nullptr;
        d_levels = nullptr;
        d_rows = nullptr;
        d_segment_ends = nullptr;
        d_pos = nullptr;
        d_neg = nullptr;
        d_merges = nullptr;
    }

    void free_buffers() noexcept {
        cudaFree(d_X);
        cudaFree(d_Y);
        cudaFree(d_scratch);

        d_X = nullptr;
        d_Y = nullptr;
        d_scratch = nullptr;
    }
};

SPR_REGISTER_KERNEL("spr.v0.cuda", SPRV0CUDA);

} // namespace
} // namespace spr
