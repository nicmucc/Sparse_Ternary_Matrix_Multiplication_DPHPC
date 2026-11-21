#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "spr/kernel.hpp"

namespace spr {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct LevelMetadata {
    uint16_t p_len, n_len, m_len;
};

struct MergeOp {
    uint16_t dst, src;
};

struct Leaf {
    uint16_t length;
    uint8_t digit;
    uint8_t flags;
};

constexpr uint8_t first_child = 1;
constexpr uint8_t store_parent = 2;

struct BlockMetadata {
    size_t row, leaf, parent, level, pos, neg, merge;
    int num_leaves, width;
};

__global__ void spr_v7_kernel(const float* X, float* Y, float* scratch, int M, int N, int k,
                              const BlockMetadata* blocks, const Leaf* leaves, const uint16_t* rows,
                              const LevelMetadata* levels, const uint16_t* pos, const uint16_t* neg,
                              const MergeOp* merges) {
    const size_t m = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= size_t(M))
        return;
    const size_t b = blockIdx.y;
    const BlockMetadata block = blocks[b];
    float* y = Y + m * N + b * k;
    size_t row = block.row;
    size_t slot = block.parent;
    float output = 0.0f;
    float parent = 0.0f;
    for (int i = 0; i < block.num_leaves; ++i) {
        const Leaf leaf = leaves[block.leaf + i];
        float sum = 0.0f;
        for (int j = 0; j < leaf.length; ++j)
            sum += X[size_t(rows[row++]) * M + m];

        if (leaf.digit == 1)
            output += sum;
        else if (leaf.digit == 2)
            output -= sum;

        parent = (leaf.flags & first_child) ? sum : parent + sum;
        if (leaf.flags & store_parent) {
            scratch[slot * M + m] = parent;
            ++slot;
        }
    }
    y[0] = output;

    size_t p = block.pos, n = block.neg, merge = block.merge;
    for (int l = 1; l < block.width; ++l) {
        const LevelMetadata level = levels[block.level + l - 1];
        float sum = 0.0f;
        for (int i = 0; i < level.p_len; ++i)
            sum += scratch[(block.parent + pos[p++]) * M + m];
        for (int i = 0; i < level.n_len; ++i)
            sum -= scratch[(block.parent + neg[n++]) * M + m];
        y[l] = sum;

        for (int i = 0; i < level.m_len; ++i) {
            const MergeOp op = merges[merge++];
            scratch[(block.parent + op.dst) * M + m] += scratch[(block.parent + op.src) * M + m];
        }
    }
}

template<class T>
void upload(T*& device, const std::vector<T>& host, const char* name) {
    if (host.empty())
        return;
    const size_t bytes = host.size() * sizeof(T);
    cuda_check(cudaMalloc(&device, bytes), name);
    cuda_check(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice), name);
}

class SPRV7CUDA final : public Kernel {
    int K_ = 0, N_ = 0, k_ = 0;
    int bound_M_ = -1;
    size_t parent_slots_ = 0, first_level_adds_ = 0;

    std::vector<Leaf> leaves_;
    std::vector<uint16_t> rows_;
    std::vector<uint16_t> pos_, neg_;
    std::vector<LevelMetadata> levels_;
    std::vector<MergeOp> merges_;
    std::vector<BlockMetadata> blocks_;

    Leaf* d_leaves_ = nullptr;
    uint16_t* d_rows_ = nullptr;
    uint16_t* d_pos_ = nullptr;
    uint16_t* d_neg_ = nullptr;
    LevelMetadata* d_levels_ = nullptr;
    MergeOp* d_merges_ = nullptr;
    BlockMetadata* d_blocks_ = nullptr;

    float* d_x_ = nullptr;
    float* d_y_ = nullptr;
    float* d_scratch_ = nullptr;

    void check_io(const IO& io) const {
        if (io.M < 0 || io.K != K_ || io.N != N_)
            throw std::invalid_argument("spr.v7.cuda: IO shape differs from prepared weights");
    }

    void free_metadata() noexcept {
        cudaFree(d_leaves_);
        cudaFree(d_rows_);
        cudaFree(d_pos_);
        cudaFree(d_neg_);
        cudaFree(d_levels_);
        cudaFree(d_merges_);
        cudaFree(d_blocks_);

        d_leaves_ = nullptr;
        d_rows_ = nullptr;
        d_pos_ = nullptr;
        d_neg_ = nullptr;
        d_levels_ = nullptr;
        d_merges_ = nullptr;
        d_blocks_ = nullptr;
    }

public:
    SPRV7CUDA() = default;
    SPRV7CUDA(const SPRV7CUDA&) = delete;
    SPRV7CUDA& operator=(const SPRV7CUDA&) = delete;

    ~SPRV7CUDA() override {
        free_metadata();
        cudaFree(d_x_);
        cudaFree(d_y_);
        cudaFree(d_scratch_);
    }

    std::string name() const override { return "spr.v7.cuda"; }
    bool uses_k() const override { return true; }

    bool supports(int M, int K, int N, const Params& p) const override {
        return M >= 0 && K >= 0 && K <= 65535 && p.k >= 1 && p.k <= 16 &&
               N >= 0 && N <= 65535 * p.k;
    }

    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        if (!supports(0, K, N, p))
            throw std::invalid_argument("spr.v7.cuda: unsupported weight shape or k");

        K_ = K;
        N_ = N;
        k_ = p.k;
        bound_M_ = -1;
        parent_slots_ = 0;
        first_level_adds_ = 0;
        free_metadata();
        leaves_.clear();
        rows_.clear();
        blocks_.clear();
        levels_.clear();
        pos_.clear();
        neg_.clear();
        merges_.clear();

        struct KeyedRow {
            uint32_t key;
            uint16_t row;
        };
        std::vector<KeyedRow> keyed;
        keyed.reserve(K);
        for (int c0 = 0; c0 < N; c0 += k_) {
            const int width = std::min(k_, N - c0);
            BlockMetadata block{rows_.size(), leaves_.size(), parent_slots_, levels_.size(),
                                pos_.size(), neg_.size(), merges_.size(), 0, width};
            keyed.clear();
            for (int r = 0; r < K; ++r) {
                uint32_t key = 0;
                for (int l = width - 1; l >= 0; --l) {
                    const int8_t w = W[size_t(r) * N + c0 + l];
                    key = 3 * key + (w == 1 ? 1u : w == -1 ? 2u : 0u);
                }
                if (key)
                    keyed.push_back({key, static_cast<uint16_t>(r)});
            }
            std::sort(keyed.begin(), keyed.end(), [](const KeyedRow& a, const KeyedRow& b) {
                return a.key < b.key;
            });

            std::vector<std::pair<int, uint16_t>> live;
            for (size_t i = 0; i < keyed.size();) {
                const size_t start = i;
                const uint32_t key = keyed[i].key;
                do {
                    rows_.push_back(keyed[i++].row);
                } while (i < keyed.size() && keyed[i].key == key);

                const uint32_t parent = key / 3;
                const uint8_t digit = static_cast<uint8_t>(key % 3);
                // Keys 1 and 2 contribute to output 0 only.
                const bool first = parent == 0 || start == 0 || keyed[start - 1].key / 3 != parent;
                const bool store = parent != 0 && (i == keyed.size() || keyed[i].key / 3 != parent);
                const uint8_t flags = (first ? first_child : 0) | (store ? store_parent : 0);
                leaves_.push_back({static_cast<uint16_t>(i - start), digit, flags});
                first_level_adds_ += (digit != 0) + !first;
                if (store)
                    live.emplace_back(parent, static_cast<uint16_t>(live.size()));
            }
            block.num_leaves = static_cast<int>(leaves_.size() - block.leaf);
            blocks_.push_back(block);
            parent_slots_ += live.size();

            for (int l = 1; l < width; ++l) {
                LevelMetadata level{};
                std::vector<std::pair<int, uint16_t>> parents;
                for (const auto& [key, slot] : live) {
                    const int digit = key % 3;
                    if (digit == 1) {
                        pos_.push_back(slot);
                        ++level.p_len;
                    } else if (digit == 2) {
                        neg_.push_back(slot);
                        ++level.n_len;
                    }

                    const int parent = key / 3;
                    if (parent == 0)
                        continue;
                    if (!parents.empty() && parents.back().first == parent) {
                        merges_.push_back({parents.back().second, slot});
                        ++level.m_len;
                    } else {
                        parents.emplace_back(parent, slot);
                    }
                }
                levels_.push_back(level);
                live = std::move(parents);
            }
        }

        upload(d_blocks_, blocks_, "upload blocks");
        upload(d_rows_, rows_, "upload rows");
        upload(d_leaves_, leaves_, "upload leaves");
        upload(d_levels_, levels_, "upload levels");
        upload(d_pos_, pos_, "upload positive slots");
        upload(d_neg_, neg_, "upload negative slots");
        upload(d_merges_, merges_, "upload merges");
    }

    void bind(const IO& io) override {
        check_io(io);
        const size_t M = io.M, K = K_, N = N_;
        const size_t max_floats = std::numeric_limits<size_t>::max() / sizeof(float);
        if (M && (K > max_floats / M || N > max_floats / M || parent_slots_ > max_floats / M))
            throw std::overflow_error("SPR CUDA allocation size overflow");

        const size_t x_bytes = M * K * sizeof(float);
        const size_t y_bytes = M * N * sizeof(float);
        bound_M_ = -1;
        cuda_check(cudaFree(d_x_), "cudaFree old X");
        d_x_ = nullptr;
        cuda_check(cudaFree(d_y_), "cudaFree old Y");
        d_y_ = nullptr;

        cuda_check(cudaFree(d_scratch_), "cudaFree old scratch");
        d_scratch_ = nullptr;
        if (M && parent_slots_)
            cuda_check(cudaMalloc(&d_scratch_, M * parent_slots_ * sizeof(float)), "cudaMalloc scratch");
        if (y_bytes > 0)
            cuda_check(cudaMalloc(&d_y_, y_bytes), "cudaMalloc Y");
        if (x_bytes > 0) {
            std::vector<float> transposed(M * K);
            for (size_t r = 0; r < K; ++r)
                for (size_t m = 0; m < M; ++m)
                    transposed[r * M + m] = io.X[m * K + r];

            cuda_check(cudaMalloc(&d_x_, x_bytes), "cudaMalloc X");
            cuda_check(cudaMemcpy(d_x_, transposed.data(), x_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy X");
        }
        bound_M_ = io.M;
    }

    void run(IO& io) override {
        check_io(io);
        if (!io.M || !N_)
            return;
        if (bound_M_ != io.M)
            throw std::logic_error("spr.v7.cuda: call bind() before run()");

        constexpr int threads = 128;
        const dim3 grid((size_t(io.M) + threads - 1) / threads, blocks_.size());
        spr_v7_kernel<<<grid, threads>>>(d_x_, d_y_, d_scratch_, io.M, N_, k_, d_blocks_, d_leaves_,
                                         d_rows_, d_levels_, d_pos_, d_neg_, d_merges_);
        cuda_check(cudaGetLastError(), "launch spr_v7_kernel");
        cuda_check(cudaDeviceSynchronize(), "execute spr_v7_kernel");
    }

    void finish(IO& io) override {
        check_io(io);
        if (!io.M || !N_)
            return;
        if (bound_M_ != io.M)
            throw std::logic_error("spr.v7.cuda: call bind() before finish()");

        cuda_check(cudaMemcpy(io.Y, d_y_, size_t(io.M) * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost), "download Y");
    }

    size_t bytes() const override {
        return leaves_.size() * sizeof(Leaf) + rows_.size() * sizeof(uint16_t) + blocks_.size() * sizeof(BlockMetadata) +
               levels_.size() * sizeof(LevelMetadata) + (pos_.size() + neg_.size()) * sizeof(uint16_t) +
               merges_.size() * sizeof(MergeOp);
    }

    double adds_per_row() const override {
        return static_cast<double>(rows_.size()) + first_level_adds_ + pos_.size() + neg_.size() + merges_.size();
    }
};

SPR_REGISTER_KERNEL("spr.v7.cuda", SPRV7CUDA);
} // namespace
} // namespace spr
