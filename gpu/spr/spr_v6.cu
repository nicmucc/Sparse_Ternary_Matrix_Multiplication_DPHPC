#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "spr/kernel.hpp"

namespace spr {
namespace {

constexpr int kWeightBlockWidth = 2;
constexpr int kPatternSlots = 8;
constexpr int kWarpSize = 32;
constexpr int kWeightBlocksPerWarp = 2;
constexpr int kRowsPerWarp = kWarpSize / kWeightBlocksPerWarp;
constexpr int kWarpsPerBlock = 4;
constexpr int kRowsPerBlock = kRowsPerWarp * kWarpsPerBlock;

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

struct BlockMetadata {
    size_t level, pos, neg, merge;
    int width;
};

__global__ void spr_v6_kernel(const float* X, float* Y, int M, int K, int N,
                              size_t num_blocks, const uint8_t* patterns,
                              const BlockMetadata* blocks, const LevelMetadata* levels,
                              const uint16_t* pos, const uint16_t* neg,
                              const MergeOp* merges) {
    __shared__ float sums[kWarpsPerBlock][kPatternSlots][kWarpSize];

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const size_t m = size_t(blockIdx.x) * kRowsPerBlock + warp * kRowsPerWarp + lane % kRowsPerWarp;
    const size_t b = size_t(blockIdx.y) * kWeightBlocksPerWarp + lane / kRowsPerWarp;
    if (m >= size_t(M) || b >= num_blocks)
        return;

    auto& warp_sums = sums[warp];
    for (int slot = 0; slot < kPatternSlots; ++slot)
        warp_sums[slot][lane] = 0.0f;

    // Both halves visit the same feature and read the same activation rows.
    for (int r = 0; r < K; ++r) {
        const uint8_t pattern = patterns[b * K + r];
        if (pattern != 0)
            warp_sums[pattern - 1][lane] += X[size_t(r) * M + m];
    }

    const BlockMetadata block = blocks[b];
    float* y = Y + m * N + b * kWeightBlockWidth;
    size_t p = block.pos, n = block.neg, merge = block.merge;
    for (int l = 0; l < block.width; ++l) {
        const LevelMetadata level = levels[block.level + l];

        float sum = 0.0f;
        for (int i = 0; i < level.p_len; ++i)
            sum += warp_sums[pos[p++]][lane];
        for (int i = 0; i < level.n_len; ++i)
            sum -= warp_sums[neg[n++]][lane];
        y[l] = sum;

        for (int i = 0; i < level.m_len; ++i) {
            const MergeOp op = merges[merge++];
            warp_sums[op.dst][lane] += warp_sums[op.src][lane];
        }
    }
}

class SPRV6CUDA final : public Kernel {
    int K_ = 0, N_ = 0;
    int bound_M_ = -1;
    size_t nonzero_patterns_ = 0;

    std::vector<uint8_t> patterns_;
    std::vector<uint16_t> pos_, neg_;
    std::vector<LevelMetadata> levels_;
    std::vector<MergeOp> merges_;
    std::vector<BlockMetadata> blocks_;

    uint8_t* d_patterns_ = nullptr;
    uint16_t* d_pos_ = nullptr;
    uint16_t* d_neg_ = nullptr;
    LevelMetadata* d_levels_ = nullptr;
    MergeOp* d_merges_ = nullptr;
    BlockMetadata* d_blocks_ = nullptr;

    float* d_x_ = nullptr;
    float* d_y_ = nullptr;

    void check_io(const IO& io) const {
        if (io.M < 0 || io.K != K_ || io.N != N_)
            throw std::invalid_argument("spr.v6.cuda: IO shape differs from prepared weights");
    }

    void free_metadata() noexcept {
        cudaFree(d_patterns_);
        cudaFree(d_pos_);
        cudaFree(d_neg_);
        cudaFree(d_levels_);
        cudaFree(d_merges_);
        cudaFree(d_blocks_);

        d_patterns_ = nullptr;
        d_pos_ = nullptr;
        d_neg_ = nullptr;
        d_levels_ = nullptr;
        d_merges_ = nullptr;
        d_blocks_ = nullptr;
    }

public:
    SPRV6CUDA() = default;
    SPRV6CUDA(const SPRV6CUDA&) = delete;
    SPRV6CUDA& operator=(const SPRV6CUDA&) = delete;

    ~SPRV6CUDA() override {
        free_metadata();
        cudaFree(d_x_);
        cudaFree(d_y_);
    }

    std::string name() const override { return "spr.v6.cuda"; }
    bool uses_k() const override { return true; }

    bool supports(int M, int K, int N, const Params& p) const override {
        constexpr int max_columns = 65535 * kWeightBlocksPerWarp * kWeightBlockWidth;
        return M >= 0 && K >= 0 && K <= 65535 && N >= 0 && N <= max_columns &&
               p.k == kWeightBlockWidth;
    }

    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        if (!supports(0, K, N, p))
            throw std::invalid_argument("spr.v6.cuda requires k=2 and a supported weight shape");

        K_ = K;
        N_ = N;
        bound_M_ = -1;
        nonzero_patterns_ = 0;
        free_metadata();

        const size_t num_blocks = (size_t(N) + kWeightBlockWidth - 1) / kWeightBlockWidth;
        patterns_.resize(num_blocks * K);
        pos_.clear();
        neg_.clear();
        levels_.clear();
        merges_.clear();
        blocks_.clear();

        for (size_t b = 0; b < num_blocks; ++b) {
            const int c0 = static_cast<int>(b * kWeightBlockWidth);
            const int width = std::min(kWeightBlockWidth, N - c0);
            blocks_.push_back({levels_.size(), pos_.size(), neg_.size(), merges_.size(), width});

            std::array<bool, kPatternSlots + 1> seen{};
            for (int r = 0; r < K; ++r) {
                int pattern = 0;
                int place = 1;
                for (int c = 0; c < width; ++c) {
                    const int8_t weight = W[size_t(r) * N + c0 + c];
                    pattern += (weight == 1 ? 1 : weight == -1 ? 2 : 0) * place;
                    place *= 3;
                }
                patterns_[b * K + r] = static_cast<uint8_t>(pattern);
                seen[pattern] = true;
                nonzero_patterns_ += pattern != 0;
            }

            std::vector<std::pair<int, uint16_t>> live;
            for (int pattern = 1; pattern <= kPatternSlots; ++pattern) {
                if (seen[pattern])
                    live.emplace_back(pattern, pattern - 1);
            }

            for (int l = 0; l < width; ++l) {
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

        if (!patterns_.empty()) {
            const size_t bytes = patterns_.size() * sizeof(uint8_t);
            cuda_check(cudaMalloc(&d_patterns_, bytes), "cudaMalloc patterns");
            cuda_check(cudaMemcpy(d_patterns_, patterns_.data(), bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy patterns");
        }
        if (!blocks_.empty()) {
            const size_t bytes = blocks_.size() * sizeof(BlockMetadata);
            cuda_check(cudaMalloc(&d_blocks_, bytes), "cudaMalloc blocks");
            cuda_check(cudaMemcpy(d_blocks_, blocks_.data(), bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy blocks");
        }
        if (!levels_.empty()) {
            const size_t bytes = levels_.size() * sizeof(LevelMetadata);
            cuda_check(cudaMalloc(&d_levels_, bytes), "cudaMalloc levels");
            cuda_check(cudaMemcpy(d_levels_, levels_.data(), bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy levels");
        }
        if (!pos_.empty()) {
            const size_t bytes = pos_.size() * sizeof(uint16_t);
            cuda_check(cudaMalloc(&d_pos_, bytes), "cudaMalloc pos");
            cuda_check(cudaMemcpy(d_pos_, pos_.data(), bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy pos");
        }
        if (!neg_.empty()) {
            const size_t bytes = neg_.size() * sizeof(uint16_t);
            cuda_check(cudaMalloc(&d_neg_, bytes), "cudaMalloc neg");
            cuda_check(cudaMemcpy(d_neg_, neg_.data(), bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy neg");
        }
        if (!merges_.empty()) {
            const size_t bytes = merges_.size() * sizeof(MergeOp);
            cuda_check(cudaMalloc(&d_merges_, bytes), "cudaMalloc merges");
            cuda_check(cudaMemcpy(d_merges_, merges_.data(), bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy merges");
        }
    }

    void bind(const IO& io) override {
        check_io(io);
        const size_t M = io.M, K = K_, N = N_;
        const size_t max_floats = std::numeric_limits<size_t>::max() / sizeof(float);
        if (M && (K > max_floats / M || N > max_floats / M))
            throw std::overflow_error("SPR CUDA allocation size overflow");

        const size_t x_bytes = M * K * sizeof(float);
        const size_t y_bytes = M * N * sizeof(float);
        bound_M_ = -1;
        cuda_check(cudaFree(d_x_), "cudaFree old X");
        d_x_ = nullptr;
        cuda_check(cudaFree(d_y_), "cudaFree old Y");
        d_y_ = nullptr;

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
            throw std::logic_error("spr.v6.cuda: call bind() before run()");

        const dim3 grid((size_t(io.M) + kRowsPerBlock - 1) / kRowsPerBlock,
                        (blocks_.size() + kWeightBlocksPerWarp - 1) / kWeightBlocksPerWarp);
        spr_v6_kernel<<<grid, kWarpsPerBlock * kWarpSize>>>(
            d_x_, d_y_, io.M, K_, N_, blocks_.size(), d_patterns_, d_blocks_,
            d_levels_, d_pos_, d_neg_, d_merges_);
        cuda_check(cudaGetLastError(), "launch spr_v6_kernel");
        cuda_check(cudaDeviceSynchronize(), "execute spr_v6_kernel");
    }

    void finish(IO& io) override {
        check_io(io);
        if (!io.M || !N_)
            return;
        if (bound_M_ != io.M)
            throw std::logic_error("spr.v6.cuda: call bind() before finish()");

        cuda_check(cudaMemcpy(io.Y, d_y_, size_t(io.M) * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost), "download Y");
    }

    size_t bytes() const override {
        return patterns_.size() * sizeof(uint8_t) + blocks_.size() * sizeof(BlockMetadata) +
               levels_.size() * sizeof(LevelMetadata) + (pos_.size() + neg_.size()) * sizeof(uint16_t) +
               merges_.size() * sizeof(MergeOp);
    }

    double adds_per_row() const override {
        return static_cast<double>(nonzero_patterns_) + pos_.size() + neg_.size() + merges_.size();
    }
};

SPR_REGISTER_KERNEL("spr.v6.cuda", SPRV6CUDA);
} // namespace
} // namespace spr
