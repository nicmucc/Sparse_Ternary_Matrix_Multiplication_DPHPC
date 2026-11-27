#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "spr/kernel.hpp"

namespace spr {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Preserve the smaller slot format through k=32. Wider masks hold bits 32..63.
template<class Mask>
struct Slot {
    uint16_t len;
    Mask pos_mask, neg_mask;
};


struct BlockMetadata {
    size_t row, slot;
    int num_slots, width;
};

template<int KW, class Mask>
__global__ void spr_v8_wide_kernel(const float* X, float* Y, int M,
                              const BlockMetadata* blocks, const Slot<Mask>* slots,
                              const uint16_t* rows) {
    const size_t m = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= size_t(M))
        return;

    const size_t b = blockIdx.y;
    const BlockMetadata block = blocks[b];
    float y[KW] = {};
    size_t row = block.row;

    
    for (int d = 0; d < block.num_slots; ++d) {
        const auto slot = slots[block.slot + d];
        float sum = 0.0f;
        for (int t = 0; t < slot.len; ++t)
            sum += X[size_t(rows[row++]) * M + m];

#pragma unroll
        for (int j = 0; j < KW; ++j) {
            if (slot.pos_mask & (Mask{1} << j)) y[j] += sum;
            if (slot.neg_mask & (Mask{1} << j)) y[j] -= sum;
        }
    }


#pragma unroll
    for (int j = 0; j < KW; ++j)
        if (j < block.width)
            Y[(b * KW + j) * size_t(M) + m] = y[j];
}

template<class T>
void upload(T*& device, const std::vector<T>& host, const char* name) {
    if (host.empty())
        return;
    const size_t bytes = host.size() * sizeof(T);
    cuda_check(cudaMalloc(&device, bytes), name);
    cuda_check(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice), name);
}

class SPRV8WideCUDA final : public Kernel {
    int K_ = 0, N_ = 0, k_ = 0;
    size_t output_adds_ = 0;

    size_t slot_bytes_ = 0;
    std::vector<uint16_t> rows_;
    std::vector<BlockMetadata> blocks_;

    // gpu copies
    void* d_slots_ = nullptr;
    uint16_t* d_rows_ = nullptr;
    BlockMetadata* d_blocks_ = nullptr;

    struct Binding {
        const float* host_x;
        float* host_y;
        int M;
        float* d_x;
        float* d_y;
    };
    std::vector<Binding> bindings_;

    void check_io(const IO& io) const {
        if (io.M < 0 || io.K != K_ || io.N != N_)
            throw std::invalid_argument("spr.v8.wide.cuda: IO shape differs from prepared weights");
    }

    void free_metadata() noexcept {
        cudaFree(d_slots_);
        cudaFree(d_rows_);
        cudaFree(d_blocks_);
        d_slots_ = nullptr;
        d_rows_ = nullptr;
        d_blocks_ = nullptr;
    }

    void free_bindings() noexcept {
        for (const Binding& binding : bindings_) {
            cudaFree(binding.d_x);
            cudaFree(binding.d_y);
        }
        bindings_.clear();
    }

    const Binding& binding_for(const IO& io) const {
        for (const Binding& binding : bindings_)
            if (binding.host_x == io.X && binding.host_y == io.Y && binding.M == io.M)
                return binding;
        throw std::logic_error("spr.v8.wide.cuda: call bind() before run() or finish()");
    }

    template<int KW>
    void launch(const Binding& binding) {
        constexpr int threads = 128;
        const dim3 grid((size_t(binding.M) + threads - 1) / threads, blocks_.size());
        using Mask = std::conditional_t<(KW <= 32), uint32_t, uint64_t>;
        spr_v8_wide_kernel<KW><<<grid, threads>>>(binding.d_x, binding.d_y, binding.M,
                                            d_blocks_, static_cast<const Slot<Mask>*>(d_slots_), d_rows_);
    }

public:
    SPRV8WideCUDA() = default;
    SPRV8WideCUDA(const SPRV8WideCUDA&) = delete;
    SPRV8WideCUDA& operator=(const SPRV8WideCUDA&) = delete;

    ~SPRV8WideCUDA() override {
        free_metadata();
        free_bindings();
    }

    std::string name() const override { return "spr.v8.wide.cuda"; }
    bool uses_k() const override { return true; }
    Layout layout() const override { return Layout::ColMajor; }

    bool supports(int M, int K, int N, const Params& p) const override {
        return M >= 0 && K >= 0 && K <= 65535 &&
               ((p.k >= 1 && p.k <= 16) || p.k == 24 || p.k == 32 || p.k == 48 || p.k == 64) &&
               N >= 0 && N <= 65535 * p.k;
    }

    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        if (!supports(0, K, N, p))
            throw std::invalid_argument("spr.v8.wide.cuda: unsupported weight shape or k");

        K_ = K;
        N_ = N;
        k_ = p.k;
        output_adds_ = 0;
        free_bindings();
        free_metadata();
        slot_bytes_ = 0;
        rows_.clear();
        blocks_.clear();

        if (k_ <= 32)
            prepare_slots<uint64_t, uint32_t>(W, K, N);
        else
            prepare_slots<unsigned __int128, uint64_t>(W, K, N);

        upload(d_blocks_, blocks_, "upload blocks");
        upload(d_rows_, rows_, "upload rows");
    }

private:
    // Base-3 needs 102 bits at width 64. Keys live only on the host.
    template<class Key, class Mask>
    void prepare_slots(const int8_t* W, int K, int N) {
        std::vector<Slot<Mask>> slots;
        struct KeyedRow {
            Key key;
            uint16_t row;
        };
        std::vector<KeyedRow> keyed;
        keyed.reserve(K);
        for (int c0 = 0; c0 < N; c0 += k_) {
            const int width = std::min(k_, N - c0);
            BlockMetadata block{rows_.size(), slots.size(), 0, width};
            keyed.clear();
            for (int r = 0; r < K; ++r) {
                Key key = 0;
                for (int j = width - 1; j >= 0; --j) {
                    const int8_t w = W[size_t(r) * N + c0 + j];
                    key = 3 * key + (w == 1 ? 1u : w == -1 ? 2u : 0u);
                }
                if (key)
                    keyed.push_back({key, static_cast<uint16_t>(r)});
            }
            std::sort(keyed.begin(), keyed.end(), [](const KeyedRow& a, const KeyedRow& b) {
                return a.key < b.key;
            });

            for (size_t i = 0; i < keyed.size();) {
                const size_t start = i;
                Key key = keyed[i].key;
                do {
                    rows_.push_back(keyed[i++].row);
                } while (i < keyed.size() && keyed[i].key == key);

                Slot<Mask> slot{static_cast<uint16_t>(i - start), 0, 0};
                for (int j = 0; j < width; ++j, key /= 3) {
                    const unsigned digit = key % 3;
                    if (digit == 1) slot.pos_mask |= Mask{1} << j;
                    if (digit == 2) slot.neg_mask |= Mask{1} << j;
                }
                output_adds_ += __builtin_popcountll(slot.pos_mask | slot.neg_mask);
                slots.push_back(slot);
            }
            block.num_slots = static_cast<int>(slots.size() - block.slot);
            blocks_.push_back(block);
        }

        slot_bytes_ = slots.size() * sizeof(Slot<Mask>);
        if (slot_bytes_) {
            cuda_check(cudaMalloc(&d_slots_, slot_bytes_), "allocate slots");
            cuda_check(cudaMemcpy(d_slots_, slots.data(), slot_bytes_, cudaMemcpyHostToDevice),
                       "upload slots");
        }
    }

public:
    void bind(const IO& io) override {
        check_io(io);
        const size_t M = io.M, K = K_, N = N_;
        const size_t max_floats = std::numeric_limits<size_t>::max() / sizeof(float);
        if (M && (K > max_floats / M || N > max_floats / M))
            throw std::overflow_error("SPR CUDA allocation size overflow");

        const size_t x_bytes = M * K * sizeof(float);
        const size_t y_bytes = M * N * sizeof(float);

        // The shared correctness suite interleaves several live Bound objects for
        // one prepared kernel. Keep each binding's device buffers until rebind or
        // the next prepare(), rather than retaining only the most recent shape.
        for (auto it = bindings_.begin(); it != bindings_.end(); ++it) {
            if (it->host_x == io.X && it->host_y == io.Y && it->M == io.M) {
                cuda_check(cudaFree(it->d_x), "cudaFree old X");
                cuda_check(cudaFree(it->d_y), "cudaFree old Y");
                bindings_.erase(it);
                break;
            }
        }

        Binding binding{io.X, io.Y, io.M, nullptr, nullptr};
        try {
            if (y_bytes > 0)
                cuda_check(cudaMalloc(&binding.d_y, y_bytes), "cudaMalloc Y");
            if (x_bytes > 0) {
                cuda_check(cudaMalloc(&binding.d_x, x_bytes), "cudaMalloc X");
                cuda_check(cudaMemcpy(binding.d_x, io.X, x_bytes, cudaMemcpyHostToDevice),
                           "cudaMemcpy X");
            }
            bindings_.push_back(binding);
        } catch (...) {
            cudaFree(binding.d_x);
            cudaFree(binding.d_y);
            throw;
        }
    }

    void run(IO& io) override {
        check_io(io);
        if (!io.M || !N_)
            return;
        const Binding& binding = binding_for(io);

        switch (k_) {
            case 1: launch<1>(binding); break;
            case 2: launch<2>(binding); break;
            case 3: launch<3>(binding); break;
            case 4: launch<4>(binding); break;
            case 5: launch<5>(binding); break;
            case 6: launch<6>(binding); break;
            case 7: launch<7>(binding); break;
            case 8: launch<8>(binding); break;
            case 9: launch<9>(binding); break;
            case 10: launch<10>(binding); break;
            case 11: launch<11>(binding); break;
            case 12: launch<12>(binding); break;
            case 13: launch<13>(binding); break;
            case 14: launch<14>(binding); break;
            case 15: launch<15>(binding); break;
            case 16: launch<16>(binding); break;
            case 24: launch<24>(binding); break;
            case 32: launch<32>(binding); break;
            case 48: launch<48>(binding); break;
            case 64: launch<64>(binding); break;
            default: throw std::logic_error("spr.v8.wide.cuda: invalid prepared k");
        }
        cuda_check(cudaGetLastError(), "launch spr_v8_wide_kernel");
        cuda_check(cudaDeviceSynchronize(), "execute spr_v8_wide_kernel");
    }

    void finish(IO& io) override {
        check_io(io);
        if (!io.M || !N_)
            return;
        const Binding& binding = binding_for(io);

        cuda_check(cudaMemcpy(io.Y, binding.d_y, size_t(io.M) * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost), "download Y");
    }

    size_t bytes() const override {
        return slot_bytes_ + rows_.size() * sizeof(uint16_t) +
               blocks_.size() * sizeof(BlockMetadata);
    }

    double adds_per_row() const override {
        return static_cast<double>(rows_.size()) + output_adds_;
    }
};

SPR_REGISTER_KERNEL("spr.v8.wide.cuda", SPRV8WideCUDA);
} // namespace
} // namespace spr
