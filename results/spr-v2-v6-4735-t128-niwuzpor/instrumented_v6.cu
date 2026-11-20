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

struct LvlMetadata {
    // instructions for one tree level
    uint16_t p_len; // number of positive buffer contributions
    uint16_t n_len; // number of negative buffer contributions
    uint16_t m_len; // number of merge operations
};

struct MergeOp { // op: buf[dst] += buf[src]
    // merging of two sums together
    uint16_t dst; // destination buffer slot
    uint16_t src; // source buffer slot
};

struct BlockMetadata {
    // we process different weight blocks simultaneously on the GPU
    // each thread must maintain independent cursors into read only metadata arrays
    // that we have precomputed in preprocess on the CPU
    // this is similar to what was done for the OpenMP code

    size_t seg; // start of the blocks segments lengths
    size_t row; // start of the blocks input row indices
    size_t level; // start of the blocks tree level records
    size_t pos; // start of positive slot instructions
    size_t neg; // start of negative slot instructions
    size_t merge; // start index of merge instructions
    int num_seg; // num of distinct non zero patterns
    int width; // num of weight columns in the block
};

// in v1 the access to x was X + m * K + r for each thread
// hence we had jumps of stride K between threads which leads to poor memory coalescing
// instead we can transpose the matrix (a bit like in the CPU version)
// and process the X + r*M + m so that each thread works on neighboring memory regions
__global__ void spr_v6_kernel(const float* X, float* Y, float* scratch,
                              int M, int K, int N, int k, size_t num_blocks,
                              size_t total_segments, const BlockMetadata* blocks,
                              const uint16_t* seg_len, const uint16_t* rows,
                              const LvlMetadata* levels, const uint16_t* pos,
                              const uint16_t* neg, const MergeOp* merges) {
    
    size_t m = blockDim.x * blockIdx.x + threadIdx.x;
    size_t b = blockIdx.y;
    if(m >= M) return;
    
    const BlockMetadata block = blocks[b];

    const float* x = X + m;
    float* y = Y + m * N + b * k;
    
    // buf for that thread
    float* buf = scratch + block.seg * M + m;
    
    // accumulate segmented sum
    size_t row = block.row;
    for(int i = 0; i < block.num_seg; i++) {
        float t1 = 0.0f, t2 = 0.0f;
        int len = seg_len[block.seg + i];
        for(int j = 0; j + 1 < len; j += 2) {
            t1 += x[rows[row++] * M];
            t2 += x[rows[row++] * M];
        }
        if (len % 2 != 0)
            t1 += x[rows[row++] * M];

        buf[i * M] = t1 + t2;
    }

    size_t p = block.pos, n = block.neg, merge = block.merge;
    for(int l = 0; l < block.width; l++) {
        LvlMetadata lvl = levels[block.level + l];

        // calculate the output for that level
        float t = 0.0f;
        for (int i = 0; i < lvl.p_len; ++i) t += buf[pos[p++] * M];
        for (int i = 0; i < lvl.n_len; ++i) t -= buf[neg[n++] * M];
        y[l] = t; 

        // compute the merges
        for(int i = 0; i < lvl.m_len; ++i) {
            const MergeOp op = merges[merge++];
            buf[op.dst * M] += buf[op.src * M];
        }
    }

}

class SPRV2CUDA final : public Kernel {
    int K_ = 0, N_ = 0, k_ = 0; // K_: number of input features, N_: number of output features, k_: number of weight columns per SPR block
    int bound_M_ = -1;

    // CPU metadata buffers
    std::vector<uint16_t> seg_len_, rows_, pos_, neg_;

    std::vector<LvlMetadata> lvl_meta_;
    std::vector<MergeOp> merges_;
    std::vector<BlockMetadata> blocks_;

    // GPU metadata buffers
    uint16_t* d_seg_len_ = nullptr;
    uint16_t* d_rows_ = nullptr;
    uint16_t* d_pos_ = nullptr;
    uint16_t* d_neg_ = nullptr;
    
    LvlMetadata* d_lvl_meta_ = nullptr;
    MergeOp* d_merges_ = nullptr;
    BlockMetadata* d_blocks_ = nullptr;

    // GPU input, output and scratch buffers
    float* d_x_ = nullptr;
    float* d_y_ = nullptr;
    float* d_scratch_ = nullptr;

    void check_io(const IO& io) const {
        if (io.M < 0 || io.K != K_ || io.N != N_)
            throw std::invalid_argument("spr.v6.cuda: IO shape differs from prepared weights");
    }

public:
    std::string name() const override { return "spr.v6.cuda"; }
    bool uses_k() const override { return true; }
    bool supports(int M, int K, int N, const Params& p) const override {
        return M >= 0 && K >= 0 && K <= 65535 && N >= 0 && p.k >= 1 && p.k <= 16;
    }

    ~SPRV2CUDA() override {
        free_metadata();

        cudaFree(d_x_);
        cudaFree(d_y_);
        cudaFree(d_scratch_);
    }

    SPRV2CUDA() = default;

    // prevent copying the owning pointers
    SPRV2CUDA(const SPRV2CUDA&) = delete;
    SPRV2CUDA& operator=(const SPRV2CUDA&) = delete;
    
    void prepare(const int8_t* W, int K, int N, const Params& p) override {
        if (!supports(0, K, N, p))
            throw std::invalid_argument("spr.v6.cuda requires 0 <= K <= 65535, N >= 0 and 1 <= k <= 16");

        K_ = K; N_ = N;
        k_ = p.k;
        bound_M_ = -1;
        free_metadata();

        seg_len_.clear(); rows_.clear();
        lvl_meta_.clear();
        pos_.clear(); neg_.clear(); merges_.clear();
        blocks_.clear();

        for(int c0 = 0; c0 < N; c0 += k_) {
            int width = std::min(k_, N - c0);

            // starting offsets for each weight block --> independent threads need the startinc cursor for each weight block
            blocks_.push_back({seg_len_.size(), rows_.size(), lvl_meta_.size(),
                               pos_.size(), neg_.size(), merges_.size(), 0, width});

            // generate base 3 keys
            struct keyed_t {
                uint32_t key;
                uint16_t row; // acts as a row or slot

                bool operator<(const keyed_t& other) const {
                    return key < other.key;
                }
            };

            std::vector<keyed_t> keyed; //(key, row)
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

            blocks_.back().num_seg = num_segs;


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

        if(!blocks_.empty()) { // Block metadata
            // alloc GPU array and transfer to device
            size_t bytes = blocks_.size() * sizeof(BlockMetadata);
            cuda_check(cudaMalloc((void**)&d_blocks_, bytes), "cudaMalloc blocks");
            cuda_check(cudaMemcpy(d_blocks_, blocks_.data(), bytes,
                          cudaMemcpyHostToDevice), "cudaMemcpy blocks");
        }

        if(!seg_len_.empty()) { // Segment lengths
            // alloc GPU array and transfer to device
            size_t bytes = seg_len_.size() * sizeof(uint16_t);
            cuda_check(cudaMalloc((void**)&d_seg_len_, bytes), "cudaMalloc seg_len");
            cuda_check(cudaMemcpy(d_seg_len_, seg_len_.data(), bytes,
                          cudaMemcpyHostToDevice), "cudaMemcpy seg_len");
        }

        if(!rows_.empty()) { // Row indices
            // alloc GPU array and transfer to device
            size_t bytes = rows_.size() * sizeof(uint16_t);
            cuda_check(cudaMalloc((void**)&d_rows_, bytes), "cudaMalloc rows");
            cuda_check(cudaMemcpy(d_rows_, rows_.data(), bytes,
                          cudaMemcpyHostToDevice), "cudaMemcpy rows");
        }

        if(!lvl_meta_.empty()) { // Tree level metadata
            // alloc GPU array and transfer to device
            size_t bytes = lvl_meta_.size() * sizeof(LvlMetadata);
            cuda_check(cudaMalloc((void**)&d_lvl_meta_, bytes), "cudaMalloc levels");
            cuda_check(cudaMemcpy(d_lvl_meta_, lvl_meta_.data(), bytes,
                          cudaMemcpyHostToDevice), "cudaMemcpy levels");
        }

        if(!pos_.empty()) { // Positive slot indices
            // alloc GPU array and transfer to device
            size_t bytes = pos_.size() * sizeof(uint16_t);
            cuda_check(cudaMalloc((void**)&d_pos_, bytes), "cudaMalloc pos");
            cuda_check(cudaMemcpy(d_pos_, pos_.data(), bytes,
                          cudaMemcpyHostToDevice), "cudaMemcpy pos");
        }

        if(!neg_.empty()) { // Negative slot indices
            // alloc GPU array and transfer to device
            size_t bytes = neg_.size() * sizeof(uint16_t);
            cuda_check(cudaMalloc((void**)&d_neg_, bytes), "cudaMalloc neg");
            cuda_check(cudaMemcpy(d_neg_, neg_.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy neg");
        }
        
        if(!merges_.empty()) { // Merge instructions
           // alloc GPU array and transfer to device
           size_t bytes = merges_.size() * sizeof(MergeOp);
           cuda_check(cudaMalloc((void**)&d_merges_, bytes), "cudaMalloc merges");
           cuda_check(cudaMemcpy(d_merges_, merges_.data(), bytes,
                          cudaMemcpyHostToDevice), "cudaMemcpy merges");
        }
    }

    void bind(const IO& io) override {
        check_io(io);

        const size_t M = static_cast<size_t>(io.M);
        const size_t K = static_cast<size_t>(K_);
        const size_t N = static_cast<size_t>(N_);

        const size_t total_segments = seg_len_.size();

        const size_t max_floats = std::numeric_limits<size_t>::max() / sizeof(float);
        if(M && (K > max_floats / M || N > max_floats / M || total_segments > max_floats / M)) {
            throw std::overflow_error("SPR CUDA allocation size overflow");
        }

        const size_t x_bytes = M * K * sizeof(float);
        const size_t y_bytes = M * N * sizeof(float);
        const size_t scratch_bytes = M * total_segments * sizeof(float);

        // free previously allocated buffers
        cuda_check(cudaFree(d_x_), "cudaFree old X");
        cuda_check(cudaFree(d_y_), "cudaFree old Y");
        cuda_check(cudaFree(d_scratch_), "cudaFree old scratch");

        d_x_ = nullptr;
        d_y_ = nullptr;
        d_scratch_ = nullptr;

        // alocate GPU memory
        if (x_bytes > 0) {
            cuda_check(cudaMalloc((void**)&d_x_, x_bytes), "cudaMalloc X");
        }
        
        if (y_bytes > 0)
            cuda_check(cudaMalloc((void**)&d_y_, y_bytes), "cudaMalloc Y");
        
        if (scratch_bytes > 0)
            cuda_check(cudaMalloc((void**)&d_scratch_, scratch_bytes), "cudaMalloc scratch");

        // copy the matrix from host to device
        if (x_bytes > 0) {
            std::vector<float> Xc(M * K);

            for (size_t r = 0; r < K; ++r) {
                for (size_t m = 0; m < M; ++m) {
                    Xc[r * M + m] = io.X[m * K + r]; // we need to transpose the input for better coalescing
                }
            }

            cuda_check(cudaMemcpy(d_x_, Xc.data(), x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy X");
        }

        bound_M_ = io.M; // input is sucessfully bound
    }

     void run(IO& io) override {
        check_io(io);

        if(!io.M || !N_) return;

        if(bound_M_ != io.M) {
            throw std::logic_error("spr.v6.cuda: call bind() before run()");
        }

        size_t num_threads = 128; // chose some value but we have to tune this to the device. Please watch respective cuda training series video

        // In v1 we have very bad thread access of the form:
        // Thread 0 --> input row 0, SPR block 0
        // Thread 1 --> input row 0, SPR block 1
        // Thread 2 --> input row 0, SPR block 2
        // since neighboring threads process different blocks this leads to warp divergence as well as poor memory access

        // What we want instead within one warp is something of the form
        // Thread 0 --> input row 0, SPR block b
        // Thread 1 --> input row 1, SPR block b
        // Thread 2 --> input row 2, SPR block b
        // and thus neighboring threads process the same exact block 
        // this is SIMILAR tot he SIMD approach we did for CPU

        dim3 grid(
            (io.M + num_threads - 1) / num_threads, 
            blocks_.size()
        );
        
        spr_bench::mark_start();
        spr_v6_kernel<<<grid, num_threads>>>(
            d_x_, d_y_, d_scratch_, io.M, K_, N_, k_,
            blocks_.size(), seg_len_.size(),
            d_blocks_, d_seg_len_, d_rows_, d_lvl_meta_,
            d_pos_, d_neg_, d_merges_);
        
        spr_bench::mark_stop();
        cuda_check(cudaGetLastError(), "launch spr_v6_kernel");
        // the harness uses wall clock timing so I think the run must wait for completion
        cuda_check(cudaDeviceSynchronize(), "execute spr_v6_kernel");
    }

    void finish(IO& io) override {
        check_io(io);
        if (!io.M || !N_) return;
        if (bound_M_ != io.M)
            throw std::logic_error("spr.v6.cuda: call bind() before finish()");
        
        // copy Y from device to host
        cuda_check(cudaMemcpy(io.Y, d_y_, io.M * N_ * sizeof(float),
                              cudaMemcpyDeviceToHost), "download Y");
    }

    size_t bytes() const override {
        return blocks_.size() * sizeof(BlockMetadata) + seg_len_.size() * sizeof(uint16_t) +
               rows_.size() * sizeof(uint16_t) + lvl_meta_.size() * sizeof(LvlMetadata) +
               pos_.size() * sizeof(uint16_t) + neg_.size() * sizeof(uint16_t) +
               merges_.size() * sizeof(MergeOp);
    }

    double adds_per_row() const override {
        return static_cast<double>(rows_.size()) + pos_.size() + neg_.size() + merges_.size();
    }

    void free_metadata() noexcept {
        cudaFree(d_seg_len_);
        cudaFree(d_rows_);
        cudaFree(d_pos_);
        cudaFree(d_neg_);
        cudaFree(d_lvl_meta_);
        cudaFree(d_merges_);
        cudaFree(d_blocks_);

        d_seg_len_ = nullptr;
        d_rows_ = nullptr;
        d_pos_ = nullptr;
        d_neg_ = nullptr;
        d_lvl_meta_ = nullptr;
        d_merges_ = nullptr;
        d_blocks_ = nullptr;
    }
};

SPR_REGISTER_KERNEL("spr.v6.cuda", SPRV2CUDA);
} // namespace
} // namespace spr
