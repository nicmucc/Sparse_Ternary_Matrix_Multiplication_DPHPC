#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "spr/kernel.hpp"

namespace spr {
namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void cublas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
}

class DenseCublas final : public Kernel {
    cublasHandle_t handle_ = nullptr;
    float* d_w_ = nullptr;
    float* d_x_ = nullptr;
    float* d_y_ = nullptr;
    size_t weight_bytes_ = 0;

public:
    ~DenseCublas() override {
        cudaFree(d_y_);
        cudaFree(d_x_);
        cudaFree(d_w_);
        if (handle_) cublasDestroy(handle_);
    }

    std::string name() const override { return "lib.cublas.cuda"; }

    void prepare(const int8_t* W, int K, int N, const Params&) override {
        if (!handle_) cublas_check(cublasCreate(&handle_), "cublasCreate");
        cuda_check(cudaFree(d_w_), "cudaFree W");
        d_w_ = nullptr;
        const size_t count = size_t(K) * N;
        weight_bytes_ = count * sizeof(float);
        if (!count) return;
        const std::vector<float> weights(W, W + count);
        cuda_check(cudaMalloc(&d_w_, weight_bytes_), "cudaMalloc W");
        cuda_check(cudaMemcpy(d_w_, weights.data(), weight_bytes_, cudaMemcpyHostToDevice), "upload W");
    }

    void bind(const IO& io) override {
        if (!io.M || !io.N) return;
        cuda_check(cudaFree(d_x_), "cudaFree X");
        d_x_ = nullptr;
        cuda_check(cudaFree(d_y_), "cudaFree Y");
        d_y_ = nullptr;
        cuda_check(cudaMalloc(&d_y_, size_t(io.M) * io.N * sizeof(float)), "cudaMalloc Y");
        if (!io.K) return;
        const size_t x_bytes = size_t(io.M) * io.K * sizeof(float);
        cuda_check(cudaMalloc(&d_x_, x_bytes), "cudaMalloc X");
        cuda_check(cudaMemcpy(d_x_, io.X, x_bytes, cudaMemcpyHostToDevice), "upload X");
    }

    void run(IO& io) override {
        if (!io.M || !io.N) return;
        if (!io.K) {
            cuda_check(cudaMemset(d_y_, 0, size_t(io.M) * io.N * sizeof(float)), "zero Y");
        } else {
            const float alpha = 1.f, beta = 0.f;
            // cuBLAS sees the row-major buffers as column-major transposes: Y^T = W^T X^T.
            cublas_check(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, io.N, io.M, io.K,
                                     &alpha, d_w_, io.N, d_x_, io.K, &beta, d_y_, io.N), "cublasSgemm");
        }
        cuda_check(cudaDeviceSynchronize(), "execute cuBLAS baseline");
    }

    void finish(IO& io) override {
        if (!io.M || !io.N) return;
        cuda_check(cudaMemcpy(io.Y, d_y_, size_t(io.M) * io.N * sizeof(float), cudaMemcpyDeviceToHost),
                   "download Y");
    }

    size_t bytes() const override { return weight_bytes_; }
};
SPR_REGISTER_KERNEL("lib.cublas.cuda", DenseCublas);

}  // namespace
}  // namespace spr
