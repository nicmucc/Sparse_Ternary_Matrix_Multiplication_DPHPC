#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(status));
        std::exit(1);
    }
}

int main() {
    int l2 = 0, sms = 0;
    check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0), "L2 size");
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "SM count");
    std::printf("L2 MiB: %.1f, SMs: %d\n", l2 / 1048576.0, sms);

    for (size_t bytes : {size_t(8) << 20, size_t(64) << 20, size_t(512) << 20}) {
        void* host = nullptr;
        void* device = nullptr;
        cudaEvent_t start, stop;
        check(cudaMallocHost(&host, bytes), "allocate pinned host buffer");
        std::memset(host, 0, bytes);
        check(cudaMalloc(&device, bytes), "allocate GPU buffer");
        check(cudaEventCreate(&start), "create start event");
        check(cudaEventCreate(&stop), "create stop event");
        check(cudaMemset(device, 0, bytes), "initialize GPU buffer");
        for (auto direction : {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost}) {
            std::vector<float> times;
            for (int run = 0; run < 7; ++run) {
                check(cudaEventRecord(start), "record start");
                check(cudaMemcpyAsync(direction == cudaMemcpyHostToDevice ? device : host,
                                      direction == cudaMemcpyHostToDevice ? host : device,
                                      bytes, direction), "copy");
                check(cudaEventRecord(stop), "record stop");
                check(cudaEventSynchronize(stop), "synchronize copy");
                float elapsed = 0;
                check(cudaEventElapsedTime(&elapsed, start, stop), "elapsed time");
                if (run >= 2) times.push_back(elapsed);
            }
            std::sort(times.begin(), times.end());
            const float median_ms = times[times.size() / 2];
            std::printf("%s,%.0f,%.4f,%.3f\n",
                        direction == cudaMemcpyHostToDevice ? "H2D" : "D2H",
                        bytes / 1048576.0, median_ms, bytes / (median_ms * 1e6));
        }
        check(cudaEventDestroy(start), "destroy start event");
        check(cudaEventDestroy(stop), "destroy stop event");
        check(cudaFree(device), "free GPU buffer");
        check(cudaFreeHost(host), "free pinned host buffer");
    }
}
