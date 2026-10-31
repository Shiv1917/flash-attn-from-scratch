#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define check_cuda(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Times a region of GPU work with CUDA events.
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        check_cuda(cudaEventCreate(&start_));
        check_cuda(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { check_cuda(cudaEventRecord(start_)); }
    // Returns elapsed milliseconds.
    float stop() {
        check_cuda(cudaEventRecord(stop_));
        check_cuda(cudaEventSynchronize(stop_));
        float ms = 0.f;
        check_cuda(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

inline void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (auto& x : v) x = dist(gen);
}

inline void dump_binary(const std::string& path, const std::vector<float>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s for writing\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}
