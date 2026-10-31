// Smoke test: toolchain, device visibility, kernel launch, event timing.
#include "../common/common.cuh"

__global__ void hello_kernel(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = i * 2;
}

int main() {
    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (sm_%d%d), %.1f GB, %d SMs\n", prop.name, prop.major,
           prop.minor, prop.totalGlobalMem / 1e9, prop.multiProcessorCount);

    const int n = 1024;
    int* d_out;
    check_cuda(cudaMalloc(&d_out, n * sizeof(int)));

    GpuTimer t;
    t.start();
    hello_kernel<<<n / 256, 256>>>(d_out);
    check_cuda(cudaGetLastError());
    float ms = t.stop();

    std::vector<int> h(n);
    check_cuda(cudaMemcpy(h.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        if (h[i] != i * 2) {
            fprintf(stderr, "mismatch at %d: %d\n", i, h[i]);
            return 1;
        }
    }
    printf("hello kernel OK (%.3f ms)\n", ms);
    check_cuda(cudaFree(d_out));
    return 0;
}
