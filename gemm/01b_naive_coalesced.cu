// Ablation between step 1 and step 2: still one thread per output element and no shared
// memory, but threadIdx.x now indexes the *column*, so a warp's loads of B and stores of C
// are contiguous (coalesced) and its load of A is a single broadcast address.
// Isolates the effect of memory coalescing from the effect of tiling.
#include "../common/gemm_harness.cuh"

__global__ void gemm_naive_coalesced(int M, int N, int K, const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

static void launch(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
    gemm_naive_coalesced<<<grid, block>>>(M, N, K, A, B, C);
}

int main(int argc, char** argv) { return gemm_main(argc, argv, "01b_naive_coalesced", launch); }
