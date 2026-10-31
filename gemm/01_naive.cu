// Naive GEMM: one thread per output element, every operand read from global memory.
#include "../common/gemm_harness.cuh"

// x indexes rows of C (threadIdx.x varies fastest), so a warp reads a column of A
// and writes a column of C: strided, uncoalesced. That is deliberate for the baseline;
// step 3 fixes it.
__global__ void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

static void launch(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(M, block.x), CEIL_DIV(N, block.y));
    gemm_naive<<<grid, block>>>(M, N, K, A, B, C);
}

int main(int argc, char** argv) { return gemm_main(argc, argv, "01_naive", launch); }
