// Coalesced + shared-memory tiled GEMM.
// Each block computes a TILE x TILE tile of C. For every step along K it stages a
// TILE x TILE tile of A and of B in shared memory (global loads are coalesced: consecutive
// threadIdx.x read consecutive addresses), syncs, accumulates, syncs, and moves on.
// Each global element is loaded once per block instead of once per thread: TILE-fold
// less global traffic.
#include "../common/gemm_harness.cuh"

#ifndef TILE
#define TILE 32
#endif

__global__ void gemm_tiled(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // threadIdx.x -> column, so a warp writes/reads contiguous row segments.
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // Out-of-range elements are zero-filled so partial tiles contribute nothing.
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

static void launch(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    gemm_tiled<<<grid, block>>>(M, N, K, A, B, C);
}

int main(int argc, char** argv) { return gemm_main(argc, argv, "02_tiled_smem", launch); }
