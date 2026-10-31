// Register-tiled GEMM.
// A block computes a BM x BN tile of C with (BM/TM)*(BN/TN) threads; each thread owns a
// TM x TN patch kept in registers. Per k step a thread loads TM values of A and TN of B
// from shared memory and does TM*TN FMAs, raising arithmetic intensity at the
// shared-memory level from 1 FMA per 2 loads (step 2) to TM*TN per (TM+TN) loads.
// A is stored transposed in shared memory so the TM-wide fragment reads are contiguous.
#include "../common/gemm_harness.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NTHREADS ((BM / TM) * (BN / TN))  // 256

__global__ void __launch_bounds__(NTHREADS)
gemm_regtiled(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[BK][BM];  // transposed: As[k][m]
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int thread_row = tid / (BN / TN);  // 0..15
    const int thread_col = tid % (BN / TN);  // 0..15

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Cooperative load mapping. A tile is BM x BK (128x8 = 1024 floats, 4 per thread),
    // B tile is BK x BN (8x128 = 1024 floats, 4 per thread).
    const int a_row = tid / BK, a_col = tid % BK;     // 32 rows x 8 cols per pass
    const int a_stride = NTHREADS / BK;               // 32
    const int b_row = tid / BN, b_col = tid % BN;     // 2 rows x 128 cols per pass
    const int b_stride = NTHREADS / BN;               // 2

    float acc[TM][TN] = {};
    float a_frag[TM], b_frag[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = 0; i < BM; i += a_stride) {
            int gr = block_row + a_row + i, gc = k0 + a_col;
            As[a_col][a_row + i] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.f;
        }
        for (int i = 0; i < BK; i += b_stride) {
            int gr = k0 + b_row + i, gc = block_col + b_col;
            Bs[b_row + i][b_col] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int i = 0; i < TM; i++) a_frag[i] = As[k][thread_row * TM + i];
#pragma unroll
            for (int j = 0; j < TN; j++) b_frag[j] = Bs[k][thread_col * TN + j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int r = block_row + thread_row * TM + i;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            int c = block_col + thread_col * TN + j;
            if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
        }
    }
}

static void launch(int M, int N, int K, const float* A, const float* B, float* C) {
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    gemm_regtiled<<<grid, NTHREADS>>>(M, N, K, A, B, C);
}

int main(int argc, char** argv) { return gemm_main(argc, argv, "03_register_tiled", launch); }
