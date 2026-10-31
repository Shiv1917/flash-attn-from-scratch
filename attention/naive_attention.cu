// Naive attention: materializes the full N x N score matrix S in global memory.
//   S = Q K^T / sqrt(d);  P = softmax(S) (in place);  O = P V
// Three kernels, O(N^2) extra memory. Intermediate baseline for the flash kernel.
// The caller supplies the scratch buffer S (N*N floats) so its memory is visible to the
// caller's allocator accounting.
#include <cfloat>
#include <cmath>

#include "../common/common.cuh"

#define TILE 32

// S[i][j] = scale * dot(Q[i], K[j]) over d; entries with j > i are -inf when causal.
__global__ void scores_kernel(const float* Q, const float* K, float* S, int N, int d, float scale, int causal) {
    __shared__ float Qs[TILE][TILE];
    __shared__ float Ks[TILE][TILE + 1];  // +1: Ks[tx][k] is read down a column
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < d; k0 += TILE) {
        Qs[ty][tx] = (row < N && k0 + tx < d) ? Q[(size_t)row * d + k0 + tx] : 0.f;
        int kr = blockIdx.x * TILE + ty;  // K row this thread helps load
        Ks[ty][tx] = (kr < N && k0 + tx < d) ? K[(size_t)kr * d + k0 + tx] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += Qs[ty][k] * Ks[tx][k];
        __syncthreads();
    }
    if (row < N && col < N) S[(size_t)row * N + col] = (causal && col > row) ? -INFINITY : acc * scale;
}

// One block per row: numerically stable softmax in place.
__global__ void softmax_kernel(float* S, int N) {
    __shared__ float red[32];
    float* s = S + (size_t)blockIdx.x * N;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;

    float mx = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) mx = fmaxf(mx, s[j]);
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    mx = lane < nwarps ? red[lane] : -INFINITY;
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    __syncthreads();  // red[] is reused below

    float sum = 0.f;
    for (int j = tid; j < N; j += blockDim.x) {
        float e = expf(s[j] - mx);
        s[j] = e;
        sum += e;
    }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    sum = lane < nwarps ? red[lane] : 0.f;
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    float inv = 1.f / sum;
    for (int j = tid; j < N; j += blockDim.x) s[j] *= inv;
}

// O (N x d) = P (N x N) * V (N x d), shared-memory tiled.
__global__ void pv_kernel(const float* P, const float* V, float* O, int N, int d) {
    __shared__ float Ps[TILE][TILE];
    __shared__ float Vs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < N; k0 += TILE) {
        Ps[ty][tx] = (row < N && k0 + tx < N) ? P[(size_t)row * N + k0 + tx] : 0.f;
        Vs[ty][tx] = (k0 + ty < N && col < d) ? V[(size_t)(k0 + ty) * d + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += Ps[ty][k] * Vs[k][tx];
        __syncthreads();
    }
    if (row < N && col < d) O[(size_t)row * d + col] = acc;
}

// S: scratch of N*N floats. Launches on the default stream; caller synchronizes.
extern "C" int naive_attention(const float* Q, const float* K, const float* V, float* O, float* S, int N, int d,
                               int causal) {
    float scale = 1.f / sqrtf((float)d);
    dim3 block(TILE, TILE);
    scores_kernel<<<dim3(CEIL_DIV(N, TILE), CEIL_DIV(N, TILE)), block>>>(Q, K, S, N, d, scale, causal);
    softmax_kernel<<<N, 256>>>(S, N);
    pv_kernel<<<dim3(CEIL_DIV(d, TILE), CEIL_DIV(N, TILE)), block>>>(S, V, O, N, d);
    return (int)cudaGetLastError();
}
