// Flash attention forward (fp32, d = 64, single head), tiled online softmax.
//
// Block layout: a block owns BR query rows. Each row is handled by TPR cooperating
// threads that split the d = 64 head dimension between them (interleaved float4 chunks,
// so neighbouring threads read neighbouring 16 B of a shared-memory row). The partial
// dot products of the TPR threads are combined with warp shuffles, after which every
// thread of the row holds the full score and runs the identical online-softmax update,
// so no further communication is needed.
//
// Loop over K/V tiles of BC rows staged in shared memory:
//   s_j   = q . K_j                (scores for the tile, kept in registers)
//   m_new = max(m, max_j s_j);  p_j = exp(s_j - m_new)
//   l     = l * exp(m - m_new) + sum_j p_j
//   o     = o * exp(m - m_new) + sum_j p_j V_j
// and finally O = o / l. The N x N matrix never exists; memory is O(N d).
//
// Scores are pre-scaled by log2(e)/sqrt(d) (folded into q) so exp() becomes exp2f().
#include <cfloat>
#include <cmath>

#include "../common/common.cuh"

#define D 64
#define V4 (D / 4)  // float4 per row

template <int BR, int BC, int TPR, bool CAUSAL>
__global__ void __launch_bounds__(BR* TPR)
flash_fwd_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
                 float* __restrict__ O, int N, float scale_log2) {
    constexpr int NT = BR * TPR;
    constexpr int NV = V4 / TPR;  // float4 chunks of the head dim owned by each thread
    static_assert(V4 % TPR == 0 && 32 % TPR == 0, "TPR must divide 16 and 32");

    __shared__ float4 Ks[BC][V4];
    __shared__ float4 Vs[BC][V4];

    const int tid = threadIdx.x;
    const int r = tid / TPR, sub = tid % TPR;
    const int row0 = blockIdx.x * BR;
    const int row = row0 + r;

    const float4* Q4 = reinterpret_cast<const float4*>(Q);
    const float4* K4 = reinterpret_cast<const float4*>(K);
    const float4* V4p = reinterpret_cast<const float4*>(V);

    float4 q[NV], o[NV];
#pragma unroll
    for (int t = 0; t < NV; t++) {
        float4 v = row < N ? Q4[(size_t)row * V4 + sub + TPR * t] : make_float4(0.f, 0.f, 0.f, 0.f);
        q[t] = make_float4(v.x * scale_log2, v.y * scale_log2, v.z * scale_log2, v.w * scale_log2);
        o[t] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float m = -INFINITY, l = 0.f;

    // Causal: no row of this block attends beyond its last row.
    const int kv_end = CAUSAL ? min(N, row0 + BR) : N;

    for (int kv0 = 0; kv0 < kv_end; kv0 += BC) {
        for (int i = tid; i < BC * V4; i += NT) {
            int c = i / V4, f = i % V4, gc = kv0 + c;
            bool in = gc < N;
            Ks[c][f] = in ? K4[(size_t)gc * V4 + f] : make_float4(0.f, 0.f, 0.f, 0.f);
            Vs[c][f] = in ? V4p[(size_t)gc * V4 + f] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        __syncthreads();

        float s[BC];
#pragma unroll
        for (int j = 0; j < BC; j++) {
            float acc = 0.f;
#pragma unroll
            for (int t = 0; t < NV; t++) {
                float4 k = Ks[j][sub + TPR * t];
                acc += q[t].x * k.x + q[t].y * k.y + q[t].z * k.z + q[t].w * k.w;
            }
#pragma unroll
            for (int off = 1; off < TPR; off <<= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
            int col = kv0 + j;
            bool valid = col < N && (!CAUSAL || col <= row);
            s[j] = valid ? acc : -INFINITY;
        }

        // Tile 0 always has column 0 valid for every row, so m_new is finite here.
        float mx = s[0];
#pragma unroll
        for (int j = 1; j < BC; j++) mx = fmaxf(mx, s[j]);
        float m_new = fmaxf(m, mx);
        float alpha = exp2f(m - m_new);
        l *= alpha;
#pragma unroll
        for (int t = 0; t < NV; t++) {
            o[t].x *= alpha;
            o[t].y *= alpha;
            o[t].z *= alpha;
            o[t].w *= alpha;
        }
#pragma unroll
        for (int j = 0; j < BC; j++) {
            float p = exp2f(s[j] - m_new);
            l += p;
#pragma unroll
            for (int t = 0; t < NV; t++) {
                float4 v = Vs[j][sub + TPR * t];
                o[t].x += p * v.x;
                o[t].y += p * v.y;
                o[t].z += p * v.z;
                o[t].w += p * v.w;
            }
        }
        m = m_new;
        __syncthreads();
    }

    if (row < N) {
        float inv = 1.f / l;
        float4* O4 = reinterpret_cast<float4*>(O);
#pragma unroll
        for (int t = 0; t < NV; t++)
            O4[(size_t)row * V4 + sub + TPR * t] = make_float4(o[t].x * inv, o[t].y * inv, o[t].z * inv, o[t].w * inv);
    }
}

template <int BR, int BC, int TPR>
static void launch_cfg(const float* Q, const float* K, const float* V, float* O, int N, int causal) {
    float scale_log2 = 1.4426950408889634f / sqrtf((float)D);
    dim3 grid(CEIL_DIV(N, BR));
    if (causal)
        flash_fwd_kernel<BR, BC, TPR, true><<<grid, BR * TPR>>>(Q, K, V, O, N, scale_log2);
    else
        flash_fwd_kernel<BR, BC, TPR, false><<<grid, BR * TPR>>>(Q, K, V, O, N, scale_log2);
}

// cfg selects a (BR, BC, TPR) tile configuration, for tuning. The default, cfg 11
// <32,32,2>, was best or within 5% of best for N = 512..16384 on an RTX 4080.
// Launches on the default stream; caller synchronizes. Returns a cudaError_t (0 = ok);
// 1 is returned for an unsupported d.
extern "C" int flash_fwd_cfg(const float* Q, const float* K, const float* V, float* O, int N, int d, int causal,
                             int cfg) {
    if (d != D) return 1;
    switch (cfg) {
        case 0: launch_cfg<32, 32, 4>(Q, K, V, O, N, causal); break;
        case 1: launch_cfg<16, 32, 4>(Q, K, V, O, N, causal); break;
        case 2: launch_cfg<32, 64, 4>(Q, K, V, O, N, causal); break;
        case 3: launch_cfg<64, 32, 4>(Q, K, V, O, N, causal); break;
        case 4: launch_cfg<16, 32, 2>(Q, K, V, O, N, causal); break;
        case 5: launch_cfg<32, 16, 4>(Q, K, V, O, N, causal); break;
        case 6: launch_cfg<16, 64, 4>(Q, K, V, O, N, causal); break;
        case 7: launch_cfg<32, 32, 8>(Q, K, V, O, N, causal); break;
        case 8: launch_cfg<32, 16, 1>(Q, K, V, O, N, causal); break;
        case 9: launch_cfg<64, 16, 1>(Q, K, V, O, N, causal); break;
        case 10: launch_cfg<64, 32, 1>(Q, K, V, O, N, causal); break;
        case 11: launch_cfg<32, 32, 2>(Q, K, V, O, N, causal); break;
        case 12: launch_cfg<64, 16, 2>(Q, K, V, O, N, causal); break;
        case 13: launch_cfg<16, 16, 2>(Q, K, V, O, N, causal); break;
        case 14: launch_cfg<16, 64, 2>(Q, K, V, O, N, causal); break;
        case 15: launch_cfg<128, 16, 1>(Q, K, V, O, N, causal); break;
        default: return 2;
    }
    return (int)cudaGetLastError();
}

extern "C" int flash_fwd(const float* Q, const float* K, const float* V, float* O, int N, int d, int causal) {
    return flash_fwd_cfg(Q, K, V, O, N, d, causal, 11);
}
