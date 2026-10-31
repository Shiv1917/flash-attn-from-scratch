#pragma once
// Shared host driver for the GEMM kernels. A kernel file defines a launcher and calls
//   return gemm_main(argc, argv, "name", launch);
// Usage of the resulting binary: <bin> M N K [outdir]
// With outdir, A.bin/B.bin/C.bin (fp32 row-major) are written for tests/test_gemm.py.
#include "common.cuh"

typedef void (*GemmLauncher)(int M, int N, int K, const float* A, const float* B, float* C);

inline int gemm_main(int argc, char** argv, const char* name, GemmLauncher launch) {
    int M = argc > 1 ? atoi(argv[1]) : 1024;
    int N = argc > 2 ? atoi(argv[2]) : 1024;
    int K = argc > 3 ? atoi(argv[3]) : 1024;
    std::string outdir = argc > 4 ? argv[4] : "";

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
    fill_random(hA, 1);
    fill_random(hB, 2);

    float *dA, *dB, *dC;
    check_cuda(cudaMalloc(&dA, hA.size() * sizeof(float)));
    check_cuda(cudaMalloc(&dB, hB.size() * sizeof(float)));
    check_cuda(cudaMalloc(&dC, hC.size() * sizeof(float)));
    check_cuda(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemset(dC, 0, hC.size() * sizeof(float)));

    for (int i = 0; i < 3; i++) launch(M, N, K, dA, dB, dC);  // warmup
    check_cuda(cudaGetLastError());

    const int iters = 20;
    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; i++) launch(M, N, K, dA, dB, dC);
    float ms = t.stop() / iters;
    check_cuda(cudaGetLastError());

    double gflops = 2.0 * M * N * K / (ms * 1e-3) / 1e9;
    printf("%s M=%d N=%d K=%d: %.3f ms, %.1f GFLOPS\n", name, M, N, K, ms, gflops);

    check_cuda(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (!outdir.empty()) {
        dump_binary(outdir + "/A.bin", hA);
        dump_binary(outdir + "/B.bin", hB);
        dump_binary(outdir + "/C.bin", hC);
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
