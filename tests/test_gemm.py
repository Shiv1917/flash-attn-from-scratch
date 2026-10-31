"""Run a GEMM binary and compare its output against torch.matmul.

Usage: python tests/test_gemm.py build/01_naive [build/02_tiled_smem ...]
"""
import subprocess
import sys
import tempfile

import numpy as np
import torch

# (M, N, K): square, non-multiples of any tile size, and skinny shapes.
SIZES = [(128, 128, 128), (1000, 1000, 1000), (513, 257, 129), (64, 1000, 33), (1024, 1024, 1024)]
RTOL, ATOL = 1e-4, 1e-3  # fp32 accumulation order differs from cuBLAS; ATOL covers near-zero entries


def run_case(binary, M, N, K):
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([binary, str(M), str(N), str(K), d], capture_output=True, text=True, check=True)
        load = lambda name, shape: torch.from_numpy(
            np.fromfile(f"{d}/{name}.bin", dtype=np.float32).reshape(shape)
        )
        A, B, C = load("A", (M, K)), load("B", (K, N)), load("C", (M, N))
    # Reference in fp64 so the comparison measures our error, not cuBLAS's.
    ref = (A.double() @ B.double()).float()
    err = (C - ref).abs().max().item()
    ok = torch.allclose(C, ref, rtol=RTOL, atol=ATOL)
    return ok, err, out.stdout.strip()


def main():
    failed = False
    for binary in sys.argv[1:]:
        print(f"== {binary} (rtol={RTOL}, atol={ATOL})")
        for M, N, K in SIZES:
            ok, err, line = run_case(binary, M, N, K)
            failed |= not ok
            print(f"  {'PASS' if ok else 'FAIL'} {M}x{N}x{K}  max_abs_err={err:.2e}  | {line}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
