"""GEMM benchmark: custom kernels vs cuBLAS (torch.matmul, TF32 disabled).

Usage: python bench/bench_gemm.py            # writes results/gemm.md, results/gemm.png
"""
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ["01_naive", "01b_naive_coalesced", "02_tiled_smem", "03_register_tiled"]
SIZES = [256, 512, 1000, 1024, 2048, 4096]
PATTERN = re.compile(r"([\d.]+) ms, ([\d.]+) GFLOPS")


def run_kernel(name, n):
    out = subprocess.run([str(ROOT / "build" / name), str(n), str(n), str(n)], capture_output=True, text=True, check=True).stdout
    return float(PATTERN.search(out).group(2))


def run_cublas(n):
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    iters = 20
    s.record()
    for _ in range(iters):
        a @ b
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return 2 * n**3 / (ms * 1e-3) / 1e9


def main():
    res = {k: [] for k in KERNELS + ["cuBLAS"]}
    for n in SIZES:
        for k in KERNELS:
            res[k].append(run_kernel(k, n))
        res["cuBLAS"].append(run_cublas(n))
        print(n, {k: round(v[-1]) for k, v in res.items()})

    lines = ["| kernel | " + " | ".join(f"N={n}" for n in SIZES) + " |", "|---|" + "---|" * len(SIZES)]
    for k, v in res.items():
        lines.append(f"| {k} | " + " | ".join(f"{g:,.0f}" for g in v) + " |")
    lines += ["", "% of cuBLAS:", "", "| kernel | " + " | ".join(f"N={n}" for n in SIZES) + " |", "|---|" + "---|" * len(SIZES)]
    for k in KERNELS:
        lines.append(f"| {k} | " + " | ".join(f"{100 * g / c:.1f}%" for g, c in zip(res[k], res["cuBLAS"])) + " |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "gemm.md").write_text("GFLOPS (square M=N=K):\n\n" + "\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, v in res.items():
        ax.plot(SIZES, v, marker="o", label=k, linestyle="--" if k == "cuBLAS" else "-")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("M = N = K")
    ax.set_ylabel("GFLOPS")
    ax.set_title(f"SGEMM on {torch.cuda.get_device_name()}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "gemm.png", dpi=150)


if __name__ == "__main__":
    main()
