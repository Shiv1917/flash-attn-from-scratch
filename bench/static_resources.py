"""Compile-time resource usage and theoretical occupancy of each GEMM kernel (no profiler needed).

Compiles every gemm/*.cu with `ptxas -v`, parses registers / shared memory / spills per kernel,
and computes theoretical occupancy for the target GPU (Ada sm_89 limits). Block sizes are the
launch configs in the sources. Writes results/gemm_static.md.
Usage: python bench/static_resources.py [sm_89]
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCH = sys.argv[1] if len(sys.argv) > 1 else "sm_89"
# sm_89: 65536 regs/SM, 1536 threads/SM, 24 blocks/SM, 100 KB smem/SM; regs allocated per warp in units of 256.
REGS_SM, THREADS_SM, BLOCKS_SM, SMEM_SM = 65536, 1536, 24, 100 * 1024
BLOCK = {"01_naive": 1024, "01b_naive_coalesced": 1024, "02_tiled_smem": 1024, "03_register_tiled": 256}


def occupancy(regs, smem, threads):
    warps = (threads + 31) // 32
    regs_per_block = ((regs * 32 + 255) // 256 * 256) * warps
    blocks = min(BLOCKS_SM, THREADS_SM // threads, REGS_SM // regs_per_block, SMEM_SM // max(smem, 1))
    return blocks, 100.0 * blocks * threads / THREADS_SM


rows = []
for src in sorted((ROOT / "gemm").glob("0[1-9]*.cu")):
    name = src.stem
    out = subprocess.run(
        ["nvcc", "-O3", "-std=c++17", f"-arch={ARCH}", "-Xptxas", "-v", "-I", str(ROOT / "build/compat"), "-c", str(src), "-o", "/dev/null"],
        capture_output=True, text=True,
    ).stderr
    regs = int(re.search(r"Used (\d+) registers", out).group(1))
    smem = int(m.group(1)) if (m := re.search(r"(\d+) bytes smem", out)) else 0
    spill = int(re.search(r"(\d+) bytes spill stores", out).group(1))
    threads = BLOCK[name]
    blocks, occ = occupancy(regs, smem, threads)
    rows.append((name, threads, regs, smem, spill, blocks, occ))

lines = ["| kernel | threads/block | regs/thread | static smem (B) | spill stores (B) | blocks/SM | theoretical occupancy |", "|---|---|---|---|---|---|---|"]
lines += [f"| {n} | {t} | {r} | {s} | {sp} | {b} | {o:.0f}% |" for n, t, r, s, sp, b, o in rows]
text = "\n".join(lines) + "\n"
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "gemm_static.md").write_text(text)
print(text)
