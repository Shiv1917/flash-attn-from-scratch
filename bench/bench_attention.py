"""Attention benchmark: naive (materializes S), our flash kernel, and PyTorch SDPA.

Usage:
  python bench/bench_attention.py --tune     # latency of every flash tile config at N=4096
  python bench/bench_attention.py            # sweep N; writes results/attention.{md,json,png}
Peak memory is torch.cuda.max_memory_allocated minus the Q/K/V inputs, i.e. the extra memory
each implementation needs (scratch + output).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "attention"))
from attn_lib import DEFAULT_CFG, FLASH_CFGS, flash_fwd, naive_attention  # noqa: E402

D = 64
REF_NAME = "torch flash fp16 (tensor cores, ref)"
HALF = [None]  # fp16 copies of the current q, k, v, created outside the timed region
NS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def sdpa(q, k, v, backend=None):
    q4, k4, v4 = q[None, None], k[None, None], v[None, None]
    if backend is None:
        return F.scaled_dot_product_attention(q4, k4, v4)
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q4, k4, v4)


def time_ms(fn, warmup=3, min_iters=5, budget_ms=300.0, max_iters=100):
    """Median latency in ms, using CUDA events around each call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times, total = [], 0.0
    while (total < budget_ms or len(times) < min_iters) and len(times) < max_iters:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        t = s.elapsed_time(e)
        times.append(t)
        total += t
    return statistics.median(times)


def peak_extra_mb(fn, inputs_bytes):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return (peak - base) / 2**20


def tune(n):
    q, k, v = (torch.randn(n, D, device="cuda") for _ in range(3))
    print(f"flash tile configs <BR,BC,TPR> at N={n}")
    for cfg, name in enumerate(FLASH_CFGS):
        ms = time_ms(lambda: flash_fwd(q, k, v, cfg=cfg))
        print(f"  cfg {cfg} {name:<10} {ms:7.3f} ms  {4 * n * n * D / (ms * 1e-3) / 1e12:6.2f} TFLOPS")


def sweep(cfg):
    impls = {
        "naive (materialize S)": lambda q, k, v: naive_attention(q, k, v),
        "flash (ours)": lambda q, k, v: flash_fwd(q, k, v, cfg=cfg),
        "torch SDPA (default)": lambda q, k, v: sdpa(q, k, v),
        "torch SDPA (math)": lambda q, k, v: sdpa(q, k, v, SDPBackend.MATH),
        # Reference only, NOT the same precision: fp16 inputs on tensor cores (real FlashAttention).
        REF_NAME: lambda q, k, v: sdpa(*HALF[0], SDPBackend.FLASH_ATTENTION),
    }
    res = {name: {"N": [], "ms": [], "tflops": [], "mem_mb": []} for name in impls}
    for n in NS:
        torch.manual_seed(0)
        q, k, v = (torch.randn(n, D, device="cuda") for _ in range(3))
        HALF[0] = tuple(x.half() for x in (q, k, v))
        in_bytes = 3 * q.numel() * 4
        free = torch.cuda.mem_get_info()[0]
        for name, fn in impls.items():
            # naive/math need ~2 N^2 fp32 (S plus softmax output); skip if the GPU can't hold it
            if ("naive" in name or "math" in name) and 2 * n * n * 4 > 0.8 * free:
                print(f"N={n:>6} {name:<24} skipped (needs {2 * n * n * 4 / 2**30:.1f} GB, {free / 2**30:.1f} GB free)")
                continue
            ms = time_ms(lambda: fn(q, k, v))
            mem = peak_extra_mb(lambda: fn(q, k, v), in_bytes)
            tf = 4 * n * n * D / (ms * 1e-3) / 1e12
            r = res[name]
            r["N"].append(n), r["ms"].append(ms), r["tflops"].append(tf), r["mem_mb"].append(mem)
            print(f"N={n:>6} {name:<24} {ms:9.3f} ms {tf:7.2f} TFLOPS {mem:10.1f} MB")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "attention.json").write_text(json.dumps(res, indent=1))

    lines = ["| N | " + " | ".join(impls) + " |", "|---|" + "---|" * len(impls)]
    for title, key, fmt in (("Latency (ms)", "ms", "{:.3f}"), ("TFLOPS", "tflops", "{:.2f}"), ("Peak extra memory (MB)", "mem_mb", "{:.1f}")):
        lines += ["", f"**{title}**", "", "| N | " + " | ".join(impls) + " |", "|---|" + "---|" * len(impls)]
        for n in NS:
            cells = []
            for name in impls:
                r = res[name]
                cells.append(fmt.format(r[key][r["N"].index(n)]) if n in r["N"] else "–")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
    (out / "attention.md").write_text(f"{torch.cuda.get_device_name()}, fp32, d={D}, single head, non-causal\n" + "\n".join(lines[3:]) + "\n")

    plot(res)


def plot(res):
    out = ROOT / "results"
    for key, ylabel, fname, logy in (("ms", "latency (ms)", "attention_latency.png", True), ("mem_mb", "peak extra memory (MB)", "attention_memory.png", True)):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, r in res.items():
            ours = name.startswith("flash")
            ax.plot(r["N"], r[key], marker="o", label=name, linestyle=":" if name == REF_NAME else "--" if "default" in name else "-",
                    lw=3 if ours else 1.5, zorder=5 if ours else 2, markersize=7 if ours else 5)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("sequence length N")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split(' (')[0].capitalize()} vs N ({torch.cuda.get_device_name()}, fp32, d={D})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw plots from results/attention.json")
    ap.add_argument("--n", type=int, default=4096, help="N for --tune")
    ap.add_argument("--cfg", type=int, default=DEFAULT_CFG, help="flash tile config for the sweep")
    a = ap.parse_args()
    if a.replot:
        plot(json.loads((ROOT / "results" / "attention.json").read_text()))
    elif a.tune:
        tune(a.n)
    else:
        sweep(a.cfg)
