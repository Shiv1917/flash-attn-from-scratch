"""Correctness tests for the attention kernels against an fp64 PyTorch reference.

Usage: python tests/test_attention.py        (after `make`)
Prints max abs error per case with the tolerance; exits non-zero on any failure.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attention"))
from attn_lib import FLASH_CFGS, flash_fwd, naive_attention  # noqa: E402

D = 64
# (N, input scale, causal, description). Tolerances are max-abs-error on O, whose entries are
# convex combinations of rows of V ~ N(0,1) (|O| <~ 4).
TOL_NORMAL = 1e-4
# Large-magnitude inputs: scores reach ~scale^2/8 * 4.5. fp32 rounding of a score s is ~6e-8*|s|,
# and exp() turns that into the same relative error in the probabilities, so the achievable
# error grows with |s| for ANY fp32 implementation. The PyTorch fp32 SDPA error is printed
# alongside as a yardstick.
TOL_LARGE = 1e-3  # scale 20 (|s| ~ 4e3)
TOL_XLARGE = 5e-3  # scale 50 (|s| ~ 1e4)
CASES = [
    (1, 1.0, False, "single token"),
    (7, 1.0, False, "N < one tile"),
    (20, 1.0, False, "N < BC"),
    (64, 1.0, False, ""),
    (100, 1.0, False, "non-multiple"),
    (513, 1.0, False, "non-multiple"),
    (1024, 1.0, False, ""),
    (4096, 1.0, False, ""),
    (513, 1.0, True, "causal"),
    (4096, 1.0, True, "causal"),
    (1024, 20.0, False, "large magnitude (scores ~ 3e3)"),
    (513, 50.0, False, "large magnitude (scores ~ 2e4)"),
    (513, 50.0, True, "large magnitude, causal"),
]


def reference(q, k, v, causal):
    qd, kd, vd = q.double(), k.double(), v.double()
    s = qd @ kd.T / D**0.5
    if causal:
        s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(1), float("-inf"))
    return (torch.softmax(s, dim=-1) @ vd).float()


def main():
    torch.manual_seed(0)
    dev = "cuda"
    failed = False
    print(f"{'impl':<8}{'N':>6} {'scale':>6} {'causal':>7} {'max_abs_err':>12} {'tol':>8}  status  note")
    for n, scale, causal, note in CASES:
        q, k = (torch.randn(n, D, device=dev) * scale for _ in range(2))
        v = torch.randn(n, D, device=dev)
        ref = reference(q, k, v, causal)
        tol = TOL_XLARGE if scale >= 50 else TOL_LARGE if scale > 1 else TOL_NORMAL
        sdpa = torch.nn.functional.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], is_causal=causal)[0, 0]
        sdpa_err = (sdpa - ref).abs().max().item()
        for name, fn in (("naive", naive_attention), ("flash", flash_fwd)):
            out = fn(q, k, v, causal)
            torch.cuda.synchronize()
            err = (out - ref).abs().max().item()
            ok = err <= tol and torch.isfinite(out).all().item()
            failed |= not ok
            print(f"{name:<8}{n:>6} {scale:>6} {str(causal):>7} {err:>12.2e} {tol:>8.0e}  {'PASS' if ok else 'FAIL'}    {note}  [torch fp32 sdpa err {sdpa_err:.1e}]")
    # Every tile config used for tuning must also be correct.
    print("\ntile configs (N=513, non-causal and causal):")
    q, k, v = (torch.randn(513, D, device=dev) for _ in range(3))
    for cfg in range(len(FLASH_CFGS)):
        for causal in (False, True):
            ref = reference(q, k, v, causal)
            err = (flash_fwd(q, k, v, causal, cfg=cfg) - ref).abs().max().item()
            ok = err <= TOL_NORMAL
            failed |= not ok
            print(f"  cfg {cfg} causal={causal!s:<5} err={err:.2e} {'PASS' if ok else 'FAIL'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
