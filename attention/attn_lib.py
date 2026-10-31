"""ctypes bindings for the attention shared libraries; operate directly on CUDA torch tensors."""
import ctypes
from pathlib import Path

import torch

BUILD = Path(__file__).resolve().parent.parent / "build"
_P = ctypes.c_void_p

_naive = ctypes.CDLL(str(BUILD / "libnaive_attention.so"))
_naive.naive_attention.argtypes = [_P, _P, _P, _P, _P, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_flash = ctypes.CDLL(str(BUILD / "libflash_fwd.so"))
_flash.flash_fwd_cfg.argtypes = [_P, _P, _P, _P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


# Flash tile configs <BR, BC, TPR>; index = cfg argument of flash_fwd. Keep in sync with flash_fwd.cu.
FLASH_CFGS = ["<32,32,4>", "<16,32,4>", "<32,64,4>", "<64,32,4>", "<16,32,2>", "<32,16,4>", "<16,64,4>", "<32,32,8>",
              "<32,16,1>", "<64,16,1>", "<64,32,1>", "<32,32,2>", "<64,16,2>", "<16,16,2>", "<16,64,2>", "<128,16,1>"]


def _check(rc, name):
    if rc != 0:
        raise RuntimeError(f"{name} returned CUDA/launch error {rc}")


def naive_attention(q, k, v, causal=False, scratch=None):
    """Materializes the N x N scores. `scratch` (N*N fp32) is allocated via torch if omitted."""
    n, d = q.shape
    if scratch is None:
        scratch = torch.empty(n * n, device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    _check(_naive.naive_attention(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), scratch.data_ptr(), n, d, int(causal)), "naive_attention")
    return o


DEFAULT_CFG = 11


def flash_fwd(q, k, v, causal=False, cfg=DEFAULT_CFG):
    n, d = q.shape
    o = torch.empty_like(q)
    _check(_flash.flash_fwd_cfg(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), n, d, int(causal), cfg), "flash_fwd")
    return o
