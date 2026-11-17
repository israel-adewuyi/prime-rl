"""Experimental single-pass online logsumexp/entropy reduction, FP32 throughout."""
import torch
import triton
import triton.language as tl


@triton.jit
def _update(M, S, T, X, MO, SO, TO, V: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * V + col, col < V, float("-inf"))
    old_m = tl.load(M + row)
    new_m = tl.maximum(old_m, tl.max(x, 0))
    old_scale = tl.exp(old_m - new_m)
    e = tl.exp(x - new_m)
    sx = tl.sum(e, 0)
    tx = tl.sum(tl.where(col < V, e * x, 0.), 0)
    tl.store(MO + row, new_m)
    tl.store(SO + row, tl.load(S + row) * old_scale + sx)
    tl.store(TO + row, tl.load(T + row) * old_scale + tx)


def online_update(m, s, t, x):
    assert x.dtype == torch.float32 and x.is_contiguous()
    mo, so, to = torch.empty_like(m), torch.empty_like(s), torch.empty_like(t)
    _update[(x.shape[0],)](m,s,t,x,mo,so,to,x.shape[1],triton.next_power_of_2(x.shape[1]),
                           num_warps=8, enable_fp_fusion=False)
    return mo, so, to
