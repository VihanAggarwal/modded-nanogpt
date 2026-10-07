"""Import track_1_short.model.gpt on a machine without CUDA, and run its validation forward there.

install(): stand-ins for the modules that need a GPU at import -- the `kernels` hub loader and FA3 (imported by
model/attention.py), and the two cross-entropy kernel modules, which compile with nvrtc at import. Call it before the
first track_1_short.model import. The training path (fp8, CUDA graphs) cannot run here.
use_torch_kernels(): plain-torch stand-ins for the GPU kernels of the eval (bf16) forward. Any deterministic function
serves: tests compare the model against itself under the same stand-ins, never against the real kernels.
"""
import os
import sys
import types

import torch
import torch.nn.functional as F


class _Anything:
    """Any attribute, call or index of a stubbed module: imports bind, nothing here is ever run."""
    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        return _Anything(f"{self._name}()")

    def __getattr__(self, name):
        return _Anything(f"{self._name}.{name}")

    def __getitem__(self, key):
        return _Anything(f"{self._name}[]")


def _module(name: str, **attrs) -> types.ModuleType:
    def missing(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Anything(f"{name}.{attr}")

    module = types.ModuleType(name)
    module.__dict__.update(attrs, __getattr__=missing)
    sys.modules[name] = module
    return module


def _no_gpu(*args, **kwargs):
    raise RuntimeError("GPU kernel stub")


STUBBED = ("flash_attn_interface", "kernels", "track_1_short.perf.kernels.cross_entropy",
           "track_1_short.perf.kernels.cplm_cross_entropy")


def install():
    os.environ["FA3_NATIVE"] = "1"  # model/attention.py then imports flash_attn_interface, stubbed here
    _module("flash_attn_interface", flash_attn_varlen_func=_no_gpu)
    _module("kernels", get_kernel=_no_gpu)
    _module("track_1_short.perf.kernels.cross_entropy", CE_KERNEL_BLOCK_SIZE=256, CE_KERNEL_VOCAB_SIZE=50304,
            SOFTCAP_A=23.0, SOFTCAP_B=5.0, SOFTCAP_C=7.5, CE_KERNEL_SOURCE="", CUDA_INCLUDE_DIRS=[],
            lm_head_inverse_scale=lambda w_s, device: torch.ones(1, device=device))
    _module("track_1_short.perf.kernels.cplm_cross_entropy", COPY_TOKEN_ID=50257)


def use_torch_kernels():
    import track_1_short.model.attention as attention
    import track_1_short.model.gpt as gpt

    def ngram_embedding(cache, slots, pool, inp, sink):
        T = inp.numel()
        return (cache[slots[:T].long()].float() - 0.5 * cache[slots[T:].long()].float()).to(cache.dtype)

    def value_embed_lookup(weight, ids, num_planes, grad_accum):
        return tuple(weight.view(num_planes, -1, weight.size(-1))[:, ids.long()].unbind(0))

    def relu_sq_mlp(x, w1, w2, *fp8_args):
        return (F.relu(x @ w1.type_as(x).T) ** 2) @ w2.type_as(x)

    def qk_norm_rope(qk, factor1, factor2, num_heads, rotary_dim, paired, key_offset):
        T, _, d = qk.shape  # norms only, no rotary
        q, k = F.rms_norm(qk[:, :num_heads], (d,)), F.rms_norm(qk[:, num_heads:], (d,))
        return (q.reshape(2 * T, num_heads // 2, d), k.reshape(2 * T, num_heads // 2, d)) if paired else (q, k)

    def fa3(q, k, v, cu_seqlens_q, max_seqlen_q, softmax_scale, window_size, **kwargs):
        cu = cu_seqlens_q.tolist()  # causal attention within each document (no window)
        out = torch.zeros(q.shape[0], q.shape[1], v.shape[-1], dtype=v.dtype)
        for a, b in zip(cu[:-1], cu[1:]):
            if b <= a:
                continue
            s = torch.einsum("thd,shd->hts", q[a:b].float(), k[a:b].float()) * softmax_scale
            s = s.masked_fill(torch.ones(b - a, b - a, dtype=torch.bool).triu(1), float("-inf"))
            out[a:b] = torch.einsum("hts,shd->thd", s.softmax(-1), v[a:b].float()).to(v.dtype)
        return out

    gpt.ngram_embedding, gpt.value_embed_lookup, gpt.ReLUSqrdMLP = ngram_embedding, value_embed_lookup, relu_sq_mlp
    attention.qk_norm_rope_forward, attention._fa3_varlen = qk_norm_rope, fa3
