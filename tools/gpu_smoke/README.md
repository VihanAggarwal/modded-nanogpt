# Single-GPU smoke test (Google Colab works)

The speedrun itself needs 8xH100 in one node; Colab gives one GPU. This test checks on one GPU what the
CPU tests cannot see: the patched loader with real pinned memory and CUDA events, PyTorch's pinned-cache
reuse, and the spawned canonical-mask builder with cudaHostRegister. It also times the costs the patches remove.

In a Colab notebook with a GPU runtime (H100 or A100 if available; any CUDA GPU works for correctness), and
a high-RAM runtime if offered (the test holds ~6 GB of host memory):

```
!git clone -b claude/nanogpt-optimization-n49ur2 https://github.com/VihanAggarwal/modded-nanogpt
%cd modded-nanogpt
!pip install -q tiktoken
!python tools/gpu_smoke/smoke_1gpu.py
```

It writes three 200 MB synthetic shards under /tmp, runs in a few minutes, and ends with `N/N passed`.
Paste the whole output back. The timings in it (fresh vs reused pinned allocation, first-fetch time,
start() vs fork) are the real-hardware numbers the estimates rest on.
