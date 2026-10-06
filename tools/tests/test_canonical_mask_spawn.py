"""CPU tests for the canonical-mask builder spawned at the clock's start (track_1_short/canonical_mask.py).

Run from the repo root: python -m pytest tools/tests -q
Needs the GPT-2 tokenizer files in tiktoken's cache (TIKTOKEN_CACHE_DIR).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track_1_short import canonical_mask  # noqa: E402
from track_1_short.canonical_mask import BackgroundCanonicalMask, build_canonical_mask  # noqa: E402

VOCAB = 50304  # next_multiple_of_n(50257, 128), the trainer's padded vocabulary


class NoCudart:
    """torch.cuda.cudart() stand-in on a machine without CUDA: registering fails, as it may on a GPU node."""
    class cudaError:
        success = 0

    def cudaHostRegister(self, ptr, nbytes, flags):
        return 1


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "cudart", lambda: NoCudart())


@pytest.fixture(scope="module")
def reference_mask() -> np.ndarray:
    return build_canonical_mask(VOCAB)


@pytest.fixture(scope="module")
def gloo_group():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29534")
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def make_builder() -> BackgroundCanonicalMask:
    logged = []
    builder = BackgroundCanonicalMask(VOCAB, owner=True, print0=lambda s, console=False: logged.append(s))
    builder.logged = logged
    return builder


def test_spawned_build_matches_inline_build(reference_mask, gloo_group):
    builder = make_builder()
    assert builder.proc is None and not builder.buf.numpy().any()  # nothing runs before the clock
    builder.start()
    builder.wait()
    out = torch.zeros(VOCAB, VOCAB // 8, dtype=torch.uint8)
    builder.collect(out)
    assert np.array_equal(out.numpy(), reference_mask)
    assert builder.buf is None and builder.proc is None
    assert not any("WARNING" in line for line in builder.logged)


def test_builder_imports_neither_torch_nor_the_trainer():
    """The spawned interpreter's imports are on the clock: the build module must stay torch-free."""
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'track_1_short')!r}); import canonical_mask_build; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'track_1_short', 'triton')))")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_non_owner_spawns_nothing():
    builder = BackgroundCanonicalMask(VOCAB, owner=False, print0=print)
    builder.start()
    assert builder.proc is None
    builder.wait()


def test_failed_build_falls_back_to_inline_build(reference_mask, gloo_group, monkeypatch):
    monkeypatch.setattr(canonical_mask, "BUILD_SCRIPT", Path("/nonexistent/canonical_mask_build.py"))
    builder = make_builder()
    builder.start()
    builder.wait()  # the interpreter exits 2 (no such script): build inline
    assert any("building it inline" in line for line in builder.logged)
    out = torch.zeros(VOCAB, VOCAB // 8, dtype=torch.uint8)
    builder.collect(out)
    assert np.array_equal(out.numpy(), reference_mask)


def test_start_costs_the_trainer_little_at_any_size():
    """start() is on rank 0's critical path. With a large process (2 GB touched here), os.fork() pays
    for the page tables; a spawn does not."""
    ballast = np.ones(2 << 30, dtype=np.uint8)  # noqa: F841  (keeps the process large)
    builder = make_builder()
    started = time.perf_counter()
    builder.start()
    spawn_ms = 1000 * (time.perf_counter() - started)
    builder.proc.kill()
    builder.proc.wait()
    started = time.perf_counter()
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    fork_ms = 1000 * (time.perf_counter() - started)
    os.waitpid(pid, 0)
    print(f"\nstart(): spawn {spawn_ms:.2f} ms vs os.fork() {fork_ms:.2f} ms with 2 GB touched")
    assert spawn_ms < fork_ms
