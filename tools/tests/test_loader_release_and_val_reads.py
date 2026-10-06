"""CPU tests for the final validation's loader release and threaded val reads (train_gpt.py, data.py).

Run from the repo root: python -m pytest tools/tests -q
Uses FineWeb-format shards under $SPEEDRUN_TEST_DATA/data/fineweb10B (real or synthetic: >= 2 train
shards and the val shard).
"""
import gc
import os
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track_1_short import data  # noqa: E402
from track_1_short.config import TRAINING_STAGES, Hyperparameters  # noqa: E402

DATA = Path(os.environ.get("SPEEDRUN_TEST_DATA", "/home/user/work/synthdata"))
pytestmark = pytest.mark.skipif(not (DATA / "data/fineweb10B/fineweb_val_000000.bin").exists(),
                                reason="no FineWeb shards at $SPEEDRUN_TEST_DATA")
WORLD = 8


class HostStaging:
    """PinnedBatchStaging's interface without CUDA: 'uploads' are host copies."""
    def upload(self, *tensors):
        return tuple(t.clone() for t in tensors)


@pytest.fixture(autouse=True)
def cpu_loader(monkeypatch):
    """Rank 3 of 8, and shard reads without pinning (no CUDA here); everything else is data.py's."""
    shards = []
    real_empty = torch.empty

    def empty(*args, pin_memory=False, **kwargs):
        t = real_empty(*args, **kwargs)
        if kwargs.get("dtype") is torch.uint16:
            shards.append(weakref.ref(t))
        return t

    monkeypatch.setattr(data.torch, "empty", empty)
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    monkeypatch.setattr(dist, "get_world_size", lambda: WORLD)
    return shards


def live(refs) -> int:
    """How many of the weakly referenced tensors are alive. A function, so no strong reference survives
    into the caller (pytest's assertion rewriting keeps an assert expression's intermediate values)."""
    return sum(ref() is not None for ref in refs)


def batches_equal(a, b):
    return all(torch.equal(x, y) if isinstance(x, torch.Tensor) else (x == y).all() for x, y in zip(a, b))


def test_close_releases_both_shards(cpu_loader):
    stage = TRAINING_STAGES[0]
    pattern = str(DATA / "data/fineweb10B/fineweb_train_*.bin")
    loader = data.distributed_data_generator(pattern, stage.batch_size, stage.train_max_seq_len, HostStaging())

    class Schedule:
        def lookup(self, step):
            return stage, 0.0

    batches = data.ScheduledBatches(loader, Schedule(), steps=range(4))
    kept = [batches.take(s) for s in range(4)]
    data_shards = cpu_loader
    # The loader's background threads (the next shard's read, both shards' full BOS scans) finish long
    # before a run's last step; wait for them as that step would find them.
    # (A finishing thread can start another: the async read's Shard starts its own scan.)
    while others := [t for t in threading.enumerate() if t is not threading.main_thread()]:
        for thread in others:
            thread.join()
    # The current shard and the next one (loaded async) are held by the suspended loader.
    assert len(data_shards) == 2 and live(data_shards) == 2
    batches.close()
    gc.disable()  # the trainer runs with cyclic GC off: the shards must go by refcount alone
    try:
        assert live(data_shards) == 0, "a closed loader still holds a shard"
    finally:
        gc.enable()
    # Batches fetched before the close are independent of the shards.
    assert all(b.inputs.numel() == stage.batch_size // WORLD for b in kept)


def test_val_batches_on_a_thread_match_inline():
    args = Hyperparameters()
    files = str(DATA / "data/fineweb10B/fineweb_val_*.bin")
    val_steps = args.val_tokens // args.val_batch_size

    def read():
        it = data.distributed_data_generator(files, args.val_batch_size, -1, HostStaging(), align_to_bos=False)
        return [next(it) for _ in range(val_steps)]

    inline = read()
    with ThreadPoolExecutor(1) as pool:
        threaded = pool.submit(read).result()
    assert len(inline) == len(threaded) == val_steps == 5
    for a, b in zip(inline, threaded):
        assert batches_equal(a, b)
