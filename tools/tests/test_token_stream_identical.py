"""Rule 1 check: the loader of this branch yields byte-identical batches to the record's (upstream 4ea6b93)
for every step of the timed run's schedule, and for the validation.

Run from the repo root: python -m pytest tools/tests -q
Uses FineWeb-format shards under $SPEEDRUN_TEST_DATA/data/fineweb10B (real or synthetic; the full
schedule needs ~4-5 train shards). The record's data.py is read from git, so the repo must have 4ea6b93.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from track_1_short import data as new_data  # noqa: E402
from track_1_short.config import (LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT,  # noqa: E402
                                  Hyperparameters)
from track_1_short.schedule import TrainingSchedule  # noqa: E402

DATA = Path(os.environ.get("SPEEDRUN_TEST_DATA", "/home/user/work/synthdata"))
RECORD_COMMIT = "4ea6b93"
pytestmark = pytest.mark.skipif(not (DATA / "data/fineweb10B/fineweb_val_000000.bin").exists(),
                                reason="no FineWeb shards at $SPEEDRUN_TEST_DATA")


def record_data_module():
    source = subprocess.run(["git", "show", f"{RECORD_COMMIT}:track_1_short/data.py"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader("record_data", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, "record_data.py", "exec"), module.__dict__)
    return module


class HostStaging:
    def upload(self, *tensors):
        return tuple(t.clone() for t in tensors)


def unpinned(module, monkeypatch):
    real_empty = torch.empty
    monkeypatch.setattr(module, "torch", type(sys)("torch_unpinned"))
    module.torch.__dict__.update(torch.__dict__)
    module.torch.empty = lambda *a, pin_memory=False, **k: real_empty(*a, **k)


def same_batch(a, b) -> bool:
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            if x.dtype != y.dtype or not torch.equal(x, y):
                return False
        elif x.dtype != y.dtype or not np.array_equal(x, y):
            return False
    return True


def deterministic_index_switch(module, monkeypatch):
    """A shard starts on a partial BOS index while a thread scans the whole shard. If batches outrun the
    scan, the loader reads the partial index's end as the shard's end and skips to the next shard. A
    real run cannot outrun it (the host runs at most a few steps ahead of 17 ms GPU steps: 6M tokens
    last >~0.6 s against a ~0.07-0.3 s scan), but this test fetches back to back, so both loaders wait
    for the scan here."""
    switch = module.Shard._maybe_switch

    def waiting_switch(self):
        self._ready.wait()
        switch(self)

    monkeypatch.setattr(module.Shard, "_maybe_switch", waiting_switch)


def test_numpy_bos_indexes_equal_torch_nonzero(monkeypatch):
    unpinned(new_data, monkeypatch)
    for path in sorted((DATA / "data/fineweb10B").glob("fineweb_*.bin"))[:2]:
        tokens = new_data._load_data_shard(path)
        shard = new_data.Shard(tokens, 8)
        shard._loader_thread.join()
        partial = (tokens[:6_000_000] == new_data.BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        full = (tokens == new_data.BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        for got, want in ((shard.bos_idx, partial), (shard._full_idx, full)):
            assert got.dtype == want.dtype and np.array_equal(got, want)


@pytest.mark.parametrize("rank", [0, 5])
def test_whole_schedule_and_validation_identical(monkeypatch, rank):
    record = record_data_module()
    for module in (record, new_data):
        unpinned(module, monkeypatch)
        deterministic_index_switch(module, monkeypatch)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    args = Hyperparameters()
    schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations,
                                device="cpu", cooldown_frac=LR_COOLDOWN_FRAC, split_embed_stage=SPLIT_EMBED_STAGE,
                                ws_post_yarn_ext=WS_POST_YARN_EXT)
    train = str(DATA / "data/fineweb10B/fineweb_train_*.bin")
    first = TRAINING_STAGES[0]
    loaders = [m.ScheduledBatches(m.distributed_data_generator(train, first.batch_size, first.train_max_seq_len, HostStaging()),
                                  schedule, steps=range(schedule.total_steps)) for m in (record, new_data)]
    for step in range(schedule.total_steps):
        a, b = (loader.take(step) for loader in loaders)
        assert same_batch(a, b), f"rank {rank}: step {step} differs"
    loaders[1].close()

    val = str(DATA / "data/fineweb10B/fineweb_val_*.bin")
    steps = args.val_tokens // args.val_batch_size
    for a, b in zip(*[[next(g) for _ in range(steps)] for g in
                      (m.distributed_data_generator(val, args.val_batch_size, -1, HostStaging(), align_to_bos=False)
                       for m in (record, new_data))]):
        assert same_batch(a, b), f"rank {rank}: a validation batch differs"
