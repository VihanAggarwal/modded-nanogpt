"""CPU tests of stream retrieval v2 inside the streamret arm (STREAM_RETRIEVAL=1: track_1_short/stream_memory.py and
its helper with the P2 / P3 parts, and the trainer hooks).

They test the arm: the stack with tools/stream_retrieval/apply_overlay.sh applied (the stack itself carries no
retrieval code). tools/tests/test_stream_retrieval_arm.py builds that tree from the working tree and runs this file
(and the standalone ones: test_stream_memory_v2.py, test_stream_integration.py, test_stream_lowtables.py,
test_stream_pointer.py) inside it; in an arm (make_streamret_arm.sh) it also runs directly:
python -m pytest tools/stream_retrieval -q. The helper is compiled with cc. Most tests write their own small shards; the
tap and end-to-end tests also use the FineWeb-format shards under $SPEEDRUN_TEST_DATA/data/fineweb10B (synthetic is
fine).

  1 the memory is exactly the trained stream (8 ranks, shard switches)      4 the flags; the trainer and the model's
  2 no leakage: val and unread shard bytes never enter any part               eval forward touch the memory only
  3 the tap leaves every batch unchanged (the whole 1050-step schedule)       behind the flag
  5 end to end on 2 gloo ranks: tap, GO, collect, check, StreamEval with the shipped gates, the FIT dump
  6 failures raise; fit_batches tiles the FIT positions into val-shaped batches
"""
import ast
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "track_1_short/stream_memory.py").exists():
    pytest.skip("the stack tree: these tests run in the streamret arm (tools/tests/test_stream_retrieval_arm.py)",
                allow_module_level=True)
sys.path.insert(0, str(ROOT))

from track_1_short import data, stream_memory  # noqa: E402
from track_1_short.config import (LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT,  # noqa: E402
                                  Hyperparameters)
from track_1_short.schedule import TrainingSchedule  # noqa: E402
from track_1_short.stream_memory import StreamMemory  # noqa: E402

DATA = Path(os.environ.get("SPEEDRUN_TEST_DATA", "/home/user/work/synthdata"))
HAVE_DATA = (DATA / "data/fineweb10B/fineweb_val_000000.bin").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="no FineWeb shards at $SPEEDRUN_TEST_DATA")
BOS, SEP, KEY = 50256, 0xFFFF, 6
FLAGS = ("STREAM_RETRIEVAL", "STREAM_RETRIEVAL_LOW", "STREAM_RETRIEVAL_POINTER", "STREAM_RETRIEVAL_FIT",
         "STREAM_RETRIEVAL_FIT_K", "STREAM_RETRIEVAL_GATE", "STREAM_RETRIEVAL_THREADS")


# ------------------------------------------------------------------------------------------------ helpers

def write_shard(path: Path, tokens) -> Path:
    tokens = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros(256, dtype=np.int32)
    header[:3] = (20240520, 1, tokens.size)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    return path


def read_shard(path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16, offset=1024)


class HostStaging:
    def upload(self, *tensors):
        return tuple(t.clone() for t in tensors)


@pytest.fixture
def cpu_loader(monkeypatch):
    real_empty = torch.empty
    monkeypatch.setattr(data.torch, "empty", lambda *a, pin_memory=False, **k: real_empty(*a, **k))


def make_memory(tmp_path, train_files, val_file, steps, *, world=1, chunk=4096, val_steps=2, stream_tokens=None,
                name="m", **kw) -> StreamMemory:
    if stream_tokens is None:
        stream_tokens = sum(read_shard(f).size for f in train_files)
    kw.setdefault("hash_bits", 16)
    kw.setdefault("threads", 3)
    return StreamMemory(train_files=[str(f) for f in train_files], val_file=str(val_file), total_steps=steps,
                        stream_tokens=stream_tokens, world=world, rank=0, master=True,
                        val_tokens=world * chunk * val_steps, chunk=chunk, device="cpu", rows_dir=str(tmp_path),
                        dump_path=str(tmp_path / f"{name}.dump"), readlog_path=str(tmp_path / f"{name}.reads"), **kw)


def finish(memory: StreamMemory, tmp_path, name="m"):
    """GO and collect, then (the rows' summary, the stream dump, the read log)."""
    memory.go()
    memory.collect()
    rows = dict(records=int(memory.counts.sum()), ptr=int(memory.pcounts.sum()) if memory.pointer else 0, low=None)
    if memory.low:  # P2's rows of every val position, in val order: [positions, 5]
        w, vs, ch = memory.world, memory.val_steps, memory.chunk
        low = np.frombuffer(memory.map, dtype=stream_memory.LOW_DTYPE, count=w * vs * ch * 5,
                            offset=memory.layout["low"]).reshape(w, vs, ch, 5)
        rows["low"] = np.concatenate([low[c % w, c // w] for c in range(w * vs)])
    dump = np.fromfile(tmp_path / f"{name}.dump", dtype=np.uint16)
    reads = (tmp_path / f"{name}.reads").read_text().split("\n")
    memory.close()
    return rows, dump, [r.split() for r in reads if r]


def small_shards(tmp_path, rng, n_shards=4, tokens=2_000_000):
    """FineWeb-format shards of random documents (BOS + 1..5000 tokens, ids < 20000)."""
    paths = []
    for i in range(n_shards):
        out, n = [], 0
        while n < tokens:
            d = int(min(rng.geometric(1 / 700), 5000))
            out.append(np.array([BOS, *rng.integers(0, 20000, d)], dtype=np.uint16))
            n += d + 1
        paths.append(write_shard(tmp_path / f"fineweb_train_{i + 1:06d}.bin", np.concatenate(out)[:tokens]))
    return paths


def record_schedule():
    hp = Hyperparameters()
    return TrainingSchedule(TRAINING_STAGES, 978, hp.num_extension_iterations, device="cpu", cooldown_frac=LR_COOLDOWN_FRAC,
                            split_embed_stage=SPLIT_EMBED_STAGE, ws_post_yarn_ext=WS_POST_YARN_EXT)


# ------------------------------------------------------------------------------------------------ 1 and 2: the stream

def test_memory_is_exactly_every_ranks_trained_tokens(tmp_path, cpu_loader, monkeypatch):
    """8 ranks' loaders on the record schedule (first 40 steps, across two shard switches): the memory, fed by rank 0's
    tap alone, is every rank's (inputs + last target) of every batch, in step-then-rank order, nothing else."""
    rng = np.random.default_rng(3)
    small_shards(tmp_path, rng)
    val = write_shard(tmp_path / "fineweb_val_000000.bin", rng.integers(40000, 50000, 70000))
    pattern = str(tmp_path / "fineweb_train_*.bin")
    schedule, world, steps = record_schedule(), 8, 40
    memory = make_memory(tmp_path, sorted(glob.glob(pattern)), val, steps, world=world, chunk=4096, val_steps=2,
                         stream_tokens=steps * (TRAINING_STAGES[0].batch_size + world), hash_bits=20)
    first = TRAINING_STAGES[0]
    loaders = []
    for r in range(world):
        monkeypatch.setattr(dist, "get_rank", lambda r=r: r)
        monkeypatch.setattr(dist, "get_world_size", lambda: world)
        loaders.append(data.ScheduledBatches(data.distributed_data_generator(
            pattern, first.batch_size, first.train_max_seq_len, HostStaging(),
            on_spans=memory.on_spans if r == 0 else None), schedule, steps=range(steps)))
    expected = []
    for step in range(steps):
        for r in range(world):
            monkeypatch.setattr(dist, "get_rank", lambda r=r: r)  # read by the generator at its first fetch
            batch = loaders[r].take(step)
            expected.append(np.concatenate([batch.inputs_cpu, batch.targets_cpu[-1:].numpy()]).astype(np.uint16))
    for loader in loaders:
        loader.close()
    _, dump, reads = finish(memory, tmp_path)
    assert memory.sent == steps
    # The stream with the separators removed is the concatenation of every rank's (inputs + last target), step by step.
    assert np.array_equal(dump[dump != SEP], np.concatenate(expected))
    # And every separator ends a document span that starts at a BOS.
    seps = np.flatnonzero(dump == SEP)
    assert dump[0] == BOS and np.all(dump[seps[:-1] + 1] == BOS) and seps[-1] == dump.size - 1
    assert len({r[1] for r in reads if r[0] == "train"}) >= 3  # across shard switches


def test_no_val_or_unread_shard_token_enters_any_part(tmp_path, cpu_loader, monkeypatch):
    """With P2 and P3 on: the helper reads exactly the spans the loader handed out (in order) and the val prefix,
    nothing of an unread shard tail or an unread shard; val shares no token with the memory, so no part has a row."""
    rng = np.random.default_rng(4)
    paths = small_shards(tmp_path, rng)
    val = write_shard(tmp_path / "fineweb_val_000000.bin", np.where(rng.random(70000) < 0.002, BOS,
                                                                    rng.integers(40000, 50000, 70000)))
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    steps, first = 30, TRAINING_STAGES[0]

    def replay(on_spans):
        loader = data.ScheduledBatches(data.distributed_data_generator(
            str(tmp_path / "fineweb_train_*.bin"), first.batch_size, first.train_max_seq_len, HostStaging(),
            on_spans=on_spans), record_schedule(), range(steps))
        for step in range(steps):
            loader.take(step)
        loader.close()

    spans = []
    record = lambda f, starts, ends: spans.extend((f, int(a), int(e)) for s, en in zip(starts, ends) for a, e in zip(s, en))
    replay(record)
    # Every token past the last one read (a document cut at max_seq_len skips its tail too) becomes 30000..39999.
    # Spans depend only on the BOS positions, which stay, so the run below reads the same spans.
    last_file, last_end = max((f, e) for f, _, e in spans)
    read_mask = [np.zeros(read_shard(p).size, dtype=bool) for p in paths]
    for f, a, e in spans:
        read_mask[f][a:e] = True
    for i, p in enumerate(paths):
        toks = read_shard(p)
        unread = ~read_mask[i] & (toks != BOS)
        toks[unread] = rng.integers(30000, 40000, int(unread.sum()))
        write_shard(p, toks)
    assert last_file == 2 and last_end < 1_900_000 and not read_mask[3].any()  # an unread tail and an unread shard
    first_spans, spans = spans, []
    memory = make_memory(tmp_path, paths, val, steps, world=8, stream_tokens=steps * (first.batch_size + 8), hash_bits=20,
                         low=True, pointer=True, low_threads=2, low_entries=steps * (first.batch_size + 8))
    replay(lambda f, starts, ends: (record(f, starts, ends), memory.on_spans(f, starts, ends)))
    assert spans == first_spans
    rows, dump, reads = finish(memory, tmp_path)
    body = dump[(dump != SEP) & (dump != BOS)]
    assert body.max() < 20000, "a token of the unread shard tail or of val entered the memory"
    # The helper read exactly the spans, in order, and the val prefix: nothing else.
    train_reads = [(int(f), (int(off) - 1024) // 2, (int(off) - 1024 + int(n)) // 2) for kind, f, off, n in
                   (r for r in reads if r[0] == "train")]
    assert train_reads == spans
    assert [r for r in reads if r[0] != "train"] == [["val", "0", "1024", str(2 * (memory.world * 4096 * 2 + 1))]]
    # val shares no token with the memory but BOS: no record, no P3 row, and P2 knows only the 1-token context BOS
    assert rows["records"] == rows["ptr"] == 0
    x = read_shard(val)[:rows["low"].shape[0]]
    seen = rows["low"]["N"] > 0
    assert np.array_equal(seen[:, 0], x == BOS) and not seen[:, 1:].any() and seen[:, 0].sum() > 10


@needs_data
@pytest.mark.parametrize("rank", [0, 5])
def test_the_tap_leaves_every_batch_unchanged(cpu_loader, monkeypatch, rank):
    """The stack's whole schedule (978 scheduled, 1050 trained steps) with and without the tap: byte-identical
    batches; and the spans the tap sees reconstruct this rank's (inputs + last target) of every batch."""
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    schedule, first = record_schedule(), TRAINING_STAGES[0]
    pattern = str(DATA / "data/fineweb10B/fineweb_train_*.bin")
    files = sorted(glob.glob(pattern))
    seen = []
    tapped = data.ScheduledBatches(data.distributed_data_generator(pattern, first.batch_size, first.train_max_seq_len,
                                   HostStaging(), on_spans=lambda f, s, e: seen.append((f, s[rank], e[rank]))),
                                   schedule, range(schedule.total_steps))
    plain = data.ScheduledBatches(data.distributed_data_generator(pattern, first.batch_size, first.train_max_seq_len,
                                  HostStaging()), schedule, range(schedule.total_steps))
    shards = {}
    for step in range(schedule.total_steps):
        a, b = tapped.take(step), plain.take(step)
        for x, y in zip(a, b):
            assert (torch.equal(x, y) if isinstance(x, torch.Tensor) else np.array_equal(x, y)), f"step {step} differs"
        f, starts, ends = seen[step]
        if f not in shards:
            shards = {f: np.memmap(files[f], dtype=np.uint16, mode="r", offset=1024)}
        own = np.concatenate([shards[f][s:e] for s, e in zip(starts, ends)])
        assert np.array_equal(own, np.concatenate([a.inputs_cpu, a.targets_cpu[-1:].numpy()]).astype(np.uint16))
    assert len(seen) == schedule.total_steps
    tapped.close()
    plain.close()


# ------------------------------------------------------------------------------------------------ 4. flags

@pytest.mark.parametrize("name, value, attr, enabled", [
    ("STREAM_RETRIEVAL", None, "ENABLED", False), ("STREAM_RETRIEVAL", "0", "ENABLED", False),
    ("STREAM_RETRIEVAL", "", "ENABLED", False), ("STREAM_RETRIEVAL", "1", "ENABLED", True),
    ("STREAM_RETRIEVAL_LOW", None, "LOW", False), ("STREAM_RETRIEVAL_LOW", "1", "LOW", True),
    ("STREAM_RETRIEVAL_POINTER", None, "POINTER", True), ("STREAM_RETRIEVAL_POINTER", "0", "POINTER", False)])
def test_the_flags(name, value, attr, enabled):
    env = {k: v for k, v in os.environ.items() if k not in FLAGS}
    if value is not None:
        env[name] = value
    out = subprocess.run([sys.executable, "-c", f"from track_1_short import stream_memory as m; print(m.{attr})"],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(enabled)


def test_the_shipped_gates_fit_the_default_parts():
    """Gate.for_run: GATE_V2 for the default parts (P1 + P3), GATE_V2_LOW with STREAM_RETRIEVAL_LOW=1 (P2 + P1 + P3),
    the spec fitted at the run's trained steps; the CPU placeholders are fitted at 1050 steps on a proxy's outputs."""
    for low in (False, True):
        gate = stream_memory.Gate.for_run(low=low, pointer=True, steps=1050)
        assert gate.orders == stream_memory.chain_orders(low) and gate.pointer and gate.exact
        assert gate.tokens() == len(gate.orders) + 3 and "own last 16" in json.dumps(gate.spec["fit"])
        assert gate.proxy and gate.fit_steps == 1050


def guarded_uses(source: str, names, guards):
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def guarded(node):
        while node in parents:
            parent = parents[node]
            if isinstance(parent, (ast.If, ast.IfExp)) and any(g in ast.unparse(parent.test) for g in guards):
                return True
            if isinstance(parent, ast.BoolOp) and any(g in ast.unparse(parent) for g in guards):
                return True
            node = parent
        return False

    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id in names and not (n.value.id == "stream_memory" and n.attr == "ENABLED")]
    return uses, [ast.unparse(n) for n in uses if not guarded(n)]


def test_trainer_touches_the_memory_only_behind_the_flag():
    """train_gpt.py: the gate, the memory and the model's side-output buffers exist only under STREAM_RETRIEVAL=1, only
    the timed loader is tapped, and every use of the memory, its rows or the retrieval's eval sits under a guard on them,
    so with the flag unset no tap, process, buffer or row exists and the validation computes what the stack does."""
    source = (ROOT / "train_gpt.py").read_text()
    guards = ("memory is not None", "stream_rows is not None", "stream_rows is None", "stream_memory.ENABLED")
    uses, unguarded = guarded_uses(source, ("memory", "stream_rows", "stream_eval", "stream_gate", "stream_memory"), guards)
    assert len(uses) >= 12
    assert not unguarded, unguarded
    assert "memory = None\n    if stream_memory.ENABLED:" in source
    assert "stream_gate = None\n    if stream_memory.ENABLED:" in source
    assert "warmup_batches = ScheduledBatches(train_loader(), " in source
    assert source.count("on_spans=memory.on_spans") == 1
    assert source.count("model.stream_lm_") == 2  # the two buffers, set under the flag only (checked above)


def test_the_models_eval_forward_reports_lm_features_only_with_the_buffers():
    """model/gpt.py: the buffers are registered empty (None) and the side output runs only when they are set; it is
    stream_memory.lm_features over LM_ROWS-row sub-slabs, with CPLM's <copy> column dropped."""
    source = (ROOT / "track_1_short/model/gpt.py").read_text()
    for name in ("stream_lm_tok", "stream_lm_out"):
        assert f'self.register_buffer("{name}", None, persistent=False)' in source
    uses, unguarded = guarded_uses(source, ("self",), ("self.stream_lm_out is not None",))
    calls = [u for u in uses if u.attr == "_stream_lm"]
    assert len(calls) == 2 and all(ast.unparse(c) not in unguarded for c in calls)
    assert "stream_memory.lm_features(logits[a - lo:b - lo], self.stream_lm_tok[a:b], drop_col)" in source
    assert "self._stream_lm(logits, lo, hi, self.copy_col_full)" in source


# ------------------------------------------------------------------------------------------------ 6. failures

@pytest.fixture
def toy(tmp_path):
    """A train shard of random documents over a 4-token vocabulary, spans over it in 4 steps, and a val shard."""
    rng = np.random.default_rng(0)
    docs = [np.array([BOS, *rng.integers(0, 4, rng.integers(3, 120))], dtype=np.uint16) for _ in range(900)]
    shard = write_shard(tmp_path / "fineweb_train_000001.bin", np.concatenate(docs))
    starts = np.cumsum([0] + [d.size for d in docs])
    spans = [(int(starts[i]), int(starts[i + 1])) for i in range(840)]
    val = np.concatenate([np.array([BOS, *rng.integers(0, 4, rng.integers(3, 120))], dtype=np.uint16) for _ in range(250)])
    assert val.size > 2 * 4096 + 1
    val_file = write_shard(tmp_path / "fineweb_val_000000.bin", val)
    steps = [spans[0:150], spans[150:420], spans[420:421], spans[421:]]
    return dict(shard=shard, val_file=val_file, val=read_shard(val_file), steps=steps, spans=spans)


def send(memory: StreamMemory, file_idx: int, spans_per_rank):
    memory.on_spans(file_idx, [[a for a, _ in s] for s in spans_per_rank], [[e for _, e in s] for s in spans_per_rank])


def test_wrong_step_count_is_refused(tmp_path, toy):
    memory = make_memory(tmp_path, [toy["shard"]], toy["val_file"], 5, low=True, pointer=True, low_threads=2)
    for g in toy["steps"]:
        send(memory, 0, [g])
    memory.go()
    with pytest.raises(RuntimeError, match="GO for 4 steps after 4 step messages; the memory expects 5"):
        memory.collect()
    memory.close()


def test_a_dead_helper_raises(tmp_path, toy):
    memory = make_memory(tmp_path, [toy["shard"]], toy["val_file"], 4, pointer=True)
    send(memory, 0, [toy["steps"][0]])
    memory.proc.kill()
    memory.proc.wait()
    with pytest.raises((BrokenPipeError, RuntimeError)):
        for g in toy["steps"][1:]:
            send(memory, 0, [g])
        memory.go()
    with pytest.raises(RuntimeError, match="exited"):
        memory.collect()
    memory.close()


def test_a_rank_without_the_helper_times_out(tmp_path, toy, monkeypatch):
    memory = make_memory(tmp_path, [toy["shard"]], toy["val_file"], 4)
    memory.proc, proc = None, memory.proc  # as on ranks 1..7: only the header to go by
    monkeypatch.setattr(stream_memory, "COLLECT_TIMEOUT_S", 0.3)
    with pytest.raises(RuntimeError, match="no state 2 after"):
        memory.collect()
    memory.proc = proc
    memory.close()


def test_more_than_one_node_is_refused(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    with pytest.raises(RuntimeError, match="needs one node"):  # before anything is compiled, spawned or mapped
        StreamMemory.for_run(train_pattern="unused", val_pattern="unused", step_batch_sizes=[8], world=16, rank=0,
                             master=True, val_tokens=16 * 4096, chunk=4096, device="cpu")


@pytest.mark.parametrize("k, ok", [(8, False), (16, True), (24, False), (32, True)])
def test_a_fit_k_that_does_not_tile_the_eval_chunks_is_refused_at_startup(monkeypatch, tmp_path, k, ok):
    """STREAM_RETRIEVAL_FIT_K x the final per-rank batch must be whole eval chunks (fit_batches): the record schedule's
    16,384-token final batch per rank and 262,144-token chunk allow 16, 32, ... A wrong K fails before anything is
    built, not in fit_dump after the whole dev run."""
    monkeypatch.setenv("STREAM_RETRIEVAL_FIT", str(tmp_path / "fit"))
    monkeypatch.setenv("STREAM_RETRIEVAL_FIT_K", str(k))
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    sizes = [8 * 49152] * 40 + [8 * 16384] * 40  # the per-rank batches of the schedule's last stages
    call = lambda: StreamMemory.for_run(train_pattern=str(tmp_path / "none_*.bin"), val_pattern=str(tmp_path / "none_*"),
                                        step_batch_sizes=sizes, world=8, rank=0, master=True, val_tokens=8 * 262144,
                                        chunk=262144, device="cpu")
    if ok:
        with pytest.raises(IndexError):  # past the check: no val file here
            call()
    else:
        with pytest.raises(RuntimeError, match=f"STREAM_RETRIEVAL_FIT_K={k}: .* use a multiple of 16"):
            call()
    assert not (tmp_path / "fit").exists()


def test_mismatched_val_batches_are_caught(tmp_path, toy):
    memory = make_memory(tmp_path, [toy["shard"]], toy["val_file"], 4)
    for g in toy["steps"]:
        send(memory, 0, [g])
    memory.go()
    memory.collect()
    val = toy["val"]

    class B:
        def __init__(self, lo):
            self.inputs_cpu = val[lo:lo + 4096].astype(np.int32)
            self.targets_cpu = torch.from_numpy(val[lo + 1:lo + 4097].astype(np.int64))
    memory.check([B(0), B(4096)])
    with pytest.raises(RuntimeError, match="not the chunk"):
        memory.check([B(4096), B(0)])
    memory.close()


def test_fit_batches_tile_the_fit_positions():
    """FIT positions (two training batches of 1024 per rank) as val-shaped batches of 2048: documents restart at every
    BOS and at the second batch's start (its first token continues no document of the first)."""
    rng = np.random.default_rng(7)
    x = rng.integers(0, 50000, 2048)
    x[[0, 300, 1500]] = BOS
    y = rng.integers(0, 50000, 2048)
    out = stream_memory.fit_batches(x, y, np.array([1024, 1024]), 2048, HostStaging())
    assert len(out) == 1
    b = out[0]
    assert np.array_equal(b.inputs_cpu, x.astype(np.int32)) and torch.equal(b.targets, torch.from_numpy(y))
    cum = b.cum_seqlens
    assert cum[:5].tolist() == [0, 300, 1024, 1500, 2048] and cum[5:].eq(2048).all() and cum.dtype == torch.int32
    with pytest.raises(AssertionError, match="do not tile"):
        stream_memory.fit_batches(x[:2000], y[:2000], np.array([1000, 1000]), 2048, HostStaging())


# ------------------------------------------------------------------------------------------------ 5. end to end

WORKER = r'''
import json, math, os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np, torch, torch.distributed as dist
from track_1_short import data, stream_memory
tmp, rank = sys.argv[2], int(sys.argv[3])
cfg = json.load(open(os.path.join(tmp, "cfg.json")))
dist.init_process_group("gloo", init_method=f"file://{tmp}/store", rank=rank, world_size=2)
real_empty = torch.empty
data.torch.empty = lambda *a, pin_memory=False, **k: real_empty(*a, **k)
class Staging:
    def upload(self, *t): return tuple(x.clone() for x in t)
class Stage:
    batch_size, train_max_seq_len = cfg["batch"], cfg["seq"]
class Schedule:
    def lookup(self, step): return Stage, 0.0
class Model:  # the hooks' contract (model/gpt.py): a uniform LM over the vocabulary, its side output in the buffers
    def __init__(self, P, K):
        self.stream_lm_tok = torch.zeros(P, K, dtype=torch.long)
        self.stream_lm_out = torch.zeros(P, 2 + 3 * K)
    def forward(self, inputs):
        P = inputs.numel()
        logits = torch.zeros(1024, 50257)
        for a in range(0, P, 1024):
            self.stream_lm_out[a:a + 1024] = stream_memory.lm_features(logits[:min(1024, P - a)], self.stream_lm_tok[a:a + 1024])
        return torch.full((P,), math.log(50257.0))
gate = stream_memory.Gate.for_run()
model = Model(cfg["chunk"], gate.tokens())
pattern = os.path.join(tmp, "data/fineweb10B/fineweb_train_*.bin")
memory = stream_memory.StreamMemory.for_run(train_pattern=pattern, val_pattern=os.path.join(tmp, "data/fineweb10B/fineweb_val_*.bin"),
    step_batch_sizes=[cfg["batch"]] * cfg["steps"], world=2, rank=rank, master=rank == 0,
    val_tokens=cfg["val_tokens"], chunk=cfg["chunk"], device="cpu")
loader = data.ScheduledBatches(data.distributed_data_generator(pattern, cfg["batch"], cfg["seq"], Staging(),
    on_spans=memory.on_spans if rank == 0 else None), Schedule(), range(cfg["steps"]))
for step in range(cfg["steps"]):
    loader.take(step)
loader.close()
val_iter = data.distributed_data_generator(os.path.join(tmp, "data/fineweb10B/fineweb_val_*.bin"), 2 * cfg["chunk"], -1,
                                           Staging(), align_to_bos=False)
val_batches = [next(val_iter) for _ in range(cfg["val_tokens"] // (2 * cfg["chunk"]))]
if rank == 0:
    memory.go()
rows = memory.collect()
memory.check(val_batches)
ev = stream_memory.StreamEval(rows, gate)
nll, mixed, hit = [], [], []
for s, b in enumerate(val_batches):
    a, m = ev.evaluate(s, b, lambda: model.forward(b.inputs), model)
    nll.append(a.numpy()); mixed.append(m.numpy())
    r, low, ptr = rows.chunk_rows(s), rows.chunk_low(s), rows.chunk_ptr(s)
    h = r.hit.clone()
    h[ptr["pos"]] = True
    for o in low:
        h |= low[o]["N"] > 0
    hit.append(h.numpy())
stream_memory.fit_dump(memory, lambda b: model.forward(b.inputs), model, cfg["chunk"], Staging(),
                       lambda s, console=False: print(s))
np.savez(os.path.join(tmp, f"rank{rank}.npz"), nll=np.stack(nll), mixed=np.stack(mixed), hit=np.stack(hit),
         inputs=np.stack([b.inputs_cpu for b in val_batches]))
dist.barrier()
if rank == 0:
    print(memory.stats())
memory.close()
dist.destroy_process_group()
'''


@needs_data
def test_toy_end_to_end_on_two_gloo_ranks(tmp_path, cpu_loader, monkeypatch):
    """The whole wrapper on 2 CPU ranks over the synthetic shards, with P2 and P3 and the FIT dump: create (the gate
    checked against the parts), tap, GO, collect, check, StreamEval (the model's side output), fit_dump. The val file
    plants a copy of a document inside the trained spans in rank 1's first chunk and a document past the stream in rank
    0's second chunk: a much lower loss on the first. fit_gate_v2.py then fits a gate on the dump."""
    (tmp_path / "data/fineweb10B").mkdir(parents=True)
    for f in sorted((DATA / "data/fineweb10B").glob("fineweb_train_*.bin")):
        (tmp_path / "data/fineweb10B" / f.name).symlink_to(f)
    cfg = dict(batch=2 * 4096, seq=2048, steps=12, chunk=8192, val_tokens=2 * 8192 * 2)
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    # Which documents the run trains on: the same loader, replayed here.
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    seen = []
    pattern = str(tmp_path / "data/fineweb10B/fineweb_train_*.bin")
    gen = data.distributed_data_generator(pattern, cfg["batch"], cfg["seq"], HostStaging(),
                                          on_spans=lambda f, s, e: seen.append((f, s, e)))
    for _ in range(cfg["steps"]):
        next(gen)
    gen.close()
    shard = read_shard(sorted(glob.glob(pattern))[0])
    f, starts, ends = seen[5]
    a, e = max(zip(starts[1], ends[1]), key=lambda s: s[1] - s[0])  # a long document rank 1 trained on
    assert f == 0 and e - a > 300 and shard[a] == BOS
    inside = shard[a:e]
    last = max(int(x) for _, _, en in seen for r in en for x in r)
    outside = shard[last + 10_000:last + 10_000 + inside.size].copy()  # never read
    outside[0] = BOS
    rng = np.random.default_rng(5)
    val = rng.integers(0, 50000, cfg["val_tokens"] + 5000).astype(np.uint16)
    p_in, p_out = cfg["chunk"] * 1 + 1000, cfg["chunk"] * 2 + 1000  # chunk 1: step 0, rank 1; chunk 2: step 1, rank 0
    val[p_in:p_in + inside.size] = inside
    val[p_out:p_out + outside.size] = outside
    write_shard(tmp_path / "data/fineweb10B/fineweb_val_000000.bin", val)
    # A gate that trusts every order (lambda = sigmoid(4)) and mixes the top components evenly: the shipped constants
    # are a proxy's, tuned to its features (test_the_shipped_gates_fit_the_default_parts checks they load).
    orders = stream_memory.chain_orders(True)
    d_top = 1 + len(stream_memory.top_cols())
    gate = dict(version=2, mode=stream_memory.GATE_MODE, orders=orders, fit={},
                w=[[4.0] + [0.0] * (35 if i == 0 else 36) for i in range(len(orders))], disc=[[-4.0] * 3] * len(orders),
                mu=[[0.0] * (35 if i == 0 else 36) for i in range(len(orders))],
                sd=[[1.0] * (35 if i == 0 else 36) for i in range(len(orders))],
                top=dict(comps=list(stream_memory.TOP_COMPS), cols=stream_memory.top_cols(), W=[[0.0] * 3] * d_top,
                         mu=[0.0] * (d_top - 1), sd=[1.0] * (d_top - 1)))
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    env = {k: v for k, v in os.environ.items() if k not in FLAGS}
    env.update(STREAM_RETRIEVAL="1", STREAM_RETRIEVAL_LOW="1", STREAM_RETRIEVAL_FIT=str(tmp_path / "fitdir"),
               STREAM_RETRIEVAL_FIT_K="2", STREAM_RETRIEVAL_THREADS="2", STREAM_RETRIEVAL_GATE=str(tmp_path / "gate.json"))
    procs = [subprocess.Popen([sys.executable, "-c", WORKER, str(ROOT), str(tmp_path), str(r)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env) for r in range(2)]
    outs = [p.communicate(timeout=600) for p in procs]
    assert all(p.returncode == 0 for p in procs), "\n".join(f"rank {r}: {o[1][-3000:]}" for r, o in enumerate(outs))
    assert "stream retrieval: memory" in outs[0][0] and "P2 tables" in outs[0][0] and "P3 rows" in outs[0][0]
    assert "stream retrieval fit:" in outs[0][0]
    r0, r1 = np.load(tmp_path / "rank0.npz"), np.load(tmp_path / "rank1.npz")
    assert np.array_equal(r1["inputs"][0][1000:1000 + inside.size], inside.astype(np.int32))
    uniform = math.log(50257.0)
    assert np.allclose(np.stack([r0["nll"], r1["nll"]]), uniform)
    copied = slice(1000 + 40, 1000 + inside.size - 1)  # deep matches of the trained copy: every part predicts the target
    assert np.all(r1["mixed"][0][copied] < 0.5)
    # Nothing touched: the model's NLL bit-identical; touched positions only differ from it
    for r in (r0, r1):
        assert np.array_equal(r["mixed"][~r["hit"]], r["nll"][~r["hit"]]) and np.isfinite(r["mixed"]).all()
    # The FIT dump: the run's last 2 batches of every rank, P1 + P2 + P3, and the model's outputs; fit_gate_v2 fits it
    fit = stream_memory.read_fit(str(tmp_path / "fitdir"))
    assert fit["k"] == 2 and fit["n"] == 2 * 4096 and fit["low"] is not None and fit["ptr"] is not None
    assert fit["steps"] == cfg["steps"]  # the dump says which step count its last batches are of
    for r in range(2):
        lm = np.load(tmp_path / f"fitdir/fit.lm.rank{r}.npz")
        assert lm["nll"].shape == (fit["n"],) and lm["top_v"].shape == (fit["n"], len(stream_memory.chain_orders(True)) + 3)
    fitted = subprocess.run([sys.executable, str(ROOT / "tools/stream_retrieval/fit_gate_v2.py"), str(tmp_path / "fitdir"),
                             "--maxiter", "60", "--out", str(tmp_path / "fitted.json")], capture_output=True, text=True,
                            env=env, cwd=ROOT)
    assert fitted.returncode == 0, fitted.stderr[-3000:]
    spec = json.loads((tmp_path / "fitted.json").read_text())
    assert spec["orders"] == stream_memory.chain_orders(True) and spec["top"]["cols"] == stream_memory.top_cols()
    assert spec["fit"]["total_steps"] == cfg["steps"] and "proxy" not in spec["fit"]
