"""CPU tests for the stream-only retrieval memory (STREAM_RETRIEVAL=1: track_1_short/stream_memory.py and .c).

They test the streamret arm: the stack with tools/stream_retrieval/apply_overlay.sh applied (the stack itself carries
no retrieval code). tools/tests/test_stream_retrieval_arm.py builds that tree from the working tree and runs this
file inside it; in an arm (make_streamret_arm.sh) it also runs directly: python -m pytest tools/stream_retrieval -q.
The helper is compiled with cc. Most tests write their own small shards; the stream, tap and end-to-end tests also
use the FineWeb-format shards under $SPEEDRUN_TEST_DATA/data/fineweb10B (synthetic is fine).

  1 the memory is exactly the trained stream      6 determinism (query threads, message batching)
  2 no leakage (val, unread shard bytes)          7 flag off: no tap/process/rows; mix() identity; loader unchanged
  3 exact against a brute-force reference         8 toy end-to-end on 2 gloo ranks
  4 causality                                     9 failures raise
  5 normalisation                                 + the FIT path and fit_gate.py
"""
import ast
import glob
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
from track_1_short.stream_memory import StreamMemory, mix  # noqa: E402

DATA = Path(os.environ.get("SPEEDRUN_TEST_DATA", "/home/user/work/synthdata"))
HAVE_DATA = (DATA / "data/fineweb10B/fineweb_val_000000.bin").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="no FineWeb shards at $SPEEDRUN_TEST_DATA")
BOS, SEP = 50256, 0xFFFF
KEY, MAXLEN, CAP, MAXVISIT, LEVELS = 6, 32, 32, 128, (6, 8, 12, 16, 24, 32)
M64 = (1 << 64) - 1


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
                        dump_path=str(tmp_path / f"{name}.dump"), features_path=str(tmp_path / f"{name}.feat"),
                        readlog_path=str(tmp_path / f"{name}.reads"), **kw)


def finish(memory: StreamMemory, tmp_path, name="m"):
    """GO, then (rows in global val order [n, 2], features [n, 4], the stream dump, the read log)."""
    memory.go()
    memory.collect()
    rows = global_rows(memory)
    feats = np.fromfile(tmp_path / f"{name}.feat", dtype=np.uint32).reshape(-1, 4)
    dump = np.fromfile(tmp_path / f"{name}.dump", dtype=np.uint16)
    reads = (tmp_path / f"{name}.reads").read_text().split("\n")
    memory.close()
    return rows, feats, dump, [r.split() for r in reads if r]


def global_rows(memory: StreamMemory) -> np.ndarray:
    """f32 [world][val_steps][chunk][2] -> [val_tokens, 2]: chunk c = step * world + rank covers c * chunk + i."""
    r = memory.all_rows
    return np.ascontiguousarray(r.transpose(1, 0, 2, 3)).reshape(-1, 2).copy()


def send(memory: StreamMemory, file_idx: int, spans_per_rank):
    memory.on_spans(file_idx, [[a for a, _ in s] for s in spans_per_rank], [[e for _, e in s] for s in spans_per_rank])


# ------------------------------------------------------------------------------------------------ reference

def fmix64(k):
    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & M64
    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & M64
    return k ^ (k >> 33)


def bucket(w, bits):
    lo = int(w[0]) | int(w[1]) << 16 | int(w[2]) << 32 | int(w[3]) << 48
    hi = int(w[4]) | int(w[5]) << 16
    return fmix64(lo ^ fmix64((hi + 0x9E3779B97F4A7C15) & M64)) >> (64 - bits)


def segment_runs(x: np.ndarray, chunk: int) -> np.ndarray:
    run = np.zeros(x.size, dtype=np.int64)
    r = 0
    for t, v in enumerate(x):
        r = 1 if (t % chunk == 0 or v == BOS) else r + 1
        run[t] = r
    return run


def reference_features(stream: np.ndarray, x: np.ndarray, y: np.ndarray, run: np.ndarray, hash_bits: int | None):
    """Section 3 of the design in plain Python. stream: the memory's tok[] including tok[0] = SEP.
    hash_bits=None: candidates are ALL matching positions, most recent first (independent of the hash);
    else the bucket walk with the helper's hash, MAXVISIT and CAP."""
    seg = np.zeros(stream.size, dtype=np.int64)
    r = 0
    for j, v in enumerate(stream):
        r = 0 if v == SEP else r + 1
        seg[j] = r
    inserted = [j for j in range(stream.size - 1) if seg[j] >= KEY and stream[j + 1] != SEP]
    chains: dict = {}
    for j in inserted:  # in stream order; walked in reverse
        key = tuple(stream[j - KEY + 1:j + 1]) if hash_bits is None else bucket(stream[j - KEY + 1:j + 1], hash_bits)
        chains.setdefault(key, []).append(j)
    out = np.zeros((x.size, 4), dtype=np.int64)
    for t in range(x.size):
        if run[t] < KEY:
            continue
        ctx = x[t - KEY + 1:t + 1]
        key = tuple(ctx) if hash_bits is None else bucket(ctx, hash_bits)
        cands, visited = [], 0
        for j in reversed(chains.get(key, [])):
            if len(cands) >= CAP or (hash_bits is not None and visited >= MAXVISIT):
                break
            visited += 1
            if not np.array_equal(stream[j - KEY + 1:j + 1], ctx):
                continue
            lim, length = min(run[t], MAXLEN), KEY
            while length < lim and stream[j - length] == x[t - length]:
                length += 1
            cands.append((length, int(stream[j + 1])))
        if not cands:
            continue
        maxl = max(c[0] for c in cands)
        lstar = max(lv for lv in LEVELS if lv <= maxl)
        nxt = [c[1] for c in cands if c[0] >= lstar]
        counts = {v: nxt.count(v) for v in nxt}
        out[t] = (len(nxt), counts.get(int(y[t]), 0), max(counts.values()), lstar)
    return out


def reference_rows(feats: np.ndarray, w=stream_memory.W) -> np.ndarray:
    n, c, m, lstar = (feats[:, k].astype(np.float64) for k in range(4))
    hit = n > 0
    z = w[0] + w[1] * np.log2(np.maximum(n, 1)) + w[2] * m / np.maximum(n, 1) + w[3] * np.log2(np.maximum(lstar, 1))
    lam = 1 / (1 + np.exp(-z))
    return np.stack([np.where(hit, 1 - lam, 1.0), np.where(hit, lam * c / np.maximum(n, 1), 0.0)], 1)


def toy_corpus(rng, n_docs, vocab, copy_from=None, max_len=120):
    """Documents [BOS, ...] over a tiny vocabulary (so contexts repeat with varied continuations), a third of them
    starting with a copy of an earlier document's (or copy_from's) prefix, so long matches occur too."""
    docs = []
    for _ in range(n_docs):
        body = list(rng.integers(0, vocab, rng.integers(3, max_len)))
        pool = docs + (copy_from or [])
        if pool and rng.random() < 0.35:
            src = pool[rng.integers(len(pool))]
            body = list(src[1:rng.integers(2, len(src) + 1)]) + body[:rng.integers(0, 20)]
        docs.append(np.array([BOS, *body], dtype=np.uint16))
    return docs


@pytest.fixture
def toy(tmp_path):
    """A train shard of toy documents, spans over it (whole documents and cut ones) in 4 steps, and a val shard."""
    rng = np.random.default_rng(0)
    docs = toy_corpus(rng, 800, vocab=4)
    repeated = np.array([BOS, *rng.integers(0, 4, 70)], dtype=np.uint16)  # 40 copies: CAP binds at L* = 32
    docs = docs[:500] + [repeated] * 40 + docs[500:] + toy_corpus(rng, 100, vocab=4)  # the last 100: never spanned
    shard = write_shard(tmp_path / "fineweb_train_000001.bin", np.concatenate(docs))
    starts = np.cumsum([0] + [d.size for d in docs])
    spans = []
    for i, d in enumerate(docs[:840]):
        a = int(starts[i])
        cut = int(rng.integers(2, d.size + 1)) if rng.random() < 0.2 and d.size > 2 and i % 7 else d.size
        spans.append((a, a + cut))
    val_docs = toy_corpus(rng, 400, vocab=4, copy_from=docs)
    val_docs[150:150] = [repeated, repeated]
    val = np.concatenate(val_docs)
    assert val.size > 2 * 4096 + 1
    val_file = write_shard(tmp_path / "fineweb_val_000000.bin", val)
    steps = [spans[0:150], spans[150:420], spans[420:421], spans[421:]]
    return dict(shard=shard, val_file=val_file, val=read_shard(val_file), steps=steps, spans=spans)


def run_toy(tmp_path, toy, name="m", grouping=None, **kw):
    groups = grouping or toy["steps"]
    memory = make_memory(tmp_path, [toy["shard"]], toy["val_file"], len(groups), name=name, **kw)
    for g in groups:
        send(memory, 0, [g])
    return finish(memory, tmp_path, name)


def expected_stream(shard_tokens, spans) -> np.ndarray:
    parts = []
    for a, e in spans:
        parts += [shard_tokens[a:e], np.array([SEP], dtype=np.uint16)]
    return np.concatenate(parts)


# ------------------------------------------------------------------------------------------------ 3. exactness

@pytest.mark.parametrize("hash_bits, chunk", [(22, 4096), (5, 4096), (22, 5000)])
def test_rows_equal_a_brute_force_reference(tmp_path, toy, hash_bits, chunk):
    """chunk 5000: the helper's 4096-position query blocks start inside segments and chunks (each block finds its
    first position's segment by looking back)."""
    rows, feats, dump, _ = run_toy(tmp_path, toy, hash_bits=hash_bits, chunk=chunk)
    stream = np.concatenate([[SEP], dump]).astype(np.uint16)
    assert np.array_equal(dump, expected_stream(read_shard(toy["shard"]), toy["spans"]))
    n = rows.shape[0]
    x, y = toy["val"][:n], toy["val"][1:n + 1]
    run = segment_runs(x, chunk)
    ref = reference_features(stream, x, y, run, hash_bits)
    assert np.array_equal(feats.astype(np.int64), ref), np.flatnonzero((feats != ref).any(1))[:10]
    if hash_bits == 22:  # few collisions: the bucket walk finds exactly the most recent CAP matches
        assert np.array_equal(ref, reference_features(stream, x, y, run, None))
    else:  # 32 buckets: the MAXVISIT bound decides, and the walk still matches the reference exactly
        assert not np.array_equal(ref, reference_features(stream, x, y, run, None))
    np.testing.assert_allclose(rows, reference_rows(feats), rtol=0, atol=1e-6)
    hit = feats[:, 0] > 0
    assert np.all(rows[~hit] == (1.0, 0.0))
    if chunk % 4096:  # some block starts mid-segment with a hit right there: the look-back decided it
        assert any(feats[b, 0] > 0 and run[b] >= KEY for b in range(4096, n, 4096))
    if hash_bits == 22:  # the toy data exercises every part of the rule
        assert set(feats[hit, 3]) == set(LEVELS) and (feats[:, 0] == CAP).any() and (feats[hit, 1] == 0).any()
        assert (feats[hit, 2] > 1).any() and ((feats[hit, 1] > 0) & (feats[hit, 1] < feats[hit, 0])).any()


# ------------------------------------------------------------------------------------------------ 4. causality

def test_causality(tmp_path, toy):
    base, feats, _, _ = run_toy(tmp_path, toy, name="base")
    val = toy["val"]
    t = 4096 + 1500  # in the second chunk
    rng = np.random.default_rng(1)
    future = val.copy()
    future[t + 2:] = rng.integers(0, 4, future.size - t - 2)
    future[t + 2::97] = BOS
    write_shard(toy["val_file"], future)
    changed, _, _, _ = run_toy(tmp_path, toy, name="future")
    assert np.array_equal(base[:t + 1].view(np.uint32), changed[:t + 1].view(np.uint32))
    assert not np.array_equal(base[t + 1:], changed[t + 1:])
    # A matched position whose target is changed: a (lambda) stays, b changes only through C.
    hits = [p for p in range(t, 2 * 4096 - 1) if feats[p, 0] > 1 and feats[p, 1] > 0]
    p = hits[0]
    target = future.copy()
    target[:] = val
    target[p + 1] = (int(val[p + 1]) + 1) % 4
    write_shard(toy["val_file"], target)
    rows2, feats2, _, _ = run_toy(tmp_path, toy, name="target")
    assert rows2[p, 0] == base[p, 0] and tuple(feats2[p, [0, 2, 3]]) == tuple(feats[p, [0, 2, 3]])
    lam = 1 - float(base[p, 0])
    assert abs(rows2[p, 1] - lam * feats2[p, 1] / feats2[p, 0]) < 1e-6
    assert np.array_equal(base[:p + 1, 0], rows2[:p + 1, 0]) and np.array_equal(base[:p, 1], rows2[:p, 1])


# ------------------------------------------------------------------------------------------------ 5. normalisation

def test_rows_sum_to_at_most_one_over_every_target(tmp_path):
    """Val blocks [BOS, ctx (10 tokens), y] for every y in a 64-token vocabulary and BOS: the same context with every
    possible target. a is the same for all y and sum_y b_y = lambda, so sum_y (a p(y) + b_y) <= 1."""
    rng = np.random.default_rng(2)
    vocab = 64
    contexts = [rng.integers(0, vocab, 10).astype(np.uint16) for _ in range(6)]
    docs = []
    for k, ctx in enumerate(contexts):
        conts = rng.integers(0, vocab, 1 + k)  # k + 1 distinct-ish continuations, repeated: M > 1
        for i in range(12 + 5 * k):
            pre = rng.integers(0, vocab, rng.integers(0, 30)) if i % 3 else []
            docs.append(np.array([BOS, *pre, *ctx, conts[i % conts.size], *rng.integers(0, vocab, 5)], dtype=np.uint16))
    docs += [np.array([BOS, *rng.integers(0, vocab, 50)], dtype=np.uint16) for _ in range(200)]
    shard = write_shard(tmp_path / "fineweb_train_000001.bin", np.concatenate(docs))
    starts = np.cumsum([0] + [d.size for d in docs])
    targets = [*range(vocab), BOS]
    blocks = [np.array([BOS, *ctx, y], dtype=np.uint16) for ctx in contexts for y in targets]
    chunk = 12 * 342
    val = np.concatenate(blocks + [np.full(2 * chunk, 7, dtype=np.uint16)])
    val_file = write_shard(tmp_path / "fineweb_val_000000.bin", val)
    memory = make_memory(tmp_path, [shard], val_file, 1, chunk=chunk)
    send(memory, 0, [[(int(starts[i]), int(starts[i + 1])) for i in range(len(docs))]])
    rows, feats, _, _ = finish(memory, tmp_path)
    for k in range(len(contexts)):
        pos = [12 * (k * len(targets) + i) + 10 for i in range(len(targets))]  # the last context token
        a, b = rows[pos, 0].astype(np.float64), rows[pos, 1].astype(np.float64)
        assert feats[pos[0], 0] > 0 and np.all(a == a[0]) and abs(b.sum() - (1 - a[0])) < 1e-6
        for scale in (1.0, 0.7):
            p = rng.dirichlet(np.ones(len(targets))) * scale
            total = (a * p + b).sum()
            assert total <= 1 + 1e-6 and (scale < 1 or abs(total - 1) < 1e-6)


# ------------------------------------------------------------------------------------------------ 6. determinism

def test_rows_are_bit_identical_across_threads_and_batching(tmp_path, toy):
    ref = run_toy(tmp_path, toy, name="ref", threads=1)
    flat = [s for g in toy["steps"] for s in g]
    regrouped = [flat[:3], flat[3:500], flat[500:790], flat[790:]]
    assert len(flat) == 840
    for name, kw in (("q7", dict(threads=7)), ("q2", dict(threads=2)), ("batched", dict(grouping=regrouped))):
        got = run_toy(tmp_path, toy, name=name, **kw)
        assert got[0].tobytes() == ref[0].tobytes(), name
        assert np.array_equal(got[1], ref[1]) and np.array_equal(got[2], ref[2]), name


# ------------------------------------------------------------------------------------------------ 1 and 2: the stream

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
            own = np.concatenate([batch.inputs_cpu, batch.targets_cpu[-1:].numpy()]).astype(np.uint16)
            expected.append(own)
    for loader in loaders:
        loader.close()
    _, _, dump, reads = finish(memory, tmp_path)
    assert memory.sent == steps
    # The stream with the separators removed is the concatenation of every rank's (inputs + last target), step by step.
    assert np.array_equal(dump[dump != SEP], np.concatenate(expected))
    # And every separator ends a document span that starts at a BOS.
    seps = np.flatnonzero(dump == SEP)
    assert dump[0] == BOS and np.all(dump[seps[:-1] + 1] == BOS) and seps[-1] == dump.size - 1
    assert len({r[1] for r in reads if r[0] == "train"}) >= 3  # across shard switches


def test_no_val_or_unread_shard_token_enters_the_memory(tmp_path, cpu_loader, monkeypatch):
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
    memory = make_memory(tmp_path, paths, val, steps, world=8, stream_tokens=steps * (first.batch_size + 8), hash_bits=20)
    replay(lambda f, starts, ends: (record(f, starts, ends), memory.on_spans(f, starts, ends)))
    assert spans == first_spans
    rows, feats, dump, reads = finish(memory, tmp_path)
    body = dump[(dump != SEP) & (dump != BOS)]
    assert body.max() < 20000, "a token of the unread shard tail or of val entered the memory"
    # The helper read exactly the spans, in order, and the val prefix: nothing else.
    train_reads = [(int(f), (int(off) - 1024) // 2, (int(off) - 1024 + int(n)) // 2) for kind, f, off, n in
                   (r for r in reads if r[0] == "train")]
    assert train_reads == spans
    assert [r for r in reads if r[0] != "train"] == [["val", "0", "1024", str(2 * (memory.world * 4096 * 2 + 1))]]
    assert not (feats[:, 0] > 0).any() and np.all(rows == (1.0, 0.0))  # val shares no token with the memory


# ------------------------------------------------------------------------------------------------ 7. flag off

@pytest.mark.parametrize("value, enabled", [(None, False), ("0", False), ("", False), ("1", True)])
def test_the_flag(value, enabled):
    env = {k: v for k, v in os.environ.items() if k != "STREAM_RETRIEVAL"}
    if value is not None:
        env["STREAM_RETRIEVAL"] = value
    out = subprocess.run([sys.executable, "-c", "from track_1_short import stream_memory as m; print(m.ENABLED)"],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(enabled)


def test_trainer_touches_the_memory_only_behind_the_flag():
    """train_gpt.py: the memory exists only under STREAM_RETRIEVAL=1, only the timed loader is tapped, and every use
    of the memory or its rows sits under a guard on them, so with the flag unset no tap, process or row exists."""
    source = (ROOT / "train_gpt.py").read_text()
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    guards = ("memory is not None", "stream_rows is not None", "stream_rows is None", "stream_memory.ENABLED")

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
            and (n.value.id in ("memory", "stream_rows") or (n.value.id == "stream_memory" and n.attr != "ENABLED"))]
    assert len(uses) >= 8
    unguarded = [ast.unparse(n) for n in uses if not guarded(n)]
    assert not unguarded, unguarded
    assert "memory = None\n    if stream_memory.ENABLED:" in source
    assert "warmup_batches = ScheduledBatches(train_loader(), " in source
    assert source.count("on_spans=memory.on_spans") == 1


def test_python_mirrors_the_c_constants():
    """The header words, message kinds, states, offsets and levels are written twice (C and Python): they agree."""
    import re
    c = (ROOT / "track_1_short/stream_memory.c").read_text()

    def enum(first):
        body = re.search(r"enum \{ *" + first + r"([^}]*)\}", c, re.S).group(0)
        names, value = {}, -1
        for item in re.sub(r"enum \{|\}", "", body).split(","):
            name, _, expr = item.strip().partition("=")
            name, expr = name.strip(), expr.strip()
            if not name:
                continue
            value = eval(expr.replace("NLEVELS", str(len(LEVELS))), {}, dict(names)) if expr else value + 1
            names[name] = value
        return names

    for first in ("H_MAGIC", "MSG_STEP", "ST_STARTING"):
        for name, value in enum(first).items():
            if name != "H_NWORDS":
                assert getattr(stream_memory, name) == value, name
    for name in ("ERRMSG_OFFSET", "CHECKSUM_OFFSET", "ROWS_OFFSET", "HEADER_MAGIC"):
        assert int(re.search(rf"#define {name} (\w+)", c).group(1).rstrip("ul"), 0) == getattr(stream_memory, name), name
    levels = re.search(r"LEVELS\[NLEVELS\] = \{([^}]*)\}", c).group(1)
    assert tuple(int(v) for v in levels.split(",")) == stream_memory.LEVELS == LEVELS
    assert int(re.search(r"#define KEY (\d+)", c).group(1)) == KEY and int(re.search(r"#define CAP (\d+)", c).group(1)) == CAP
    assert enum("H_MAGIC")["H_NWORDS"] * 8 <= stream_memory.ERRMSG_OFFSET  # the header words fit before the message


def test_the_helper_requires_the_gate_constants(tmp_path):
    rows = tmp_path / "rows"
    rows.write_bytes(bytes(4096 + 2 * 4096 * 8))
    out = subprocess.run([str(stream_memory.build_helper()), f"rows={rows}", "val=/dev/null", "world=1",
                          "val_tokens=8192", "chunk=4096", "cap=16", "--", "/dev/null"], capture_output=True, text=True)
    assert out.returncode == 2 and "w=w0,w1,w2,w3 is required" in out.stderr


def test_mix_is_the_identity_where_nothing_matched():
    torch.manual_seed(0)
    nll = torch.rand(10000) * 12
    rows = torch.stack([torch.ones(10000), torch.zeros(10000)], -1)
    hit = torch.rand(10000) < 0.1
    lam = torch.rand(10000)
    c_over_n = (torch.rand(10000) * 4).floor() / 4
    rows[hit] = torch.stack([1 - lam, lam * c_over_n], -1)[hit]
    out = mix(nll, rows)
    assert torch.equal(out[~hit], nll[~hit])  # bit-identical
    p = (torch.exp(-nll) - 1e-9).clamp_min(0)
    want = -torch.log((1 - lam) * p + lam * c_over_n + 1e-9)
    torch.testing.assert_close(out[hit], want[hit])
    assert torch.equal(mix(nll, torch.stack([torch.ones(10000), torch.zeros(10000)], -1)), nll)


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


# ------------------------------------------------------------------------------------------------ 9. failures

def test_wrong_step_count_is_refused(tmp_path, toy):
    memory = make_memory(tmp_path, [toy["shard"]], toy["val_file"], 5)
    for g in toy["steps"]:
        send(memory, 0, [g])
    memory.go()
    with pytest.raises(RuntimeError, match="GO for 4 steps after 4 step messages; the memory expects 5"):
        memory.collect()
    memory.close()


def test_a_dead_helper_raises(tmp_path, toy):
    memory = make_memory(tmp_path, [toy["shard"]], toy["val_file"], 4)
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


# ------------------------------------------------------------------------------------------------ 8. end to end

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
nll = torch.full((cfg["chunk"],), math.log(50257.0))  # a uniform model
mixed = torch.stack([stream_memory.mix(nll, rows[s]) for s in range(len(val_batches))])
np.savez(os.path.join(tmp, f"rank{rank}.npz"), rows=rows.numpy(), mixed=mixed.numpy(),
         inputs=np.stack([b.inputs_cpu for b in val_batches]))
dist.barrier()
if rank == 0:
    print(memory.stats())
memory.close()
dist.destroy_process_group()
'''


@needs_data
def test_toy_end_to_end_on_two_gloo_ranks(tmp_path, cpu_loader, monkeypatch):
    """The whole wrapper (create, tap, GO, collect, check, mix) on 2 CPU ranks over the synthetic shards. The val
    file plants a copy of a document inside the trained spans in rank 1's first chunk and a document past the stream
    in rank 0's second chunk: hits and a lower loss on the first, none on the second."""
    import json
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
    procs = [subprocess.Popen([sys.executable, "-c", WORKER, str(ROOT), str(tmp_path), str(r)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for r in range(2)]
    outs = [p.communicate(timeout=300) for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    assert "stream retrieval: memory" in outs[0][0]
    r0, r1 = np.load(tmp_path / "rank0.npz"), np.load(tmp_path / "rank1.npz")
    assert np.array_equal(r1["inputs"][0][1000:1000 + inside.size], inside.astype(np.int32))
    uniform = math.log(50257.0)
    copied = slice(1000 + KEY - 1, 1000 + inside.size - 1)  # positions with a full key whose next token is in the copy
    rows_in = r1["rows"][0][copied]
    assert np.all(rows_in[:, 1] > 0.05) and np.all(r1["mixed"][0][copied] < uniform - 2)
    assert np.all(r1["rows"][0][1000 + 40:1000 + inside.size - 1, 1] > 0.9)  # L* = 32: lambda ~ 0.97
    assert np.all(r0["rows"][1] == (1.0, 0.0)) and np.allclose(r0["mixed"][1], uniform)
    # Everywhere else nothing matched (random tokens), and every rank got its own chunks.
    others = np.ones((2, 2, cfg["chunk"]), dtype=bool)
    others[1, 0, copied] = False
    assert np.all(np.stack([r0["rows"], r1["rows"]])[others] == (1.0, 0.0))


# ------------------------------------------------------------------------------------------------ STREAM_RETRIEVAL_FIT

def test_fit_features_and_gate_fit(tmp_path, toy):
    """FIT steps after GO are queried, never inserted, as val-shaped chunks (inputs buf[:-1], targets buf[1:] per
    batch); fit_gate.py recovers a gate from features and NLL."""
    memory = make_memory(tmp_path, [toy["shard"]], toy["val_file"], 2, fit_path=str(tmp_path / "fit"), chunk=4096)
    send(memory, 0, [toy["spans"][:400]])
    send(memory, 0, [toy["spans"][400:700]])
    memory.go()
    memory.collect()
    held = toy["spans"][700:]
    assert len(held) >= 80
    shard = read_shard(toy["shard"])
    memory.fitting = True
    batches = [held[i:i + 10] for i in range(0, 80, 10)]
    for g in batches:
        send(memory, 0, [g])
    memory.fit_go()
    memory._wait(stream_memory.ST_FIT_DONE, 30)
    assert int(memory.hdr[stream_memory.H_STEPS]) == 2  # nothing inserted
    feats = np.fromfile(tmp_path / "fit", dtype=np.uint32).reshape(-1, 4)
    dump = np.fromfile(tmp_path / "m.dump", dtype=np.uint16)
    memory.close()
    xs, ys = [], []
    for g in batches:
        buf = np.concatenate([shard[a:e] for a, e in g])
        xs.append(buf[:-1])
        ys.append(buf[1:])
    x, y = np.concatenate(xs), np.concatenate(ys)
    ref = reference_features(np.concatenate([[SEP], dump]).astype(np.uint16), x, y, segment_runs(x, 4096), 16)
    assert feats.shape[0] == x.size and np.array_equal(feats.astype(np.int64), ref)
    assert (feats[:, 0] > 0).mean() > 0.3

    # fit_gate.py on synthetic NLL drawn so that the target is more likely when the memory agrees.
    sys.path.insert(0, str(ROOT / "tools/stream_retrieval"))
    import fit_gate
    rng = np.random.default_rng(6)
    n = 4096 * 4
    f = np.zeros((n, 4), dtype=np.uint32)
    hit = rng.random(n) < 0.4
    f[hit, 0] = rng.integers(1, 33, hit.sum())
    f[hit, 3] = rng.choice(LEVELS, hit.sum())
    f[hit, 1] = (rng.random(hit.sum()) < f[hit, 3] / 40) * f[hit, 0]
    f[hit, 2] = np.maximum(f[hit, 1], 1)
    f.tofile(tmp_path / "syn")
    np.save(tmp_path / "syn.rank0.npy", rng.uniform(1, 8, (4, 4096)).astype(np.float32))
    w = fit_gate.main([str(tmp_path / "syn"), "--world", "1", "--chunk", "4096"])
    assert w.shape == (4,) and w[3] > 0  # longer matches earn more weight


def test_pseudo_val_batches_tile_training_batches():
    rng = np.random.default_rng(7)

    class B:
        def __init__(self):
            buf = np.concatenate([[BOS], rng.integers(0, 50000, 1024)]).astype(np.int64)
            self.inputs_cpu = buf[:-1].astype(np.int32)
            self.targets_cpu = torch.from_numpy(buf[1:])
    batches = [B() for _ in range(8)]
    out = stream_memory.pseudo_val_batches(batches, 4096, HostStaging())
    assert len(out) == 2
    assert np.array_equal(out[1].inputs_cpu, np.concatenate([b.inputs_cpu for b in batches[4:]]))
    assert torch.equal(out[1].targets, torch.cat([b.targets_cpu for b in batches[4:]]))
    cum = out[0].cum_seqlens
    assert cum[:5].tolist() == [0, 0, 1024, 2048, 3072] and cum[5:].eq(4096).all() and cum.dtype == torch.int32
