"""CPU tests of stream retrieval v2's P1 part: the helper's records (arm/track_1_short/stream_memory.c, query side),
their parsing and the GPU-side gate (stream_memory.py: Rows, chain_features, mix_v2), and fit_gate_v2.py.

Standalone: stream_memory.py is loaded from the overlay by path (no stack tree, no track_1_short), the helper is
compiled with cc into a temporary directory (stream_memory.build_helper), every shard is written here.
  python -m pytest tools/stream_retrieval/test_stream_memory_v2.py -q

  records   exact against a brute-force reference (hash walk, MAXVISIT, CAP, levels, recent / longest, FULLCAP);
            bit-identical across query threads and message batching; never read the target (changing val[t + 1:]
            leaves records <= t byte-identical); FIT = the run's last K batches against the frozen memory
  gate      features equal a numpy port of rg/combo/ds.py; causal; the chain sums to 1 over the vocabulary;
            identity where nothing matched; lm_top_features; mix_v1; the shipped constants load
  fit       fit_gate_v2.py end to end on a helper FIT dump: gain, held-out, JSON and module round trip
"""
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
MODULE = HERE / "arm/track_1_short/stream_memory.py"
_spec = importlib.util.spec_from_file_location("stream_memory_v2_under_test", MODULE)
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)
sys.path.insert(0, str(HERE))
import fit_gate_v2  # noqa: E402

BOS, SEP = 50256, 0xFFFF
KEY, MAXLEN, CAP, MAXVISIT, FULLCAP, LEVELS = 6, 32, 32, 128, 4096, (6, 7, 8, 10, 12, 16, 24, 32)
M64 = (1 << 64) - 1
torch.set_num_threads(2)


# ------------------------------------------------------------------------------------------------ data

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


def toy_corpus(rng, n_docs, vocab, copy_from=None, max_len=120):
    """Documents [BOS, ...] over a tiny vocabulary (contexts repeat with varied continuations), a third of them
    starting with a copy of an earlier document's (or copy_from's) prefix, so that long matches occur too."""
    docs = []
    for _ in range(n_docs):
        body = list(rng.integers(0, vocab, rng.integers(3, max_len)))
        pool = docs + (copy_from or [])
        if pool and rng.random() < 0.35:
            src = pool[rng.integers(len(pool))]
            body = list(src[1:rng.integers(2, len(src) + 1)]) + body[:rng.integers(0, 20)]
        docs.append(np.array([BOS, *body], dtype=np.uint16))
    return docs


def equal_batches(docs, starts, first_doc, world, length):
    """Per rank, consecutive documents from first_doc whose spans total exactly `length` tokens (the last one cut),
    as the loader gives every rank batch_size / world + 1 tokens. Returns (spans per rank, next document)."""
    out, d = [], first_doc
    for _ in range(world):
        spans, have = [], 0
        while have < length:
            a = int(starts[d])
            take = min(docs[d].size, length - have)
            spans.append((a, a + take))
            have += take
            d += 1
        out.append(spans)
    return out, d


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """Two ranks. A train shard of toy documents (vocabulary 4); 4 ordinary steps (all spans under rank 0, some
    documents cut), then 2 FIT-able steps (each rank exactly 300 tokens); a long document (5,000 tokens) trained once
    and copied into val, so full match lengths reach FULLCAP; val = toy documents copying trained ones."""
    tmp = tmp_path_factory.mktemp("toy")
    rng = np.random.default_rng(0)
    docs = toy_corpus(rng, 700, vocab=4)
    repeated = np.array([BOS, *rng.integers(0, 4, 70)], dtype=np.uint16)  # 40 copies: CAP binds at L* = 32
    long_doc = np.array([BOS, *rng.integers(0, 4, 5000)], dtype=np.uint16)
    docs = docs[:400] + [repeated] * 40 + [long_doc] + docs[400:] + toy_corpus(rng, 80, vocab=4)
    shard = write_shard(tmp / "fineweb_train_000001.bin", np.concatenate(docs))
    starts = np.cumsum([0] + [d.size for d in docs])
    plain = []
    for i in range(0, 600):
        a, d = int(starts[i]), docs[i]
        cut = int(rng.integers(2, d.size + 1)) if rng.random() < 0.2 and d.size > 2 and i % 7 and i not in range(400, 441) \
            else d.size
        plain.append((a, a + cut))
    steps = [[plain[0:150], []], [plain[150:380], []], [plain[380:381], []], [plain[381:], []]]
    fit1, nxt = equal_batches(docs, starts, 600, 2, 300)
    fit2, nxt = equal_batches(docs, starts, nxt, 2, 300)
    steps += [fit1, fit2]
    trained = [d for d in docs[:nxt]]
    val_docs = toy_corpus(rng, 330, vocab=4, copy_from=trained)
    val_docs[100:100] = [repeated, repeated]
    val_docs[200:200] = [docs[600], docs[nxt - 3]]  # documents of the FIT batches: in the memory for val, not for FIT
    val = np.concatenate([long_doc[:4500]] + val_docs)  # from a chunk start: its runs reach FULLCAP
    assert val.size > 2 * 2 * 5000 + 1 and nxt < len(docs) - 20
    val_file = write_shard(tmp / "fineweb_val_000000.bin", val)
    return dict(tmp=tmp, shard=shard, val_file=val_file, val=read_shard(val_file), steps=steps, docs=docs)


def run_helper(tmp, toy, *, name, steps=None, world=2, chunk=4096, val_steps=2, fit_k=0, threads=3, hash_bits=16,
               val_file=None):
    steps = steps or toy["steps"]
    stream_tokens = sum(e - a for st in steps for r in st for a, e in r)
    memory = sm.StreamMemory(train_files=[str(toy["shard"])], val_file=str(val_file or toy["val_file"]),
                             total_steps=len(steps), stream_tokens=stream_tokens, world=world, rank=0, master=True,
                             val_tokens=world * chunk * val_steps, chunk=chunk, device="cpu", threads=threads,
                             hash_bits=hash_bits, rows_dir=str(tmp), dump_path=str(tmp / f"{name}.dump"),
                             fit_path=str(tmp / f"{name}.fit") if fit_k else None, fit_k=fit_k or sm.FIT_K)
    for st in steps:
        memory.on_spans(0, [[a for a, _ in r] for r in st], [[e for _, e in r] for r in st])
    memory.go()
    memory.collect()
    # every chunk's records in global val order: chunk c = step * world + rank lives in region (rank, step)
    recs = []
    for c in range(world * val_steps):
        s, r = divmod(c, world)
        n = int(memory.counts[r, s])
        off = sm.ROWS_OFFSET + (r * val_steps + s) * memory.region_bytes
        recs.append(np.frombuffer(memory.map, dtype=sm.REC_DTYPE, count=n, offset=off).copy())
    if fit_k:
        memory.fit_go()
    fit = memory.fit_wait() if fit_k else None
    hdr = [int(v) for v in memory.hdr[:sm.H_T_FIT_DONE + 1]]
    memory.close()
    dump = np.fromfile(tmp / f"{name}.dump", dtype=np.uint16)
    return dict(recs=recs, fit=fit, hdr=hdr, stream=np.concatenate([[SEP], dump]).astype(np.uint16))


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


def runs_of(x: np.ndarray, chunk: int, starts=None) -> np.ndarray:
    run = np.zeros(x.size, dtype=np.int64)
    r = 0
    for t, v in enumerate(x):
        r = 1 if (t % chunk == 0 or v == BOS or (starts is not None and starts[t])) else r + 1
        run[t] = r
    return run


def common_suffix(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the common suffix of two equal-length arrays."""
    ne = np.flatnonzero(a[::-1] != b[::-1])
    return int(ne[0]) if ne.size else a.size


def reference_records(stream: np.ndarray, x: np.ndarray, run: np.ndarray, hash_bits: int, limit=None) -> dict:
    """The design's record of every position t of x (run[t] = its segment's tokens up to t) in plain Python:
    {t: dict}. stream = the memory's tok[] (tok[0] = SEP); entries >= limit are not in the memory."""
    limit = stream.size if limit is None else limit
    seg = np.zeros(stream.size, dtype=np.int64)
    r = 0
    for j, v in enumerate(stream):
        r = 0 if v == SEP else r + 1
        seg[j] = r
    chains: dict = {}
    for j in range(min(limit, stream.size - 1)):
        if seg[j] >= KEY and stream[j + 1] != SEP:
            chains.setdefault(bucket(stream[j - KEY + 1:j + 1], hash_bits), []).append(j)
    out = {}
    for t in range(x.size):
        if run[t] < KEY:
            continue
        ctx = x[t - KEY + 1:t + 1]
        cands, visited = [], 0
        for j in reversed(chains.get(bucket(ctx, hash_bits), [])):
            if len(cands) >= CAP or visited >= MAXVISIT:
                break
            visited += 1
            if not np.array_equal(stream[j - KEY + 1:j + 1], ctx):
                continue
            lim = min(run[t], MAXLEN)
            length = KEY
            while length < lim and stream[j - length] == x[t - length]:
                length += 1
            full = length
            if length == MAXLEN:
                lim2 = min(run[t], FULLCAP, j + 1)
                full = common_suffix(stream[j - lim2 + 1:j + 1], x[t - lim2 + 1:t + 1])
            cands.append((j, length, int(stream[j + 1]), full))
        if not cands:
            continue
        maxl = max(c[1] for c in cands)
        ls = max(k for k, lv in enumerate(LEVELS) if lv <= maxl)
        rec = dict(nx=[c[2] for c in cands], len=[c[1] for c in cands], ncand=len(cands), lstar=ls,
                   n=[0] * 8, d=[0] * 8, m=[0] * 8, n1=[0] * 8, n2=[0] * 8, top=[0] * 8)
        for k, L in enumerate(LEVELS):
            nxt = [c[2] for c in cands if c[1] >= L]
            if not nxt:
                continue
            cnt = {v: nxt.count(v) for v in nxt}
            mm = max(cnt.values())
            rec["n"][k], rec["d"][k], rec["m"][k] = len(nxt), len(cnt), mm
            rec["n1"][k] = sum(c == 1 for c in cnt.values())
            rec["n2"][k] = sum(c == 2 for c in cnt.values())
            rec["top"][k] = min(v for v, c in cnt.items() if c == mm)
        rec_i = next(i for i, c in enumerate(cands) if c[1] >= LEVELS[ls])
        rec["pos_recent"], rec["len_recent"] = cands[rec_i][0], cands[rec_i][3]
        best = max(c[3] for c in cands)
        lon = next(i for i, c in enumerate(cands) if c[3] == best)
        rec["pos_longest"], rec["len_longest"] = cands[lon][0], best
        out[t] = rec
    return out


def assert_records_equal(recs: np.ndarray, ref: dict, offset: int = 0):
    """recs: one chunk's records (pos within the chunk); ref: {global t: record}, chunk starting at offset."""
    got = {int(r["pos"]) + offset: r for r in recs}
    assert set(got) == set(ref), (sorted(set(ref) - set(got))[:5], sorted(set(got) - set(ref))[:5])
    for t, r in got.items():
        e = ref[t]
        nc = e["ncand"]
        assert int(r["ncand"]) == nc and int(r["lstar"]) == e["lstar"], t
        assert r["nx"][:nc].tolist() == e["nx"] and r["len"][:nc].tolist() == e["len"], t
        assert not r["len"][nc:].any() and not r["nx"][nc:].any(), t
        for f in ("n", "d", "m", "n1", "n2", "top"):
            assert r[f].tolist() == e[f], (t, f, r[f].tolist(), e[f])
        for f in ("pos_recent", "len_recent", "pos_longest", "len_longest"):
            assert int(r[f]) == e[f], (t, f, int(r[f]), e[f])


# ------------------------------------------------------------------------------------------------ records

@pytest.mark.parametrize("hash_bits, chunk", [(22, 4096), (5, 4096), (22, 5000)])
def test_records_equal_a_brute_force_reference(tmp_path, toy, hash_bits, chunk):
    """32 buckets: MAXVISIT binds. chunk 5000: query blocks start inside segments (each finds its run by looking
    back); the 4,500-token copy of the long document saturates the full match length at FULLCAP."""
    out = run_helper(tmp_path, toy, name=f"h{hash_bits}_{chunk}", chunk=chunk, hash_bits=hash_bits)
    n = 2 * 2 * chunk
    x = toy["val"][:n]
    ref = reference_records(out["stream"], x, runs_of(x, chunk), hash_bits)
    for c, recs in enumerate(out["recs"]):
        assert np.all(np.diff(recs["pos"].astype(np.int64)) > 0)
        part = {t: v for t, v in ref.items() if c * chunk <= t < (c + 1) * chunk}
        assert_records_equal(recs, part, c * chunk)
    assert out["hdr"][sm.H_HITS] == len(ref) and out["hdr"][sm.H_REC_BYTES] == sm.REC_BYTES
    assert out["hdr"][sm.H_CANDS] == sum(r["ncand"] for r in ref.values())
    if hash_bits == 22:  # the data exercises every part of the record
        allr = np.concatenate(out["recs"])
        assert set(allr["lstar"].tolist()) == set(range(8)) and (allr["ncand"] == CAP).any()
        assert (allr["len_longest"] == FULLCAP).any() and ((allr["len_longest"] > MAXLEN) & (allr["len_longest"] < FULLCAP)).any()
        assert (allr["n1"] > 0).any() and (allr["n2"] > 0).any() and (allr["m"] > 1).any()
        assert (allr["pos_recent"] != allr["pos_longest"]).any()


def test_records_are_bit_identical_across_threads_and_batching(tmp_path, toy):
    steps = toy["steps"][:4]
    flat = [s for st in steps for s in st[0]]
    regrouped = [[flat[:3], []], [flat[3:500], []], [flat[500:], []]]
    ref = run_helper(tmp_path, toy, name="t1", steps=steps, threads=1)
    for name, kw in (("t7", dict(threads=7, steps=steps)), ("batched", dict(steps=regrouped))):
        got = run_helper(tmp_path, toy, name=name, **kw)
        assert all(a.tobytes() == b.tobytes() for a, b in zip(ref["recs"], got["recs"])), name


def test_records_never_read_the_target(tmp_path, toy):
    """Changing val[t + 1:] (t's target and everything after) leaves the records of positions <= t byte-identical;
    the records after t change (the test is not vacuous)."""
    base = run_helper(tmp_path, toy, name="base")
    val = toy["val"]
    t = 4096 * 2 + 1700  # chunk 2
    allb = np.concatenate([r for r in base["recs"]])
    gpos = np.concatenate([r["pos"].astype(np.int64) + c * 4096 for c, r in enumerate(base["recs"])])
    assert (gpos == t).any(), "pick a matched position"
    rng = np.random.default_rng(1)
    future = val.copy()
    future[t + 1:] = rng.integers(0, 4, future.size - t - 1)
    future[t + 1::97] = BOS
    f = write_shard(tmp_path / "future_val.bin", future)
    changed = run_helper(tmp_path, toy, name="future", val_file=f)
    allc = np.concatenate([r for r in changed["recs"]])
    gpc = np.concatenate([r["pos"].astype(np.int64) + c * 4096 for c, r in enumerate(changed["recs"])])
    assert allb[gpos <= t].tobytes() == allc[gpc <= t].tobytes()
    assert allb[gpos > t].tobytes() != allc[gpc > t].tobytes()


def test_fit_records_are_the_last_batches_against_the_frozen_memory(tmp_path, toy):
    """fit_k = 2: per rank, the last two steps' batches (inputs buf[:-1], targets buf[1:]) queried against the memory
    before step 4 (segments restart at every batch); the val records are those of a run without FIT."""
    out = run_helper(tmp_path, toy, name="fit", fit_k=2)
    plain = run_helper(tmp_path, toy, name="nofit")
    assert all(a.tobytes() == b.tobytes() for a, b in zip(out["recs"], plain["recs"]))
    fit = out["fit"]
    shard = read_shard(toy["shard"])
    freeze = 1 + sum(e - a + 1 for st in toy["steps"][:4] for r in st for a, e in r)
    assert fit["freeze"] == freeze == out["hdr"][sm.H_FIT_FREEZE] and fit["k"] == 2 and fit["world"] == 2
    assert fit["batch_lengths"].tolist() == [299, 299] and fit["n"] == 598
    hits = 0
    for r in range(2):
        bufs = [np.concatenate([shard[a:e] for a, e in st[r]]) for st in toy["steps"][4:]]
        x = np.concatenate([b[:-1] for b in bufs])
        y = np.concatenate([b[1:] for b in bufs])
        assert np.array_equal(fit["tokens"][r, :, 0], x) and np.array_equal(fit["tokens"][r, :, 1], y)
        starts = np.zeros(x.size, dtype=bool)
        starts[[0, 299]] = True
        ref = reference_records(out["stream"], x, runs_of(x, 10 ** 9, starts), 16, limit=freeze)
        recs = np.frombuffer(fit["recs"][r].tobytes(), dtype=sm.REC_DTYPE)
        assert_records_equal(recs, ref)
        assert all(int(v) < freeze for v in np.concatenate([recs["pos_recent"], recs["pos_longest"]]))
        hits += len(ref)
        unfrozen = reference_records(out["stream"], x, runs_of(x, 10 ** 9, starts), 16)
        # without the freeze every position would find its own entry (its target as the next token, the whole run)
        assert sum(unfrozen[t]["len_longest"] > ref.get(t, {"len_longest": 0})["len_longest"] for t in unfrozen) > x.size // 2
    assert out["hdr"][sm.H_FIT_HITS] == hits > 0 and out["hdr"][sm.H_FIT_POSITIONS] == 2 * 598


def test_fit_and_step_errors(tmp_path, toy):
    helper = str(sm.build_helper())
    rows = tmp_path / "rows"
    rows.write_bytes(bytes(sm.ROWS_OFFSET + 2 * 4096 * sm.REC_BYTES))
    common = [helper, f"rows={rows}", f"val={toy['val_file']}", "world=1", "val_tokens=8192", "chunk=4096", "cap=64"]
    for extra, msg in ((["steps=3", "fit_k=2"], "go together"), (["steps=2", f"fit={tmp_path / 'f'}", "fit_k=2"], "go together")):
        r = subprocess.run(common + extra + ["--", str(toy["shard"])], capture_output=True, text=True)
        assert r.returncode == 2 and msg in r.stderr, r.stderr
    # unequal FIT batches across ranks are refused (the eval needs every rank's batches aligned)
    steps = toy["steps"][:4] + [[toy["steps"][4][0], toy["steps"][4][1][:-1]], toy["steps"][5]]
    with pytest.raises(RuntimeError, match="FIT batch 0 holds"):
        run_helper(tmp_path, toy, name="bad", steps=steps, fit_k=2)
    memory = sm.StreamMemory(train_files=[str(toy["shard"])], val_file=str(toy["val_file"]), total_steps=3,
                             stream_tokens=10 ** 5, world=2, rank=0, master=True, val_tokens=2 * 4096 * 2, chunk=4096,
                             device="cpu", threads=2, hash_bits=12, rows_dir=str(tmp_path))
    for st in toy["steps"][:2]:
        memory.on_spans(0, [[a for a, _ in r] for r in st], [[e for _, e in r] for r in st])
    memory.go()
    with pytest.raises(RuntimeError, match="GO for 2 steps after 2 step messages; the memory expects 3"):
        memory.collect()
    memory.close()


def test_collect_gives_every_rank_its_own_chunks(tmp_path, toy):
    """Rank r's records (collect -> DeviceRows -> Rows per val step) are those of val chunks c = step * world + r,
    and its val batches pass the checksum check."""
    steps = toy["steps"][:4]
    world, chunk, val_steps = 2, 4096, 2
    memory = sm.StreamMemory(train_files=[str(toy["shard"])], val_file=str(toy["val_file"]), total_steps=len(steps),
                             stream_tokens=10 ** 5, world=world, rank=0, master=True, val_tokens=world * chunk * val_steps,
                             chunk=chunk, device="cpu", threads=2, hash_bits=16, rows_dir=str(tmp_path),
                             dump_path=str(tmp_path / "c.dump"))
    for st in steps:
        memory.on_spans(0, [[a for a, _ in r] for r in st], [[e for _, e in r] for r in st])
    memory.go()
    memory.collect()
    stream = np.concatenate([[SEP], np.fromfile(tmp_path / "c.dump", dtype=np.uint16)]).astype(np.uint16)
    x = toy["val"][:world * chunk * val_steps]
    ref = reference_records(stream, x, runs_of(x, chunk), 16)
    val = toy["val"]

    class B:
        def __init__(self, lo):
            self.inputs_cpu = val[lo:lo + chunk].astype(np.int32)
            self.targets_cpu = torch.from_numpy(val[lo + 1:lo + chunk + 1].astype(np.int64))
    for rank in range(world):
        memory.rank = rank
        dr = memory.collect()
        chunks = [s * world + rank for s in range(val_steps)]
        memory.check([B(c * chunk) for c in chunks])
        for s, c in enumerate(chunks):
            rows = dr.chunk_rows(s)
            want = sorted(t - c * chunk for t in ref if c * chunk <= t < (c + 1) * chunk)
            assert rows.hit.nonzero()[:, 0].tolist() == want
            for t in want[:50]:
                assert rows.lstar[t] == ref[t + c * chunk]["lstar"] and rows.ncand[t] == ref[t + c * chunk]["ncand"]
    with pytest.raises(RuntimeError, match="not the chunk"):
        memory.check([B(0), B(chunk)])  # rank 1's batches are chunks 1 and 3
    memory.close()


def test_fit_runs_only_on_rank_0s_fit_message(tmp_path, toy):
    """The FIT queries wait for FIT (sent after the clock stops): until then the helper stays DONE; a FIT with another
    k, a second FIT, or a FIT without fit= is refused."""
    def start(name, fit):
        steps = toy["steps"]
        m = sm.StreamMemory(train_files=[str(toy["shard"])], val_file=str(toy["val_file"]), total_steps=len(steps),
                            stream_tokens=10 ** 5, world=2, rank=0, master=True, val_tokens=2 * 4096 * 2, chunk=4096,
                            device="cpu", threads=2, hash_bits=12, rows_dir=str(tmp_path),
                            fit_path=str(tmp_path / name) if fit else None, fit_k=2)
        for st in steps:
            m.on_spans(0, [[a for a, _ in r] for r in st], [[e for _, e in r] for r in st])
        m.go()
        m.collect()
        return m
    m = start("f1", True)
    import time
    time.sleep(0.3)
    assert int(m.hdr[sm.H_STATE]) == sm.ST_DONE and not (tmp_path / "f1").exists()
    m.fit_go()
    assert m.fit_wait()["n"] == 598
    m.fit_go()
    with pytest.raises(RuntimeError, match="FIT 2: twice"):
        m._wait(sm.ST_ERROR + 1, 5)
    m.close()
    m = start("f2", True)
    m._write(np.array([sm.MSG_FIT, 3], dtype=np.uint32))
    with pytest.raises(RuntimeError, match="FIT 3: not fit_k"):
        m.fit_wait()
    m.close()
    m = start("f3", False)
    m._write(np.array([sm.MSG_FIT, 2], dtype=np.uint32))
    with pytest.raises(RuntimeError, match="without fit="):
        m._wait(sm.ST_ERROR + 1, 5)
    m.close()


def test_python_mirrors_the_c_constants():
    c = (MODULE.with_name("stream_memory.c")).read_text()

    def enum(first):
        body = re.search(r"enum \{ *" + first + r"([^}]*)\}", c, re.S).group(0)
        names, value = {}, -1
        for item in re.sub(r"enum \{|\}", "", body).split(","):
            name, _, expr = item.strip().partition("=")
            name, expr = name.strip(), expr.strip()
            if name:
                value = eval(expr.replace("NLEVELS", str(len(LEVELS))), {}, dict(names)) if expr else value + 1
                names[name] = value
        return names

    for first in ("H_MAGIC", "MSG_STEP", "ST_STARTING"):
        for name, value in enum(first).items():
            if name != "H_NWORDS":
                assert getattr(sm, name) == value, name
    assert enum("H_MAGIC")["H_NWORDS"] * 8 <= sm.ERRMSG_OFFSET
    for name in ("ERRMSG_OFFSET", "CHECKSUM_OFFSET", "COUNTS_OFFSET", "PCOUNTS_OFFSET", "ROWS_OFFSET", "REGION_ALIGN",
                 "HEADER_MAGIC", "FIT_MAGIC", "FIT_HEAD_WORDS", "LATE_STEPS", "KEY", "MAXLEN", "CAP", "MAXVISIT",
                 "FULLCAP", "MAX_CHUNKS"):
        assert int(re.search(rf"#define {name} (\w+)", c).group(1).rstrip("ul"), 0) == getattr(sm, name), name
    for name in ("ARENA_REC_SHARE", "ARENA_PTR_SHARE", "PREFAULT_REC_SHARE", "PREFAULT_PTR_SHARE"):  # host_bytes' sizes
        assert float(re.search(rf"#define {name} ([0-9.]+)", c).group(1)) == getattr(sm, name), name
    assert sm.CHECKSUM_OFFSET + 8 * sm.MAX_CHUNKS <= sm.COUNTS_OFFSET and sm.COUNTS_OFFSET + 8 * sm.MAX_CHUNKS <= sm.PCOUNTS_OFFSET
    assert sm.PCOUNTS_OFFSET + 8 * sm.MAX_CHUNKS <= sm.ROWS_OFFSET
    low = re.search(r"LOW_ORDERS\[LOW_NORDERS\] = \{([^}]*)\}", c).group(1)
    assert tuple(int(v) for v in low.split(",")) == sm.LOW_ORDERS
    levels = re.search(r"LEVELS\[NLEVELS\] = \{([^}]*)\}", c).group(1)
    assert tuple(int(v) for v in levels.split(",")) == sm.LEVELS == LEVELS
    # rec_t's fields, in order, with their C types, are REC_DTYPE's
    body = re.search(r"typedef struct \{(.*?)\} rec_t;", c, re.S).group(1)
    fields = []
    for line in body.split("\n"):
        line = line.split("//")[0].strip().rstrip(";")
        if not line:
            continue
        ctype, names = line.split(None, 1)
        for nm in names.split(","):
            nm = nm.strip()
            dim = re.search(r"\[(\w+)\]", nm)
            n = {"NLEVELS": 8, "CAP": CAP}.get(dim.group(1), None) if dim else None
            n = int(dim.group(1)) if dim and n is None else n
            fields.append((nm.split("[")[0], {"uint32_t": "<u4", "uint16_t": "<u2", "uint8_t": "|u1"}[ctype], n))
    want = [(nm, sm.REC_DTYPE.fields[nm][0].base.str, sm.REC_DTYPE.fields[nm][0].shape[0] if sm.REC_DTYPE.fields[nm][0].shape else None)
            for nm in sm.REC_DTYPE.names]
    assert fields == want


# ------------------------------------------------------------------------------------------------ parsing

def random_records(rng, P, H):
    recs = np.zeros(H, dtype=sm.REC_DTYPE)
    recs["pos"] = np.sort(rng.choice(P, H, replace=False))
    for f in ("pos_recent", "pos_longest"):
        recs[f] = rng.integers(0, 2 ** 32, H, dtype=np.uint64)
    for f in ("len_recent", "len_longest"):
        recs[f] = rng.integers(0, 2 ** 16, H)
    nc = rng.integers(1, CAP + 1, H)
    recs["ncand"] = nc
    recs["lstar"] = rng.integers(0, 8, H)
    recs["top"] = rng.integers(0, 2 ** 16, (H, 8))
    for f in ("n", "d", "m", "n1", "n2"):
        recs[f] = rng.integers(0, 33, (H, 8))
    for i in range(H):
        recs["nx"][i, :nc[i]] = rng.integers(0, 2 ** 16, nc[i])
        recs["len"][i, :nc[i]] = rng.integers(KEY, MAXLEN + 1, nc[i])
    return recs


def test_rows_parse_the_records():
    rng = np.random.default_rng(2)
    P, H = 5000, 700
    recs = random_records(rng, P, H)
    rows = sm.Rows(torch.from_numpy(recs.view(np.uint8).reshape(H, sm.REC_BYTES).copy()), P)
    pos = recs["pos"].astype(np.int64)
    assert rows.hit.sum() == H and rows.hit[pos].all()
    for f in ("pos_recent", "pos_longest", "len_recent", "len_longest", "ncand", "lstar"):
        assert np.array_equal(getattr(rows, f).numpy()[pos], recs[f].astype(np.int64)), f
    for f in ("n", "d", "m", "n1", "n2"):
        assert np.array_equal(getattr(rows, f).numpy()[pos], recs[f].astype(np.float64)), f
    assert np.array_equal(rows.top.numpy()[pos], recs["top"].astype(np.int64))
    assert np.array_equal(rows.len.numpy()[pos], recs["len"].astype(np.int32))
    nx = rows.nx.numpy()[pos]
    assert np.array_equal(nx[recs["len"] > 0], recs["nx"][recs["len"] > 0]) and (nx[recs["len"] == 0] == -1).all()
    other = np.setdiff1d(np.arange(P), pos)
    assert (rows.lstar.numpy()[other] == -1).all() and not rows.n.numpy()[other].any() and (rows.nx.numpy()[other] == -1).all()
    y = torch.from_numpy(rng.integers(0, 2 ** 16, P))
    y[pos[:50]] = torch.from_numpy(recs["nx"][:50, 0].astype(np.int64))  # targets among the candidates
    counts = rows.level_counts(y)
    for k, L in enumerate(LEVELS):
        want = ((recs["nx"] == y.numpy()[pos][:, None]) & (recs["len"] >= L)).sum(1)
        assert np.array_equal(counts[L]["C"].numpy()[pos], want) and not counts[L]["C"].numpy()[other].any()
    assert counts[LEVELS[0]]["C"].numpy()[pos[:50]].min() >= 1


# ------------------------------------------------------------------------------------------------ gate features

def np_doc_ema(x, ds, d):  # rg/lmfeat/dochist.py
    xs = np.concatenate([[0.0], x[:-1]])
    S = lfilter([1.0], [1.0, -d], xs)
    t = np.arange(x.size)
    with np.errstate(under="ignore"):
        decay = np.power(d, (t - ds).astype(np.float64))
    return S - decay * S[ds]


def np_match_feats(lp, ent, L, ds):  # rg/lmfeat/lmaware.py's match_feats (fixed L)
    Q = lp.size
    t = np.arange(Q)
    s = -lp
    cs = np.concatenate([[0.0], np.cumsum(s)])
    ws = lambda c, lo, hi: np.where(hi >= lo, c[np.maximum(hi + 1, 0)] - c[np.maximum(lo, 0)], 0.0)
    lo, hi = np.maximum(t - L, ds), t - 1
    n = np.maximum(hi - lo + 1, 0)
    S = ws(cs, lo, hi)
    mx = np.array([s[a:b + 1].astype(np.float32).max() if b >= a else 0.0 for a, b in zip(lo, hi)], dtype=np.float64)
    cols = [np.log1p(S), S / np.maximum(n, 1), mx]
    if ent is not None:
        ce = np.concatenate([[0.0], np.cumsum(ent)])
        cols.append((S - ws(ce, lo, hi)) / np.maximum(n, 1))
        lo16 = np.maximum(t - 16, ds)
        cols.append(ws(ce, lo16, hi) / np.maximum(t - lo16, 1))
        for w in (1, 4, 16, 64):
            low = np.maximum(t - w, ds)
            cols.append(ws(cs, low, hi) / np.maximum(t - low, 1))
        cols.append(np.log2(np.maximum(t - ds, 1)))
    return np.stack(cols, 1)


def np_chain_phis(counts, orders, y, lp, ent, mx, lqb, ds, tp):
    """rg/combo/ds.py's chain_phis(mode='lm+su2+tp+hi') with lo.order_phi, in numpy (float64)."""
    L2 = lambda v: np.log2(np.maximum(v, 1.0))
    t = np.arange(y.size)
    phis = []
    for i, o in enumerate(orders):
        c = counts[o]
        N, D, M, n1, n2 = (c[k] for k in ("N", "D", "M", "n1", "n2"))
        Nn, Dn = np.maximum(N, 1), np.maximum(D, 1)
        cols = [np.ones_like(N), L2(N), L2(D), M / Nn, n1 / Dn, n2 / Dn, L2(N) ** 2 / 16, (N == 1) * 1.0]
        if i:
            pN = counts[orders[i - 1]]["N"]
            cols.append(np.where(pN > 0, L2(pN) - L2(N), 0.0))
        cols += [ent, mx]
        cols += list(np_match_feats(lp, ent, o, ds).T) + list(np_match_feats(lqb, None, o, ds).T)
        hit = (N > 0).astype(np.float64)
        corr = hit * (c["top"] == y)
        r = np.where(N > 0, c["C"] / Nn, 0.0)
        llr = hit * np.log(0.5 + 0.5 * r / np.exp(lqb))
        for d in (0.9, 0.98):
            eh, ec = np_doc_ema(hit, ds, d), np_doc_ema(corr, ds, d)
            cols += [np.log1p(eh), ec / (eh + 0.5), np_doc_ema(llr, ds, d) / (eh + 0.5)]
        cols += [np.r_[0.0, hit[:-1]] * (t > ds), np.r_[0.0, corr[:-1]] * (t > ds)]
        v, inn, rank = tp[o]
        cols += [v, v ** 2 / 10, inn, mx - v, np.log2(1 + rank)]
        phis.append(np.stack(cols, 1))
    return phis


def synthetic_chunk(rng, P, orders=LEVELS, vocab=6):
    """Random but consistent inputs of one chunk: tokens with BOS segments, nested level counts (N non-increasing in
    the level, C <= N, n1 + n2 <= D <= N, M <= N), LM outputs and top-token features."""
    x = rng.integers(0, vocab, P)
    x[rng.random(P) < 0.01] = BOS
    y = np.r_[x[1:], rng.integers(0, vocab)]
    counts, N = {}, rng.integers(0, 33, P) * (rng.random(P) < 0.4)
    for o in orders:
        N = np.minimum(N, rng.integers(0, 33, P)) * (rng.random(P) < 0.8)
        D = np.where(N > 0, rng.integers(1, 1 + np.maximum(N, 1)), 0)
        n1 = np.where(D > 0, rng.integers(0, 1 + D), 0)
        n2 = np.where(D > 0, rng.integers(0, 1 + D - n1), 0)
        counts[o] = dict(N=N.astype(np.float64), D=D.astype(np.float64), M=np.minimum(N, rng.integers(1, 33, P)).astype(np.float64),
                         n1=n1.astype(np.float64), n2=n2.astype(np.float64), C=np.minimum(N, rng.integers(0, 3, P)).astype(np.float64),
                         top=np.where(N > 0, rng.integers(0, vocab, P), 0))
    lp = -rng.exponential(3.0, P)
    lqb = np.log(0.3 * np.exp(lp) + 0.7 * rng.random(P) * 0.1 + 1e-6)
    ent, mx = rng.exponential(3.0, P), -rng.exponential(0.5, P)
    tp = {o: (-rng.exponential(2.0, P), (rng.random(P) < 0.6).astype(np.float64), rng.integers(0, 33, P).astype(np.float64))
          for o in orders}
    return dict(x=x, y=y, counts=counts, lp=lp, lqb=lqb, ent=ent, mx=mx, tp=tp)


def torch_features(ch, starts=None, orders=LEVELS):
    T = lambda a: torch.from_numpy(np.asarray(a))
    counts = {o: {k: T(v) for k, v in c.items()} for o, c in ch["counts"].items()}
    x = T(ch["x"])
    st = sm.seg_starts(x) if starts is None else starts
    return sm.chain_features(counts, list(orders), T(ch["y"]), x, T(ch["lp"]), T(ch["ent"]), T(ch["lqb"]), st,
                             "lm+su2+tp+hi", mx=T(ch["mx"]), tp={o: tuple(T(a) for a in v) for o, v in ch["tp"].items()})


def test_features_equal_the_research_reference():
    rng = np.random.default_rng(3)
    ch = synthetic_chunk(rng, 3000)
    mine = torch_features(ch)
    starts = (ch["x"] == BOS)
    starts[0] = True
    ds = np.maximum.accumulate(np.where(starts, np.arange(starts.size), 0))
    ref = np_chain_phis(ch["counts"], LEVELS, ch["y"], ch["lp"], ch["ent"], ch["mx"], ch["lqb"], ds, ch["tp"])
    for o, a, b in zip(LEVELS, mine, ref):
        assert a.shape == b.shape == (3000, 36 if o == 6 else 37)
        np.testing.assert_allclose(a.numpy(), b, rtol=1e-9, atol=1e-9, err_msg=f"order {o}")


def test_seg_ema_and_window_max_brute_force():
    rng = np.random.default_rng(4)
    P = 1500
    v = rng.normal(size=P)
    starts = rng.random(P) < 0.02
    starts[0] = True
    ds = np.maximum.accumulate(np.where(starts, np.arange(P), 0))
    for d in (0.9, 0.98):
        got = sm.seg_ema(torch.from_numpy(v), torch.from_numpy(starts), d).numpy()
        want = [sum(d ** (t - 1 - u) * v[u] for u in range(ds[t], t)) for t in range(P)]
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
    wm = sm._window_max(torch.from_numpy(v), torch.from_numpy(ds), (3, 32))
    for L in (3, 32):
        want = [v[max(t - L, ds[t]):t].astype(np.float32).max() if t > ds[t] else 0.0 for t in range(P)]
        np.testing.assert_array_equal(wm[L].numpy(), np.asarray(want, dtype=np.float32).astype(np.float64))


def test_features_are_causal():
    """Randomize everything that depends on targets at positions >= t0 (targets, the model's log p of them, the base's,
    the target counts C) and the target-independent per-position inputs after t0: rows <= t0 are bit-identical."""
    rng = np.random.default_rng(5)
    ch = synthetic_chunk(rng, 4000)
    ref = torch_features(ch)
    for t0 in (1000, 2500, 3999):
        e = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ch.items()}
        n = 4000 - t0
        e["y"][t0:] = rng.integers(0, 6, n)
        e["lp"][t0:] = -rng.exponential(3.0, n)
        e["lqb"][t0:] = -rng.exponential(3.0, n)
        e["counts"] = {o: {k: v.copy() for k, v in c.items()} for o, c in ch["counts"].items()}
        for o, c in e["counts"].items():
            c["C"][t0:] = np.minimum(c["N"][t0:], rng.integers(0, 3, n))
            for k in ("N", "D", "M", "n1", "n2"):
                c[k][t0 + 1:] = c[k][t0 + 1:][rng.permutation(n - 1)] if n > 1 else c[k][t0 + 1:]
            c["top"][t0 + 1:] = rng.integers(0, 6, n - 1)
        e["ent"][t0 + 1:] = rng.exponential(3.0, n - 1)
        e["mx"][t0 + 1:] = -rng.exponential(0.5, n - 1)
        e["tp"] = {o: tuple(np.r_[a[:t0 + 1], rng.random(n - 1)] for a in v) for o, v in ch["tp"].items()}
        e["x"][t0 + 1:] = rng.integers(0, 6, n - 1)  # inputs after t0 (segment starts after t0 move too)
        got = torch_features(e)
        for a, b in zip(ref, got):
            assert torch.equal(a[:t0 + 1], b[:t0 + 1]), t0
        assert any(not torch.equal(a[t0 + 1:], b[t0 + 1:]) for a, b in zip(ref, got)) or t0 == 3999


def kn_gate(rng, orders):
    d = {6: 36}
    spec = dict(version=2, mode="lm+su2+tp+hi", orders=list(orders), top=None,
                w=[(rng.normal(size=d.get(o, 37)) * 0.3).tolist() for o in orders],
                disc=[rng.normal(size=3).tolist() for _ in orders],
                mu=[rng.normal(size=d.get(o, 37) - 1).tolist() for o in orders],
                sd=[(rng.random(d.get(o, 37) - 1) + 0.5).tolist() for o in orders])
    return sm.Gate(spec)


def test_the_chain_sums_to_one_over_the_vocabulary():
    """At one position: fixed candidates (next tokens, lengths) and gate features; for every target v of a 12-token
    vocabulary, C_L(v) from the candidates; p a distribution: sum_v q(v) = sum_v p(v) = 1."""
    rng = np.random.default_rng(6)
    gate = kn_gate(rng, LEVELS)
    V = 12
    for trial in range(20):
        nc = int(rng.integers(1, CAP + 1))
        nx = rng.integers(0, V, nc)
        ln = np.sort(rng.integers(KEY, MAXLEN + 1, nc))[::-1]
        p = rng.dirichlet(np.ones(V) * 0.3)
        counts = {}
        for L in LEVELS:
            s = nx[ln >= L]
            cnt = np.bincount(s, minlength=V)
            C = torch.from_numpy(cnt.astype(np.float64))  # row v = target v
            full = lambda val: torch.full((V,), float(val), dtype=torch.float64)
            counts[L] = dict(N=full(s.size), C=C, D=full((cnt > 0).sum()), M=full(cnt.max()), n1=full((cnt == 1).sum()),
                             n2=full((cnt == 2).sum()), top=torch.full((V,), int(cnt.argmax())))
        phis = [torch.from_numpy(np.tile(np.r_[1.0, rng.normal(size=gate.w[i].numel() - 1)], (V, 1))) for i in range(len(LEVELS))]
        q = sm.chain_prob(torch.from_numpy(p), counts, phis, gate.orders, gate.w, gate.disc, gate.mu, gate.sd)
        assert abs(float(q.sum()) - 1) < 1e-12 and (q >= 0).all(), trial


def test_mix_v2_is_the_identity_where_nothing_matched_and_mixes_where_it_did():
    rng = np.random.default_rng(7)
    P = 4096
    gate = kn_gate(rng, LEVELS)
    recs = random_records(rng, P, 300)
    for i in range(300):  # consistent per-level counts from the candidates
        nx, ln = recs["nx"][i], recs["len"][i]
        for k, L in enumerate(LEVELS):
            s = nx[ln >= L]
            cnt = np.bincount(s, minlength=1) if s.size else np.zeros(1, int)
            recs["n"][i, k], recs["d"][i, k], recs["m"][i, k] = s.size, (cnt > 0).sum(), cnt.max()
            recs["n1"][i, k], recs["n2"][i, k] = (cnt == 1).sum(), (cnt == 2).sum()
            recs["top"][i, k] = cnt.argmax() if s.size else 0
    rows = sm.Rows(torch.from_numpy(recs.view(np.uint8).reshape(-1, sm.REC_BYTES).copy()), P)
    x = torch.from_numpy(rng.integers(0, 50000, P))
    y = torch.from_numpy(rng.integers(0, 50000, P))
    y[torch.from_numpy(recs["pos"][:100].astype(np.int64))] = torch.from_numpy(recs["nx"][:100, 0].astype(np.int64))
    nll = torch.from_numpy(rng.exponential(3.0, P).astype(np.float32))
    K = len(gate.orders)
    lm = dict(ent=torch.rand(P) * 5, mx=-torch.rand(P), top_v=-torch.rand(P, K) * 5, top_in=torch.rand(P, K) < 0.5,
              top_rank=torch.randint(0, 33, (P, K)))
    out = sm.mix_v2(nll, y, x, rows, lm, gate)
    assert out.dtype == nll.dtype and torch.equal(out[~rows.hit], nll[~rows.hit])
    assert not torch.equal(out[rows.hit], nll[rows.hit]) and torch.isfinite(out).all()
    toks = sm.top_tokens(rows, gate)
    assert toks.shape == (P, K) and torch.equal(toks[rows.hit], rows.top[rows.hit] * (rows.n[rows.hit] > 0))


def test_lm_top_features_follow_the_top32_convention():
    rng = np.random.default_rng(8)
    P, V = 300, 500
    logp = torch.log_softmax(torch.from_numpy(rng.normal(size=(P, V)) * 3), -1)
    toks = torch.from_numpy(rng.integers(0, V, (P, 5)))
    top = logp.topk(32, -1).indices
    toks[:, 0] = top[:, 0]
    toks[:, 1] = top[:, 31]
    f = sm.lm_top_features(logp, toks)
    lp32 = logp.topk(32, -1).values.float()
    for i in range(P):
        for j in range(5):
            tok = int(toks[i, j])
            pos = (top[i] == tok).nonzero()
            if pos.numel():
                assert f["top_in"][i, j] and int(f["top_rank"][i, j]) == int(pos[0, 0])
                assert abs(float(f["top_v"][i, j]) - float(logp[i, tok])) < 1e-5
            else:
                assert not f["top_in"][i, j] and int(f["top_rank"][i, j]) == 32
                assert abs(float(f["top_v"][i, j]) - (float(lp32[i, -1]) - 1.0)) < 1e-5
    p = logp.exp()
    np.testing.assert_allclose(f["ent"].numpy(), -(p * logp).sum(-1).numpy(), rtol=1e-5)
    np.testing.assert_allclose(f["mx"].numpy(), logp.max(-1).values.numpy(), rtol=1e-6)
    assert bool(f["top_in"][:, :2].all()) and (f["top_rank"][:, 0] == 0).all()


# v1's 4-constant gate on the deepest of the levels (6, 8, 12, 16, 24, 32), on v2's records: the paired comparison of
# the research (kept here, not in the record's source).
W_V1 = (-15.8, 0.957, 7.412, 2.528)
LEVELS_V1 = (6, 8, 12, 16, 24, 32)


def mix_v1(nll, y, rows, w=W_V1, eps: float = 1e-9):
    """v1's single link on the same records: q = (1 - lam) p + lam C/N at the deepest of LEVELS_V1 reached,
    lam = sigmoid(w . [1, log2 N, M/N, log2 L*])."""
    nll64 = nll.to(torch.float64)
    maxl = rows.len.max(1).values
    li = torch.full_like(maxl, -1, dtype=torch.int64)
    for lv in LEVELS_V1:
        li = torch.where(maxl >= lv, sm.LEVELS.index(lv), li)
    hit = li >= 0
    k = li.clamp_min(0)
    ar = torch.arange(nll.numel(), device=nll.device)
    N, M = rows.n[ar, k], rows.m[ar, k]
    L = torch.tensor(sm.LEVELS, dtype=torch.float64, device=nll.device)[k]
    eq = (rows.nx == y.to(torch.int32)[:, None]) & (rows.len >= L[:, None])
    C = eq.sum(1).to(torch.float64)
    z = w[0] + w[1] * torch.log2(N.clamp_min(1)) + w[2] * M / N.clamp_min(1) + w[3] * torch.log2(L)
    lam = torch.sigmoid(z)
    q = (1 - lam) * (torch.exp(-nll64) - eps).clamp_min(0) + lam * C / N.clamp_min(1)
    return torch.where(hit, -torch.log(q + eps), nll64).to(nll.dtype)


def test_mix_v1_is_the_single_link_gate(tmp_path, toy):
    out = run_helper(tmp_path, toy, name="v1", hash_bits=22)
    recs = out["recs"][0]
    rows = sm.Rows(torch.from_numpy(recs.view(np.uint8).reshape(-1, sm.REC_BYTES).copy()), 4096)
    y = torch.from_numpy(toy["val"][1:4097].astype(np.int64))
    nll = torch.full((4096,), 3.0)
    got = mix_v1(nll, y, rows).to(torch.float64)
    w = W_V1
    for r in recs[:200]:
        t = int(r["pos"])
        lens = r["len"][:r["ncand"]]
        L = max(lv for lv in LEVELS_V1 if lv <= lens.max())
        sel = lens >= L
        nxt = r["nx"][:r["ncand"]][sel]
        N, C, M = sel.sum(), (nxt == int(y[t])).sum(), np.bincount(nxt).max()
        lam = 1 / (1 + math.exp(-(w[0] + w[1] * math.log2(N) + w[2] * M / N + w[3] * math.log2(L))))
        q = (1 - lam) * (math.exp(-3.0) - 1e-9) + lam * C / N
        assert abs(float(got[t]) + math.log(q + 1e-9)) < 1e-5
    assert torch.equal(got[~rows.hit].float(), nll[~rows.hit])


@pytest.mark.parametrize("low", [False, True])
def test_the_shipped_gates_load_and_fit_the_features(low):
    """GATE_V2 (the memory's levels; P3's top level) and GATE_V2_LOW (P2's orders 1-5 first): their shapes match
    chain_features' and top_block's columns, fitted on a run's own last 16 batches."""
    gate = sm.Gate.load(low=low)
    assert gate.orders == sm.chain_orders(low) and gate.mode == sm.GATE_MODE
    for i, o in enumerate(gate.orders):
        d = 36 if i == 0 else 37
        assert gate.w[i].numel() == d and gate.mu[i].numel() == gate.sd[i].numel() == d - 1 and gate.disc[i].numel() == 3
        assert (gate.sd[i] > 0).all()
    assert gate.pointer and gate.top["comps"] == list(sm.TOP_COMPS) and gate.top["cols"] == sm.top_cols()
    assert tuple(gate.top["W"].shape) == (1 + len(sm.top_cols()), len(sm.TOP_COMPS)) and (gate.top["sd"] > 0).all()
    assert "own last 16 batches" in json.dumps(gate.spec["fit"])
    # the CPU proof's fit: at 1050 trained steps, on the run's own last 16 batches, a proxy's outputs (a placeholder)
    assert gate.fit_steps == 1050 and gate.spec["fit"]["batches"] == 16 and gate.proxy
    exact = sm.Gate.load(low=low, steps=1050)
    assert exact.exact and "own last batches" in exact.describe() and "PROXY" in exact.describe()
    with pytest.raises(RuntimeError, match="placeholder fitted on a proxy"):
        exact.check_record(1050)
    other = sm.Gate.load(low=low, steps=972)  # a cut's run: the nearest spec, flagged; a record refuses it
    assert not other.exact and "NOT this run's 972 steps" in other.describe()
    with pytest.raises(RuntimeError, match="no constants fitted at 972 trained steps"):
        other.check_record(972)


def test_a_run_gets_the_constants_fitted_at_its_step_count():
    """select_spec: the spec fitted at the run's trained steps, else the nearest (ties to the longer run), else the
    longest; check_record passes only for an own-model spec at the run's step count."""
    def spec(steps, proxy=None):
        return dict(sm.GATE_V2[0], fit=dict(total_steps=steps, **({"proxy": proxy} if proxy else {})))
    specs = [spec(1050, "cpu"), spec(970), spec(950)]
    assert sm.select_spec(specs, 970) == (specs[1], True)
    assert sm.select_spec(specs, 960)[0] is specs[1] and not sm.select_spec(specs, 960)[1]  # a tie: the longer run
    assert sm.select_spec(specs, 1020)[0] is specs[0] and sm.select_spec(specs, 1000)[0] is specs[1]
    assert sm.select_spec(specs, None)[0] is specs[0]
    assert sm.select_spec(specs[0], 970) == (specs[0], False)  # a single spec (a STREAM_RETRIEVAL_GATE file)
    sm.Gate(specs[1], 970).check_record(970)
    with pytest.raises(RuntimeError, match="placeholder"):
        sm.Gate(specs[0], 1050).check_record(1050)
    with pytest.raises(RuntimeError, match="no constants fitted at 950"):
        sm.Gate(spec(None), 950).check_record(950)  # a dump that does not say its step count


# ------------------------------------------------------------------------------------------------ fit_gate_v2.py

def test_fit_gate_v2_end_to_end(tmp_path, toy):
    """A helper FIT dump of the toy run's last 2 batches; a synthetic 'model' that puts 1/4 on the 4 toy tokens
    (so the memory's continuations are worth something); fit_gate_v2 fits, the gain is positive, the JSON loads into
    a Gate that mix_v2 applies, and --module rewrites the constants block of a copy of stream_memory.py."""
    out = run_helper(tmp_path, toy, name="fitfit", fit_k=2)
    path = str(tmp_path / "fitfit.fit")
    fit = out["fit"]
    K = len(LEVELS)
    for r in range(fit["world"]):
        n = fit["n"]
        rng = np.random.default_rng(10 + r)
        nll = np.full(n, math.log(4.2)) + rng.normal(0, 0.05, n)
        x, y, rows, starts = sm.fit_inputs(fit, r)
        sm.save_fit_lm(path, r, nll=nll, ent=np.full(n, math.log(4.0)), mx=np.full(n, -math.log(4.0)),
                       top_v=np.full((n, K), -math.log(4.2)), top_in=np.ones((n, K), bool), top_rank=np.zeros((n, K)))
    spec = fit_gate_v2.main([path, "--maxiter", "200", "--out", str(tmp_path / "gate.json"), "--heldout"])
    assert spec["fit"]["in_sample_mnat"] > 10 and "heldout_mnat" in spec["fit"]
    gate = sm.Gate.load(str(tmp_path / "gate.json"))
    x, y, rows, starts = sm.fit_inputs(fit, 0)
    data = np.load(path + ".lm.rank0.npz")
    lm = {k: torch.from_numpy(data[k]) for k in data.files}
    mixed = sm.mix_v2(torch.from_numpy(data["nll"]), y, x, rows, lm, gate, starts=starts)
    assert float((torch.from_numpy(data["nll"]) - mixed).mean()) > 0.01
    copy = tmp_path / "stream_memory.py"
    copy.write_text(MODULE.read_text())
    for part in ("stream_lowtables", "stream_pointer"):  # stream_memory.py imports its siblings
        (tmp_path / f"{part}.py").write_text(MODULE.with_name(f"{part}.py").read_text())
    fit_gate_v2.write_module(spec, copy)
    spec2 = importlib.util.spec_from_file_location("sm_copy", copy)
    m2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(m2)
    # the spec of the toy run's step count, fitted on its "own model": it replaces the shipped proxy placeholder
    assert spec["fit"]["total_steps"] == fit["steps"] == len(toy["steps"]) and "proxy" not in spec["fit"]
    assert m2.GATE_V2 == [json.loads(json.dumps(spec))] and m2.Gate.load(low=False).orders == list(LEVELS)
    assert m2.GATE_V2_LOW == sm.GATE_V2_LOW  # the other block is left as it was
    gate = m2.Gate.load(low=False, steps=fit["steps"])
    assert gate.exact and gate.check_record(fit["steps"]) is None
    # a second step count's spec joins it; the same step count again replaces it; a proxy's joins as a placeholder
    other = dict(spec, fit=dict(spec["fit"], total_steps=fit["steps"] + 10))
    fit_gate_v2.write_module(other, copy)
    fit_gate_v2.write_module(dict(spec, fit=dict(spec["fit"], note="again")), copy)
    fit_gate_v2.write_module(dict(spec, fit=dict(spec["fit"], total_steps=999, proxy="cpu")), copy)
    got = fit_gate_v2.read_module(copy, "GATE_V2")
    assert [(g["fit"]["total_steps"], g["fit"]["note"], g["fit"].get("proxy")) for g in got] == \
        [(999, spec["fit"]["note"], "cpu"), (fit["steps"] + 10, spec["fit"]["note"], None), (fit["steps"], "again", None)]


def test_fit_gate_v2_with_low_orders_and_a_top_component(tmp_path, toy):
    """The P2 / P3 contract: exact low-order counts (low_orders, low_<field>) join the chain below the memory's levels,
    a top component (top_comps / top_avail / top_phi) gets the softmax gate; the fitted Gate applies through mix_v2."""
    out = run_helper(tmp_path, toy, name="lowtop", fit_k=2)
    path = str(tmp_path / "lowtop.fit")
    fit = out["fit"]
    n, lows = fit["n"], [1, 2]
    for r in range(fit["world"]):
        rng = np.random.default_rng(20 + r)
        x, y, rows, starts = sm.fit_inputs(fit, r)
        yn = y.numpy()
        low = {}
        for o in lows:  # exact counts of the toy's 4 tokens: the target seen C times among N
            N = rng.integers(1, 40, n)
            C = np.minimum(N, rng.binomial(N, 0.4))
            D = np.minimum(N, 4)
            low[o] = dict(N=N, C=C, D=D, M=np.maximum(C, (N + 3) // 4), n1=np.zeros(n, int), n2=np.zeros(n, int),
                          top=np.where(rng.random(n) < 0.4, yn, (yn + 1) % 4))
        avail = rng.random(n) < 0.3
        ptr = np.where(rng.random(n) < 0.6, yn, (yn + 2) % 4)
        K = len(lows) + len(LEVELS)
        sm.save_fit_lm(path, r, nll=np.full(n, math.log(4.2)), ent=np.full(n, math.log(4.0)), mx=np.full(n, -1.3),
                       top_v=np.full((n, K), -1.4), top_in=np.ones((n, K), bool), top_rank=np.zeros((n, K)),
                       low_orders=np.array(lows), top_names=np.array(["ptr"]),
                       **{f"low_{f}": np.stack([low[o][f] for o in lows], 1) for f in fit_gate_v2.LOW_FIELDS},
                       top_comps=((ptr == yn) & avail).astype(np.float64)[:, None], top_avail=avail[:, None],
                       top_phi=np.stack([np.ones(n), rng.normal(size=n)], 1))
    spec = fit_gate_v2.main([path, "--maxiter", "150", "--out", str(tmp_path / "g.json")])
    assert spec["orders"] == lows + list(LEVELS) and spec["top"]["comps"] == ["ptr"]
    assert np.asarray(spec["top"]["W"]).shape == (2, 1) and spec["fit"]["in_sample_mnat"] > 10
    gate = sm.Gate.load(str(tmp_path / "g.json"))
    data = np.load(path + ".lm.rank0.npz")
    x, y, rows, starts = sm.fit_inputs(fit, 0)
    T = torch.from_numpy
    low = {o: {f: T(data[f"low_{f}"][:, i]) for f in fit_gate_v2.LOW_FIELDS} for i, o in enumerate(lows)}
    top = dict(comps=T(data["top_comps"]), avail=T(data["top_avail"]), phi=T(data["top_phi"]))
    lm = {k: T(data[k]) for k in ("ent", "mx", "top_v", "top_in", "top_rank")}
    nll = T(data["nll"])
    mixed = sm.mix_v2(nll, y, x, rows, lm, gate, low=low, top=top, starts=starts)
    assert torch.isfinite(mixed).all() and float((nll - mixed).mean()) > 0.01
    assert sm.top_tokens(rows, gate, low).shape == (n, K)
