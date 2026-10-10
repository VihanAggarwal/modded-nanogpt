"""The FIT path's rows (P1, P2, P3) are EXACTLY the rows those positions get when queried, as val, by a helper whose
memory physically ends at the freeze (it never received the run's last K steps): RULES_CHECK's "P1's walk skips newer
entries, P3 reads only below that point, P2 holds those steps' table insertions until their rows are queried".

test_stream_integration.py's FIT test checks P3's FIT rows against stream_pointer.run given the whole stream (the FIT
region included) with only the candidates limited to the freeze, so a P3 read past the freeze (a pointer advance, a
deletion or resync scan, a source document's bounds) would agree with that reference. Here run B's memory ends at the
freeze, so any read past it changes a row; control C (the memory with the FIT steps) shows that it would.

The data make a leak visible: the K FIT batches repeat each other (batch 2 = batch 1's documents again) and carry a
long document in a vocabulary the pre-freeze memory never saw, so any part that saw the FIT region (P1's walk, P2's
tables, P3's pointer or source copy) would give rows at FIT batch 2 that a frozen memory cannot. Also requirement (3):
doc-state and source sets restart at every chunk start (here chunk = L), not only at BOS.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_stream_memory_v2 as V  # noqa: E402  (loads the overlay's stream_memory.py by path)

sm = V.sm
LT, SP = sm.stream_lowtables, sm.stream_pointer
BOS, SEP = V.BOS, V.SEP
WORLD, K, L = 2, 3, 600          # 2 ranks, 3 FIT steps, 600 positions per rank per FIT batch


def build(tmp: Path, seed=0):
    rng = np.random.default_rng(seed)
    pre = V.toy_corpus(rng, 500, vocab=4)                                  # the pre-freeze memory's documents
    rare = np.array([BOS, *rng.integers(100, 400, 900)], dtype=np.uint16)  # never before the freeze
    # FIT documents: copies of pre-freeze documents (hits against the frozen memory) and the rare document
    fit_docs = [rare] + [pre[int(i)] for i in rng.integers(0, 500, 40)]
    docs = list(pre)
    starts_pre = len(docs)
    # three copies of the FIT documents in the shard: FIT batch s reads copy s (batch 2 repeats batch 1, etc.)
    for _ in range(K):
        docs += [d.copy() for d in fit_docs]
    docs += V.toy_corpus(rng, 50, vocab=4)
    shard = V.write_shard(tmp / "fineweb_train_000001.bin", np.concatenate(docs))
    starts = np.cumsum([0] + [d.size for d in docs])
    plain = [(int(starts[i]), int(starts[i]) + docs[i].size) for i in range(starts_pre)]
    steps = [[plain[0:200], []], [plain[200:350], []], [plain[350:], []]]
    fit_steps, nxt = [], starts_pre
    for s in range(K):
        # each FIT step: every rank exactly L + 1 tokens (the loader's batch_size / world + 1), from copy s
        first = starts_pre + s * len(fit_docs)
        b, nxt = V.equal_batches(docs, starts, first, WORLD, L + 1)
        fit_steps.append(b)
    return dict(shard=shard, docs=docs, pre_steps=steps, fit_steps=fit_steps)


def helper(tmp, d, *, name, steps, val_file, val_steps, chunk, fit_k=0, low=True, pointer=True, threads=3):
    stream_tokens = sum(e - a for st in steps for r in st for a, e in r)
    m = sm.StreamMemory(train_files=[str(d["shard"])], val_file=str(val_file), total_steps=len(steps),
                        stream_tokens=stream_tokens, world=WORLD, rank=0, master=True,
                        val_tokens=WORLD * chunk * val_steps, chunk=chunk, device="cpu", threads=threads,
                        hash_bits=16, rows_dir=str(tmp), dump_path=str(tmp / f"{name}.dump"), low=low,
                        pointer=pointer, low_threads=2, low_entries=1 << 15,
                        fit_path=str(tmp / f"{name}.fit") if fit_k else None, fit_k=fit_k or sm.FIT_K)
    for st in steps:
        m.on_spans(0, [[a for a, _ in r] for r in st], [[e for _, e in r] for r in st])
    m.go()
    m.collect()
    out = dict(recs={}, low={}, ptr={})
    lay = m.layout
    for c in range(WORLD * val_steps):
        s, r = divmod(c, WORLD)
        region = r * val_steps + s
        out["recs"][c] = np.frombuffer(m.map, dtype=sm.REC_DTYPE, count=int(m.counts[r, s]),
                                       offset=sm.ROWS_OFFSET + region * m.region_bytes).copy()
        if low:
            out["low"][c] = np.frombuffer(m.map, dtype=LT.ROW, count=chunk * 5,
                                          offset=lay["low"] + region * chunk * 5 * LT.ROW.itemsize).reshape(chunk, 5).copy()
        if pointer:
            out["ptr"][c] = np.frombuffer(m.map, dtype=SP.ROW_DTYPE, count=int(m.pcounts[r, s]),
                                          offset=lay["ptr"] + region * chunk * SP.ROW_BYTES).copy()
    if fit_k:
        m.fit_go()
        out["fit"] = m.fit_wait()
    m.close()
    out["stream"] = np.concatenate([[SEP], np.fromfile(tmp / f"{name}.dump", dtype=np.uint16)]).astype(np.uint16)
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("adv")
    d = build(tmp)
    # a dummy val for run A (its val rows are not compared)
    dummy_val = V.write_shard(tmp / "dummy_val.bin", np.concatenate(V.toy_corpus(np.random.default_rng(9), 300, 4)))
    A = helper(tmp, d, name="A", steps=d["pre_steps"] + d["fit_steps"], val_file=dummy_val, val_steps=2, chunk=2048,
               fit_k=K)
    fit = A["fit"]
    assert fit["n"] == K * L and list(fit["batch_lengths"]) == [L] * K
    assert fit["steps"] == len(d["pre_steps"]) + K  # the FIT header carries the run's trained steps
    # run B: the memory ends at the freeze (it never gets the K FIT steps); its "val" is the FIT batches laid out so
    # that val chunk c = s * WORLD + r holds rank r's FIT batch s (inputs), chunk = L, val_steps = K
    toks = fit["tokens"]  # [world][n][2]
    val = np.zeros(WORLD * K * L + 2, dtype=np.uint16)
    for s in range(K):
        for r in range(WORLD):
            c = s * WORLD + r
            val[c * L:(c + 1) * L] = toks[r, s * L:(s + 1) * L, 0]
    val[-2:] = BOS
    val_file = V.write_shard(tmp / "fitval.bin", val)
    B = helper(tmp, d, name="B", steps=d["pre_steps"], val_file=val_file, val_steps=K, chunk=L)
    # control C: the same "val" against the memory WITH the FIT steps (what a leak would look like)
    C = helper(tmp, d, name="C", steps=d["pre_steps"] + d["fit_steps"], val_file=val_file, val_steps=K, chunk=L)
    return d, A, B, C


def fit_rows_by_chunk(A):
    """A's FIT rows regrouped per (s, r) -> val chunk c = s * WORLD + r, positions relative to the batch."""
    fit = A["fit"]
    recs, low, ptr = {}, {}, {}
    for r in range(WORLD):
        rr = np.frombuffer(fit["recs"][r].tobytes(), dtype=sm.REC_DTYPE)
        pr = np.frombuffer(fit["ptr"][r].tobytes(), dtype=SP.ROW_DTYPE)
        for s in range(K):
            c = s * WORLD + r
            sel = (rr["pos"] >= s * L) & (rr["pos"] < (s + 1) * L)
            x = rr[sel].copy()
            x["pos"] -= np.uint32(s * L)
            recs[c] = x
            selp = (pr["pos"] >= s * L) & (pr["pos"] < (s + 1) * L)
            p = pr[selp].copy()
            p["pos"] -= np.uint32(s * L)
            ptr[c] = p
            low[c] = fit["low"][r, s * L:(s + 1) * L]
    return recs, low, ptr


def test_memory_of_B_is_A_before_the_freeze(runs):
    d, A, B, C = runs
    freeze = A["fit"]["freeze"]
    assert B["stream"].size == freeze and np.array_equal(A["stream"][:freeze], B["stream"])
    assert B["stream"][-1] == SEP


def test_fit_rows_equal_rows_against_a_memory_that_ends_at_the_freeze(runs):
    d, A, B, C = runs
    recs, low, ptr = fit_rows_by_chunk(A)
    for c in range(WORLD * K):
        assert recs[c].tobytes() == B["recs"][c].tobytes(), ("P1", c)
        assert ptr[c].tobytes() == B["ptr"][c].tobytes(), ("P3", c)
        fa, fb = low[c], B["low"][c]
        for f in ("N", "M", "D", "n1", "n2", "top"):
            assert np.array_equal(fa[f], fb[f]), ("P2", c, f)
        # C(y): B's last position of a chunk has a different target (the next chunk's first input)
        assert np.array_equal(fa["C"][:-1], fb["C"][:-1]), ("P2 C", c)


def test_the_control_would_have_caught_a_leak(runs):
    """Not vacuous: against the memory WITH the FIT steps (control C) the rows of FIT batches 2.. differ (they hit
    batch 1's copy, the rare document included) in every part."""
    d, A, B, C = runs
    recs, low, ptr = fit_rows_by_chunk(A)
    for s in range(1, K):
        for r in range(WORLD):
            c = s * WORLD + r
            assert recs[c].tobytes() != C["recs"][c].tobytes(), ("P1", c)
            assert ptr[c].tobytes() != C["ptr"][c].tobytes(), ("P3", c)
            assert not np.array_equal(low[c]["N"], C["low"][c]["N"]), ("P2", c)
    # the rare tokens never match before the freeze: no FIT record at a rare-vocabulary context
    toks = A["fit"]["tokens"]
    for r in range(WORLD):
        rr = np.frombuffer(A["fit"]["recs"][r].tobytes(), dtype=sm.REC_DTYPE)
        x = toks[r, :, 0]
        assert not ((x[rr["pos"]] >= 100) & (x[rr["pos"]] < 400)).any()


def test_not_vacuous(runs):
    d, A, B, C = runs
    recs, low, ptr = fit_rows_by_chunk(A)
    nrec = sum(v.size for v in recs.values())
    nptr = sum(v.size for v in ptr.values())
    hp = sum(int(((v["flags"] & SP.F_HP) != 0).sum()) for v in ptr.values())
    hs = sum(int(((v["flags"] & SP.F_HS) != 0).sum()) for v in ptr.values())
    print(f"FIT records {nrec}, P3 rows {nptr} (pointer {hp}, source {hs}) of {WORLD * K * L} positions")
    assert nrec > 300 and hp > 100 and hs > 100


def test_no_state_crosses_a_chunk_boundary(runs, tmp_path):
    """Requirement (3): doc-state and source sets reset at the 262,144-token chunk boundary (here chunk = L). Scrambling
    every token of chunk 0 (BOS-free, so its last segment would otherwise run into chunk 1) leaves every row of every
    later chunk byte-identical (P1, P2, P3)."""
    d, A, B, C = runs
    # val: pre-freeze documents (so P1 / P3 / P2 are active), with no BOS within 300 tokens of the chunk-0/1 boundary,
    # so one document runs across it (a missing reset would carry chunk 0's state into chunk 1)
    pre = np.concatenate([V.read_shard(d["shard"])[:40000]])
    val = pre[:WORLD * K * L + 2].copy()
    win = val[L - 300:L + 300]
    win[win == BOS] = 1
    assert val[L] != BOS
    rng = np.random.default_rng(5)
    val2 = val.copy()
    val2[:L] = rng.integers(0, 4, L)  # no BOS in chunk 0
    f1 = V.write_shard(tmp_path / "v1.bin", val)
    f2 = V.write_shard(tmp_path / "v2.bin", val2)
    R1 = helper(tmp_path, d, name="R1", steps=d["pre_steps"], val_file=f1, val_steps=K, chunk=L)
    R2 = helper(tmp_path, d, name="R2", steps=d["pre_steps"], val_file=f2, val_steps=K, chunk=L)
    assert R1["ptr"][0].tobytes() != R2["ptr"][0].tobytes()  # chunk 0 did change
    early = R1["ptr"][1]["pos"] < 300
    print(f"chunk 1: {int(early.sum())} P3 rows in its first 300 positions (one document from chunk 0 runs on)")
    assert early.sum() > 20
    for c in range(1, WORLD * K):
        assert R1["recs"][c].tobytes() == R2["recs"][c].tobytes(), ("P1", c)
        assert R1["ptr"][c].tobytes() == R2["ptr"][c].tobytes(), ("P3", c)
        assert R1["low"][c].tobytes() == R2["low"][c].tobytes(), ("P2", c)
