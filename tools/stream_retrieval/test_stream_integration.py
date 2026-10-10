"""CPU tests of stream retrieval v2 as one helper: P2 (stream_lowtables.c) and P3 (stream_pointer.c) wired into
stream_memory.c, their rows in the rows file and the FIT dump, and the eval side that turns them into the mixture
(stream_memory.py: parse_low / parse_ptr, DeviceRows, top_block, chunk_mix, StreamEval, lm_features, fit_dump's
fit_lm_outputs) and fit_gate_v2.py on such a dump.

Standalone, like test_stream_memory_v2.py (whose toy data and brute-force walk this reuses): the overlay's modules are
loaded by path and the helper is compiled into a temporary directory.
  python -m pytest tools/stream_retrieval/test_stream_integration.py -q

  rows     the helper's P2 rows equal stream_lowtables.brute_force_rows, its P3 rows equal stream_pointer.run on the
           brute-force walk's candidates, P1's records are unchanged by the parts; bit-identical across query threads,
           insertion threads and message batching; FIT: P2 against the tables before the last K steps (held), P3
           against the frozen memory; nothing reads t's target before row t (P2: C(y) only)
  eval     collect gives every rank its own P2 / P3 rows; the top block's columns are causal; the whole mixture (chain
           over P2's orders and the memory's levels, then the top softmax) sums to 1 over the vocabulary;
           lm_features = lm_top_features; StreamEval through a model's side-output buffers = chunk_mix
  fit      fit_lm_outputs -> save_fit_lm -> fit_gate_v2.py on a P1 + P2 + P3 dump -> a Gate that Gate.for_run accepts
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_stream_memory_v2 as V  # noqa: E402  (its toy data and brute-force walk; the module under test is V.sm)
import fit_gate_v2  # noqa: E402

sm, toy = V.sm, V.toy
LT, SP = sm.stream_lowtables, sm.stream_pointer
BOS, SEP, KEY, CAP, MAXLEN, MAXVISIT = V.BOS, V.SEP, V.KEY, V.CAP, V.MAXLEN, V.MAXVISIT
WORLD, CHUNK, VAL_STEPS = 2, 4096, 2
torch.set_num_threads(2)


# ------------------------------------------------------------------------------------------------ helpers

def run_parts(tmp, toy, *, name, steps=None, low=True, pointer=True, fit_k=0, threads=3, low_threads=3, hash_bits=22,
              val_file=None, keep=False):
    """The helper on the toy run with its parts; returns the rows of every val chunk in global order and the stream."""
    steps = steps or toy["steps"]
    stream_tokens = sum(e - a for st in steps for r in st for a, e in r)
    m = sm.StreamMemory(train_files=[str(toy["shard"])], val_file=str(val_file or toy["val_file"]),
                        total_steps=len(steps), stream_tokens=stream_tokens, world=WORLD, rank=0, master=True,
                        val_tokens=WORLD * CHUNK * VAL_STEPS, chunk=CHUNK, device="cpu", threads=threads,
                        hash_bits=hash_bits, rows_dir=str(tmp), dump_path=str(tmp / f"{name}.dump"), low=low,
                        pointer=pointer, low_threads=low_threads,
                        fit_path=str(tmp / f"{name}.fit") if fit_k else None, fit_k=fit_k or sm.FIT_K)
    for st in steps:
        m.on_spans(0, [[a for a, _ in r] for r in st], [[e for _, e in r] for r in st])
    m.go()
    m.collect()
    out = dict(recs=[], low=[], ptr=[], hdr=None)
    lay = m.layout
    for c in range(WORLD * VAL_STEPS):  # chunk c = step * world + rank lives in region (rank, step)
        s, r = divmod(c, WORLD)
        region = r * VAL_STEPS + s
        out["recs"].append(np.frombuffer(m.map, dtype=sm.REC_DTYPE, count=int(m.counts[r, s]),
                                         offset=sm.ROWS_OFFSET + region * m.region_bytes).copy())
        if low:
            out["low"].append(np.frombuffer(m.map, dtype=LT.ROW, count=CHUNK * 5,
                                            offset=lay["low"] + region * CHUNK * 5 * LT.ROW.itemsize).reshape(CHUNK, 5).copy())
        if pointer:
            out["ptr"].append(np.frombuffer(m.map, dtype=SP.ROW_DTYPE, count=int(m.pcounts[r, s]),
                                            offset=lay["ptr"] + region * CHUNK * SP.ROW_BYTES).copy())
    if fit_k:
        m.fit_go()
        out["fit"] = m.fit_wait()
    out["hdr"] = [int(v) for v in m.hdr[:sm.H_LOW_LATE_PENDING + 1]]
    out["stats"] = m.stats()
    if keep:
        out["memory"] = m
    else:
        m.close()
    out["stream"] = np.concatenate([[SEP], np.fromfile(tmp / f"{name}.dump", dtype=np.uint16)]).astype(np.uint16)
    return out


def reference_candidates(stream, x, run, hash_bits, limit=None):
    """The helper's walk in plain Python (test_stream_memory_v2's reference): every position's candidates (memory
    entry, match length capped at min(run, MAXLEN)) in walk order, as CSR arrays for stream_pointer.run."""
    limit = stream.size if limit is None else limit
    seg = np.zeros(stream.size, dtype=np.int64)
    r = 0
    for j, v in enumerate(stream):
        r = 0 if v == SEP else r + 1
        seg[j] = r
    chains = {}
    for j in range(min(limit, stream.size - 1)):
        if seg[j] >= KEY and stream[j + 1] != SEP:
            chains.setdefault(V.bucket(stream[j - KEY + 1:j + 1], hash_bits), []).append(j)
    off, pos, ln = [0], [], []
    for t in range(x.size):
        if run[t] >= KEY:
            ctx = x[t - KEY + 1:t + 1]
            visited = nc = 0
            for j in reversed(chains.get(V.bucket(ctx, hash_bits), [])):
                if nc >= CAP or visited >= MAXVISIT:
                    break
                visited += 1
                if not np.array_equal(stream[j - KEY + 1:j + 1], ctx):
                    continue
                length, lim = KEY, min(run[t], MAXLEN)
                while length < lim and stream[j - length] == x[t - length]:
                    length += 1
                pos.append(j)
                ln.append(length)
                nc += 1
        off.append(len(pos))
    return np.array(off, np.uint64), np.array(pos, np.uint32), np.array(ln, np.uint8)


def global_ptr(rows: list) -> np.ndarray:
    out = []
    for c, r in enumerate(rows):
        r = r.copy()
        r["pos"] += np.uint32(c * CHUNK)
        out.append(r)
    return np.concatenate(out)


def val_xy(toy):
    n = WORLD * CHUNK * VAL_STEPS
    return toy["val"][:n], toy["val"][1:n + 1]


# ------------------------------------------------------------------------------------------------ rows

def test_parts_equal_their_references(tmp_path, toy):
    out = run_parts(tmp_path, toy, name="ref")
    x, y = val_xy(toy)
    # P2: the exact counts of the stream (sort-based brute force), every val position, segments at BOS and chunks
    want = LT.brute_force_rows(out["stream"][1:], x, y, chunk=CHUNK)
    got = np.concatenate(out["low"])
    assert got.tobytes() == want.tobytes()
    assert (got["N"] > 0).mean() > 0.9 and (got["C"][:, 4] > 0).any() and (got["N"][:, 4] == 0).any()
    # P3: stream_pointer.run on the brute-force walk's candidates
    runs = V.runs_of(x, CHUNK)
    off, pos, ln = reference_candidates(out["stream"], x, runs, 22)
    want = SP.run(out["stream"], 0, x, off, pos, ln, y=y, runs=runs.astype(np.uint32), chunk=CHUNK)
    got = global_ptr(out["ptr"])
    assert got.size == want.size > 1000 and got.tobytes() == want.tobytes()
    assert ((got["flags"] & SP.F_HP) != 0).any() and ((got["flags"] & SP.F_HS) != 0).any()
    assert out["hdr"][sm.H_PTR_ROWS] == got.size and out["hdr"][sm.H_LOW_ORDERS] == 5 and out["hdr"][sm.H_PTR] == 1
    # P2's drain ran beside the P1 / P3 queries (its own thread from GO), then its rows: the statistics say so
    h = out["hdr"]
    assert h[sm.H_LOW_DRAIN_NS] > 0 and h[sm.H_LOW_WAIT_NS] <= h[sm.H_T_DONE] - h[sm.H_T_GO]
    assert h[sm.H_LOW_LATE_PENDING] <= h[sm.H_LOW_MAX_PENDING] and "drained" in out["stats"]
    # P1's records do not change with the parts (P3's whole-segment query blocks included)
    plain = run_parts(tmp_path, toy, name="plain", low=False, pointer=False)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(out["recs"], plain["recs"]))
    assert not plain["low"] and not plain["ptr"] and plain["hdr"][sm.H_LOW_ORDERS] == 0


def test_parts_are_bit_identical_across_threads_and_batching(tmp_path, toy):
    steps = toy["steps"][:4]
    flat = [s for st in steps for s in st[0]]
    regrouped = [[flat[:3], []], [flat[3:500], []], [flat[500:], []]]
    ref = run_parts(tmp_path, toy, name="t1", steps=steps, threads=1, low_threads=1)
    for name, kw in (("t5", dict(threads=5, low_threads=4, steps=steps)), ("batched", dict(steps=regrouped))):
        got = run_parts(tmp_path, toy, name=name, **kw)
        for part in ("recs", "low", "ptr"):
            assert all(a.tobytes() == b.tobytes() for a, b in zip(ref[part], got[part])), (name, part)


def test_fit_parts_are_the_last_batches_against_the_memory_before_them(tmp_path, toy):
    """fit_k = 2 with P2 and P3: P2's FIT rows against the tables before the last 2 steps (their blocks held, then
    inserted: the val rows equal a run without FIT), P3's against the frozen memory (segments restart at each batch).
    test_stream_fit_freeze.py checks every part against a helper whose memory physically ends at the freeze."""
    out = run_parts(tmp_path, toy, name="fit", fit_k=2)
    plain = run_parts(tmp_path, toy, name="nofit")
    for part in ("recs", "low", "ptr"):
        assert all(a.tobytes() == b.tobytes() for a, b in zip(out[part], plain[part])), part
    fit = out["fit"]
    assert fit["low"] is not None and fit["ptr"] is not None and fit["low"].shape == (2, 598, 5)
    freeze = fit["freeze"]
    frozen = out["stream"][:freeze]
    assert frozen[-1] == SEP
    for r in range(2):
        x, y = fit["tokens"][r, :, 0], fit["tokens"][r, :, 1]
        want = LT.brute_force_rows(frozen[1:], x, y, chunk=299)  # two batches of 299 positions: chunk = batch
        assert fit["low"][r].tobytes() == want.tobytes(), r
        unfrozen = LT.brute_force_rows(out["stream"][1:], x, y, chunk=299)
        assert unfrozen.tobytes() != want.tobytes()  # the last batches themselves are not in the FIT tables
        starts = np.zeros(x.size, dtype=bool)
        starts[[0, 299]] = True
        runs = V.runs_of(x, 10 ** 9, starts)
        off, pos, ln = reference_candidates(out["stream"], x, runs, 22, limit=freeze)
        want = SP.run(out["stream"], 0, x, off, pos, ln, y=y, runs=runs.astype(np.uint32), chunk=0)
        got = np.frombuffer(fit["ptr"][r].tobytes(), dtype=SP.ROW_DTYPE)
        assert got.size == want.size > 50 and got.tobytes() == want.tobytes(), r
    assert out["hdr"][sm.H_FIT_PTR_ROWS] == sum(p.shape[0] for p in fit["ptr"])


def test_no_part_reads_the_target_before_its_row(tmp_path, toy):
    """Changing val[t + 1:] (t's target and everything after): P3's rows <= t and P2's rows < t are byte-identical;
    at t P2 changes only C(y) (the target's count, read as a gather); later rows change (not vacuous)."""
    base = run_parts(tmp_path, toy, name="base")
    val = toy["val"]
    t = 4096 * 2 + 1700  # chunk 2
    rng = np.random.default_rng(1)
    future = val.copy()
    future[t + 1:] = rng.integers(0, 4, future.size - t - 1)
    future[t + 1::97] = BOS
    future[t + 1] = (int(val[t + 1]) + 1) % 4
    f = V.write_shard(tmp_path / "future_val.bin", future)
    changed = run_parts(tmp_path, toy, name="future", val_file=f)
    lb, lc = np.concatenate(base["low"]), np.concatenate(changed["low"])
    assert lb[:t].tobytes() == lc[:t].tobytes() and lb[t + 1:].tobytes() != lc[t + 1:].tobytes()
    for fld in ("N", "M", "D", "n1", "n2", "top"):
        assert np.array_equal(lb[t][fld], lc[t][fld]), fld
    pb, pc = global_ptr(base["ptr"]), global_ptr(changed["ptr"])
    assert pb[pb["pos"] <= t].tobytes() == pc[pc["pos"] <= t].tobytes()
    assert pb[pb["pos"] > t].tobytes() != pc[pc["pos"] > t].tobytes()


def test_collect_gives_every_rank_its_own_parts(tmp_path, toy):
    out = run_parts(tmp_path, toy, name="col", keep=True)
    m = out.pop("memory")
    for rank in range(WORLD):
        m.rank = rank
        dr = m.collect()
        for s in range(VAL_STEPS):
            c = s * WORLD + rank
            low = dr.chunk_low(s)
            for i, o in enumerate(sm.LOW_ORDERS):
                for fld in sm.LOW_FIELDS:
                    assert np.array_equal(low[o][fld].numpy(), out["low"][c][fld][:, i].astype(np.int64)), (rank, s, o, fld)
            ptr = dr.chunk_ptr(s)
            want = out["ptr"][c]
            for fld in ("pos", "flags", "pred", "vtop", "src_li", "mem_top", "d_npos"):
                assert np.array_equal(ptr[fld].numpy(), want[fld].astype(np.int64)), fld
            for fld in ("vw", "score", "d_ema90"):
                assert np.array_equal(ptr[fld].numpy(), want[fld]), fld
            assert np.array_equal(ptr["src_n"].numpy(), want["src_n"].astype(np.int64))
    m.close()


# ------------------------------------------------------------------------------------------------ eval

def chunk_inputs(out, toy, c, rng, vocab_p=None):
    """Chunk c's parts as the eval sees them, with a synthetic model: p puts 0.95 on the toy's 4 tokens."""
    x, y = val_xy(toy)
    lo = c * CHUNK
    xc = torch.from_numpy(x[lo:lo + CHUNK].astype(np.int64))
    yc = torch.from_numpy(y[lo:lo + CHUNK].astype(np.int64))
    rows = sm.Rows(torch.from_numpy(out["recs"][c].view(np.uint8).reshape(-1, sm.REC_BYTES).copy()), CHUNK)
    low = sm.parse_low(torch.from_numpy(out["low"][c].view(np.uint8).reshape(-1).copy()), CHUNK)
    ptr = sm.parse_ptr(torch.from_numpy(out["ptr"][c].view(np.uint8).reshape(-1, SP.ROW_BYTES).copy()))
    return xc, yc, rows, low, ptr


def random_gate(rng, orders, pointer=True):
    spec = dict(version=2, mode=sm.GATE_MODE, orders=list(orders), w=[], disc=[], mu=[], sd=[], top=None, fit={})
    for i, _ in enumerate(orders):
        d = 36 if i == 0 else 37  # chain_features' columns in GATE_MODE (the first order has no "narrowed" column)
        spec["w"].append(list(rng.normal(0, 0.3, d) + np.r_[-1.0, np.zeros(d - 1)]))
        spec["disc"].append(list(rng.normal(0, 0.5, 3)))
        spec["mu"].append([0.0] * (d - 1))
        spec["sd"].append([1.0] * (d - 1))
    if pointer:
        d = 1 + len(sm.top_cols())
        spec["top"] = dict(comps=list(sm.TOP_COMPS), cols=sm.top_cols(), W=rng.normal(0, 0.2, (d, 3)).tolist(),
                           mu=[0.0] * (d - 1), sd=[1.0] * (d - 1))
    return sm.Gate(spec)


def fake_lm(rng, P, K):
    return dict(ent=torch.from_numpy(rng.uniform(0.5, 3, P)).float(), mx=torch.from_numpy(-rng.uniform(0.1, 2, P)).float(),
                top_v=torch.from_numpy(-rng.uniform(0, 8, (P, K))).float(), top_in=torch.from_numpy(rng.random((P, K)) < 0.6),
                top_rank=torch.from_numpy(rng.integers(0, 33, (P, K))))


def test_top_block_is_causal_and_has_its_columns(tmp_path, toy):
    out = run_parts(tmp_path, toy, name="tb")
    rng = np.random.default_rng(3)
    x, y, rows, low, ptr = chunk_inputs(out, toy, 2, rng)
    orders = sm.chain_orders(True)
    K = len(orders) + 3
    nll = torch.from_numpy(rng.exponential(2.0, CHUNK))
    lm = fake_lm(rng, CHUNK, K)
    counts = dict(rows.level_counts(y))
    counts.update(low)
    starts = sm.seg_starts(x)
    top = sm.top_block(ptr, CHUNK, x, y, starts, nll, lm, len(orders), counts, orders, rows)
    idx = ptr["pos"]
    assert top["phi"].shape == (CHUNK, 1 + len(sm.top_cols())) and torch.isfinite(top["phi"]).all()
    off = torch.ones(CHUNK, dtype=torch.bool)
    off[idx] = False
    assert not top["phi"][off].any() and not top["avail"][off].any() and (top["phi"][idx, 0] == 1).all()
    assert top["avail"][idx].any(1).all()  # every P3 row has a component
    # t's target and its NLL (and everything after) leave the columns at <= t unchanged; t's components follow y
    t = int(idx[len(idx) // 2])
    y2, nll2 = y.clone(), nll.clone()
    y2[t:] = (y2[t:] + 1) % 4
    nll2[t:] = nll2[t:] + 1.5
    counts2 = dict(rows.level_counts(y2))
    counts2.update({o: dict(c, C=torch.zeros_like(c["C"])) for o, c in low.items()})
    top2 = sm.top_block(ptr, CHUNK, x, y2, starts, nll2, lm, len(orders), counts2, orders, rows)
    assert torch.equal(top["phi"][:t + 1], top2["phi"][:t + 1]) and not torch.equal(top["phi"], top2["phi"])
    assert torch.equal(top["comps"][:t], top2["comps"][:t])


def test_the_mixture_sums_to_one_over_the_vocabulary(tmp_path, toy):
    """At P3-active positions t of a chunk: q_t(v) for every v of the toy's 4 tokens (P2's C(v), the candidates' C_L(v),
    the components at v; all gate features fixed by positions < t) sums to sum_v p_t(v) = 1."""
    out = run_parts(tmp_path, toy, name="norm")
    rng = np.random.default_rng(4)
    x, y, rows, low, ptr = chunk_inputs(out, toy, 1, rng)
    orders = sm.chain_orders(True)
    gate = random_gate(rng, orders)
    lm = fake_lm(rng, CHUNK, gate.tokens())
    p = rng.dirichlet(np.ones(4), CHUNK)  # the model: a distribution over the 4 toy tokens at every position
    stream = out["stream"]
    lt = LT.LowTables(sm.LOW_ORDERS, threads=1, expected_positions=stream.size)
    lt.insert_block(stream[1:])
    lt.finish()
    xs = x.numpy().astype(np.uint16)
    runs = V.runs_of(xs, CHUNK)
    checked = 0
    for t in [int(v) for v in ptr["pos"][::97][:12]]:
        total = 0.0
        for v in range(4):
            yv = y.clone()
            yv[t] = v
            nll = torch.from_numpy(-np.log(p[np.arange(CHUNK), np.clip(yv.numpy(), 0, 3)] + 1e-9))
            lowv = {o: {f: c[f].clone() for f in c} for o, c in low.items()}
            row = lt.query_one(xs, t, v, int(runs[t]))
            for i, o in enumerate(sm.LOW_ORDERS):
                lowv[o]["C"][t] = float(row["C"][i])
            q = torch.exp(-sm.chunk_mix(nll, x, yv, rows, lowv, ptr, lm, gate).to(torch.float64))[t] - 1e-9
            total += float(q)
        assert abs(total - 1) < 1e-6, (t, total)
        checked += 1
    lt.close()
    assert checked >= 10


def test_lm_features_follow_lm_top_features():
    rng = np.random.default_rng(5)
    R, Vv, K = 300, 600, 7
    logits = torch.from_numpy(rng.normal(size=(R, Vv)) * 3).float()
    logits[:, 5] = 40.0  # a column that is no token (CPLM's <copy>): dropped to the mask's -60 first
    toks = torch.from_numpy(rng.integers(0, Vv, (R, K)))
    out = sm.lm_features(logits, toks, torch.tensor([5], dtype=torch.int32))
    ref_logits = logits.clone()
    ref_logits[:, 5] = -60.0
    ref = sm.lm_top_features(ref_logits.log_softmax(-1), toks)
    got = sm.unpack_lm(out, K)
    for k in ("ent", "mx", "top_v"):
        torch.testing.assert_close(got[k], ref[k].float(), rtol=1e-5, atol=1e-5)
    assert torch.equal(got["top_in"], ref["top_in"]) and torch.equal(got["top_rank"], ref["top_rank"])
    assert out.shape == (R, 2 + 3 * K) and out.dtype == torch.float32


class FakeModel:
    """The hooks' contract: query tokens in stream_lm_tok, the eval forward fills stream_lm_out (model/gpt.py)."""

    def __init__(self, P, K, logits):
        self.stream_lm_tok = torch.zeros(P, K, dtype=torch.long)
        self.stream_lm_out = torch.zeros(P, 2 + 3 * K)
        self.logits = logits

    def forward(self, y):
        for a in range(0, self.logits.shape[0], 1000):  # in slabs, as the eval CE loop
            self.stream_lm_out[a:a + 1000] = sm.lm_features(self.logits[a:a + 1000], self.stream_lm_tok[a:a + 1000])
        return -self.logits.log_softmax(-1).gather(1, y.clamp(max=self.logits.shape[1] - 1)[:, None])[:, 0]


def test_stream_eval_through_the_models_side_output(tmp_path, toy):
    out = run_parts(tmp_path, toy, name="se", keep=True)
    m = out.pop("memory")
    m.rank = 1
    dr = m.collect()
    rng = np.random.default_rng(6)
    gate = random_gate(rng, sm.chain_orders(True))
    ev = sm.StreamEval(dr, gate)
    x, y = val_xy(toy)
    for s in range(VAL_STEPS):
        c = s * WORLD + 1
        xc = torch.from_numpy(x[c * CHUNK:(c + 1) * CHUNK].astype(np.int64))
        yc = torch.from_numpy(y[c * CHUNK:(c + 1) * CHUNK].astype(np.int64))
        logits = torch.from_numpy(rng.normal(size=(CHUNK, 64)) * 2).float()
        model = FakeModel(CHUNK, gate.tokens(), logits)
        batch = type("B", (), dict(inputs=xc.to(torch.int32), targets=yc))
        nll, mixed = ev.evaluate(s, batch, lambda: model.forward(yc), model)
        tok = model.stream_lm_tok
        rows, low, ptr = dr.chunk_rows(s), dr.chunk_low(s), dr.chunk_ptr(s)
        assert torch.equal(tok, sm.query_tokens(rows, gate.orders, low, ptr, True))
        assert torch.equal(tok[ptr["pos"], -3], torch.where((ptr["flags"] & SP.F_HP) != 0, ptr["pred"], 0))
        lm = sm.lm_top_features(logits.log_softmax(-1), tok)
        want = sm.chunk_mix(nll, xc, yc, rows, low, ptr, lm, gate)
        torch.testing.assert_close(mixed, want, rtol=1e-5, atol=1e-5)
        touched = torch.zeros(CHUNK, dtype=torch.bool)
        touched[ptr["pos"]] = True
        touched |= rows.hit
        for o in sm.LOW_ORDERS:
            touched |= low[o]["N"] > 0
        assert torch.equal(mixed[~touched], nll[~touched]) and not torch.equal(mixed[touched], nll[touched])
    m.close()


# ------------------------------------------------------------------------------------------------ fit

def test_fit_dump_to_gate_with_all_parts(tmp_path, toy):
    """The FIT path end to end on the toy run: the helper's P1 + P2 + P3 dump, the model's outputs through
    fit_lm_outputs (a synthetic LM that knows the 4 toy tokens), fit_gate_v2.py, then Gate.for_run's checks and the
    mixture on the FIT positions."""
    out = run_parts(tmp_path, toy, name="ff", fit_k=2)
    fit = out["fit"]
    d = tmp_path / "ff.fit"
    orders = sm.chain_orders(True)
    rng = np.random.default_rng(7)
    for r in range(fit["world"]):
        n = fit["n"]
        logits = torch.full((n, 50257), -30.0)
        logits[:, :4] = torch.from_numpy(rng.normal(0, 0.3, (n, 4))).float()
        y = torch.from_numpy(fit["tokens"][r, :, 1].astype(np.int64))

        def run_chunk(lo, tok, logits=logits, y=y):
            P = tok.shape[0]
            model = FakeModel(P, tok.shape[1], logits[lo:lo + P])
            model.stream_lm_tok.copy_(tok)
            return model.forward(y[lo:lo + P]), model.stream_lm_out.clone()
        lmo = sm.fit_lm_outputs(fit, r, orders, True, 299, run_chunk, "cpu")
        assert lmo["top_v"].shape == (n, len(orders) + 3)
        sm.save_fit_lm(str(d), r, **lmo)
    spec = fit_gate_v2.main([str(d), "--maxiter", "120", "--out", str(tmp_path / "g.json")])
    assert spec["orders"] == orders and spec["top"]["comps"] == list(sm.TOP_COMPS) and spec["top"]["cols"] == sm.top_cols()
    assert np.asarray(spec["top"]["W"]).shape == (1 + len(sm.top_cols()), 3) and spec["fit"]["in_sample_mnat"] > 50
    assert fit_gate_v2.module_block(spec) == "GATE_V2_LOW"
    gate = sm.Gate(json.loads((tmp_path / "g.json").read_text()))
    x, y, rows, starts = sm.fit_inputs(fit, 0)
    low, ptr = sm.fit_parts(fit, 0)
    data = np.load(f"{d}.lm.rank0.npz")
    lm = {k: torch.from_numpy(data[k]) for k in ("ent", "mx", "top_v", "top_in", "top_rank")}
    nll = torch.from_numpy(data["nll"]).to(torch.float64)
    mixed = sm.chunk_mix(nll, x, y, rows, low, ptr, lm, gate, starts=starts)
    assert torch.isfinite(mixed).all() and float((nll - mixed).mean()) > 0.05


def test_gate_for_run_checks_the_parts(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    path = tmp_path / "g.json"
    for orders, pointer, low_flag, ptr_flag, msg in (
            (sm.chain_orders(False), True, True, True, "orders"),
            (sm.chain_orders(True), False, True, True, "no P3 top level"),
            (sm.chain_orders(True), True, True, False, "needs P3"),
            (sm.chain_orders(True), True, True, True, None)):
        path.write_text(json.dumps(random_gate(rng, orders, pointer).spec))
        monkeypatch.setenv("STREAM_RETRIEVAL_GATE", str(path))
        if msg:
            with pytest.raises(RuntimeError, match=msg):
                sm.Gate.for_run(low=low_flag, pointer=ptr_flag)
        else:
            assert sm.Gate.for_run(low=low_flag, pointer=ptr_flag).tokens() == len(orders) + 3


def test_rows_layout_is_the_helpers(tmp_path, toy):
    for low, pointer in ((False, False), (True, False), (False, True), (True, True)):
        out = run_parts(tmp_path, toy, name=f"lay{int(low)}{int(pointer)}", low=low, pointer=pointer,
                        steps=toy["steps"][:4])
        lay = sm.rows_layout(WORLD, VAL_STEPS, CHUNK, low, pointer)
        assert out["hdr"][sm.H_LOW_OFFSET] == lay["low"] and out["hdr"][sm.H_PTR_OFFSET] == lay["ptr"]
        assert out["hdr"][sm.H_LOW_ROW_BYTES] == LT.ROW.itemsize and out["hdr"][sm.H_PTR_ROW_BYTES] == SP.ROW_BYTES
        assert lay["low"] % sm.REGION_ALIGN == 0 and lay["ptr"] % sm.REGION_ALIGN == 0
