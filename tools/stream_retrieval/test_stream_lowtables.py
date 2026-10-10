"""CPU tests for the exact low-order tables (STREAM_RETRIEVAL_LOW=1: track_1_short/stream_lowtables.c and its binding
stream_lowtables.py). Standalone: they use the overlay's copy (tools/stream_retrieval/arm/track_1_short), compile it
with cc into a temp dir and need no GPU, torch or shards.

  1 exact: every field of every row equals a brute-force reference (sorting, no hashing), on small-vocabulary
    streams (many ties, top swaps, internal BOS, chunk restarts), a FineWeb-like Zipf stream with repeats, any order
    set (1-8, gaps), table sizes down to ~98% load, and the real stream when STREAM_LOWTABLES_STREAM/_VAL are set
  2 the fields' definitions, by enumerating the vocabulary at a position: sum_v C(v) = N, #{C > 0} = D, max = M,
    #{C = 1} = n1, #{C = 2} = n2, top = the lowest-id argmax, C(BOS) = 0 (so C/N is a distribution off BOS)
  3 determinism: the same table bytes (lt_digest) and rows for any thread count and any block boundaries
  (1-3 with every order in the index tier, with none, and with the default: orders >= 4)
  4 the three query paths (lt_query_rows / lt_query_block / lt_query_one) agree
  5 hold / release (FIT positions against the tables before their own blocks), finish refusing held blocks and
    refusing blocks after it, errors that name the full table
  6 causality: a row depends on x[<= t] and y[t] only
  7 the C file compiles warning-free alone and in one translation unit with the helper (no name clashes), and the
    default sizes cover the counts measured on the real 1050-step stream
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ARM = HERE / "arm/track_1_short"
sys.path.insert(0, str(ARM))
import stream_lowtables as L  # noqa: E402

BOS, SEP = L.BOS, L.SEP


# ------------------------------------------------------------------------------------------------ data

def toy_stream(rng, nspans=400, vocab=10, maxlen=60, bos_rate=0.03):
    """Spans of documents (BOS first, a few internal BOS), each followed by SEP, over a tiny vocabulary: every context
    recurs, so counts, ties and top swaps are dense."""
    parts = []
    for _ in range(nspans):
        n = int(rng.integers(1, maxlen))
        doc = rng.integers(0, vocab, n).astype(np.uint16)
        doc[0] = BOS
        doc[rng.random(n) < bos_rate] = BOS
        parts += [doc, np.array([SEP], np.uint16)]
    return np.concatenate(parts)


def toy_val(rng, n, vocab=10, bos_rate=0.03):
    v = rng.integers(0, vocab, n + 1).astype(np.uint16)
    v[rng.random(n + 1) < bos_rate] = BOS
    return v[:-1], v[1:]


_ZM = None


def zipf_tokens(rng, n, vocab=50257):
    """Zipf-Mandelbrot ranks (p ~ (r + 3)^-1.1: the top token ~4%, as GPT-2 tokens of web text) over shuffled ids."""
    global _ZM
    if _ZM is None:
        w = (np.arange(1, vocab) + 3.0) ** -1.1
        _ZM = np.cumsum(w / w.sum())
    ranks = np.minimum(np.searchsorted(_ZM, rng.random(n)), vocab - 2)
    perm = np.random.default_rng(99).permutation(vocab - 1)  # token ids are not ranks; BOS (50256) never drawn here
    return perm[ranks].astype(np.uint16)


def zipf_stream(rng, ntok=300_000, copy_rate=0.3):
    """FineWeb-like: Zipf tokens, documents of geometric lengths, and passages copied from earlier documents (the
    boilerplate the memory exists for)."""
    base = zipf_tokens(rng, ntok)
    out, pos = [], 0
    while pos < ntok:
        n = int(min(rng.geometric(1 / 400), ntok - pos))
        doc = base[pos:pos + n].copy()
        pos += n
        if out and rng.random() < copy_rate:
            src = out[int(rng.integers(len(out)))]
            a = int(rng.integers(0, max(1, src.size - 1)))
            seg = src[a:a + int(rng.integers(5, 80))]
            seg = seg[(seg != BOS) & (seg != SEP)]
            at = int(rng.integers(0, max(1, doc.size)))
            doc = np.concatenate([doc[:at], seg, doc[at:]])
        doc = np.concatenate([[BOS], doc]).astype(np.uint16)
        out.append(doc)
    spans = []
    for d in out:
        spans += [d, np.array([SEP], np.uint16)]
    return np.concatenate(spans)


def zipf_val(rng, stream, n):
    """Val positions: fresh Zipf text with passages copied from the stream, BOS-separated documents."""
    v = zipf_tokens(rng, n + 1)
    for _ in range(n // 200):
        a = int(rng.integers(0, stream.size - 100))
        seg = stream[a:a + int(rng.integers(5, 60))]
        seg = seg[(seg != SEP)]
        at = int(rng.integers(0, n + 1 - seg.size))
        v[at:at + seg.size] = seg
    v[rng.random(n + 1) < 1 / 300] = BOS
    return v[:-1], v[1:]


def split_blocks(stream, nblocks, rng=None):
    """Whole spans per block (cut right after a SEP)."""
    seps = np.flatnonzero(stream == SEP) + 1
    k = min(nblocks - 1, seps.size - 1)
    cuts = np.sort(rng.choice(seps[:-1], k, replace=False)) if rng is not None else \
        seps[np.linspace(0, seps.size - 2, k).astype(int)] if k > 0 else []
    return np.split(stream, np.unique(cuts))


def build(stream, orders=L.ORDERS, threads=3, blocks=1, **kw):
    t = L.LowTables(orders, threads=threads, expected_positions=stream.size, **kw)
    for b in split_blocks(stream, blocks):
        t.insert_block(b)
    t.finish()
    return t


def assert_rows_equal(rows, ref):
    for f in L.FIELDS:
        bad = np.argwhere(rows[f] != ref[f])
        assert not bad.size, f"field {f} differs at (position, order) {bad[:5].tolist()}: {rows[f][tuple(bad[0])]} vs " \
                             f"{ref[f][tuple(bad[0])]}"


# ------------------------------------------------------------------------------------------------ 1 exact

TIERS = [None, 1, 9]  # the default (orders >= 4 in the index tier), every order, none


@pytest.mark.parametrize("tier_from", TIERS)
@pytest.mark.parametrize("seed,vocab,chunk", [(0, 6, 1000), (1, 10, 257), (2, 30, 0), (3, 3, 64)])
def test_rows_equal_brute_force_on_small_vocabularies(seed, vocab, chunk, tier_from):
    rng = np.random.default_rng(seed)
    stream = toy_stream(rng, vocab=vocab)
    x, y = toy_val(rng, 6000, vocab=vocab)
    ref = L.brute_force_rows(stream, x, y, chunk)
    with build(stream, threads=1 + seed % 4, blocks=7, tier_from=tier_from) as t:
        assert_rows_equal(t.query_rows(x, y, chunk, threads=3), ref)
    assert (ref["N"] > 0).mean() > 0.3 and (ref["D"] > 1).any() and (ref["n2"] > 0).any()


@pytest.mark.parametrize("tier_from", [None, 2])
@pytest.mark.parametrize("orders", [(1,), (2, 4), (1, 3, 6, 8), (1, 2, 3, 4, 5, 6, 7, 8)])
def test_any_order_set(orders, tier_from):
    rng = np.random.default_rng(5)
    stream = toy_stream(rng, vocab=4, maxlen=40)
    x, y = toy_val(rng, 4000, vocab=4)
    with build(stream, orders=orders, threads=2, blocks=3, tier_from=tier_from) as t:
        assert_rows_equal(t.query_rows(x, y, 500), L.brute_force_rows(stream, x, y, 500, orders))


@pytest.mark.parametrize("tier_from", TIERS)
def test_rows_equal_brute_force_on_a_zipf_stream_with_repeats(tier_from):
    rng = np.random.default_rng(7)
    stream = zipf_stream(rng)
    x, y = zipf_val(rng, stream, 60_000)
    ref = L.brute_force_rows(stream, x, y, 262144 // 16)
    with build(stream, threads=4, blocks=40, tier_from=tier_from) as t:
        rows = t.query_rows(x, y, 262144 // 16)
        s = t.stats()
    assert_rows_equal(rows, ref)
    assert (ref["N"][:, 4] > 0).mean() > 0.005 and (ref["C"][:, 4] > 0).any()  # the copied passages reach order 5
    assert all(s["orders"][k]["positions"] > 0 for k in L.ORDERS) and not s["failed"]


@pytest.mark.parametrize("tier_from", [None, 1])
def test_rows_stay_exact_at_high_load(tier_from):
    """First-level tables sized for exactly their entries at 0.9 load: long probe runs, wrap-around, overflow
    flags (the stats tier of a tiered order, whose partitions hold a few dozen entries here, gets twice its own)."""
    rng = np.random.default_rng(11)
    stream = zipf_stream(rng, ntok=120_000)
    x, y = zipf_val(rng, stream, 20_000)
    with build(stream, threads=2, tier_from=tier_from) as t:
        s = t.stats()["orders"]
    big = (3, 4, 5)  # orders 1-2 keep their defaults: their pair partitions are Zipf-skewed on a stream this small
    ctx = [s[k]["contexts"] if k in big else 0 for k in L.ORDERS]
    pairs = [max(s[k]["pairs"], 1) if k in big else 0 for k in L.ORDERS]
    promoted = [2 * max(s[k]["promoted"], 1) if k in big else 0 for k in L.ORDERS]
    with build(stream, threads=3, blocks=9, ctx_entries=ctx, pair_entries=pairs, promoted_entries=promoted,
               load=0.9, tier_from=tier_from) as t:
        loads = t.stats()["orders"]
        assert_rows_equal(t.query_rows(x, y, 0), L.brute_force_rows(stream, x, y, 0))
    assert min(loads[k]["idx_maxload" if loads[k]["tiered"] else "ctx_maxload"] for k in big) > 0.85
    if tier_from == 1:
        assert all(loads[k]["promoted"] < loads[k]["contexts"] for k in L.ORDERS)


@pytest.mark.skipif(not (os.environ.get("STREAM_LOWTABLES_STREAM") and os.environ.get("STREAM_LOWTABLES_VAL")),
                    reason="set STREAM_LOWTABLES_STREAM (a .u16 stream with SEPs) and STREAM_LOWTABLES_VAL (a val .bin)")
def test_rows_equal_brute_force_on_the_real_stream():
    stream = np.fromfile(os.environ["STREAM_LOWTABLES_STREAM"], dtype=np.uint16, count=3_000_000)
    stream = stream[:np.flatnonzero(stream == SEP)[-1] + 1]
    v = np.fromfile(os.environ["STREAM_LOWTABLES_VAL"], dtype=np.uint16, offset=1024, count=300_001)
    with build(stream, threads=4, blocks=11) as t:
        rows = t.query_rows(v[:-1], v[1:], 262144)
    assert_rows_equal(rows, L.brute_force_rows(stream, v[:-1], v[1:], 262144))


# ------------------------------------------------------------------------------------------------ 2 definitions

@pytest.mark.parametrize("tier_from", [1, 9])
def test_fields_are_the_next_token_distribution_by_enumeration(tier_from):
    rng = np.random.default_rng(3)
    vocab = 7
    stream = toy_stream(rng, nspans=300, vocab=vocab)
    x, _ = toy_val(rng, 300, vocab=vocab)
    run = L.runs(x, 0, sep=False)
    with build(stream, threads=2, blocks=5, tier_from=tier_from) as t:
        checked = 0
        for pos in range(5, 300, 7):
            per_y = np.stack([t.query_one(x, pos, yy, run[pos]) for yy in [*range(vocab), BOS]])  # [vocab + 1, orders]
            for oi in range(len(L.ORDERS)):
                r = per_y[:, oi]
                assert len({(int(a["N"]), int(a["M"]), int(a["D"]), int(a["n1"]), int(a["n2"]), int(a["top"]))
                            for a in r}) == 1  # everything but C is target-independent
                C, N = r["C"].astype(np.int64), int(r["N"][0])
                assert C[-1] == 0  # BOS is never a next token
                if N == 0:
                    assert not C.any() and not r["top"].any()
                    continue
                checked += 1
                assert C.sum() == N and (C > 0).sum() == r["D"][0] and C.max() == r["M"][0]
                assert (C == 1).sum() == r["n1"][0] and (C == 2).sum() == r["n2"][0]
                assert r["top"][0] == int(np.flatnonzero(C == C.max())[0])  # ties: the lowest id
    assert checked > 100


@pytest.mark.parametrize("tier_from", [1, 9])
def test_top_swaps_and_ties_follow_the_lowest_id_rule(tier_from):
    """Context (5) followed, one span at a time, by 7, 3, 7, 3, 9, 9, 9, 2: the top moves 7 -> 3 (tie, lower id) ->
    7 -> 3 (tie) -> 9, and every count stays exact through the moves between the ctx slot and the pair table."""
    seq = [7, 3, 7, 3, 9, 9, 9, 2]
    want_top = [7, 3, 7, 3, 3, 3, 9, 9]
    x = np.array([BOS, 5], np.uint16)
    with L.LowTables((1, 2), threads=2, expected_positions=100, tier_from=tier_from) as t:
        stream = []
        for v, top in zip(seq, want_top):
            span = np.array([BOS, 5, v, SEP], np.uint16)
            stream.append(span)
            t.insert_block(span)
            t.sync()
            for yy in (2, 3, 7, 9):
                got = t.query_one(x, 1, yy, 2)
                ref = L.brute_force_rows(np.concatenate(stream), x, np.array([0, yy], np.uint16), 0, (1, 2))[1]
                assert got.tolist() == ref.tolist()
            assert got[0]["top"] == top and got[1]["top"] == top


# ------------------------------------------------------------------------------------------------ 3 determinism

@pytest.mark.parametrize("tier_from", TIERS)
def test_tables_are_identical_across_threads_and_blocks(tier_from):
    rng = np.random.default_rng(13)
    stream = zipf_stream(rng, ntok=150_000)
    x, y = zipf_val(rng, stream, 20_000)
    digests, rows = set(), []
    for threads, nblocks in [(1, 1), (2, 13), (3, 200), (8, 57), (5, 10_000)]:
        with L.LowTables(threads=threads, expected_positions=stream.size, tier_from=tier_from) as t:
            for b in split_blocks(stream, nblocks, rng):
                t.insert_block(b)
            t.finish()
            digests.add(t.digest())
            rows.append(t.query_rows(x, y, 4096, threads=threads))
    assert len(digests) == 1
    for r in rows[1:]:
        assert r.tobytes() == rows[0].tobytes()


def test_prefault_and_nice_change_nothing():
    rng = np.random.default_rng(31)
    stream = toy_stream(rng, vocab=12)
    with build(stream, threads=2, blocks=5) as a, build(stream, threads=3, blocks=2, prefault=False, nice=5) as b:
        assert a.digest() == b.digest()


# ------------------------------------------------------------------------------------------------ 4 query paths

@pytest.mark.parametrize("tier_from", TIERS)
def test_query_paths_agree(tier_from):
    rng = np.random.default_rng(17)
    stream = toy_stream(rng, vocab=8)
    x, y = toy_val(rng, 5000, vocab=8)
    chunk = 700
    run = L.runs(x, chunk, sep=False)
    with build(stream, threads=3, blocks=4, tier_from=tier_from) as t:
        a = t.query_rows(x, y, chunk, threads=4)
        b = np.concatenate([t.query_block(x, y[s:s + 333], run[s:s + 333], start=s) for s in range(0, x.size, 333)])
        c = np.stack([t.query_one(x, i, y[i], run[i]) for i in range(x.size)])
    assert a.tobytes() == b.tobytes() == c.tobytes()
    assert (run[::chunk] == 1).all() and (run[x == BOS] == 1).all()


# ------------------------------------------------------------------------------------------------ 5 lifecycle

@pytest.mark.parametrize("tier_from", [None, 1])
def test_hold_queries_the_tables_as_they_stood_before_the_held_blocks(tier_from):
    rng = np.random.default_rng(19)
    stream = toy_stream(rng, nspans=600, vocab=9)
    blocks = split_blocks(stream, 12)
    early, late = blocks[:8], blocks[8:]
    x, y = toy_val(rng, 3000, vocab=9)
    with L.LowTables(threads=3, expected_positions=stream.size, tier_from=tier_from) as t:
        for b in early:
            t.insert_block(b)
        t.hold(True)
        for b in late:
            t.insert_block(b)
        t.sync()
        s = t.stats()
        assert s["held"] == len(late) and s["blocks"] == len(blocks)
        assert_rows_equal(t.query_rows(x, y, 0), L.brute_force_rows(np.concatenate(early), x, y, 0))
        with pytest.raises(RuntimeError, match="held blocks"):
            t.finish()
    with L.LowTables(threads=3, expected_positions=stream.size, tier_from=tier_from) as t:
        for b in early:
            t.insert_block(b)
        t.hold(True)
        for b in late:
            t.insert_block(b)
        t.sync()
        t.query_rows(x, y, 0)
        t.hold(False)
        t.finish()
        assert_rows_equal(t.query_rows(x, y, 0), L.brute_force_rows(stream, x, y, 0))
        with build(stream, threads=2, tier_from=tier_from) as plain:
            assert plain.digest() == t.digest()
        assert t.stats()["finished"]
        with pytest.raises(RuntimeError, match="after lt_finish"):
            t.insert_block(blocks[0])


@pytest.mark.parametrize("which,match", [("ctx", "order-3 stats table is full"), ("pair", "order-3 pair table is full"),
                                         ("idx", "order-4 index table is full"),
                                         ("promoted", "order-4 stats table is full.*promoted_entries")])
def test_a_full_table_fails_loudly(which, match):
    rng = np.random.default_rng(23)
    stream = toy_stream(rng, nspans=3000, vocab=20)
    big = [stream.size] * 4
    kw = dict(ctx_entries=list(big), pair_entries=list(big), promoted_entries=list(big))
    kw[{"ctx": "ctx_entries", "pair": "pair_entries", "idx": "ctx_entries", "promoted": "promoted_entries"}[which]][
        3 if which in ("idx", "promoted") else 2] = 300
    with L.LowTables((1, 2, 3, 4), threads=2, expected_positions=stream.size, **kw) as t:
        t.insert_block(stream)
        with pytest.raises(RuntimeError, match=match):
            t.finish()
        assert t.stats()["failed"]
        with pytest.raises(RuntimeError):
            t.insert_block(stream[:10])


def test_bad_configurations_are_refused():
    with pytest.raises(RuntimeError, match="ascending"):
        L.LowTables((3, 2))
    with pytest.raises(RuntimeError, match="threads"):
        L.LowTables((1,), threads=0)
    with pytest.raises(RuntimeError, match="ascending"):
        L.LowTables((9,))


def test_empty_blocks_and_spans_without_separators():
    """A block is whole spans: one without a trailing SEP loses only its last token's (absent) next."""
    a = np.array([BOS, 1, 2, 3, 1, 2, 4], np.uint16)
    with L.LowTables((1, 2), threads=2, expected_positions=100) as t:
        t.insert_block(np.zeros(0, np.uint16))
        t.insert_block(a)
        t.insert_block(np.array([SEP, SEP], np.uint16))
        t.finish()
        x = np.array([BOS, 1, 2], np.uint16)
        rows = t.query_rows(x, np.array([1, 2, 3], np.uint16), 0)
    assert_rows_equal(rows, L.brute_force_rows(np.concatenate([a, [SEP]]).astype(np.uint16), x,
                                               np.array([1, 2, 3], np.uint16), 0, (1, 2)))
    assert rows[2]["N"].tolist() == [2, 2] and rows[2]["C"].tolist() == [1, 1] and rows[2]["D"].tolist() == [2, 2]


# ------------------------------------------------------------------------------------------------ 6 causality

def test_rows_depend_on_the_past_and_the_target_only():
    rng = np.random.default_rng(29)
    stream = toy_stream(rng, vocab=8)
    v = rng.integers(0, 8, 4001).astype(np.uint16)
    v[rng.random(4001) < 0.02] = BOS
    with build(stream, threads=2) as t:
        base = t.query_rows(v[:-1], v[1:], 1000)
        for t0 in (999, 1000, 2500):
            w = v.copy()
            w[t0 + 1:] = rng.integers(0, 8, w.size - t0 - 1)
            rows = t.query_rows(w[:-1], w[1:], 1000)
            assert rows[:t0].tobytes() == base[:t0].tobytes()
            for f in L.FIELDS:  # at t0 only C (which reads the target) may move
                if f != "C":
                    assert (rows[t0][f] == base[t0][f]).all()
            assert (rows[t0 + 1:] != base[t0 + 1:]).any()


# ------------------------------------------------------------------------------------------------ 7 build, sizing

def test_compiles_warning_free_and_with_the_helper(tmp_path):
    cc = subprocess.run(["cc", "-O2", "-std=c11", "-pthread", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                         str(ARM / "stream_lowtables.c"), "-o", str(tmp_path / "lt.so"), "-lm"], capture_output=True,
                        text=True)
    assert cc.returncode == 0, cc.stderr
    helper = ARM / "stream_memory.c"
    if helper.exists():  # one translation unit, as the helper may #include it: no clashing names or macros
        unit = tmp_path / "unit.c"
        unit.write_text(f'#include "{helper}"\n#include "{ARM / "stream_lowtables.c"}"\n')
        cc = subprocess.run(["cc", "-O2", "-std=c11", "-pthread", "-c", str(unit), "-o", str(tmp_path / "unit.o")],
                            capture_output=True, text=True)
        assert cc.returncode == 0, cc.stderr


# Exact counts of the real 1050-step stream (289,983,448 tokens; bench_lowtables.py) and of its first 30M tokens:
# (contexts, pair entries, contexts seen twice or more) per order 1-5.
MEASURED = {
    29_999_704: [(49_579, 5_212_201, 49_277), (5_211_077, 12_366_628, 1_905_335), (16_463_948, 9_221_979, 2_867_130),
                 (24_231_838, 4_086_109, 1_896_095), (27_455_504, 1_521_791, 970_139)],
    289_983_448: [(49_982, 22_627_026, None), (22_612_661, 89_668_226, None), (106_722_631, 100_846_811, None),
                  (195_038_534, 60_983_879, 22_870_867), (245_415_383, 27_451_588, 14_432_433)],
}


@pytest.mark.parametrize("tokens", sorted(MEASURED))
def test_default_sizes_cover_the_real_stream(tokens):
    for k, counts in zip(L.ORDERS, MEASURED[tokens]):
        for what, c in zip(("contexts", "pairs", "promoted"), counts):
            if c is not None and not (k == 1 and what != "pairs"):  # order-1 contexts: 65536, the bound
                assert c <= L.default_entries(k, tokens, what) <= 1.12 * c + 70_000, (k, what)


def test_default_sizes_are_monotone_and_bounded():
    for k in range(1, 9):
        prev = 0
        for n in [10, 1000, 10**6, 3 * 10**6, 3 * 10**7, 10**8, 3 * 10**8, 10**9]:
            for what in ("contexts", "pairs", "promoted"):
                e = L.default_entries(k, n, what)
                assert e <= n + 64
            e = L.default_entries(k, n)
            assert e >= prev
            prev = e
