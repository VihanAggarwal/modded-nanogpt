"""Exact low-order count tables of the run's trained stream (STREAM_RETRIEVAL_LOW=1; off by default; part P2 of
stream retrieval v2). The Python binding of stream_lowtables.c (ctypes), plus a brute-force reference.

What it is. For orders 1-5, every context's exact next-token statistics in the tokens this run trained on: N, M, D,
n1, n2, the top next token, and C(y) for a queried y. The helper process (stream_memory.c) links the C file and feeds
it each timed step's spans on the clock; at GO it finishes the tables (asserting every step is in) and writes one row
(N, C, M, D, n1, n2, top: 20 bytes) per val position and order, before the clock stops. The untimed eval turns the rows
into the gated chain over orders (KN components, stick-breaking gates) on top of the stream memory's own levels. The
FIT path queries the run's own last K timed batches against the tables as they stood before those batches
(LowTables.hold). This module is the same code for tests, benchmarks (tools/stream_retrieval/bench_lowtables.py) and
fitting tools.

Credits. The recipe is PR #380's (Deven): exact n-gram counts at low orders, a gated chain over the orders, its gate
fitted on training positions. Here the counts are of this run's consumed stream only (no corpus statistics), built on
the clock. No code from #380; the C file ports this branch's research tools (rg/loworders/lotable.c, ngcount.c).

Why it is off by default (STREAM_RETRIEVAL_LOW=1 turns it on). On the llm.c proxy it is worth ~10 of the ~34 millinats
of the v2 stack, but its cost is the kind the maintainers have objected to ("run-away CPU farms", #367). Measured on
the real 1050-step stream (289,983,448 tokens; bench_lowtables.py, a 4-vCPU VM shared with other jobs):
  - host RAM: 14.2 GB of tables (orders 1-3 6.8 GB, orders 4-5 7.3 GB with their index tier; 21.7 GB without it),
    allocated and touched before the clock;
  - CPU during training: 125-185 CPU-s of insertion (86-112 ns per position and order on average: random DRAM lines,
    ~2 per update; the cost per token doubles at orders 4-5 as the tables fill), i.e. ~4 cores on average over a ~35 s
    run and 5-6 over its last ~480 steps, on 8 insertion threads that wake at each step's block; this VM's 4 vCPUs
    could not keep pace with 119 ns per token (a dev run must show the 8xH100 host's step time unchanged and the
    backlog over the last steps small: it drains at GO, beside the other parts' queries);
  - at GO: ~0.4-0.45 us per val position and thread for the 5 lookups (~4-5 CPU-s for 10,485,760 positions), on the
    clock.
Turn it on for the maximum gain once a dev run has shown the step time and GO->rows unchanged.
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ENABLED = os.environ.get("STREAM_RETRIEVAL_LOW") == "1"
SOURCE = Path(__file__).resolve().with_name("stream_lowtables.c")
ORDERS = (1, 2, 3, 4, 5)
THREADS = 8                     # insertion threads (partition owners)
LOAD = 0.75                     # table slots = expected entries / LOAD
CFLAGS = ("-O2", "-std=c11", "-pthread")
SEP, BOS = 0xFFFF, 50256
MAX_ORDERS, MAX_K = 8, 8
TIER_FROM = 4                   # orders >= this keep a context seen once as an 8-byte index slot (stream_lowtables.c)

# One row per (position, order): lt_row_t, 20 bytes. All zero where the context never occurred in the stream.
ROW = np.dtype([("N", "<u4"), ("C", "<u4"), ("M", "<u4"), ("D", "<u2"), ("n1", "<u2"), ("n2", "<u2"), ("top", "<u2")])
FIELDS = ROW.names

# lt_stats() words (stream_lowtables.c's LT_S_* / LT_SO_*)
S_NORDERS, S_THREADS, S_BLOCKS, S_HELD, S_TOKENS, S_BUSY_NS, S_BYTES, S_STATE, S_FINISH_NS, S_CPU_NS, S_PENDING, \
    S_ORDER0 = range(12)
SO_K, SO_TIERED, SO_POSITIONS, SO_CONTEXTS, SO_PROMOTED, SO_PAIRS, SO_IDX_SLOTS, SO_CTX_SLOTS, SO_PAIR_SLOTS, \
    SO_IDX_MAXLOAD_PPM, SO_CTX_MAXLOAD_PPM, SO_PAIR_MAXLOAD_PPM, SO_BYTES, SO_WORDS = range(14)
STATS_WORDS = S_ORDER0 + MAX_ORDERS * SO_WORDS
ST_FINISHED, ST_FAILED = 1, 2


class _Config(ctypes.Structure):  # lt_config_t
    _fields_ = [("norders", ctypes.c_int32), ("orders", ctypes.c_int32 * MAX_ORDERS), ("threads", ctypes.c_int32),
                ("prefault", ctypes.c_int32), ("nice", ctypes.c_int32), ("tier_from", ctypes.c_int32),
                ("reserved", ctypes.c_int32), ("expected_positions", ctypes.c_uint64),
                ("ctx_entries", ctypes.c_uint64 * MAX_ORDERS), ("pair_entries", ctypes.c_uint64 * MAX_ORDERS),
                ("promoted_entries", ctypes.c_uint64 * MAX_ORDERS), ("load", ctypes.c_double)]


def build_library() -> Path:
    """Compile stream_lowtables.c into a shared library (cached by the source's hash)."""
    source = SOURCE.read_bytes()
    out = Path(tempfile.gettempdir()) / "stream_memory_build" / f"liblowtables_{hashlib.sha256(source).hexdigest()[:16]}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang") or "cc"
        cc = subprocess.run([compiler, *CFLAGS, "-shared", "-fPIC", str(SOURCE), "-o", str(tmp), "-lm"],
                            capture_output=True, text=True)
        if cc.returncode:
            raise RuntimeError(f"cannot compile {SOURCE}: {cc.stderr.strip()}")
        os.replace(tmp, out)
    return out


_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(str(build_library()))
        vp, u16p, u32p, u64p = ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint32), \
            ctypes.POINTER(ctypes.c_uint64)
        L.lt_create.argtypes = [ctypes.POINTER(_Config), ctypes.c_char_p, ctypes.c_size_t]
        L.lt_create.restype = vp
        L.lt_destroy.argtypes = [vp]
        L.lt_destroy.restype = None
        L.lt_error.argtypes = [vp]
        L.lt_error.restype = ctypes.c_char_p
        L.lt_insert_block.argtypes = [vp, u16p, ctypes.c_size_t]
        L.lt_insert_block.restype = ctypes.c_int
        for name in ("lt_sync", "lt_finish"):
            getattr(L, name).argtypes = [vp]
            getattr(L, name).restype = ctypes.c_int
        L.lt_hold.argtypes = [vp, ctypes.c_int]
        L.lt_hold.restype = ctypes.c_int
        L.lt_query_rows.argtypes = [vp, u16p, u16p, ctypes.c_size_t, ctypes.c_uint64, vp, ctypes.c_int]
        L.lt_query_rows.restype = ctypes.c_int
        L.lt_query_block.argtypes = [vp, u16p, u16p, u32p, ctypes.c_size_t, vp]
        L.lt_query_block.restype = None
        L.lt_query_one.argtypes = [vp, u16p, ctypes.c_uint16, ctypes.c_uint32, vp]
        L.lt_query_one.restype = None
        L.lt_stats.argtypes = [vp, u64p, ctypes.c_int]
        L.lt_stats.restype = ctypes.c_int
        L.lt_digest.argtypes = [vp]
        L.lt_digest.restype = ctypes.c_uint64
        L.lt_default_entries.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_int]
        L.lt_default_entries.restype = ctypes.c_uint64
        L.lt_census.argtypes = [vp, ctypes.c_int, u64p]
        L.lt_census.restype = ctypes.c_int
        L.lt_abi.argtypes = [ctypes.c_int]
        L.lt_abi.restype = ctypes.c_uint64
        abi = [L.lt_abi(i) for i in range(7)]
        want = [ctypes.sizeof(_Config), ROW.itemsize, STATS_WORDS, MAX_ORDERS, MAX_K, _Config.load.offset, TIER_FROM]
        if abi != want:
            raise RuntimeError(f"stream_lowtables.py does not mirror stream_lowtables.c: {abi} != {want}")
        _LIB = L
    return _LIB


def _u16(a):
    a = np.ascontiguousarray(a, dtype=np.uint16)
    return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def default_entries(k: int, positions: int, what: str = "contexts") -> int:
    """The tables' default expected entries for a stream of `positions` tokens: what = "contexts", "pairs" (pair
    entries) or "promoted" (contexts seen twice or more: a tiered order's stats slots)."""
    return int(lib().lt_default_entries(k, positions, ("contexts", "pairs", "promoted").index(what)))


class LowTables:
    """One set of tables. orders: ascending in 1..8; threads: insertion threads; expected_positions: the stream's size
    for the default table sizes (ctx_entries / pair_entries / promoted_entries override them per order); tier_from:
    orders >= it have the index tier (None: TIER_FROM; 9: none); prefault: touch every page now."""

    def __init__(self, orders=ORDERS, *, threads: int = THREADS, expected_positions: int = 0, ctx_entries=None,
                 pair_entries=None, promoted_entries=None, tier_from: int | None = None, load: float = LOAD,
                 prefault: bool = True, nice: int = 0):
        self.orders = tuple(int(k) for k in orders)
        cfg = _Config()
        cfg.norders = len(self.orders)
        for i, k in enumerate(self.orders):
            cfg.orders[i] = k
            cfg.ctx_entries[i] = int(ctx_entries[i]) if ctx_entries is not None else 0
            cfg.pair_entries[i] = int(pair_entries[i]) if pair_entries is not None else 0
            cfg.promoted_entries[i] = int(promoted_entries[i]) if promoted_entries is not None else 0
        cfg.threads, cfg.prefault, cfg.nice, cfg.tier_from = int(threads), int(prefault), int(nice), int(tier_from or 0)
        cfg.expected_positions, cfg.load = int(expected_positions), float(load)
        err = ctypes.create_string_buffer(256)
        self._lib = lib()
        self._h = self._lib.lt_create(ctypes.byref(cfg), err, 256)
        if not self._h:
            raise RuntimeError(f"stream_lowtables: {err.value.decode()}")

    def _check(self, rc):
        if rc:
            raise RuntimeError(f"stream_lowtables: {self._lib.lt_error(self._h).decode() or 'failed'}")

    def insert_block(self, tokens):
        """One step's tokens: whole spans, SEP (0xFFFF) between them. Copied; inserted asynchronously, in order."""
        a, p = _u16(tokens)
        self._check(self._lib.lt_insert_block(self._h, p, a.size))

    def hold(self, on: bool):
        """on: queue the next blocks without inserting them (FIT positions are queried against the tables as they
        stood before their own blocks); off: release them, in order."""
        self._check(self._lib.lt_hold(self._h, int(bool(on))))

    def sync(self):
        self._check(self._lib.lt_sync(self._h))

    def finish(self):
        """Every block is in (none held), then the tables are frozen: the helper's check at GO."""
        self._check(self._lib.lt_finish(self._h))

    def query_rows(self, x, y, chunk: int = 262144, threads: int | None = None, out: np.ndarray | None = None) -> np.ndarray:
        """Rows [n, norders] of positions t = 0..n-1 (input x[t], target y[t]); segments restart at every BOS and every
        multiple of chunk (0: never). out: a C-contiguous ROW array [n, norders] to fill (else a new one)."""
        x, px = _u16(x)
        y, py = _u16(y)
        assert x.size == y.size
        if out is None:
            out = np.empty((x.size, len(self.orders)), dtype=ROW)
        assert out.dtype == ROW and out.shape == (x.size, len(self.orders)) and out.flags.c_contiguous
        self._check(self._lib.lt_query_rows(self._h, px, py, x.size, int(chunk), out.ctypes.data,
                                            int(threads or os.cpu_count() or 1)))
        return out

    def query_block(self, x, y, run, start: int = 0) -> np.ndarray:
        """lt_query_block on positions start.. of x (x[start - 4:] must hold the contexts), with the caller's run[]."""
        x, _ = _u16(x)
        y = np.ascontiguousarray(y, dtype=np.uint16)
        run = np.ascontiguousarray(run, dtype=np.uint32)
        n = y.size
        out = np.empty((n, len(self.orders)), dtype=ROW)
        self._lib.lt_query_block(self._h, ctypes.cast(x.ctypes.data + 2 * start, ctypes.POINTER(ctypes.c_uint16)),
                                 y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                                 run.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), n, out.ctypes.data)
        return out

    def query_one(self, x, t: int, y: int, run: int) -> np.ndarray:
        x, _ = _u16(x)
        out = np.empty(len(self.orders), dtype=ROW)
        self._lib.lt_query_one(self._h, ctypes.cast(x.ctypes.data + 2 * t, ctypes.POINTER(ctypes.c_uint16)), int(y),
                               int(run), out.ctypes.data)
        return out

    def stats(self) -> dict:
        w = np.zeros(STATS_WORDS, dtype=np.uint64)
        moving = self._lib.lt_stats(self._h, w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), STATS_WORDS)
        out = dict(threads=int(w[S_THREADS]), blocks=int(w[S_BLOCKS]), held=int(w[S_HELD]), tokens=int(w[S_TOKENS]),
                   busy_s=int(w[S_BUSY_NS]) / 1e9, cpu_s=int(w[S_CPU_NS]) / 1e9, pending=int(w[S_PENDING]),
                   bytes=int(w[S_BYTES]), finished=bool(w[S_STATE] & ST_FINISHED),
                   failed=bool(w[S_STATE] & ST_FAILED), finish_ms=int(w[S_FINISH_NS]) / 1e6, moving=bool(moving),
                   orders={})
        for i in range(int(w[S_NORDERS])):
            o = w[S_ORDER0 + i * SO_WORDS:S_ORDER0 + (i + 1) * SO_WORDS]
            out["orders"][int(o[SO_K])] = dict(
                tiered=bool(o[SO_TIERED]), positions=int(o[SO_POSITIONS]), contexts=int(o[SO_CONTEXTS]),
                promoted=int(o[SO_PROMOTED]), pairs=int(o[SO_PAIRS]), idx_slots=int(o[SO_IDX_SLOTS]),
                ctx_slots=int(o[SO_CTX_SLOTS]), pair_slots=int(o[SO_PAIR_SLOTS]),
                idx_maxload=int(o[SO_IDX_MAXLOAD_PPM]) / 1e6, ctx_maxload=int(o[SO_CTX_MAXLOAD_PPM]) / 1e6,
                pair_maxload=int(o[SO_PAIR_MAXLOAD_PPM]) / 1e6, bytes=int(o[SO_BYTES]))
        return out

    def census(self, k: int) -> dict:
        """Order k's contexts with N == 1, N == 2, D == 1, and N >= 2 with D == 1 (a scan of its table)."""
        w = np.zeros(4, dtype=np.uint64)
        self._check(self._lib.lt_census(self._h, int(k), w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))))
        return dict(zip(("n_eq_1", "n_eq_2", "d_eq_1", "n_ge_2_d_eq_1"), map(int, w)))

    def digest(self) -> int:
        return int(self._lib.lt_digest(self._h))

    def close(self):
        if getattr(self, "_h", None):
            self._lib.lt_destroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------ reference

def runs(tokens: np.ndarray, chunk: int = 0, sep: bool = True) -> np.ndarray:
    """Tokens of each position's segment up to and including it: restarts at BOS (1), after SEP (SEP itself: 0) and at
    multiples of chunk (1)."""
    t = np.asarray(tokens)
    j = np.arange(t.size, dtype=np.int64)
    anchor = np.full(t.size, -1, dtype=np.int64)
    if sep:
        anchor = np.where(t == SEP, j, anchor)
    anchor = np.where(t == BOS, j - 1, anchor)
    if chunk:
        anchor = np.where(j % chunk == 0, np.maximum(anchor, j - 1), anchor)
    return j - np.maximum.accumulate(anchor)


def _ctx_keys(tokens: np.ndarray, ends: np.ndarray, k: int) -> np.ndarray:
    cols = np.stack([tokens[ends - k + 1 + i] for i in range(k)], 1).astype(np.uint16)
    return np.ascontiguousarray(cols).view(np.dtype((np.void, 2 * k))).ravel()


def brute_force_rows(stream, x, y, chunk: int = 262144, orders=ORDERS) -> np.ndarray:
    """The exact rows by sorting (numpy; independent of the C file's hashing): rows [n, norders]."""
    stream = np.asarray(stream, dtype=np.uint16)
    x, y = np.asarray(x, dtype=np.uint16), np.asarray(y, dtype=np.uint16)
    out = np.zeros((x.size, len(orders)), dtype=ROW)
    mrun = runs(stream)
    vrun = runs(x, chunk, sep=False)
    j = np.arange(stream.size - 1)
    nxt = stream[1:]
    for oi, k in enumerate(orders):
        mj = j[(mrun[:-1] >= k) & (nxt != SEP) & (nxt != BOS)]
        qt = np.flatnonzero(vrun >= k)
        if not mj.size or not qt.size:
            continue
        keys = np.concatenate([_ctx_keys(stream, mj, k), _ctx_keys(x, qt, k)])
        _, cid = np.unique(keys, return_inverse=True)
        mc, qc = cid[:mj.size].astype(np.int64), cid[mj.size:].astype(np.int64)
        nv = stream[mj + 1].astype(np.int64)
        pair, pc = np.unique(mc * 65536 + nv, return_counts=True)
        pctx, ptok = pair // 65536, pair % 65536
        nctx = int(cid.max()) + 1
        N = np.bincount(pctx, weights=pc, minlength=nctx)
        D = np.bincount(pctx, minlength=nctx)
        n1 = np.bincount(pctx, weights=pc == 1, minlength=nctx)
        n2 = np.bincount(pctx, weights=pc == 2, minlength=nctx)
        M = np.zeros(nctx, np.int64)
        np.maximum.at(M, pctx, pc)
        order = np.lexsort((ptok, -pc, pctx))  # by context, count descending, token ascending
        first = order[np.r_[True, pctx[order][1:] != pctx[order][:-1]]]
        top = np.zeros(nctx, np.int64)
        top[pctx[first]] = ptok[first]
        q = qc
        seen = N[q] > 0
        r = out[qt, oi]
        r["N"], r["M"], r["D"] = N[q], np.where(seen, M[q], 0), D[q]
        r["n1"], r["n2"], r["top"] = n1[q], n2[q], np.where(seen, top[q], 0)
        want = q * 65536 + y[qt].astype(np.int64)
        pos = np.searchsorted(pair, want)
        hit = (pos < pair.size) & (pair[np.minimum(pos, pair.size - 1)] == want)
        r["C"] = np.where(hit, pc[np.minimum(pos, pair.size - 1)], 0)
        out[qt, oi] = r
    return out
