"""Stream-only retrieval at the final validation (STREAM_RETRIEVAL=1; off by default), v2: the helper's rows (P1
match records, P2 exact low-order counts, P3 pointer / vote / source-copy rows) and a gated mixture of them with the
model's probability, computed on the GPU in the untimed eval.

At the final validation, the probability p of each val token (CPLM with the canonical mask) is mixed with next-token
distributions read from memories of the tokens this run trained on, and nothing else.

- The memory holds only the tokens this run trains on. Rank 0's loader passes the document spans that
  Shard.next_batch computes for all ranks (data.py, `on_spans`) to a helper process (stream_memory.c), which reads
  those token ranges from the shard files and indexes them as the batches are fetched: on the clock, in parallel with
  training. Only the timed loader is tapped; the warmup loader is not.
- At the last step rank 0 sends GO. Before the clock stops the helper reads the val shard and writes, per val
  position (stream_memory.c's header has the details):
    P1  a RECORD where the position's 6-token context occurred in the memory: up to 32 verified candidates (next
        token, match length), and per match level L in LEVELS the counts N, D, M, n1, n2 and the top next token over
        the candidates matching at least L deep, plus the most recent / longest candidate (target-independent);
    P2  (STREAM_RETRIEVAL_LOW=1) the exact counts of orders 1-5 of the stream (stream_lowtables.c, filled during
        training on 8 insertion threads): (N, C(y), M, D, n1, n2, top) per order, C(y) read at the target as a
        cross-entropy gather reads p(y);
    P3  (STREAM_RETRIEVAL_POINTER, on by default) per segment, sequentially (stream_pointer.c): a pointer beam over
        the memory (alignment hypotheses advanced on correct predictions), its vote, a copy from up to 8 source
        documents of the memory, and doc-state counters of earlier positions' outcomes.
  Each rank copies its rows to the device (`collect`).
- The mixing runs in the untimed eval loop, like CPLM's own mixture (StreamEval):
    chain  P_0 = p; P_i = (1 - lam_i) P_{i-1} + lam_i r_i over the orders ascending (P2's 1-5, then the memory's
           levels 6..32), r_i the modified-KN distribution of order i's next tokens, lam_i = sigmoid(w_i . phi_i) on
           target-independent features (chain_features: counts, the model's entropy / max log p / log p of the
           order's top token, the model's surprisal over the matched context, the order's causal history);
    top    q = softmax-gated mixture of [the chain, P3's pointer, vote, source copy] where any of them is available
           (top_block: P3's 92 helper features plus model-aware ones: entropy, the model's log p of the pointer's /
           vote's / memory's tokens, the surprisal of each component's matched context, the longest chain order, and
           the log-likelihood-ratio history of the memory and the pointer over EARLIER positions).
  Positions where nothing matched keep p bit-identical. Training, the model and the token streams are untouched: the
  same weights are scored with and without the mixture in one run, so one run measures the gain exactly.

The trainer's side (hooks.patch; every rank, rank 0 also feeds the helper with on_spans and sends go / fit_go):
  gate = Gate.for_run(); memory = StreamMemory.for_run(...)       before the clock
  rows = memory.collect()                                          on the clock, after GO: this rank's rows on device
  ev = StreamEval(rows, gate); nll, nll_mixed = ev.evaluate(s, batch, forward, model)    untimed eval, val step s
  (the model's eval forward reports the LM's entropy, max log p and top-32 log-probs at ev.tokens(): lm_features)
STREAM_RETRIEVAL_FIT=<dir> (dev runs, untimed, after the final validation): fit_dump(...) writes the helper's rows of
the run's own last STREAM_RETRIEVAL_FIT_K batches and the model's eval outputs there; tools/stream_retrieval/
fit_gate_v2.py <dir> fits the constants.

Why the result is a valid probability model. Every r_i sums to 1 over the vocabulary: sum_v max(C_v - d(C_v), 0)
= N - d1 n1 - d2 n2 - d3 (D - n1 - n2), its denominator; the pointer is one token, the vote's shares sum to 1, the
source copy's counts sum to N (or less where its distribution was truncated). All are functions of the memory, val[<=
t] and EARLIER positions' outcomes. Every gate reads only target-independent quantities (the history features read
targets and the model's probabilities of positions < t only). Stick-breaking and the softmax are convex combinations,
so sum_v q_t(v) = sum_v p_t(v) <= 1 (CPLM's p sums to <= 1; retrieved mass on a masked token is dropped, never
renormalized). The eval reads q at the realised token (C_L(y) from the candidate list, P2's C(y), the pointer's /
vote's / source's share of y: gathers, as cross-entropy reads p(y)). Forward-only, nothing learned from val.

The gate's constants (GATE_V2: P1 + P3; GATE_V2_LOW: P2 + P1 + P3) are fitted on the run's OWN consumed data, never on
val: with STREAM_RETRIEVAL_FIT the helper also writes the rows of the run's last STREAM_RETRIEVAL_FIT_K timed batches,
queried against the memory as it stood before the first of them was inserted, and fit_gate_v2.py fits the constants
on those positions and the model's eval outputs there. Each block holds one spec per step count a FIT dev run was
fitted at, and a run uses the spec of its own step count: the loader is deterministic, so a dev run at a record leg's
NUM_SCHEDULED_ITERATIONS has the leg's own last batches (a record attempt requires such a spec, fitted on our model,
for every cut: Gate.check_record). The shipped specs are the CPU proof's (provenance in each "fit"): fitted at 1050
steps on the replayed run's own last 16 batches with the llm.c GPT-2 124M proxy's outputs, placeholders until the
dev runs fit them on our model.

Credits. No code is taken from another PR; the design builds on two of them:
- PR #367 (Herman Brunborg): exact-match retrieval of continuations from training data on this track, and its
  StreamIndex (exact_match/src/stream.rs), whose memory and match rule this one follows: the stream of every rank's
  documents (inputs plus the last target), step by step, each followed by a STOP; positions keyed on their last 6
  tokens and resolved against the most recent occurrences; the match levels; the next tokens of the occurrences that
  match at least that deep; their (length, count, top-token share) as features.
- PR #380 (Deven): mixing CPLM's probability at the output with retrieved count distributions under sigmoid gates
  fitted on training positions, chained over the match orders (its ChainFit). P2 is #380's recipe: exact n-gram counts
  at low orders and a gated chain over the orders, fitted on training positions (here counts of the run's own consumed
  stream, built on the clock, and the gate fitted on the run's own last batches; no #380 code).
- kNN-LM (Khandelwal et al., 2020), Infini-gram (Liu et al., 2024), interpolated Kneser-Ney (Chen & Goodman, 1998)
  for retrieval / n-gram interpolation; the LZ77/zlib hash chain for the index.
This branch's part: restricting the memories to this run's consumed stream and using them at the final validation,
the hash-chain index and C helper filled from the loader's spans on the clock, the records and their GPU-side gated
chain with model-aware features, the pointer beam / vote / source copy / doc-state (P3), the FIT path on the run's own
last batches, and the tests. Nothing from #381.
"""
import atexit
import fcntl
import glob
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import mmap
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


def _sibling(name: str):
    """track_1_short.<name> in the trainer; by path when this file is loaded on its own (tests, fit_gate_v2.py)."""
    if __package__:
        return importlib.import_module(f"{__package__}.{name}")
    path = Path(__file__).resolve().with_name(f"{name}.py")
    key = f"_stream_retrieval_{name}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


stream_lowtables = _sibling("stream_lowtables")
stream_pointer = _sibling("stream_pointer")

ENABLED = os.environ.get("STREAM_RETRIEVAL") == "1"
LOW = stream_lowtables.ENABLED                                   # STREAM_RETRIEVAL_LOW=1 (P2; off by default)
POINTER = os.environ.get("STREAM_RETRIEVAL_POINTER", "1") != "0"  # P3 (on by default; =0 for an ablation)
SOURCE = Path(__file__).resolve().with_name("stream_memory.c")
SOURCES = (SOURCE, SOURCE.with_name("stream_lowtables.c"), SOURCE.with_name("stream_pointer.c"))  # one binary
HASH_BITS = 29                  # 2^29 buckets, 2 GB of heads
QUERY_THREADS = 96              # at most, and 16 of the CPUs this process may use left to the trainer
                                # (STREAM_RETRIEVAL_THREADS overrides): GO's queries are ~1.2 (P1 + P3) + ~0.4 (P2) us
                                # per val position and thread
LOW_THREADS = 8                 # P2's insertion threads (partition owners)
CFLAGS = ("-O2", "-std=c11", "-pthread")
FIT_K = 16                      # STREAM_RETRIEVAL_FIT: the run's last 16 timed batches (at the schedule's final batch
                                # size, 16 x 16,384 = one 262,144-token chunk per rank; K x the last per-rank batch
                                # must be a whole number of eval chunks, checked at startup)
COLLECT_TIMEOUT_S = 60.0
FIT_TIMEOUT_S = 600.0
READY_TIMEOUT_S = 600.0         # P2's tables (~13 GiB) are allocated and touched before READY
PINNED_SHARE = 0.25             # the pinned staging for the P1 records: this share of the rank's val positions
PINNED_PTR_SHARE = 0.6          # ... for the P3 rows (33% of positions are active on the real stream)
READ_PIECE = 8 << 20            # collect(): the rows are read in pieces of 8 MB ...
READ_THREADS = 8                # ... on this many threads
LM_ROWS = 8192                  # the eval forward's LM features (lm_features) are computed this many rows at a time

# The rows file (stream_memory.c): header words (u64) as its H_* enum, the error message, the chunk checksums, the P1
# record count and the P3 row count of every (rank, step) region, then the regions [world][val_steps]: P1 records
# (chunk records of room each), P2 rows (chunk x 5 orders, dense), P3 rows (chunk of room each). These constants
# mirror the C file's (the tests check that they match).
H_MAGIC, H_STATE, H_ERROR, H_WORLD, H_VAL_STEPS, H_CHUNK, H_STEPS, H_ENTRIES, H_INSERTED, H_HITS, H_CANDS, \
    H_QUERIED, H_T_READY, H_T_GO, H_T_DONE, H_INSERT_NS, H_QUERY_NS, H_BYTES_READ, H_REC_BYTES, H_LEVEL0 = range(20)
LEVELS = (6, 7, 8, 10, 12, 16, 24, 32)
H_FIT_POSITIONS, H_FIT_HITS, H_FIT_FREEZE, H_T_FIT_DONE, H_LOW_ORDERS, H_PTR, H_LOW_OFFSET, H_PTR_OFFSET, H_PTR_ROWS, \
    H_LOW_WAIT_NS, H_LOW_MAX_PENDING, H_LOW_BYTES, H_LOW_CPU_NS, H_FIT_PTR_ROWS, H_LOW_FIT_NS, H_LOW_ROW_BYTES, \
    H_PTR_ROW_BYTES, H_LOW_DRAIN_NS, H_LOW_LATE_PENDING = range(H_LEVEL0 + len(LEVELS), H_LEVEL0 + len(LEVELS) + 19)
HEADER_MAGIC = 0x324D454D52545353
FIT_MAGIC = 0x3354494652545353  # "SSTRFIT3": 16 header words, word 8 the run's trained steps
FIT_MAGIC_V2 = 0x3254494652545353  # "SSTRFIT2" (8 header words, no step count): still read
FIT_HEAD_WORDS = 16
LATE_STEPS = 100
ST_STARTING, ST_READY, ST_DONE, ST_FIT_DONE, ST_ERROR = 0, 1, 2, 3, 14
ERRMSG_OFFSET, CHECKSUM_OFFSET, COUNTS_OFFSET, PCOUNTS_OFFSET, ROWS_OFFSET, REGION_ALIGN = 512, 1024, 4096, 8192, 16384, 4096
MAX_CHUNKS = 384
MSG_STEP, MSG_GO, MSG_FIT = 1, 2, 3
KEY, MAXLEN, CAP, MAXVISIT, FULLCAP = 6, 32, 32, 128, 4096
BOS = 50256
F_SETPIPE_SZ = 1031
PIPE_BYTES = 1 << 20

# One matched position's record (stream_memory.c's rec_t).
REC_DTYPE = np.dtype([("pos", "<u4"), ("pos_recent", "<u4"), ("pos_longest", "<u4"), ("len_recent", "<u2"),
                      ("len_longest", "<u2"), ("top", "<u2", (8,)), ("nx", "<u2", (CAP,)), ("len", "u1", (CAP,)),
                      ("n", "u1", (8,)), ("d", "u1", (8,)), ("m", "u1", (8,)), ("n1", "u1", (8,)), ("n2", "u1", (8,)),
                      ("ncand", "u1"), ("lstar", "u1"), ("pad", "u1", (6,))])
REC_BYTES = REC_DTYPE.itemsize
assert REC_BYTES == 176
LOW_ORDERS = stream_lowtables.ORDERS            # P2's orders (stream_memory.c LOW_ORDERS)
LOW_DTYPE = stream_lowtables.ROW                # lt_row_t, 20 bytes per (position, order)
LOW_BYTES = LOW_DTYPE.itemsize
LOW_FIELDS = ("N", "C", "D", "M", "n1", "n2", "top")
PTR_DTYPE = stream_pointer.ROW_DTYPE            # sp_row_t, 380 bytes per active position
PTR_BYTES = PTR_DTYPE.itemsize
TOP_COMPS = ("ptr", "vote", "src")              # P3's components in the top-level softmax


def build_helper() -> Path:
    """Compile stream_memory.c (which #includes stream_lowtables.c and stream_pointer.c; cached by the three sources'
    hash). cc is on every node: Triton needs it too."""
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in SOURCES)).hexdigest()[:16]
    out = Path(tempfile.gettempdir()) / "stream_memory_build" / f"stream_memory_{digest}"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang") or "cc"
        cc = subprocess.run([compiler, *CFLAGS, str(SOURCE), "-o", str(tmp), "-lm"], capture_output=True, text=True)
        if cc.returncode:
            raise RuntimeError(f"cannot compile {SOURCE}: {cc.stderr.strip()}")
        os.replace(tmp, out)
    return out


def chunk_checksum(tokens: np.ndarray) -> int:
    """The checksum the helper stores for each chunk's L + 1 val tokens (its inputs and its last target)."""
    i = np.arange(tokens.size, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return int(((tokens.astype(np.uint64) + np.uint64(1)) * (i * np.uint64(0x9E3779B97F4A7C15) + np.uint64(1)))
                   .sum(dtype=np.uint64))


def rows_layout(world: int, val_steps: int, chunk: int, low: bool, pointer: bool) -> dict:
    """Byte offsets of the rows file's regions (stream_memory.c main()) and its size."""
    align = lambda v: (v + REGION_ALIGN - 1) // REGION_ALIGN * REGION_ALIGN
    nreg = world * val_steps
    low_off = align(ROWS_OFFSET + nreg * chunk * REC_BYTES)
    ptr_off = align(low_off + (nreg * chunk * len(LOW_ORDERS) * LOW_BYTES if low else 0))
    return dict(low=low_off, ptr=ptr_off, total=ptr_off + (nreg * chunk * PTR_BYTES if pointer else 0))


ARENA_REC_SHARE, ARENA_PTR_SHARE = 0.10, 0.40      # stream_memory.c: the query threads' arenas, touched before GO
PREFAULT_REC_SHARE, PREFAULT_PTR_SHARE = 0.15, 0.50  # ... and the share of each rows region touched before GO
LOW_TABLE_BYTES_PER_TOKEN = 49.0  # P2's tables at the default sizes: 13.2 GiB for the 1050-step stream's 289.5M tokens


def host_bytes(*, stream_tokens: int, val_tokens: int, world: int, low: bool, pointer: bool,
               hash_bits: int = HASH_BITS) -> dict:
    """The retrieval's host RAM on the node (estimated from the sizes the code allocates; all of it before the clock):
    the helper's memory (stream, chain, heads), its query arenas, P2's tables, the rows file in /dev/shm (the pages
    touched before GO) and every rank's pinned staging. GiB per item and their total."""
    entries = int(1.02 * stream_tokens) + (1 << 20)
    n_rank = val_tokens // world
    out = dict(memory=2 * entries + 4 * entries + (4 << hash_bits),
               arenas=val_tokens * (ARENA_REC_SHARE * REC_BYTES + (ARENA_PTR_SHARE * PTR_BYTES if pointer else 0)),
               low_tables=LOW_TABLE_BYTES_PER_TOKEN * stream_tokens if low else 0,
               rows_file=val_tokens * (PREFAULT_REC_SHARE * REC_BYTES + (len(LOW_ORDERS) * LOW_BYTES if low else 0)
                                       + (PREFAULT_PTR_SHARE * PTR_BYTES if pointer else 0)),
               pinned=world * (PINNED_SHARE * n_rank * REC_BYTES + (n_rank * len(LOW_ORDERS) * LOW_BYTES if low else 0)
                               + (PINNED_PTR_SHARE * n_rank * PTR_BYTES if pointer else 0)))
    out = {k: v / 2 ** 30 for k, v in out.items()}
    out["total"] = sum(out.values())
    return out


def available_cpus() -> int:
    """The CPUs this process may run on: its affinity mask, capped by a cgroup v2 CPU quota (containers)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n)


def chain_orders(low: bool) -> list:
    """The gated chain's orders: P2's exact orders 1-5 (STREAM_RETRIEVAL_LOW), then the memory's levels."""
    return (list(LOW_ORDERS) if low else []) + list(LEVELS)


def _sync():
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


# ==================================================================================================== records

def _u16(b: Tensor) -> Tensor:
    return b.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _u32(b: Tensor) -> Tensor:
    return b.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


class Rows:
    """The per-position fields of one chunk of P positions (val step or FIT sequence), on the records' device.
    Unmatched positions: hit False, len 0, n 0, lstar -1. nx [P, CAP] int32 (-1 = no candidate), len [P, CAP] int32
    (match length capped at MAXLEN), n / d / m / n1 / n2 [P, 8] float64, top [P, 8] int64 (per level of LEVELS),
    lstar [P] int64 (index of L* in LEVELS), ncand, pos_recent, pos_longest, len_recent, len_longest [P] int64."""

    def __init__(self, recs: Tensor, P: int):
        if recs.dtype != torch.uint8 or recs.dim() != 2 or recs.shape[1] != REC_BYTES:
            raise ValueError(f"records must be uint8 [H, {REC_BYTES}], got {recs.dtype} {tuple(recs.shape)}")
        dev, H = recs.device, recs.shape[0]
        pos = _u32(recs[:, 0:4]).view(H)
        if H and int(pos.max()) >= P:
            raise ValueError(f"a record at position {int(pos.max())} of a chunk of {P}")
        self.P = P
        self.hit = torch.zeros(P, dtype=torch.bool, device=dev)
        self.hit[pos] = True
        ln = recs[:, 96:128].to(torch.int32)
        self.len = torch.zeros((P, CAP), dtype=torch.int32, device=dev)
        self.len[pos] = ln
        self.nx = torch.full((P, CAP), -1, dtype=torch.int32, device=dev)
        self.nx[pos] = torch.where(ln > 0, _u16(recs[:, 32:96]).view(H, CAP), -1)
        for name, a in (("n", 128), ("d", 136), ("m", 144), ("n1", 152), ("n2", 160)):
            v = torch.zeros((P, len(LEVELS)), dtype=torch.float64, device=dev)
            v[pos] = recs[:, a:a + 8].to(torch.float64)
            setattr(self, name, v)
        self.top = torch.zeros((P, len(LEVELS)), dtype=torch.int64, device=dev)
        self.top[pos] = _u16(recs[:, 16:32]).view(H, len(LEVELS)).to(torch.int64)
        self.lstar = torch.full((P,), -1, dtype=torch.int64, device=dev)
        self.lstar[pos] = recs[:, 169].to(torch.int64)
        self.ncand = torch.zeros(P, dtype=torch.int64, device=dev)
        self.ncand[pos] = recs[:, 168].to(torch.int64)
        for name, a, b, conv in (("pos_recent", 4, 8, _u32), ("pos_longest", 8, 12, _u32), ("len_recent", 12, 14, _u16),
                                 ("len_longest", 14, 16, _u16)):
            v = torch.zeros(P, dtype=torch.int64, device=dev)
            v[pos] = conv(recs[:, a:b]).view(H).to(torch.int64)
            setattr(self, name, v)

    def level_counts(self, y: Tensor) -> dict:
        """{L: dict(N, C, D, M, n1, n2, top)} for every level, float64 [P] (top int64): C = the candidates at level L
        whose next token is the target y [P]; everything else is target-independent."""
        eq = self.nx == y.to(device=self.nx.device, dtype=torch.int32)[:, None]
        out = {}
        for k, L in enumerate(LEVELS):
            out[L] = dict(N=self.n[:, k], C=(eq & (self.len >= L)).sum(1).to(torch.float64), D=self.d[:, k],
                          M=self.m[:, k], n1=self.n1[:, k], n2=self.n2[:, k], top=self.top[:, k])
        return out


def _field(raw: Tensor, off: int, dtype: np.dtype) -> Tensor:
    """A field of packed rows raw (uint8 [H, row bytes], any device) at byte offset off: int64 / float32, [H] or
    [H, count]."""
    base, shape = dtype.base, dtype.shape
    count = int(np.prod(shape)) if shape else 1
    b = raw[:, off:off + base.itemsize * count].contiguous()
    kind = base.kind + str(base.itemsize)
    if kind == "u1":
        v = b.to(torch.int64)
    elif kind == "u2":
        v = (b.view(torch.int16).to(torch.int32) & 0xFFFF).to(torch.int64)
    elif kind == "u4":
        v = b.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    elif kind == "i4":
        v = b.view(torch.int32).to(torch.int64)
    elif kind == "f4":
        v = b.view(torch.float32)
    else:
        raise TypeError(f"no parser for {base}")
    return v.reshape(raw.shape[0], *shape) if shape else v.reshape(raw.shape[0])


def parse_low(raw: Tensor, P: int, orders=LOW_ORDERS) -> dict:
    """P2's rows of P positions (uint8, P x len(orders) x 20 bytes) -> {order: dict(N, C, D, M, n1, n2 float64 [P],
    top int64 [P])}, all 0 where the context never occurred."""
    r = raw.reshape(P * len(orders), LOW_BYTES)
    out = {}
    fields = {n: _field(r, LOW_DTYPE.fields[n][1], LOW_DTYPE.fields[n][0]).view(P, len(orders)) for n in LOW_DTYPE.names}
    for i, o in enumerate(orders):
        out[o] = {f: (fields[f][:, i] if f == "top" else fields[f][:, i].to(torch.float64)) for f in LOW_FIELDS}
    return out


def parse_ptr(raw: Tensor) -> dict:
    """P3's rows (uint8 [H, 380]) -> {field: int64 / float32 tensor [H] or [H, count]} (stream_pointer.ROW_DTYPE)."""
    if raw.dim() != 2 or raw.shape[1] != PTR_BYTES:
        raise ValueError(f"P3 rows must be uint8 [H, {PTR_BYTES}], got {tuple(raw.shape)}")
    return {n: _field(raw, PTR_DTYPE.fields[n][1], PTR_DTYPE.fields[n][0]) for n in PTR_DTYPE.names
            if not n.startswith("pad")}


def records_numpy(buf) -> np.ndarray:
    return np.frombuffer(buf, dtype=REC_DTYPE)


def read_fit(path: str) -> dict:
    """The helper's FIT dump (PATH, or a STREAM_RETRIEVAL_FIT directory holding PATH = <dir>/fit): dict(world, n
    (positions per rank), k (batches), freeze (the memory's entries before the last k steps), steps (the run's trained
    steps; None in a dump from before the header held it), batch_lengths [k], recs: per rank a uint8 [H_r, REC_BYTES]
    array, tokens: u16 [world][n][2], low: LOW_DTYPE [world][n][5] or None (P2), ptr: per rank a uint8 [H_r,
    PTR_BYTES] array or None (P3))."""
    path = fit_file(path)
    raw = np.fromfile(path, dtype=np.uint8)
    magic = int(raw[:8].view(np.uint64)[0])
    words = FIT_HEAD_WORDS if magic == FIT_MAGIC else 8
    head = raw[:8 * words].view(np.uint64)
    if magic not in (FIT_MAGIC, FIT_MAGIC_V2) or int(head[3]) != REC_BYTES:
        raise ValueError(f"{path} is not a v2 FIT dump")
    world, n, k, freeze, parts, norders = (int(head[i]) for i in (1, 2, 4, 5, 6, 7))
    steps = int(head[8]) if magic == FIT_MAGIC else None
    counts = raw[8 * words:8 * words + 8 * world].view(np.uint64).astype(np.int64)
    off = 8 * words + 8 * world
    blen = raw[off:off + 8 * k].view(np.uint64).astype(np.int64)
    off += 8 * k
    recs, o = [], off
    for c in counts:
        recs.append(raw[o:o + c * REC_BYTES].reshape(-1, REC_BYTES))
        o += c * REC_BYTES
    if o != raw.size or blen.sum() != n:
        raise ValueError(f"{path}: sizes do not add up")
    tokens = np.fromfile(f"{path}.tokens", dtype=np.uint16).reshape(world, n, 2)
    low = ptr = None
    if parts & 1:
        low = np.fromfile(f"{path}.low", dtype=LOW_DTYPE).reshape(world, n, norders)
    if parts & 2:
        praw = np.fromfile(f"{path}.ptr", dtype=np.uint8)
        pc = praw[:8 * world].view(np.uint64).astype(np.int64)
        ptr, o = [], 8 * world
        for c in pc:
            ptr.append(praw[o:o + c * PTR_BYTES].reshape(-1, PTR_BYTES))
            o += c * PTR_BYTES
        if o != praw.size:
            raise ValueError(f"{path}.ptr: sizes do not add up")
    return dict(world=world, n=n, k=k, freeze=freeze, steps=steps, batch_lengths=blen, recs=recs, tokens=tokens, low=low,
                ptr=ptr, path=path)


def fit_file(path: str) -> str:
    """STREAM_RETRIEVAL_FIT names a directory; the dump is <dir>/fit (a plain path is taken as is)."""
    return os.path.join(path, "fit") if os.path.isdir(path) else path


def fit_inputs(fit: dict, rank: int, device="cpu"):
    """Rank `rank`'s FIT positions: (x, y int64 [n], Rows, seg_start bool [n]) on device; segments restart at every
    batch (and every BOS, as on val)."""
    xy = torch.from_numpy(fit["tokens"][rank].astype(np.int64)).to(device)
    starts = torch.zeros(fit["n"], dtype=torch.bool, device=device)
    starts[torch.from_numpy(np.concatenate([[0], np.cumsum(fit["batch_lengths"])[:-1]])).to(device)] = True
    rows = Rows(torch.from_numpy(np.ascontiguousarray(fit["recs"][rank])).to(device), fit["n"])
    return xy[:, 0], xy[:, 1], rows, seg_starts(xy[:, 0], starts)


def fit_parts(fit: dict, rank: int, device="cpu"):
    """Rank `rank`'s P2 counts ({order: counts}, or None) and P3 rows (parse_ptr dict, or None) at its FIT positions."""
    low = ptr = None
    if fit["low"] is not None:
        raw = torch.from_numpy(np.ascontiguousarray(fit["low"][rank]).view(np.uint8).reshape(-1)).to(device)
        low = parse_low(raw, fit["n"])
    if fit["ptr"] is not None:
        ptr = parse_ptr(torch.from_numpy(np.ascontiguousarray(fit["ptr"][rank])).to(device))
    return low, ptr


def save_fit_lm(path: str, rank: int, **arrays):
    """The model's eval outputs at this rank's FIT positions, for fit_gate_v2.py: nll [n] (-log(p + 1e-9)), ent [n],
    mx [n], and top_v, top_in, top_rank [n, K] at query_tokens()' tokens (lm_features), as <path>.lm.rank<r>.npz."""
    need = {"nll", "ent"}
    if not need <= arrays.keys():
        raise ValueError(f"save_fit_lm needs {sorted(need)}")
    np.savez(f"{fit_file(path)}.lm.rank{rank}.npz", **{k: (v.detach().cpu().numpy() if isinstance(v, Tensor) else
                                                           np.asarray(v)) for k, v in arrays.items()})


# ==================================================================================================== the gate

GATE_MODE = "lm+su2+tp+hi"
# The shipped constants, filled below from GATE_V2_JSON / GATE_V2_LOW_JSON (fit_gate_v2.py --module writes them): per
# chain (without / with P2's orders) a list of specs, one per step count a FIT dev run was fitted at (its "fit":
# "total_steps"). A run uses the spec fitted at its own step count, whose FIT batches are its own last batches (the
# loader is deterministic); a record attempt requires one per cut, fitted on our model (tools/record_attempt/
# preflight.py: Gate.check_record).
GATE_V2 = GATE_V2_LOW = None


def select_spec(specs, steps: int | None) -> tuple[dict, bool]:
    """(the spec for a run of `steps` trained steps, whether it was fitted at exactly that step count): the one fitted
    at `steps`, else the nearest (dev runs at other step counts; ties to the longer run), else (steps None) the
    longest run's."""
    specs = [specs] if isinstance(specs, dict) else list(specs)
    fitted = lambda sp: (sp.get("fit") or {}).get("total_steps")
    for sp in specs:
        if steps is not None and fitted(sp) == steps:
            return sp, True
    known = [sp for sp in specs if fitted(sp) is not None]
    if not known:
        return specs[0], False
    if steps is None:
        return max(known, key=fitted), False
    return min(known, key=lambda sp: (abs(fitted(sp) - steps), -fitted(sp))), False


class Gate:
    """The constants of mix_v2: the chain's orders (ascending; the memory's levels, after P2's orders 1-5 when
    STREAM_RETRIEVAL_LOW is on), the feature mode, and per order the gate weights w (on the standardized phi, whose
    column 0 is the constant), the KN discount logits (3), and the standardization mu / sd of phi[:, 1:]; optionally
    the top-level softmax over extra components (P3: comps TOP_COMPS on the columns top_cols()): its weights
    W [d, K] and mu / sd."""

    def __init__(self, spec: dict, steps: int | None = None, exact: bool | None = None):
        self.spec = spec
        fit = spec.get("fit") or {}
        self.fit_steps = fit.get("total_steps")  # the trained steps of the dev run whose last batches it was fitted on
        self.proxy = fit.get("proxy")            # set when the model outputs it was fitted on are a proxy's
        self.steps = steps                       # the run's own trained steps (when known)
        self.exact = (self.fit_steps is not None and self.fit_steps == steps) if exact is None else exact
        self.orders = [int(o) for o in spec["orders"]]
        self.mode = spec["mode"]
        self.w = [torch.tensor(v, dtype=torch.float64) for v in spec["w"]]
        self.disc = [torch.tensor(v, dtype=torch.float64) for v in spec["disc"]]
        self.mu = [torch.tensor(v, dtype=torch.float64) for v in spec["mu"]]
        self.sd = [torch.tensor(v, dtype=torch.float64) for v in spec["sd"]]
        top = spec.get("top")
        self.top = None if not top else dict(comps=list(top["comps"]), cols=top.get("cols"),
                                              W=torch.tensor(top["W"], dtype=torch.float64),
                                              mu=torch.tensor(top["mu"], dtype=torch.float64),
                                              sd=torch.tensor(top["sd"], dtype=torch.float64))
        assert len(self.w) == len(self.disc) == len(self.mu) == len(self.sd) == len(self.orders)

    @property
    def pointer(self) -> bool:
        """The top level is P3's (its components and the columns of top_block)."""
        return self.top is not None and self.top["cols"] is not None

    @classmethod
    def load(cls, path: str | None = None, low: bool | None = None, steps: int | None = None) -> "Gate":
        """STREAM_RETRIEVAL_GATE=<json> (dev runs, e.g. a fresh fit_gate_v2.py output), else the shipped constants
        for the chain with (GATE_V2_LOW) or without (GATE_V2) P2's orders (default: STREAM_RETRIEVAL_LOW): the spec
        fitted at the run's `steps` (trained steps), else the nearest one (select_spec)."""
        path = path or os.environ.get("STREAM_RETRIEVAL_GATE")
        if path:
            with open(path) as f:
                spec, exact = select_spec(json.load(f), steps)
            return cls(spec, steps, exact)
        low = LOW if low is None else low
        specs = GATE_V2_LOW if low else GATE_V2
        if not specs:
            raise RuntimeError(f"stream_memory.{'GATE_V2_LOW' if low else 'GATE_V2'} is empty: fit it with "
                               "tools/stream_retrieval/fit_gate_v2.py")
        spec, exact = select_spec(specs, steps)
        return cls(spec, steps, exact)

    def describe(self) -> str:
        """One line for the log: where the constants come from, and whether they are this run's."""
        fit = self.spec.get("fit") or {}
        where = (f"the last {fit.get('batches', '?')} batches of a {self.fit_steps}-step dev run" if self.fit_steps
                 else "a FIT dump of unknown step count")
        own = ("this run's step count: its own last batches" if self.exact else
               f"NOT this run's {self.steps} steps: a dev-run gate, not a record's" if self.steps is not None else "")
        return (f"stream retrieval gate: {len(self.orders)} chain orders{' + P3' if self.pointer else ''}, fitted on "
                f"{where} ({own})" + (f"; PROXY model outputs ({self.proxy}): a placeholder" if self.proxy else "")
                + (f"; {fit['note']}" if fit.get("note") else ""))

    def check_record(self, steps: int):
        """A record's legs may only use constants fitted at their own step count, on our model's outputs at a dev run's
        last batches: raises otherwise (tools/record_attempt/preflight.py, for every cut)."""
        problems = []
        if self.fit_steps != steps:
            problems.append(f"no constants fitted at {steps} trained steps (the nearest is fitted at "
                            f"{self.fit_steps}): run a FIT dev run at that step count and fit_gate_v2.py --module")
        if self.proxy:
            problems.append(f"the constants are a placeholder fitted on a proxy's outputs ({self.proxy})")
        if problems:
            raise RuntimeError("; ".join(problems))

    @classmethod
    def for_run(cls, low: bool = LOW, pointer: bool = POINTER, steps: int | None = None) -> "Gate":
        """The run's gate (for a run of `steps` trained steps), checked against its parts before anything is built:
        its chain is chain_orders(low), and its top level is P3's exactly when P3 runs."""
        gate = cls.load(low=low, steps=steps)
        if gate.orders != chain_orders(low):
            raise RuntimeError(f"the stream retrieval gate's orders {gate.orders} are not this run's {chain_orders(low)} "
                               f"(STREAM_RETRIEVAL_LOW={int(low)}): fit_gate_v2.py on a dump of this configuration")
        if pointer != gate.pointer:
            raise RuntimeError("the stream retrieval gate " + ("has no P3 top level: STREAM_RETRIEVAL_POINTER=0, or a gate "
                               "fitted with P3" if pointer else "needs P3's rows: leave STREAM_RETRIEVAL_POINTER on"))
        if gate.pointer and gate.top["cols"] != top_cols():
            raise RuntimeError("the stream retrieval gate's top columns are not top_cols(): refit it")
        return gate

    def tokens(self) -> int:
        """Columns of query_tokens(): one per chain order, plus P3's 3 tokens."""
        return len(self.orders) + (len(TOP_COMPS) if self.pointer else 0)

    def to(self, device):
        for name in ("w", "disc", "mu", "sd"):
            setattr(self, name, [v.to(device) for v in getattr(self, name)])
        if self.top:
            self.top = {k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in self.top.items()}
        return self


def L2(v: Tensor) -> Tensor:
    return torch.log2(v.clamp_min(1.0))


def seg_starts(x: Tensor, extra: Tensor | None = None) -> Tensor:
    """Segment starts of one chunk: its first position and every BOS (and `extra`, e.g. FIT batch starts)."""
    s = x == BOS
    s[0] = True
    return s if extra is None else s | extra


def seg_start_index(starts: Tensor) -> Tensor:
    """ds[t] = the last segment start at or before t."""
    t = torch.arange(starts.numel(), device=starts.device)
    return torch.cummax(torch.where(starts, t, torch.zeros_like(t)), 0).values


def seg_ema(v: Tensor, starts: Tensor, d: float) -> Tensor:
    """S[t] = sum over EARLIER positions t' of t's segment of d^(t-1-t') v[t'] (S = 0 at a segment start): the
    recurrence S[t] = d S[t-1] + v[t-1], reset at the starts, as a log-depth scan of affine maps (float64)."""
    a = torch.where(starts, torch.zeros_like(v), torch.full_like(v, d))
    b = torch.zeros_like(v)
    b[1:] = v[:-1]
    b = torch.where(starts, torch.zeros_like(v), b)
    k, n = 1, v.numel()
    while k < n:
        b = torch.cat([b[:k], a[k:] * b[:-k] + b[k:]])
        a = torch.cat([a[:k], a[k:] * a[:-k]])
        k *= 2
    return b


def _wsum(cs: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """sum of v[lo..hi] inclusive from cs = [0, cumsum(v)]; 0 where hi < lo (here lo <= hi + 1 always)."""
    return torch.where(hi >= lo, cs[hi + 1] - cs[lo], torch.zeros((), dtype=cs.dtype, device=cs.device))


def _cs(v: Tensor) -> Tensor:
    return torch.cat([torch.zeros(1, dtype=torch.float64, device=v.device), torch.cumsum(v.to(torch.float64), 0)])


def _window_max(s: Tensor, ds: Tensor, lens) -> dict:
    """{L: max of s[t-L..t-1] clipped at the segment start ds (float32; 0 where empty)} for every L in lens."""
    s32 = s.to(torch.float32)
    P = s.numel()
    t = torch.arange(P, device=s.device)
    neg = torch.tensor(-math.inf, dtype=torch.float32, device=s.device)
    m = torch.full((P,), -math.inf, dtype=torch.float32, device=s.device)
    out, want = {}, set(lens)
    for sft in range(1, max(lens) + 1):
        shifted = torch.cat([torch.full((sft,), -math.inf, dtype=torch.float32, device=s.device), s32[:-sft]])
        m = torch.maximum(m, torch.where(t - sft >= ds, shifted, neg))
        if sft in want:
            zero = torch.zeros((), dtype=torch.float32, device=s.device)
            out[sft] = torch.where(torch.isfinite(m), m, zero).to(torch.float64)
    return out


def chain_features(counts: dict, orders, y: Tensor, x: Tensor, lp_lm: Tensor, ent: Tensor, lq_base: Tensor,
                   starts: Tensor, mode: str = GATE_MODE, mx: Tensor | None = None, tp: dict | None = None,
                   extra: Tensor | None = None, bad: Tensor | None = None) -> list:
    """The raw gate features phi_o [P, d] (column 0 = 1) of every order of the chain, at ALL P positions of a chunk,
    as rg/combo/ds.py's chain_phis (lo.order_phi + the mode's blocks):
      counts   [1, log2 N, log2 D, M/N, n1/D, n2/D, log2^2 N / 16, N == 1] (+ how much the extension from the previous
               order narrowed the context, log2 N_prev - log2 N, for every order but the first)
      'lm'     the model's entropy and max log p at t
      extra    caller-supplied columns (e.g. document-local copy statistics), in place of rg's 'cp' block
      'su2'    surprisal of the order's context: the model's NLL over the o tokens before t (log1p sum, mean, max), the
               same minus its entropy, the mean entropy / NLL over the last 16 / 1 / 4 / 16 / 64 tokens, log2 position;
               and the base's NLL over the context (log1p sum, mean, max)
      'hi'     the order's own history in the segment, EMAs 0.9 / 0.98 over earlier positions of: hit, top == target,
               log(1/2 + r / 2p) (r = C/N); the previous position's hit and top == target
      'tp'     the model's log p of the order's top token (v, v^2/10, in its top 32, max log p - v, log2(1 + rank))
    All target-independent at t: y, lp_lm and lq_base enter only through positions < t.
    counts: {o: dict(N, C, D, M, n1, n2, top)} float64 [P]; y, x [P]; lp_lm = the model's log p of the target [P]
    (for 'su2'); lq_base = log p of the base mixture's P_0 at the target [P] ('su2' second block, 'hi' log ratio);
    starts: segment starts [P] bool; mx [P] ('lm', 'tp'); tp: {o: (v, inn, rank)} [P] each."""
    dev = y.device
    P = y.numel()
    t = torch.arange(P, device=dev)
    ds = seg_start_index(starts)
    f64 = lambda v: v.to(device=dev, dtype=torch.float64)
    ent, lp_lm, lq_base = f64(ent), f64(lp_lm), f64(lq_base)
    good = torch.ones(P, dtype=torch.bool, device=dev) if bad is None else ~bad
    blocks = mode.split("+")
    shared = {}
    if "su2" in blocks:
        s, sb = -lp_lm, -lq_base
        cs, ce, csb = _cs(s), _cs(ent), _cs(sb)
        hi = t - 1
        lo16 = torch.maximum(t - 16, ds)
        shared["ent16"] = _wsum(ce, lo16, hi) / (t - lo16).clamp_min(1)
        for w in (1, 4, 16, 64):
            low = torch.maximum(t - w, ds)
            shared[f"last{w}"] = _wsum(cs, low, hi) / (t - low).clamp_min(1)
        shared["pos"] = torch.log2((t - ds).clamp_min(1).to(torch.float64))
        wmax, wmax_b = _window_max(s, ds, orders), _window_max(sb, ds, orders)
    phis = []
    for i, o in enumerate(orders):
        c = counts[o]
        N, D, M, n1, n2 = (f64(c[k]) for k in ("N", "D", "M", "n1", "n2"))
        Nn, Dn = N.clamp_min(1), D.clamp_min(1)
        cols = [torch.ones_like(N), L2(N), L2(D), M / Nn, n1 / Dn, n2 / Dn, L2(N) ** 2 / 16, (N == 1).to(torch.float64)]
        if i > 0:
            Np = f64(counts[orders[i - 1]]["N"])
            cols.append(torch.where(Np > 0, L2(Np) - L2(N), torch.zeros_like(N)))
        if "lm" in blocks:
            cols += [ent, f64(mx)]
        if extra is not None:
            cols += list(f64(extra).unbind(1))
        if "su2" in blocks:
            lo = torch.maximum(t - o, ds)
            n = (t - lo).clamp_min(0)
            S, E, Sb = _wsum(cs, lo, t - 1), _wsum(ce, lo, t - 1), _wsum(csb, lo, t - 1)
            nn = n.clamp_min(1)
            cols += [torch.log1p(S), S / nn, wmax[o], (S - E) / nn, shared["ent16"], shared["last1"], shared["last4"],
                     shared["last16"], shared["last64"], shared["pos"], torch.log1p(Sb), Sb / nn, wmax_b[o]]
        if "hi" in blocks:
            hit = (N > 0).to(torch.float64)
            top = c["top"].to(device=dev, dtype=torch.int64)
            corr = hit * (top == y.to(torch.int64)).to(torch.float64) * good.to(torch.float64)
            r = torch.where(N > 0, f64(c["C"]) / Nn, torch.zeros_like(N))
            llr = hit * torch.log(0.5 + 0.5 * r / torch.exp(lq_base))
            for d in (0.9, 0.98):
                eh, ec, el = seg_ema(hit, starts, d), seg_ema(corr, starts, d), seg_ema(llr, starts, d)
                cols += [torch.log1p(eh), ec / (eh + 0.5), el / (eh + 0.5)]
            not_first = (t > ds).to(torch.float64)
            prev = lambda v: torch.cat([v.new_zeros(1), v[:-1]]) * not_first
            cols += [prev(hit), prev(corr)]
        if "tp" in blocks:
            v, inn, rank = (f64(a) for a in tp[o])
            cols += [v, v ** 2 / 10, inn, f64(mx) - v, torch.log2(1 + rank)]
        phis.append(torch.stack(cols, 1))
    return phis


def kn_component(c: dict, disc: Tensor) -> Tensor:
    """r(y) of the modified-KN distribution at the target: max(C - d(C), 0) / (N - d1 n1 - d2 n2 - d3 (D - n1 - n2)),
    discounts d1 <= d2 <= d3 in (0, 1) from 3 logits (lo.Backoff._comp)."""
    u = torch.sigmoid(disc)
    d1 = u[0]
    d2 = u[0] + (1 - u[0]) * u[1]
    d3 = u[0] + (1 - u[0]) * u[1] + (1 - u[0] - (1 - u[0]) * u[1]) * u[2]
    N, C, n1, n2, D = c["N"], c["C"], c["n1"], c["n2"], c["D"]
    mass = N - d1 * n1 - d2 * n2 - d3 * (D - n1 - n2)
    dc = torch.where(C == 1, d1, torch.where(C == 2, d2, d3))
    num = torch.where(C > 0, C - dc, torch.zeros_like(C))
    return num / mass.clamp_min(1e-6)


def chain_prob(p0: Tensor, counts: dict, phis: list, orders, w, disc, mu, sd) -> Tensor:
    """P after the stick-breaking chain: P_0 = p0; P_i = (1 - lam_i) P_{i-1} + lam_i r_i where order i matched
    (N_i > 0), lam_i = sigmoid(w_i . [1, (phi_i[:, 1:] - mu_i) / sd_i]). Differentiable in w and disc (the fit)."""
    P = p0
    for i, o in enumerate(orders):
        h = counts[o]["N"] > 0
        if not bool(h.any()):
            continue
        with torch.no_grad():  # constants of the fit: standardized in place, one copy per order
            ph = phis[i][h].to(torch.float64, copy=True)
            ph[:, 1:] -= mu[i]
            ph[:, 1:] /= sd[i]
        lam = torch.sigmoid(ph @ w[i])
        r = kn_component({k: counts[o][k][h] for k in ("N", "C", "D", "n1", "n2")}, disc[i])
        idx = h.nonzero()[:, 0]
        P = P.index_put((idx,), (1 - lam) * P[idx] + lam * r)
    return P


def top_mixture(lq_chain: Tensor, top: dict, gate_top: dict) -> Tensor:
    """The top-level softmax over [the chain, extra components] where any extra component is available (joint.py):
    pi = softmax([0, z]) over the available ones, z = phi_std W; log q = logsumexp(log pi + log comps)."""
    comps, avail, phi = top["comps"], top["avail"], top["phi"]
    has = avail.any(1)
    if not bool(has.any()):
        return lq_chain
    ph = phi[has].to(torch.float64)
    ph = torch.cat([ph[:, :1], (ph[:, 1:] - gate_top["mu"]) / gate_top["sd"]], 1)
    z = ph @ gate_top["W"]
    z = torch.cat([torch.zeros_like(z[:, :1]), z], 1)
    av = torch.cat([torch.ones_like(avail[has][:, :1]), avail[has]], 1)
    z = torch.where(av, z, torch.full_like(z, -math.inf))
    lpi = z - torch.logsumexp(z, 1, keepdim=True)
    with torch.no_grad():
        lb_comp = torch.log(comps[has].to(torch.float64).clamp_min(0))
    lb = torch.cat([lq_chain[has][:, None], lb_comp], 1)
    out = lq_chain.clone()
    out[has] = torch.logsumexp(lpi + lb, 1)
    return out


def top_tokens(rows: Rows, gate, low: dict | None = None) -> Tensor:
    """[P, len(orders)] the top next token of every order of the chain (gate: a Gate, or the orders), 0 where the order
    did not match."""
    cols = []
    for o in (gate.orders if isinstance(gate, Gate) else gate):
        if o in LEVELS:
            k = LEVELS.index(o)
            cols.append(torch.where(rows.n[:, k] > 0, rows.top[:, k], torch.zeros_like(rows.top[:, k])))
        else:
            c = low[o]
            cols.append(torch.where(c["N"] > 0, c["top"].to(torch.int64), torch.zeros_like(c["top"], dtype=torch.int64)))
    return torch.stack(cols, 1)


def ptr_tokens(ptr: dict, P: int, device) -> Tensor:
    """[P, 3] P3's predicted tokens: the pointer's, the vote's top and the memory's top at L* (0 where unavailable)."""
    out = torch.zeros((P, len(TOP_COMPS)), dtype=torch.int64, device=device)
    if ptr is None or ptr["pos"].numel() == 0:
        return out
    flags = ptr["flags"]
    hp = (flags & stream_pointer.F_HP) != 0
    has = (flags & stream_pointer.F_HAS) != 0
    out[ptr["pos"]] = torch.stack([torch.where(hp, ptr["pred"], 0), torch.where(hp, ptr["vtop"], 0),
                                   torch.where(has, ptr["mem_top"], 0)], 1)
    return out


def query_tokens(rows: Rows, orders, low: dict | None = None, ptr: dict | None = None, pointer: bool = False) -> Tensor:
    """[P, len(orders) + 3 pointer] the tokens whose log p the model's eval forward reports (lm_features): the chain's
    top tokens (the 'tp' block), then P3's (top_block's 'tp_*' columns)."""
    tok = top_tokens(rows, orders, low)
    if pointer:
        tok = torch.cat([tok, ptr_tokens(ptr, rows.P, tok.device)], 1)
    return tok


@torch.no_grad()
def lm_top_features(logp: Tensor, tokens: Tensor, k: int = 32) -> dict:
    """From the model's log-probabilities logp [P, V] (any chunk of positions): entropy, max log p, and for tokens
    [P, K]: v = log p of the token if it is among the k most likely, else (the k-th log p) - 1; in = whether it is;
    rank = its index among the k (k if not) (rg/combo/ds.py's _lp_of over the LM's top-32)."""
    lp32, ix = logp.float().topk(k, dim=-1)
    eq = ix[:, None, :] == tokens[:, :, None]
    inn = eq.any(-1)
    v = torch.where(inn, (lp32[:, None, :] * eq).sum(-1), lp32[:, -1:] - 1.0)
    rank = torch.where(inn, eq.to(torch.int64).argmax(-1), torch.full_like(tokens, k))
    p = logp.float().exp()
    ent = -(p * logp.float()).nan_to_num().sum(-1)
    return dict(ent=ent, mx=lp32[:, 0], top_v=v, top_in=inn, top_rank=rank)


def lm_features(logits: Tensor, tokens: Tensor, drop_col: Tensor | None = None, k: int = 32) -> Tensor:
    """The model eval forward's side output for the gate (hooks.patch: model/gpt.py's eval CE loop, LM_ROWS rows at a
    time): from its masked LM logits [R, V] and the query tokens [R, K]: float32 [R, 2 + 3K] = the LM's entropy, max
    log p, then lm_top_features' top_v, top_in, top_rank (unpack_lm). drop_col: a column that is no token (CPLM's
    <copy> gate), set to the mask's -60 first. Traceable (torch.compile fullgraph)."""
    logits = logits.float()
    if drop_col is not None:
        logits = logits.index_fill(1, drop_col.long(), -60.0)
    logp = logits.log_softmax(-1)
    lp32, ix = logp.topk(k, dim=-1)
    eq = ix[:, None, :] == tokens[:, :, None]
    inn = eq.any(-1)
    v = torch.where(inn, (lp32[:, None, :] * eq).sum(-1), lp32[:, -1:] - 1.0)
    rank = torch.where(inn, eq.to(torch.int32).argmax(-1), torch.full_like(tokens, k, dtype=torch.int32))
    ent = -(logp.exp() * logp).sum(-1)
    return torch.cat([ent[:, None], lp32[:, :1], v, inn.float(), rank.float()], 1)


def unpack_lm(out: Tensor, K: int) -> dict:
    """lm_features' columns -> dict(ent, mx [P], top_v [P, K], top_in bool, top_rank int64)."""
    return dict(ent=out[:, 0], mx=out[:, 1], top_v=out[:, 2:2 + K], top_in=out[:, 2 + K:2 + 2 * K] > 0.5,
                top_rank=out[:, 2 + 2 * K:2 + 3 * K].round().to(torch.int64))


# ------------------------------------------------------------------------------------------------ P3's top block

_TOP_LM_COLS = (["lm_ent", "lm_mx"] + [f"tp_{c}_{f}" for c in ("ptr", "vote", "mem") for f in ("v", "in", "gap")]
                + [f"su_{c}_{f}" for c in ("ptr", "src", "mem") for f in ("log", "mean", "max", "ment")]
                + ["chain_logo", "chain_logn", "chain_purity", "chain_logd"]
                + ["llr_ret_32", "llr_ret_seg", "llr_ptr_32", "llr_ptr_seg"])


def top_cols() -> list:
    """The top level's feature columns after the constant: P3's helper features (stream_pointer.features, the
    research's names) and the model-aware ones of top_block."""
    return list(stream_pointer.features(np.zeros(0, PTR_DTYPE))) + _TOP_LM_COLS


def _ctx_surprisal(cs: Tensor, ce: Tensor, s32: Tensor, ds: Tensor, t: Tensor, L: Tensor, lmax: int) -> list:
    """At positions t [H] with context lengths L [H] (0: none): over [max(t - L, ds_t), t - 1], the window of the eval
    NLL s of EARLIER positions (cs, ce: cumsums of s and of the entropy): log1p(S), S / n, max s, (S - E) / n
    (rg/lmfeat/lmaware.py match_feats' first four)."""
    lo = torch.maximum(t - L, ds)
    hi = t - 1
    n = (hi - lo + 1).clamp_min(0)
    S, E = _wsum(cs, lo, hi), _wsum(ce, lo, hi)
    m = torch.full(t.shape, -math.inf, dtype=torch.float32, device=t.device)
    for sft in range(1, lmax + 1):
        ok = (sft <= L) & (t - sft >= ds)
        m = torch.maximum(m, torch.where(ok, s32[(t - sft).clamp_min(0)], torch.full_like(m, -math.inf)))
    nn = n.clamp_min(1)
    return [torch.log1p(S), S / nn, torch.where(n > 0, m.to(torch.float64), torch.zeros_like(S)), (S - E) / nn]


def top_block(ptr: dict, P: int, x: Tensor, y: Tensor, starts: Tensor, nll: Tensor, lm: dict, k0: int, counts: dict,
              orders, rows: Rows) -> dict:
    """P3's components and the top level's features at one chunk of P positions: dict(comps [P, 3] float64 (the
    pointer's, the vote's and the source copy's probability of the target), avail [P, 3] bool, phi [P, 1 + d] float32
    (0 where nothing is available; columns [1] + top_cols())).
    ptr: parse_ptr's rows (positions ptr['pos']); nll [P]: the model's eval NLL (-log(p + 1e-9)); lm: ent, mx [P],
    top_v / top_in [P, K] whose columns k0, k0 + 1, k0 + 2 are at ptr_tokens(); counts: the chain's {order: counts}
    (with the target's C); rows: P1's Rows (L* for the memory's own history). Every column is target-independent at t:
    targets and the model's probabilities enter only through EARLIER positions."""
    dev = nll.device
    d = 1 + len(top_cols())
    out = dict(comps=torch.zeros((P, len(TOP_COMPS)), dtype=torch.float64, device=dev),
               avail=torch.zeros((P, len(TOP_COMPS)), dtype=torch.bool, device=dev),
               phi=torch.zeros((P, d), dtype=torch.float32, device=dev))
    idx = ptr["pos"].to(device=dev, dtype=torch.int64)
    if idx.numel() == 0:
        return out
    if int(idx.max()) >= P:
        raise ValueError(f"a P3 row at position {int(idx.max())} of a chunk of {P}")
    yv = y.to(device=dev, dtype=torch.int64)
    flags = ptr["flags"]
    hp = (flags & stream_pointer.F_HP) != 0
    hs = (flags & stream_pointer.F_HS) != 0
    has = (flags & stream_pointer.F_HAS) != 0
    p_ptr, p_vote, p_src = stream_pointer.component_probs(ptr, yv[idx], xp=torch)
    out["comps"][idx] = torch.stack([p_ptr, p_vote, p_src], 1).to(torch.float64)
    out["avail"][idx] = torch.stack([hp, hp, hs], 1)
    cols = [v.to(torch.float64) for v in stream_pointer.features(ptr, xp=torch).values()]
    # the model at t
    ent, mx = lm["ent"].to(torch.float64), lm["mx"].to(torch.float64)
    cols += [ent[idx], mx[idx]]
    for j, av in enumerate((hp, hp, has)):
        v = lm["top_v"][idx, k0 + j].to(torch.float64)
        inn = lm["top_in"][idx, k0 + j].to(torch.float64)
        z = torch.zeros_like(v)
        cols += [torch.where(av, v, z), torch.where(av, inn, z), torch.where(av, mx[idx] - v, z)]
    # surprisal of each component's matched context: the pointer's current run, the source's longest match, the
    # memory's longest full match
    ds = seg_start_index(starts.to(dev))
    s = nll.to(torch.float64)
    cs, ce, s32 = _cs(s), _cs(ent), s.to(torch.float32)
    lens = ((torch.where(hp, ptr["run"].clamp(max=64), 0), 64), (torch.where(hs, ptr["src_lbest"].clamp(max=32), 0), 32),
            (torch.where(has, ptr["mem_lenl"].clamp(max=64), 0), 64))
    for L, lmax in lens:
        cols += _ctx_surprisal(cs, ce, s32, ds[idx], idx, L.to(torch.int64), lmax)
    # the longest chain order matched at t: its order, log N, purity, log D
    lo_ = torch.zeros(P, dtype=torch.float64, device=dev)
    Nl, Ml, Dl = torch.zeros_like(lo_), torch.zeros_like(lo_), torch.zeros_like(lo_)
    for o in orders:
        c = counts[o]
        m = c["N"].to(dev) > 0
        lo_ = torch.where(m, float(o), lo_)
        Nl, Ml, Dl = (torch.where(m, c[f].to(device=dev, dtype=torch.float64), a) for f, a in (("N", Nl), ("M", Ml), ("D", Dl)))
    cols += [torch.log2(lo_.clamp_min(1))[idx], L2(Nl)[idx], (Ml / Nl.clamp_min(1))[idx], L2(Dl)[idx]]
    # the memory's and the pointer's log-likelihood ratio against p over EARLIER positions of the segment (last 32,
    # whole segment): rg/align/feats.py hist2 with llr
    t = torch.arange(P, device=dev)
    pbase = torch.exp(-s)
    hit = rows.hit.to(dev)
    li = rows.lstar.clamp_min(0)
    Lstar = torch.tensor(LEVELS, dtype=torch.int32, device=dev)[li]
    Cl = ((rows.nx == yv.to(torch.int32)[:, None]) & (rows.len >= Lstar[:, None])).sum(1).to(torch.float64)
    Nst = rows.n.gather(1, li[:, None])[:, 0]
    b_ret = torch.where(hit, Cl / Nst.clamp_min(1), torch.zeros_like(Cl))
    b_ptr = torch.zeros(P, dtype=torch.float64, device=dev)
    b_ptr[idx] = torch.where(hp, p_ptr.to(torch.float64), torch.zeros_like(b_ptr[idx]))
    hp_d = torch.zeros(P, dtype=torch.bool, device=dev)
    hp_d[idx] = hp
    for b, av in ((b_ret, hit), (b_ptr, hp_d)):
        llr = torch.where(av, torch.log(0.5 * pbase + 0.5 * b) - torch.log(pbase), torch.zeros_like(b))
        cl = _cs(llr)
        for lo in (torch.maximum(ds, t - 32), ds):
            v = _wsum(cl, lo, t - 1)[idx]
            cols.append(torch.sign(v) * torch.log1p(v.abs()))
    phi = torch.stack([torch.ones_like(cols[0])] + cols, 1)
    if phi.shape[1] != d:
        raise RuntimeError(f"top_block built {phi.shape[1]} columns, top_cols() has {d - 1} + 1")
    out["phi"][idx] = phi.to(torch.float32)
    return out


def mix_v2(nll: Tensor, y: Tensor, x: Tensor, rows: Rows, lm: dict, gate: Gate, *, low: dict | None = None,
           top: dict | None = None, starts: Tensor | None = None, eps: float = 1e-9, return_logq: bool = False):
    """Per-token NLL of the mixture at one chunk of P positions (the untimed eval).
    nll [P]: the model's eval NLL, -log(p + eps); y, x [P]: targets, inputs; rows: the chunk's Rows (the memory's
    levels); lm: dict(ent [P], and for the gate's mode: mx [P], top_v / top_in / top_rank [P, K] at top_tokens()'s
    tokens, nll_lm [P] if the model's own NLL differs from the mixture base's); low: P2's {order: counts} for orders
    1-5 (STREAM_RETRIEVAL_LOW); top: the top level's dict(comps [P, K'], avail [P, K'] bool, phi [P, d]) (P3:
    top_block); starts: segment starts (default: the chunk start and every BOS).
    Returns -log(q + eps) where some order (or top component) is available, else nll bit-identical."""
    dev = nll.device
    if gate.w[0].device != dev:
        gate.to(dev)
    nll64 = nll.to(torch.float64)
    p0 = (torch.exp(-nll64) - eps).clamp_min(0)
    counts = dict(rows.level_counts(y))
    if low:
        counts.update(low)
    counts = {o: {k: (v.to(dev) if k == "top" else v.to(device=dev, dtype=torch.float64)) for k, v in counts[o].items()}
              for o in gate.orders}
    starts = seg_starts(x) if starts is None else starts
    lq_feat = -nll64  # the base's log-prob at the target, as the features see it
    lp_lm = -lm["nll_lm"].to(torch.float64) if "nll_lm" in lm else lq_feat
    tp = None
    if "tp" in gate.mode.split("+"):
        tp = {o: (lm["top_v"][:, i], lm["top_in"][:, i], lm["top_rank"][:, i]) for i, o in enumerate(gate.orders)}
    phis = chain_features(counts, gate.orders, y, x, lp_lm, lm["ent"], lq_feat, starts, gate.mode, mx=lm.get("mx"),
                          tp=tp, extra=lm.get("extra"))
    P = chain_prob(p0, counts, phis, gate.orders, gate.w, gate.disc, gate.mu, gate.sd)
    hit = torch.zeros_like(p0, dtype=torch.bool)
    for o in gate.orders:
        hit |= counts[o]["N"] > 0
    lq = torch.log(P.clamp_min(1e-300))
    if top is not None and gate.top is not None:
        lq = top_mixture(lq, top, gate.top)
        hit |= top["avail"].any(1)
    if return_logq:
        return lq, hit
    return torch.where(hit, -torch.log(torch.exp(lq) + eps), nll64).to(nll.dtype)


def chunk_mix(nll: Tensor, x: Tensor, y: Tensor, rows: Rows, low: dict | None, ptr: dict | None, lm: dict, gate: Gate,
              starts: Tensor | None = None) -> Tensor:
    """One chunk's mixed NLL from its parts (P1 rows, P2 counts, P3 rows) and the model's outputs (nll, lm): the
    top block built from P3's rows when the gate has P3's top level, then mix_v2. The eval and the fitter's check."""
    starts = seg_starts(x) if starts is None else starts
    top = None
    if gate.pointer:
        counts = dict(rows.level_counts(y))
        if low:
            counts.update(low)
        top = top_block(ptr, rows.P, x, y, starts, nll, lm, len(gate.orders), counts, gate.orders, rows)
    return mix_v2(nll, y, x, rows, lm, gate, low=low, top=top, starts=starts)


class StreamEval:
    """The retrieval's side of the untimed final validation on one rank: per val step s, the query tokens for the
    model's LM side output (hooks.patch: the model's stream_lm_tok / stream_lm_out buffers), then the mixture."""

    def __init__(self, rows: "DeviceRows", gate: Gate):
        self.rows, self.gate = rows, gate
        if gate.pointer and rows.ptr is None:
            raise RuntimeError("the gate has P3's top level but the helper wrote no P3 rows")
        if any(o not in LEVELS for o in gate.orders) and rows.low is None:
            raise RuntimeError("the gate's chain has P2's orders but the helper wrote no P2 rows")

    def parts(self, s: int):
        return self.rows.chunk_rows(s), self.rows.chunk_low(s), self.rows.chunk_ptr(s)

    def tokens(self, s: int) -> Tensor:
        rows, low, ptr = self.parts(s)
        return query_tokens(rows, self.gate.orders, low, ptr, self.gate.pointer)

    def mix(self, s: int, nll: Tensor, x: Tensor, y: Tensor, lm: dict) -> Tensor:
        rows, low, ptr = self.parts(s)
        return chunk_mix(nll, x, y, rows, low, ptr, lm, self.gate)

    @torch.no_grad()
    def evaluate(self, s: int, batch, forward, model):
        """(the model's NLL, the mixture's NLL) of val step s: the query tokens go to the model's stream_lm_tok
        buffer, forward() runs the eval forward, its side output (stream_lm_out) feeds the gate."""
        tok = self.tokens(s)
        model.stream_lm_tok.copy_(tok)
        nll = forward()
        lm = unpack_lm(model.stream_lm_out, tok.shape[1])
        x, y = batch.inputs.to(torch.int64), batch.targets.to(torch.int64)
        return nll, self.mix(s, nll, x, y, lm)


# ==================================================================================================== the handle

class StreamMemory:
    """The memory's handle on every rank: rank 0 owns the helper and feeds it; every rank maps the rows.

    world: ranks whose spans each step message carries (the loader's world size) and the rows' layout.
    low / pointer: the helper's P2 / P3 parts (for_run: STREAM_RETRIEVAL_LOW, STREAM_RETRIEVAL_POINTER); low_entries
    (tests): P2's table entries per order, in place of the sizes measured on FineWeb (a random stream needs more).
    fit_path / fit_k: STREAM_RETRIEVAL_FIT (dev runs): the rows of the run's last fit_k batches (fit_go, fit_wait).
    """

    def __init__(self, *, train_files: list[str], val_file: str, total_steps: int, stream_tokens: int, world: int,
                 rank: int, master: bool, val_tokens: int, chunk: int, device, print0=None, threads: int | None = None,
                 hash_bits: int = HASH_BITS, fit_path: str | None = None, fit_k: int = FIT_K, low: bool = False,
                 pointer: bool = False, low_threads: int = LOW_THREADS, low_entries: int = 0, dump_path: str | None = None,
                 readlog_path: str | None = None, rows_dir: str | None = None):
        assert val_tokens % (world * chunk) == 0, "val_tokens must be a whole number of world x chunk steps"
        assert world * (val_tokens // (world * chunk)) <= MAX_CHUNKS
        self.world, self.rank, self.master, self.chunk = world, rank, master, chunk
        self.val_steps = val_tokens // (world * chunk)
        self.total_steps = total_steps
        self.low, self.pointer = bool(low), bool(pointer)
        self.device = torch.device(device)
        self.print0 = print0 or (lambda s, console=False: None)
        self.fit_path, self.fit_k = (fit_file(fit_path) if fit_path else None), (fit_k if fit_path else 0)
        self.sent = self.max_lag = 0
        self.proc = self.fd = self._queue = self._writer = self._writer_error = None
        self.waited_ms = self.collect_ms = 0.0
        self.layout = rows_layout(world, self.val_steps, chunk, self.low, self.pointer)
        nbytes = self.layout["total"]  # reserved; tmpfs holds what is written
        path = None
        if master:
            rows_dir = rows_dir or self._rows_dir(val_tokens)
            fd, path = tempfile.mkstemp(prefix="stream_rows_", dir=rows_dir)
            self._path = path
            atexit.register(self.close)
            os.ftruncate(fd, nbytes)
            os.close(fd)
            # Allocation only: the stream (cap entries reserved, the expected ones prefaulted), the heads, P2's tables.
            cap = 2 * stream_tokens + 2  # a span holds >= 1 token, so separators <= tokens
            prefault = min(cap, int(1.02 * stream_tokens) + (1 << 20))
            threads = threads or int(os.environ.get("STREAM_RETRIEVAL_THREADS", 0)) or \
                max(1, min(QUERY_THREADS, available_cpus() - 16))
            args = [f"rows={path}", f"val={val_file}", f"world={world}", f"val_tokens={val_tokens}", f"chunk={chunk}",
                    f"steps={total_steps}", f"cap={cap}", f"prefault={prefault}", f"hash_bits={hash_bits}",
                    f"threads={threads}", f"low={int(self.low)}", f"low_threads={low_threads}",
                    f"low_tokens={stream_tokens}", f"low_entries={low_entries}", f"pointer={int(self.pointer)}"]
            if self.fit_path:
                Path(self.fit_path).parent.mkdir(parents=True, exist_ok=True)
                args += [f"fit={self.fit_path}", f"fit_k={fit_k}"]
            args += [f"{k}={v}" for k, v in (("dump", dump_path), ("readlog", readlog_path)) if v]
            self.proc = subprocess.Popen([str(build_helper()), *args, "--", *train_files], stdin=subprocess.PIPE)
            self.fd = self.proc.stdin.fileno()
            try:
                fcntl.fcntl(self.fd, F_SETPIPE_SZ, PIPE_BYTES)
            except OSError:
                pass  # the default 64 KB still holds ~16 steps of spans
            # The messages are packed and written by a thread of their own: the tap only queues the loader's lists.
            self._queue = queue.SimpleQueue()
            self._writer = threading.Thread(target=self._write_loop, name="stream_memory_writer", daemon=True)
            self._writer.start()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            box = [path]
            dist.broadcast_object_list(box, src=0)
            path = box[0]
        with open(path, "r+b") as f:
            self.map = mmap.mmap(f.fileno(), nbytes)
        self._fd = os.open(path, os.O_RDONLY)  # collect() reads the rows with preadv: no page faults on this mapping
        self.hdr = np.frombuffer(self.map, dtype=np.uint64, count=64)
        if master:
            self._wait(ST_READY, timeout=READY_TIMEOUT_S)
            parts = "P1" + (" + P2 (orders 1-5)" if self.low else "") + (" + P3" if self.pointer else "")
            self.print0(f"stream retrieval: helper ready ({parts}; {stream_tokens} stream tokens expected, {threads} "
                        f"query threads, 2^{hash_bits} buckets{f', FIT on the last {fit_k} batches' if fit_path else ''})",
                        console=True)
        _sync()
        if master:
            os.unlink(path)  # the mappings stay
            self._path = None
        if int(self.hdr[H_LOW_OFFSET]) != self.layout["low"] or int(self.hdr[H_PTR_OFFSET]) != self.layout["ptr"]:
            raise RuntimeError("stream retrieval: the helper's rows layout is not stream_memory.py's")
        self.counts = np.frombuffer(self.map, dtype=np.uint64, count=world * self.val_steps,
                                    offset=COUNTS_OFFSET).reshape(world, self.val_steps)
        self.pcounts = np.frombuffer(self.map, dtype=np.uint64, count=world * self.val_steps,
                                     offset=PCOUNTS_OFFSET).reshape(world, self.val_steps)
        self.region_bytes = chunk * REC_BYTES
        pin = self.device.type == "cuda"
        n = self.val_steps * chunk
        self.host = torch.empty(max(1, int(PINNED_SHARE * n)) * REC_BYTES, dtype=torch.uint8, pin_memory=pin)
        self.host_low = torch.empty(n * len(LOW_ORDERS) * LOW_BYTES if self.low else 1, dtype=torch.uint8, pin_memory=pin)
        self.host_ptr = torch.empty(max(1, int(PINNED_PTR_SHARE * n)) * PTR_BYTES if self.pointer else 1,
                                    dtype=torch.uint8, pin_memory=pin)

    @staticmethod
    def _rows_dir(val_tokens: int) -> str:
        """/dev/shm when it has room for the rows (~1-3 GB with P2 / P3), else the temp directory (page cache)."""
        if os.path.isdir("/dev/shm"):
            try:
                st = os.statvfs("/dev/shm")
                if st.f_bavail * st.f_frsize > 400 * val_tokens + (1 << 30):
                    return "/dev/shm"
            except OSError:
                pass
        return tempfile.gettempdir()

    @classmethod
    def for_run(cls, *, train_pattern: str, val_pattern: str, step_batch_sizes: list[int], world: int, rank: int,
                master: bool, val_tokens: int, chunk: int, device, print0=None, low: bool = LOW,
                pointer: bool = POINTER):
        """The trainer's memory: the shards in data.py's order, sized from the schedule (each rank's spans hold
        its batch tokens + 1), STREAM_RETRIEVAL_FIT (a directory) / STREAM_RETRIEVAL_FIT_K from the environment."""
        local = int(os.environ.get("LOCAL_WORLD_SIZE", world))
        if local != world:
            raise RuntimeError(f"STREAM_RETRIEVAL needs one node: world size {world}, local world size {local}")
        fit_dir = os.environ.get("STREAM_RETRIEVAL_FIT") or None
        fit_k = int(os.environ.get("STREAM_RETRIEVAL_FIT_K", FIT_K))
        if fit_dir:  # fit_dump evaluates the FIT positions in val-shaped chunks (fit_batches): check now, not at the end
            if not 0 < fit_k < len(step_batch_sizes):
                raise RuntimeError(f"STREAM_RETRIEVAL_FIT_K={fit_k}: between 1 and the run's {len(step_batch_sizes)} steps")
            n = sum(b // world for b in step_batch_sizes[-fit_k:])
            if n % chunk:
                last = step_batch_sizes[-1] // world
                raise RuntimeError(f"STREAM_RETRIEVAL_FIT_K={fit_k}: the last {fit_k} batches hold {n} positions per rank, "
                                   f"not a whole number of {chunk}-position eval chunks; with the final batch of {last} "
                                   f"per rank use a multiple of {chunk // math.gcd(chunk, last)}")
        train_files = sorted(glob.glob(train_pattern))
        val_file = sorted(glob.glob(val_pattern))[0]
        stream_tokens = sum(batch_size + world for batch_size in step_batch_sizes)
        if fit_dir and master:
            os.makedirs(fit_dir, exist_ok=True)
        return cls(train_files=train_files, val_file=val_file, total_steps=len(step_batch_sizes),
                   stream_tokens=stream_tokens, world=world, rank=rank, master=master, val_tokens=val_tokens,
                   chunk=chunk, device=device, print0=print0, low=low, pointer=pointer,
                   fit_path=os.path.join(fit_dir, "fit") if fit_dir else None, fit_k=fit_k)

    # ---------------------------------------------------------------- rank 0: the messages

    @staticmethod
    def _pack(step: int, file_idx: int, starts: list, ends: list) -> np.ndarray:
        """One STEP message from the loader's span lists (one list per rank)."""
        counts = [len(s) for s in starts]
        n, head_words = sum(counts), 4 + len(counts)
        msg = np.empty(head_words + 2 * n, dtype=np.uint32)
        msg[:4] = (MSG_STEP, step, file_idx, len(starts))
        msg[4:head_words] = counts
        msg[head_words::2] = np.fromiter(itertools.chain.from_iterable(starts), dtype=np.int64, count=n)
        msg[head_words + 1::2] = np.fromiter(itertools.chain.from_iterable(ends), dtype=np.int64, count=n)
        return msg

    def _write_now(self, words: np.ndarray):
        view = memoryview(words.astype(np.uint32, copy=False).tobytes())
        while view:
            n = os.write(self.fd, view)
            view = view[n:]

    def _write_loop(self):
        """The writer thread (rank 0): packs and writes the queued messages in order, until close() queues None."""
        while (item := self._queue.get()) is not None:
            if self._writer_error is not None:
                continue
            try:
                self._write_now(self._pack(*item) if isinstance(item, tuple) else item)
            except Exception as e:  # noqa: BLE001 - a dead helper: raised to the trainer at its next message
                self._writer_error = e

    def _check_alive(self):
        """Rank 0: raise if the helper has died or a message to it failed (the writer thread only records that)."""
        if self._writer_error is not None:
            raise RuntimeError(f"stream retrieval: the message to the helper failed: {self._writer_error!r}")
        if self.proc.poll() is not None:
            raise RuntimeError(f"stream retrieval helper exited ({self.proc.returncode}) during training")

    def _write(self, words: np.ndarray):
        """Queue a message (rank 0); the writer thread writes every message in the order queued."""
        self._check_alive()
        self._queue.put(np.asarray(words, dtype=np.uint32).copy())

    def on_spans(self, file_idx: int, starts: list, ends: list):
        """The loader's tap (rank 0): one fetched step's spans for every rank, from shard file `file_idx`. Runs on the
        thread that fetches (rank 0's main thread) and only queues the loader's lists; the writer thread packs them
        (the loader never changes a list it has returned). The helper's liveness (a waitpid) is checked every 16 steps
        and at GO."""
        if self.sent % 16 == 0:
            self._check_alive()
        elif self._writer_error is not None:
            raise RuntimeError(f"stream retrieval: the message to the helper failed: {self._writer_error!r}")
        self._queue.put((self.sent, file_idx, starts, ends))
        self.sent += 1
        self.max_lag = max(self.max_lag, self.sent - int(self.hdr[H_STEPS]))

    def go(self):
        """Rank 0, at the last step: the memory is complete; read val and compute the rows."""
        self._write(np.array([MSG_GO, self.sent], dtype=np.uint32))

    # ---------------------------------------------------------------- every rank

    def _wait(self, state: int, timeout: float):
        """Until the helper's state is `state` or later (READY < DONE < FIT_DONE); raises on its error or death."""
        t0 = time.perf_counter()
        while (current := int(self.hdr[H_STATE])) < state or current == ST_ERROR:
            if current == ST_ERROR:
                msg = bytes(self.map[ERRMSG_OFFSET:ERRMSG_OFFSET + 256]).split(b"\0")[0].decode(errors="replace")
                raise RuntimeError(f"stream retrieval helper failed: {msg}")
            if self.proc is not None and self.proc.poll() is not None and int(self.hdr[H_STATE]) < state:
                raise RuntimeError(f"stream retrieval helper exited ({self.proc.returncode}) before state {state}")
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError(f"stream retrieval: no state {state} after {timeout:.1f} s (state {current})")
            time.sleep(1e-4)
        return 1000 * (time.perf_counter() - t0)

    def own_records(self) -> list:
        """This rank's records per val step: numpy uint8 [H_s, REC_BYTES] views of the shared map."""
        out = []
        for s in range(self.val_steps):
            off = ROWS_OFFSET + (self.rank * self.val_steps + s) * self.region_bytes
            n = int(self.counts[self.rank, s])
            out.append(np.frombuffer(self.map, dtype=np.uint8, count=n * REC_BYTES, offset=off).reshape(n, REC_BYTES))
        return out

    def own_low(self) -> np.ndarray | None:
        """This rank's P2 rows, every val step's in turn: uint8 [val_steps x chunk x 5 x 20] (one contiguous slab)."""
        if not self.low:
            return None
        size = self.chunk * len(LOW_ORDERS) * LOW_BYTES
        return np.frombuffer(self.map, dtype=np.uint8, count=self.val_steps * size,
                             offset=self.layout["low"] + self.rank * self.val_steps * size)

    def own_ptr(self) -> list | None:
        """This rank's P3 rows per val step: numpy uint8 [H_s, PTR_BYTES] views."""
        if not self.pointer:
            return None
        out = []
        for s in range(self.val_steps):
            off = self.layout["ptr"] + (self.rank * self.val_steps + s) * self.chunk * PTR_BYTES
            n = int(self.pcounts[self.rank, s])
            out.append(np.frombuffer(self.map, dtype=np.uint8, count=n * PTR_BYTES, offset=off).reshape(n, PTR_BYTES))
        return out

    def _read(self, pieces: list, host: Tensor) -> Tensor:
        """The byte ranges pieces [(offset, nbytes)] of the rows file, concatenated into the pinned buffer host (a
        plain one if they do not fit): preadv from the file, split across threads (no page faults on the mapping)."""
        total = sum(n for _, n in pieces)
        buf = host[:total] if total <= host.numel() else torch.empty(total, dtype=torch.uint8)
        dst = memoryview(buf.numpy())
        jobs, o = [], 0
        for off, n in pieces:
            for a in range(0, n, READ_PIECE):
                b = min(n, a + READ_PIECE)
                jobs.append((off + a, o + a, b - a))
            o += n

        def read(job):
            off, at, n = job
            while n:
                got = os.preadv(self._fd, [dst[at:at + n]], off)
                if got <= 0:
                    raise RuntimeError("stream retrieval: short read of the rows file")
                off, at, n = off + got, at + got, n - got
        if len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(READ_THREADS, len(jobs))) as pool:
                list(pool.map(read, jobs))
        elif jobs:
            read(jobs[0])
        return buf

    def _to_device(self, buf: Tensor) -> Tensor:
        return buf.to(self.device, non_blocking=True) if self.device.type == "cuda" else buf.clone()

    def collect(self) -> "DeviceRows":
        """This rank's rows on its device, once the helper is done (raises if it failed): on the clock, a copy of ~7.5%
        x 176 bytes (P1) + 100 bytes (P2) + ~33% x 380 bytes (P3) per val position."""
        self.waited_ms = self._wait(ST_DONE, COLLECT_TIMEOUT_S)
        t0 = time.perf_counter()
        r, vs, ch = self.rank, self.val_steps, self.chunk
        sizes = [int(self.counts[r, s]) for s in range(vs)]
        recs = self._read([(ROWS_OFFSET + (r * vs + s) * self.region_bytes, n * REC_BYTES) for s, n in enumerate(sizes)],
                          self.host)
        low = ptr = psizes = None
        if self.low:
            size = ch * len(LOW_ORDERS) * LOW_BYTES
            low = self._read([(self.layout["low"] + r * vs * size, vs * size)], self.host_low)
        if self.pointer:
            psizes = [int(self.pcounts[r, s]) for s in range(vs)]
            ptr = self._read([(self.layout["ptr"] + (r * vs + s) * ch * PTR_BYTES, n * PTR_BYTES)
                              for s, n in enumerate(psizes)], self.host_ptr)
        recs = self._to_device(recs).view(-1, REC_BYTES)
        low = self._to_device(low) if low is not None else None
        ptr = self._to_device(ptr).view(-1, PTR_BYTES) if ptr is not None else None
        self.collect_ms = 1000 * (time.perf_counter() - t0)
        return DeviceRows(recs, sizes, ch, low=low, ptr=ptr, psizes=psizes)

    def check(self, val_batches):
        """Every val batch of this rank is the chunk its rows were computed for (a checksum of its tokens)."""
        sums = np.frombuffer(self.map, dtype=np.uint64, count=self.world * self.val_steps, offset=CHECKSUM_OFFSET)
        for s, batch in enumerate(val_batches):
            tokens = np.empty(self.chunk + 1, dtype=np.uint64)
            tokens[:-1] = batch.inputs_cpu
            tokens[-1] = int(batch.targets_cpu[-1])
            if chunk_checksum(tokens) != int(sums[s * self.world + self.rank]):
                raise RuntimeError(f"stream retrieval: rank {self.rank}'s val batch {s} is not the chunk its rows hold")

    def stats(self) -> str:
        h = [int(v) for v in self.hdr[:H_LOW_LATE_PENDING + 1]]
        q = max(h[H_QUERIED], 1)
        levels = " ".join(f"{lv}:{h[H_LEVEL0 + k] / q:.4f}" for k, lv in enumerate(LEVELS))
        out = (f"stream retrieval: memory {h[H_ENTRIES] - 1} entries ({h[H_INSERTED]} indexed) from {h[H_STEPS]} steps, "
               f"insert {h[H_INSERT_NS] / 1e9:.2f} s ({h[H_INSERT_NS] / max(h[H_INSERTED], 1):.1f} ns/entry), max lag "
               f"{self.max_lag} steps; GO->rows {(h[H_T_DONE] - h[H_T_GO]) / 1e6:.1f} ms (queries "
               f"{h[H_QUERY_NS] / 1e6:.1f} ms); waited {self.waited_ms:.1f} ms at collect, copied in "
               f"{self.collect_ms:.1f} ms; hit {h[H_HITS] / q:.4f} ({h[H_CANDS] / max(h[H_HITS], 1):.1f} candidates), "
               f"L* {levels}")
        if self.pointer:
            out += f"; P3 rows {h[H_PTR_ROWS] / q:.4f}"
        if self.low:
            out += (f"; P2 tables {h[H_LOW_BYTES] / 2 ** 30:.2f} GiB, insertion {h[H_LOW_CPU_NS] / 1e9:.1f} CPU-s, max "
                    f"backlog {h[H_LOW_MAX_PENDING]} steps ({h[H_LOW_LATE_PENDING]} over the last {LATE_STEPS}), drained "
                    f"{h[H_LOW_DRAIN_NS] / 1e6:.1f} ms after GO, GO waited {h[H_LOW_WAIT_NS] / 1e6:.1f} ms past the "
                    "P1 / P3 queries")
        return out + self._huge_pages()

    def _huge_pages(self) -> str:
        """Rank 0: the helper's memory in transparent huge pages, and the kernel's THP mode (they matter to insertion)."""
        if self.proc is None:
            return ""
        try:
            mode = Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text().split("[")[1].split("]")[0]
        except (OSError, IndexError):
            mode = "?"
        try:
            kb = next(int(line.split()[1]) for line in Path(f"/proc/{self.proc.pid}/smaps_rollup").read_text().splitlines()
                      if line.startswith("AnonHugePages:"))
            return f"; helper huge pages {kb / 2 ** 20:.2f} GiB (THP {mode})"
        except (OSError, StopIteration, ValueError):
            return f"; THP {mode}"

    # ---------------------------------------------------------------- STREAM_RETRIEVAL_FIT (dev, untimed)

    def fit_go(self):
        """Rank 0, after the clock has stopped: compute the FIT rows (queries of the frozen memory)."""
        if not self.fit_path:
            raise RuntimeError("no STREAM_RETRIEVAL_FIT path")
        self._write(np.array([MSG_FIT, self.fit_k], dtype=np.uint32))

    def fit_wait(self) -> dict:
        """Every rank, after rank 0's fit_go: the FIT rows of the run's last fit_k batches (read_fit's dict)."""
        if not self.fit_path:
            raise RuntimeError("no STREAM_RETRIEVAL_FIT path")
        self._wait(ST_FIT_DONE, FIT_TIMEOUT_S)
        return read_fit(self.fit_path)

    def close(self):
        if self._writer is not None:
            self._queue.put(None)
            self._writer.join(timeout=30)
            if self._writer.is_alive() and self.proc is not None:  # a helper that stopped reading: its pipe is full
                self.proc.kill()
                self._writer.join()
            self._writer = None
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None
        if getattr(self, "_path", None):
            try:
                os.unlink(self._path)
            except OSError:
                pass
            self._path = None


class DeviceRows:
    """A rank's rows on its device: step s's records are recs[off[s]:off[s + 1]] (chunk_rows(s) -> Rows); P2's rows
    (chunk_low(s) -> {order: counts}) and P3's (chunk_ptr(s) -> parse_ptr dict) when the helper wrote them."""

    def __init__(self, recs: Tensor, sizes: list, chunk: int, low: Tensor | None = None, ptr: Tensor | None = None,
                 psizes: list | None = None):
        self.recs, self.chunk, self.low, self.ptr = recs, chunk, low, ptr
        self.off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.poff = np.concatenate([[0], np.cumsum(psizes)]).astype(np.int64) if psizes is not None else None

    def chunk_rows(self, s: int) -> Rows:
        return Rows(self.recs[int(self.off[s]):int(self.off[s + 1])], self.chunk)

    def chunk_low(self, s: int) -> dict | None:
        if self.low is None:
            return None
        size = self.chunk * len(LOW_ORDERS) * LOW_BYTES
        return parse_low(self.low[s * size:(s + 1) * size], self.chunk)

    def chunk_ptr(self, s: int) -> dict | None:
        if self.ptr is None:
            return None
        return parse_ptr(self.ptr[int(self.poff[s]):int(self.poff[s + 1])])


def fit_batches(x: np.ndarray, y: np.ndarray, batch_lengths, chunk: int, staging):
    """Val-shaped batches (chunk positions each, documents from their BOS and from every training batch's start) of a
    rank's FIT positions (read_fit's tokens): the same compiled eval forward applies."""
    from track_1_short.data import BOS_ID, Batch, cu_seqlens_rows
    from track_1_short.ngram_table import ngram_row_ids
    n = x.size
    assert n % chunk == 0, f"{n} FIT positions do not tile chunks of {chunk}"
    batch_start = np.zeros(n, dtype=bool)
    batch_start[np.concatenate([[0], np.cumsum(batch_lengths)[:-1]])] = True
    out = []
    for lo in range(0, n, chunk):
        inputs = torch.from_numpy(x[lo:lo + chunk].astype(np.int32))
        targets = torch.from_numpy(y[lo:lo + chunk].astype(np.int64))
        starts = torch.from_numpy(np.flatnonzero((x[lo:lo + chunk] == BOS_ID) | batch_start[lo:lo + chunk]))
        starts = starts[starts > 0]
        cum = torch.full((cu_seqlens_rows(chunk),), chunk)
        cum[0] = 0
        cum[1:len(starts) + 1] = starts
        cum = cum.to(torch.int32)
        ngram_ids = ngram_row_ids(inputs)
        dev = staging.upload(inputs, targets, cum, ngram_ids)
        out.append(Batch(*dev, ngram_ids_cpu=ngram_ids.numpy(), targets_cpu=targets, inputs_cpu=inputs.numpy()))
    return out


@torch.no_grad()
def fit_lm_outputs(fit: dict, rank: int, gate_orders, pointer: bool, chunk: int, run_chunk, device) -> dict:
    """The model's eval outputs at rank `rank`'s FIT positions (the arrays save_fit_lm writes): the FIT positions in
    chunks of `chunk`; run_chunk(lo, tokens [chunk, K]) runs the eval forward of positions [lo, lo + chunk) with the
    query tokens in the model's buffer and returns (nll [chunk], lm_features [chunk, 2 + 3K])."""
    x, y, rows, starts = fit_inputs(fit, rank, device)
    low, ptr = fit_parts(fit, rank, device)
    tok = query_tokens(rows, gate_orders, low, ptr, pointer)
    nll, lmo = [], []
    for lo in range(0, fit["n"], chunk):
        a, b = run_chunk(lo, tok[lo:lo + chunk])
        nll.append(a.float())
        lmo.append(b.float())
    lm = unpack_lm(torch.cat(lmo), tok.shape[1])
    return dict(nll=torch.cat(nll), ent=lm["ent"], mx=lm["mx"], top_v=lm["top_v"], top_in=lm["top_in"],
                top_rank=lm["top_rank"])


def fit_dump(memory: StreamMemory, forward, model, chunk: int, staging, print0):
    """STREAM_RETRIEVAL_FIT (dev runs only, after the final validation, untimed): rank 0 sends FIT, the helper writes
    the rows of the run's own last fit_k batches (against the memory before them); every rank runs the eval forward on
    its FIT positions (val-shaped chunks: fit_batches) and writes the model's outputs (save_fit_lm). Then
    tools/stream_retrieval/fit_gate_v2.py <dir> fits the gate's constants."""
    if memory.master:
        memory.fit_go()
    fit = memory.fit_wait()
    x, y = fit["tokens"][memory.rank, :, 0], fit["tokens"][memory.rank, :, 1]
    batches = fit_batches(x, y, fit["batch_lengths"], chunk, staging)

    def run_chunk(lo, tok):
        model.stream_lm_tok.copy_(tok)
        nll = forward(batches[lo // chunk])
        return nll, model.stream_lm_out.clone()
    out = fit_lm_outputs(fit, memory.rank, chain_orders(memory.low), memory.pointer, chunk, run_chunk, memory.device)
    save_fit_lm(memory.fit_path, memory.rank, **out)
    _sync()
    if memory.master:
        print0(f"stream retrieval fit: {fit['n']} positions per rank (the run's last {fit['k']} batches); rows in "
               f"{memory.fit_path}*, the model's outputs in {memory.fit_path}.lm.rank*.npz; fit with "
               f"python tools/stream_retrieval/fit_gate_v2.py {os.path.dirname(memory.fit_path)}", console=True)


# ==================================================================================================== constants

# fit_gate_v2.py --module rewrites the blocks between the markers. Provenance: see "fit" inside each.
# GATE_V2_BEGIN
GATE_V2_JSON = (
    '[{"version":2,"mode":"lm+su2+tp+hi","orders":[6,7,8,10,12,16,24,32],"w":[[-4.097389,1.652412,0.2902048,1.471781,'
    '-0.6381171,-0.1485156,-0.8241709,0.1492781,0.2146953,0.02051869,0.425216,0.3014594,0.09227908,-0.156866,0.103278'
    '6,0.2522945,0.2065089,-0.267553,-0.1372861,-0.0879553,0.425216,0.3014594,0.09227908,0.03672161,0.1917954,0.18646'
    '97,-0.3196842,0.3266522,-0.210671,-0.4676525,0.6414666,0.2228073,-0.2126502,0.2019348,-0.2453829,0.2591498],[-6.'
    '010948,0.7237577,-0.9234949,-0.1856773,-1.511011,-0.04796901,0.06016725,-0.7464012,-0.5558488,1.073292,0.504195,'
    '-0.4330831,-0.3146798,0.4615566,-0.6209653,-0.1510221,0.2572864,-0.2680022,0.1976863,-0.3049441,-0.7388849,-0.43'
    '30831,-0.3146798,0.4615566,-0.3126121,0.1616404,1.188802,-0.7261254,0.6941469,0.6450063,-0.190475,-0.1307081,0.2'
    '172214,-0.5228035,0.6376864,-0.05180835,0.668708],[-5.00111,-0.3036382,-0.3232125,0.03500872,0.1765097,0.6268579'
    ',-0.2236774,0.7755908,-0.6497762,-0.391167,-0.04125533,1.152849,-0.4612335,-0.1308236,-0.1444292,-0.025913,0.296'
    '8262,-0.122375,0.6305557,-0.4093486,0.660344,1.152849,-0.4612335,-0.1308236,0.1369798,1.609458,-0.1690835,-0.981'
    '1797,1.938333,0.2801299,0.04487511,-0.01523948,0.3542438,-0.263662,0.04270064,-0.4432246,0.881695],[-3.791496,0.'
    '948397,-0.01473052,0.5089971,-1.793483,-0.1828002,-0.06384433,-1.179073,-0.00201824,0.2855448,0.05946177,1.22646'
    '2,-0.7967222,0.152838,0.05206234,-0.4217799,0.03760758,-0.216936,-0.1196367,0.09052609,0.4507101,1.226462,-0.796'
    '7222,0.152838,0.2809472,-0.5558151,-0.07298093,0.3470127,0.6203853,-0.2672084,-0.08138059,0.0199075,0.2461213,-0'
    '.4713292,-0.532526,-0.2796922,0.9042994],[-2.131705,0.5374281,-0.7355794,0.4511978,0.1312087,0.1032571,0.6741016'
    ',0.213924,-0.1034018,-0.619061,-0.186478,1.488844,-0.2742567,-0.6403801,0.5190265,-0.2263336,0.04495585,-0.80063'
    '13,-0.1051182,0.2196701,-0.3265402,1.488844,-0.2742567,-0.6403801,-0.8979016,0.4617942,0.01912652,-0.14662,0.055'
    '17465,0.7428399,-0.3703063,0.2243935,-0.02049349,-0.006366483,0.3192335,-0.06397041,-0.3817403],[-0.8379099,0.00'
    '3661529,-0.4622957,0.009838161,-0.4045107,-0.407665,0.002470607,-0.2494846,-0.07082119,-0.07206691,0.5378867,1.5'
    '76773,-0.6004505,0.2019423,-0.3274406,-0.5354259,-0.02058298,-0.1673268,-0.6004505,-0.1984883,-0.5826229,1.57677'
    '3,-0.6004505,0.2019423,-0.9681007,0.2316171,0.3469233,0.2194392,1.169281,0.2188669,-0.2502207,0.4781963,-0.17653'
    '79,-0.7976176,0.113855,0.4904391,-0.02650255],[-1.144499,-0.3956205,-0.1767077,0.3037713,0.4211153,-0.5162392,-0'
    '.2854209,0.5288624,-1.131965,0.4970597,-0.0400963,0.405761,-0.2453976,-0.6342703,0.004905957,0.07606692,0.418042'
    '2,-0.01540363,-0.5156059,-0.09674756,0.6413467,0.405761,-0.2453976,-0.6342703,-0.102669,-0.3765374,0.2316091,1.0'
    '43502,0.4872805,0.7463405,-0.439619,0.4939338,0.6072119,-0.2542787,-0.1803381,-0.8024372,-0.1638542],[-1.599501,'
    '0.1196398,-0.2883464,0.6076235,-0.3185853,0.04439757,0.03566486,-0.1920752,-1.173705,-0.006123917,0.2227258,-0.0'
    '2375363,-0.3536142,0.5824561,-0.0149926,-1.184028,-0.1693819,0.5795257,0.1731932,0.6614337,-0.03286009,-0.023753'
    '63,-0.3536142,0.5824561,0.2759049,0.3918035,0.4395295,0.7002745,0.6864911,-0.09838971,0.3875056,0.4296062,-0.254'
    '8473,-0.5573772,0.0520334,0.4369952,1.318659]],"disc":[[3.598689,-1.208645,-0.9889401],[2.386821,-0.9312428,-1.1'
    '18867],[0.5007911,-1.05091,-1.01528],[-1.678479,-0.3482955,-0.9798695],[-0.03423171,-1.013572,-1.003721],[-0.052'
    '47704,-0.9666884,-1.01432],[0.0219651,-1.004202,-1.000616],[0.005568299,-1.000685,-1.000095]],"mu":[[1.853095,1.'
    '119055,0.7227726,0.7259818,0.09539455,0.4329809,0.3727337,2.912656,-1.107016,2.316534,1.846004,4.850346,-0.56761'
    '18,2.772662,1.545154,1.666235,2.470341,2.86123,8.363672,2.316534,1.846004,4.850346,1.118565,0.3976477,0.282904,1'
    '.951981,0.4223017,0.2026659,0.6142402,0.4389131,-2.910853,1.517282,0.8000354,1.803837,1.890451],[1.487185,0.6632'
    '375,0.8198941,0.6829162,0.09925845,0.3149265,0.4403391,0.5768926,2.736516,-1.031989,2.389321,1.834006,5.017076,-'
    '0.4998196,2.641199,1.609731,1.657404,2.322287,2.753392,8.344165,2.389321,1.834006,5.017076,1.232481,0.5177348,0.'
    '6281942,1.943342,0.5472972,0.5967031,0.6706396,0.568836,-2.648338,1.355184,0.82556,1.616348,1.715098],[1.326236,'
    '0.3687756,0.8953884,0.6326101,0.1050694,0.2755866,0.4865384,0.2830404,2.605685,-0.9746266,2.488484,1.912885,5.32'
    '0991,-0.4286294,2.56574,1.732333,1.786667,2.247908,2.674519,8.347356,2.488484,1.912885,5.320991,1.462083,0.64400'
    '25,0.9609096,2.190406,0.6743671,0.975493,0.7651599,0.7047338,-2.411476,1.200252,0.8479568,1.436849,1.553034],[1.'
    '274826,0.1402771,0.9616196,0.5688599,0.1127184,0.2688812,0.513814,0.1790659,2.487354,-0.9202128,2.710327,2.07678'
    '4,5.935518,-0.3426357,2.519811,1.949528,1.991951,2.216328,2.592533,8.369929,2.710327,2.076784,5.935518,1.805291,'
    '0.8020096,1.341813,2.6501,0.8367055,1.395199,0.8901685,0.8683837,-2.143022,1.025521,0.8726953,1.222809,1.365945]'
    ',[1.261074,0.08153846,0.9778121,0.5519118,0.1123058,0.2657169,0.5199833,0.09023417,2.472545,-0.9137505,2.896528,'
    '2.14367,6.335237,-0.3145918,2.503539,2.036716,2.071456,2.203975,2.554426,8.404047,2.896528,2.14367,6.335237,1.93'
    '06,0.8507764,1.463016,2.816417,0.8851992,1.518344,0.9296924,0.9173467,-2.095614,0.996811,0.876055,1.181864,1.334'
    '836],[1.225935,0.04926209,0.9871587,0.547118,0.1085529,0.2540897,0.5282603,0.09794144,2.458683,-0.9093228,3.1781'
    '57,2.175427,6.841743,-0.2961208,2.471548,2.086811,2.116587,2.175427,2.501789,8.478206,3.178157,2.175427,6.841743'
    ',2.029427,0.8812762,1.557823,2.938346,0.9148114,1.614355,0.9546777,0.9470064,-2.065839,0.9753456,0.8777886,1.156'
    '516,1.324079],[1.140961,0.02561758,0.993245,0.5544417,0.1011725,0.2266322,0.5441555,0.09839703,2.486987,-0.92170'
    '46,3.601227,2.219325,7.561978,-0.2784283,2.472411,2.170636,2.181455,2.193378,2.443822,8.606383,3.601227,2.219325'
    ',7.561978,2.111902,0.9028602,1.647586,3.062205,0.9363087,1.707251,0.9684938,0.9643945,-2.078785,0.9784151,0.8774'
    '889,1.15708,1.333674],[1.080022,0.02045777,0.9946937,0.5681899,0.09242571,0.208544,0.5604931,0.05531594,2.518795'
    ',-0.9323902,3.920704,2.268703,8.124846,-0.2653551,2.511741,2.239489,2.257011,2.251425,2.415295,8.715098,3.920704'
    ',2.268703,8.124846,2.162362,0.9123309,1.723877,3.148193,0.9453001,1.775575,0.9744839,0.9717603,-2.124342,1.00829'
    '3,0.8719897,1.191952,1.366475]],"sd":[[1.869153,1.515747,0.3392253,0.3853267,0.2386505,0.5739786,0.4835321,1.870'
    '177,0.8709112,0.6668528,1.06821,2.548192,0.7009974,0.8435125,1.95636,1.191319,0.9606063,0.7708105,1.772397,0.666'
    '8528,1.06821,2.548192,0.7556,0.3490122,0.9358284,0.9329375,0.3043557,0.9206047,0.4867742,0.4962544,2.588389,2.01'
    '8144,0.3999735,2.277437,2.040715],[1.6814,1.175001,0.2944011,0.4287016,0.2657862,0.4756473,0.4964278,1.096286,1.'
    '900447,0.8763529,0.7872265,1.209391,2.835688,0.682876,0.9375937,2.12284,1.370149,1.062767,0.8469476,1.782595,0.7'
    '872265,1.209391,2.835688,0.8599082,0.3824978,1.122184,1.177884,0.3540122,1.111468,0.469981,0.4952389,2.55698,1.9'
    '5333,0.3794874,2.192284,1.999652],[1.628031,0.884823,0.2356337,0.4611673,0.2876981,0.4539674,0.4998188,0.7492151'
    ',1.935875,0.887121,0.903034,1.345903,3.148693,0.6798502,1.035854,2.308523,1.542944,1.176445,0.9209577,1.785309,0'
    '.903034,1.345903,3.148693,0.874761,0.3667036,1.21897,1.271171,0.3497389,1.18066,0.4238989,0.4561623,2.487429,1.8'
    '1416,0.3590628,2.07063,1.949346],[1.636129,0.5539535,0.1462154,0.4854802,0.3083635,0.4574399,0.4998091,0.6127354'
    ',1.965601,0.8970625,1.039058,1.445359,3.501519,0.6543962,1.155459,2.52704,1.691157,1.315092,1.010516,1.766876,1.'
    '039058,1.445359,3.501519,0.7377848,0.2679214,1.230454,1.125326,0.2518331,1.14455,0.3126796,0.3380731,2.379636,1.'
    '65601,0.333314,1.9033,1.872934],[1.631307,0.4166696,0.1109338,0.4910545,0.3107348,0.4538637,0.4996005,0.4331948,'
    '1.973993,0.9006995,1.094645,1.435454,3.625151,0.6162037,1.195259,2.587078,1.715614,1.361659,1.037769,1.72787,1.0'
    '94645,1.435454,3.625151,0.6523713,0.2166874,1.213547,1.024041,0.1988753,1.118207,0.2556647,0.2753574,2.361464,1.'
    '631016,0.3295188,1.869082,1.860873],[1.600787,0.3278273,0.0843014,0.4937772,0.3079399,0.4388628,0.4992007,0.4505'
    '516,1.976056,0.8996796,1.141642,1.389699,3.728496,0.5536907,1.219859,2.618171,1.722465,1.389699,1.054142,1.64412'
    '2,1.141642,1.389699,3.728496,0.5855021,0.1835242,1.212075,0.9736193,0.169085,1.11546,0.20801,0.2240206,2.342171,'
    '1.594606,0.3275299,1.829806,1.856154],[1.524573,0.2356493,0.06044602,0.4947696,0.300137,0.4059181,0.4980465,0.46'
    '46856,1.989224,0.9089414,1.166769,1.321654,3.809237,0.4864143,1.211411,2.673007,1.739282,1.384946,1.077351,1.515'
    '803,1.166769,1.321654,3.809237,0.524758,0.1561906,1.204915,0.9268382,0.1496916,1.116639,0.1746813,0.1853045,2.33'
    '7264,1.578076,0.3278752,1.814048,1.858595],[1.473179,0.2201601,0.05493597,0.4936661,0.2885323,0.380889,0.4963271'
    ',0.3404354,1.992402,0.9095507,1.16655,1.280817,3.86139,0.4477918,1.19449,2.721179,1.76356,1.381808,1.105252,1.42'
    '3329,1.16655,1.280817,3.86139,0.4788791,0.1418871,1.212887,0.8748084,0.1360729,1.124578,0.1576864,0.1656569,2.36'
    '0106,1.600691,0.3341013,1.839946,1.878322]],"top":{"comps":["ptr","vote","src"],"cols":["ptr_run","ptr_hits","pt'
    'r_miss16","ptr_acc64","ptr_since","ptr_age","ptr_share","ptr_tcnt","ptr_ntok","ptr_seedlen","ptr_nrec","ptr_marg'
    'in","ptr_score","ptr_how0","ptr_how1","ptr_how2","ptr_how3","ptr_how4","ptr_how5","hist16_ret_c","hist16_ret_acc'
    '","hist16_ptr_c","hist16_ptr_acc","hist16_either","hist64_ret_c","hist64_ret_acc","hist64_ptr_c","hist64_ptr_acc'
    '","hist64_either","histseg_ret_c","histseg_ret_acc","histseg_ptr_c","histseg_ptr_acc","histseg_either","hist2_re'
    't_r32","hist2_ret_rseg","hist2_ptr_r32","hist2_ptr_rseg","mem_acc_h8","mem_n_h8","mem_acc_p8","mem_hit_p8","mem_'
    'acc_h32","mem_n_h32","mem_acc_p32","mem_hit_p32","mem_acc_h128","mem_n_h128","mem_acc_p128","mem_hit_p128","mem_'
    'r_h32","mem_streak","mem_since_wrong","mem_since_hit","mem_prev_corr","mem_prev_hit","mem_ema90","mem_ema98","me'
    'm_nlong32","mem_nlong16","mem_acc_doc","mem_lenl","mem_lenr","mem_cont","mem_cont_run","mem_cont_corr_run","mem_'
    'npos","mem_ncand","src_n","src_purity","src_lbest","src_lbest_l2","src_nsrc","src_since","src_newlen","src_n2","'
    'src_acc_h8","src_acc_h32","src_acc_doc","src_nhit","src_streak","src_prev_corr","src_r_h32","ret_logn","ret_puri'
    'ty","ret_logl","ind_has","ind_hp","ptr_agree","both_has","both_lstar","both_hs","lm_ent","lm_mx","tp_ptr_v","tp_'
    'ptr_in","tp_ptr_gap","tp_vote_v","tp_vote_in","tp_vote_gap","tp_mem_v","tp_mem_in","tp_mem_gap","su_ptr_log","su'
    '_ptr_mean","su_ptr_max","su_ptr_ment","su_src_log","su_src_mean","su_src_max","su_src_ment","su_mem_log","su_mem'
    '_mean","su_mem_max","su_mem_ment","chain_logo","chain_logn","chain_purity","chain_logd","llr_ret_32","llr_ret_se'
    'g","llr_ptr_32","llr_ptr_seg"],"W":[[-6.70895,-7.862703,-8.911077],[-0.4680385,0.2232993,-0.3436568],[-0.4701843'
    ',0.05273903,0.002461355],[0.01015218,0.1472508,0.35989],[0.4689096,0.01683389,-0.1163723],[-0.1824768,0.2699651,'
    '-0.2254688],[0.9938403,0.5258764,-0.02026543],[0.9942158,0.2630362,0.2436381],[0.5068576,0.742321,0.08112065],[-'
    '0.07577732,-0.2142082,0.2406642],[-0.1659116,0.0860366,-0.03240603],[-0.3836957,-0.4617983,-0.5065199],[-0.21022'
    '67,-0.2664129,0.09880203],[0.1998654,0.007815813,0.2566999],[-0.01840029,0.1525569,-0.006172644],[0.07641465,0.1'
    '074039,-0.02864345],[0.01840978,-0.216307,0.05816709],[-0.1419892,-0.01279452,-0.1395292],[-0.0003223125,0.01600'
    '478,-0.07865367],[-0.2241428,0.08069304,0.02199624],[-0.1097623,0.01363377,0.01880979],[0.1443434,-0.1142531,0.0'
    '08814958],[0.4461511,0.4580825,0.2090092],[0.04907393,0.2425032,-0.1875622],[-0.3943169,0.06460397,-0.03951106],'
    '[-0.2825337,0.0806643,-0.3769801],[0.003264532,-0.09340569,0.08028972],[0.5637819,0.3402869,0.3247517],[0.153267'
    '1,0.2145067,0.07530714],[-0.06136083,-0.182953,-0.06078908],[-0.04494671,0.03472719,-0.4299776],[-0.1677162,-0.0'
    '239799,-0.07049968],[0.5620785,0.3000442,0.3346976],[0.02220665,-0.0653041,-0.1363534],[-0.009193697,0.1050747,-'
    '0.1075884],[-0.007893683,0.3052297,-0.09726329],[0.3960525,0.003915232,-0.1869061],[-0.3854836,-0.2062431,-0.027'
    '18919],[0.02220665,-0.0653041,-0.1363534],[0.001568367,-0.1449764,0.06384533],[-0.1007822,-0.1337282,0.152029],['
    '0.1237218,-0.04380597,-0.1085144],[-0.115598,-0.1253401,-0.03569097],[-0.07914036,-0.008681316,0.2110348],[-0.15'
    '87617,0.1993796,0.07652854],[0.08460286,-0.1037804,-0.3389726],[-0.06782861,0.1640399,0.2598068],[0.1131079,-0.1'
    '293546,0.186046],[0.06875316,-0.2502449,-0.1252389],[-0.01681381,-0.07671773,0.07225369],[-0.05923698,0.05901424'
    ',-0.05730303],[0.3057187,0.1720146,-0.1400575],[-0.00631647,-0.2204045,-0.3210788],[-0.101771,-0.05745421,0.2279'
    '078],[0.05753682,0.0985106,-0.2504221],[0.009271181,0.05181849,-0.1944637],[0.001998818,-0.05952984,0.1880509],['
    '-0.0603444,-0.06404246,0.07824555],[-0.3084118,-0.01506809,0.1832513],[-0.01754435,-0.1407824,-0.02099281],[-0.1'
    '207706,0.01127869,-0.20042],[-0.3902648,0.1476495,0.1425354],[-0.05339654,-0.05082856,-0.394507],[-0.07368278,-0'
    '.08273039,-0.4313027],[0.07569011,0.06728072,-0.1634649],[-0.04841294,0.04956565,-0.2204253],[-0.2012641,0.19616'
    '05,0.2586211],[-0.3140942,-0.1460335,-0.4958407],[-0.120515,-0.346758,-0.01231972],[0.1895783,-0.08109404,1.0228'
    '9],[0.2473304,-0.1150065,1.366041],[-0.1394422,0.157182,-0.3440436],[-0.05523252,0.05514203,0.7530921],[-0.06243'
    '764,0.005969675,0.006943946],[-0.09299103,-0.2027595,-0.1453262],[-0.05034664,-0.3909241,-0.1872107],[-0.163482,'
    '-0.120921,-0.4096139],[0.1266563,0.01444361,0.272121],[0.06802363,-0.04745944,0.06558443],[0.08869694,0.04520756'
    ',0.4512132],[0.2232985,0.10746,0.9187998],[-0.07554307,0.115204,0.1953309],[-0.09974122,0.04638976,0.2953033],[-'
    '0.08996047,-0.3291243,0.3205431],[-0.1227548,0.1270116,0.086917],[0.03182057,0.03052714,0.2196485],[-0.03355733,'
    '-0.1612317,-0.3689687],[0.2283342,0.2297391,0.4613106],[0.0001727951,-0.0001414248,-0.004131567],[-0.1700692,-0.'
    '1427092,-0.1295122],[0.2283342,0.2297391,0.4613106],[-0.03355733,-0.1612317,-0.3689687],[-0.07338148,0.4008529,-'
    '0.01890705],[0.4541412,0.6667337,-0.02934826],[0.4653558,0.1889413,0.05651395],[0.6962635,0.05285502,0.1764696],'
    '[0.5805988,-0.03096208,-0.01601015],[-0.632055,-0.008053672,-0.09815348],[0.1563936,0.09088392,0.07400208],[0.05'
    '800564,-0.2188274,0.008270112],[-0.04711849,-0.04867381,0.01348327],[-0.2124585,0.0466404,-0.1445572],[-0.151977'
    '6,0.1989389,0.1765445],[-0.3049392,0.02623462,-0.1339369],[-0.2316516,0.1017746,0.3173649],[0.4404754,-0.0115913'
    '7,0.1345802],[0.2765298,0.2648989,-0.4031986],[-0.1244223,-0.1310609,0.03473732],[-0.09353883,-0.1139428,1.17242'
    '8],[0.01061484,-0.04535135,0.3810113],[-0.06826339,-0.07297139,0.2041356],[-0.1337434,0.1970553,-0.1482205],[0.0'
    '495744,-0.03100333,0.2902115],[-0.2380718,-0.05129059,-0.4052974],[0.04466017,-0.1440805,0.08220338],[0.03215979'
    ',0.09316971,-0.03347086],[-0.03355733,-0.1612317,-0.3689687],[-0.1227548,0.1270116,0.086917],[0.03182057,0.03052'
    '714,0.2196485],[0.06283621,-0.1176742,-0.0609384],[0.1028235,-0.1448317,0.09543018],[0.2258012,0.3550458,0.69939'
    '61],[0.38241,0.7570825,0.03250868],[-0.0385255,-0.0177418,-0.02048382]],"mu":[0.946565,2.252345,1.387342,0.56937'
    '24,0.8808414,1.406066,0.2761396,0.8826586,2.023591,1.974277,0.9249767,0.4106763,0.340256,0.1154892,0.187034,0.34'
    '69062,0.001321445,0.03303896,0.01515559,0.6937338,0.2161065,0.830794,0.1677096,0.1156381,1.371334,0.2773227,1.59'
    '5177,0.1699133,0.09182444,3.524109,0.3374238,3.875528,0.175867,0.07257551,0.2254923,0.3110421,0.1687515,0.175867'
    ',0.3190534,2.104793,0.09255117,0.1970967,0.3424942,3.041684,0.07443454,0.144278,0.3487991,3.463757,0.06167636,0.'
    '120197,0.317393,0.1911678,2.065082,1.741853,0.105161,0.23919,0.08724728,0.06267443,0.2144928,0.4250836,0.3474067'
    ',0.7604555,0.7591582,0.09253228,0.1676293,0.1645313,6.001597,0.4447992,0.7289668,0.390178,2.574169,0.9000111,4.9'
    '37626,4.146497,2.837219,0.9930464,0.2788928,0.2802502,0.2795573,3.962516,0.2623642,0.1653872,0.2549437,0.330489,'
    '0.1868287,0.7261494,0.2400304,0.6989454,0.1853192,0.2400304,0.7261494,0.5394722,3.155593,-1.204437,-3.825046,0.2'
    '941474,3.026026,-3.67405,0.3206527,2.87503,-0.7200656,0.1886384,0.4543387,0.7783221,0.6558379,1.624161,-0.219239'
    '2,0.9675617,1.018098,2.006969,-0.3952058,0.6586018,0.4622775,1.374684,-0.1349111,0.7261494,0.330489,0.1868287,0.'
    '1978862,-0.3058332,-1.289003,-1.121753,-3.040511],"sd":[1.508098,1.65903,1.806775,0.3961814,1.514147,1.583502,0.'
    '3333589,0.7879523,1.597942,1.297753,1.099968,1.240058,0.9951256,0.3196114,0.3899388,0.4759856,0.03632766,0.17873'
    '83,0.1221716,1.059818,0.3156154,1.15842,0.2738906,0.2246423,1.432073,0.280017,1.528778,0.2300792,0.1856514,1.848'
    '485,0.2140384,1.920921,0.1851664,0.1343039,0.2884036,0.2155811,0.2523852,0.1851664,0.2136245,0.3069468,0.1955558'
    ',0.2279399,0.2037516,0.7166681,0.1705172,0.1823373,0.2004177,1.078014,0.1389674,0.145662,0.2068037,0.7079799,1.4'
    '99443,1.476558,0.3067608,0.426589,0.1889123,0.1394605,0.8991131,1.180745,0.1963658,1.460007,1.456291,0.2897759,0'
    '.6654864,0.6619767,1.143108,1.210383,1.459528,0.4417801,5.336538,1.145198,2.912622,2.706835,0.9873643,1.708515,0'
    '.2253715,0.2115415,0.1936232,1.770653,0.7737358,0.3715296,0.2118025,1.008603,0.3672361,1.338872,0.4271017,0.4587'
    '164,0.3885563,0.4271017,1.338872,0.4984395,1.862598,0.8821905,3.672766,0.4556586,3.396519,3.639401,0.4667275,3.3'
    '69231,1.816523,0.3912211,1.393727,1.250888,1.14174,2.867108,0.6344036,1.115548,1.313648,2.717582,0.8328125,1.244'
    '711,0.9474916,2.80609,0.3914908,1.338872,1.008603,0.3672361,0.7382721,1.189213,2.598557,1.569042,2.832158]},"fit'
    '":{"what":"CPU placeholder (the CPU proof\'s fit): replaced by fits on our model, one per step count, from FIT de'
    'v runs (tools/stream_retrieval/README.md)","proxy":"llm.c GPT-2 124M on CPU, not our model: P_0 a CPLM-like copy'
    ' mixture with its constants fitted on training positions; the side output its entropy, max log p and top-32 log-'
    'probs","total_steps":1050,"freeze":287882714,"source":"the CPU proof: the hooked timed loader replaying the reco'
    'rd schedule (978 scheduled + 72 = 1050 steps, 8 ranks, the real shards) into the helper, its FIT dump (ranks 0-3'
    ')","data":"the run\'s own last 16 batches (steps 1034-1049) of that 1050-step run, queried by the helper\'s FIT pa'
    'th against the memory as it stood before step 1034 (entry 287,882,714)","ranks":[0,1,2,3],"positions":1048576,"m'
    'atched":0.0809,"batches":16,"mode":"lm+su2+tp+hi","parts":"P1 + P3","top_comps":["ptr","vote","src"],"in_sample_'
    'mnat":46.395,"note":"CPU proof, P1 + P3","val_mnat_cpu_proxy":"24.65 [19.41, 30.64] millinats over the CPLM-like'
    ' base on the first 1,048,576 val positions (research A+C, 2-fold on val: 24.10)"}}]'
)
# GATE_V2_END
# GATE_V2_LOW_BEGIN
GATE_V2_LOW_JSON = (
    '[{"version":2,"mode":"lm+su2+tp+hi","orders":[1,2,3,4,5,6,7,8,10,12,16,24,32],"w":[[-9.351432,0.2001589,0.186502'
    '7,-0.1421163,-1.174961,0.08797105,-1.638542,0.0,0.8633232,0.3380668,1.037057,0.1246602,0.1246602,-0.3470048,0.06'
    '330583,0.1246602,0.06588963,-0.2586709,0.1292195,-0.0227447,1.037057,0.1246602,0.1246602,0.13169,-0.01914062,0.0'
    '3607512,-0.2452088,-0.1368232,0.03851539,-0.5250788,0.5126279,0.2676932,0.4543867,0.2823699,-0.1586116,0.0322693'
    '],[-6.53322,1.387375,-0.4863399,0.3090939,-0.5139529,-0.08829174,-2.833812,0.00648353,-0.3957862,0.766945,0.0554'
    '8368,0.1572391,0.5863946,-0.1355604,-0.2376038,0.106891,-0.3362073,-0.038404,-0.1218921,-0.186652,0.008157189,0.'
    '1572391,0.5863946,-0.1355604,-0.03650147,0.03451478,0.0364602,-0.1170253,-0.01852287,0.06003582,-0.08752638,0.06'
    '443008,-0.01937113,-0.7228732,0.03591509,0.03775879,0.08260428],[-5.740157,2.808531,-0.1753101,0.7971269,-0.4786'
    '085,-0.09794045,-4.394201,0.1267028,-0.4740679,0.4485047,-0.0444261,-0.04427516,0.4214712,-0.02795614,-0.2122263'
    ',-0.02420458,-0.01773188,-0.03028577,-0.1314145,-0.04008835,-0.05869034,-0.04427516,0.4214712,-0.02795614,-0.129'
    '4406,-0.1001115,0.1261022,-0.002689096,0.1368904,-0.01149668,-0.09580717,0.3717724,-0.0585132,-0.769125,0.152334'
    '8,0.04597872,0.07094354],[-5.208403,2.792144,-0.300374,0.9106237,-0.5061515,-0.03344081,-3.233474,0.3343377,-0.3'
    '822207,0.373345,0.07494146,0.01929497,0.1886427,-0.003334115,-0.03244077,0.007957456,0.01274774,0.1886427,-0.117'
    '5822,-0.171862,-0.08809037,0.01929497,0.1886427,-0.003334115,-0.1496715,-0.09979496,0.213487,-0.130613,0.2720047'
    ',-0.01567931,-0.1160417,0.4666723,0.03308969,-0.8526396,0.1669795,-0.008904438,0.1494223],[-4.874403,0.616658,0.'
    '1940989,0.8245652,-0.7117377,-0.1706019,-0.6853059,0.2323542,-0.2695922,0.2332892,0.07493743,0.05709424,0.553083'
    '5,0.002732405,-0.1481043,0.1201433,0.09521789,-0.5597112,-0.2625638,-0.3961855,0.1554355,0.05709424,0.5530835,0.'
    '002732405,0.1123968,-0.1306184,0.08823146,-0.4713869,0.3460071,0.1637605,-0.4166996,0.665294,0.1712105,-0.899072'
    ',0.1878199,-0.161775,0.229618],[-7.533395,1.399392,0.3134052,1.642489,-0.6855072,-0.03900557,-0.08314711,-0.1987'
    '247,-1.349674,0.1742032,-0.2926006,0.8082634,0.1851433,0.2063788,-0.01265749,0.1664357,0.3603739,-0.4669493,-0.4'
    '232481,0.05210148,0.1043227,0.8082634,0.1851433,0.2063788,-0.2377812,0.5310346,0.4580668,-0.573248,0.3090419,-0.'
    '4716408,-0.2776793,0.5996691,0.2638072,0.5229821,0.3749486,-0.411718,1.194256],[-6.115781,-0.3722797,-1.736945,-'
    '0.3155634,-0.007454451,-0.2399263,0.02433543,-1.10343,-0.6736649,0.114697,0.7568917,1.267618,-0.6613782,0.104661'
    '9,-0.1308927,0.2211597,0.3952054,-0.1622423,-0.6998909,-0.440122,-1.108752,1.267618,-0.6613782,0.1046619,0.31011'
    '13,0.9737809,0.6757154,0.3622749,0.712173,0.4978164,-0.6771895,0.002651914,0.07403689,-0.5247826,0.3929142,0.216'
    '21,0.4590779],[-5.195397,-0.2833658,-0.240173,0.1235399,-0.004714896,0.7766414,-0.2404879,0.877938,-0.6515606,-0'
    '.4139324,-0.09942949,1.054093,-0.3660895,-0.1657741,-0.1721661,-0.013417,0.4057439,-0.07957833,0.864961,-0.45539'
    '56,0.7730077,1.054093,-0.3660895,-0.1657741,0.6511206,1.679221,-0.09048295,-1.6165,1.642179,0.08126084,-0.390396'
    ',0.07114576,0.3743553,-0.1862677,0.004603539,-0.4923081,0.8752887],[-3.967562,0.9092707,-0.06382937,0.415262,-2.'
    '081377,-0.2353742,0.1915985,-1.008839,-0.08006365,0.3320312,0.04018555,1.332294,-0.8289017,0.1052139,0.04897285,'
    '-0.4241862,0.1242725,-0.6114778,-0.06230221,0.1584084,-0.02640334,1.332294,-0.8289017,0.1052139,0.4148632,-0.674'
    '1695,0.05595496,-0.1289098,0.7213113,-0.1189813,-0.07787707,0.04912742,0.08080776,-0.5374665,-0.421556,-0.082091'
    '16,0.7648183],[-2.068017,0.3138853,-0.6141042,0.4073822,0.01583208,0.1408624,0.4030519,0.1819343,-0.1561644,-0.4'
    '643245,-0.3304083,1.585454,-0.6862596,-0.3953553,0.5965512,-0.04965777,-0.293624,-0.77686,-0.1449337,2.471948e-0'
    '5,0.09376969,1.585454,-0.6862596,-0.3953553,-0.3030656,0.3009063,-0.05352638,-0.3790364,0.1266971,0.9856609,-0.4'
    '660219,0.1970559,-0.2150262,-0.5351999,-0.1503013,0.1124498,-0.4591911],[-0.7326915,0.3626727,-0.2490778,-0.1471'
    '623,-0.728659,-0.08100639,1.001195,0.9239275,-0.395412,0.7542326,0.5561068,1.153242,-0.4614004,-0.2026327,-0.256'
    '8426,-0.4090601,-0.191842,-0.184842,-0.4614004,-0.02442908,0.1699711,1.153242,-0.4614004,-0.2026327,-0.7055766,0'
    '.2593809,0.0554595,0.7510249,0.2572496,0.1868376,-0.1751262,0.2733859,0.3090611,-0.5509633,1.038475,-0.1221746,0'
    '.5725864],[-2.077163,0.09987487,-0.2619559,0.5987402,-0.476516,-0.220231,-0.05298598,-0.3014625,-0.9002976,-0.14'
    '14245,-0.2432922,0.2557225,-0.7269933,0.322512,0.5408477,-0.8572134,0.06442442,0.4541903,0.02763314,0.9403733,0.'
    '3409721,0.2557225,-0.7269933,0.322512,0.5957889,0.3326138,0.3604666,1.660282,0.5348896,-0.09666073,0.3002446,0.4'
    '997575,-0.2551366,-0.3600498,-0.1494383,0.2068217,1.229859],[-1.115871,-0.2831222,-0.3218484,0.3700991,0.5055457'
    ',-0.6430158,-0.002749905,0.6405652,0.1304678,0.2754925,0.1315914,-0.2299566,0.3932239,-0.6236538,-0.9739686,-0.0'
    '5100391,-0.08522299,0.03148006,-0.1996542,-0.2261382,-0.07908073,-0.2299566,0.3932239,-0.6236538,0.3004857,0.312'
    '2779,0.3931386,0.6395168,0.2947099,0.5109469,-0.05931467,0.4971501,0.4464656,-0.6568774,-0.09921545,-0.5076328,-'
    '0.6985376]],"disc":[[-4.938702,-4.391081,-3.655451],[0.8645879,-3.362601,-2.779843],[2.325635,-0.1707764,-1.2435'
    '59],[2.820444,-1.12703,-0.9469789],[4.866579,-1.096619,-1.024856],[0.377502,-1.340576,-0.8824228],[0.7448205,-0.'
    '8591545,-1.096879],[0.3273765,-1.023487,-1.017156],[-1.231934,-0.5643628,-0.9903242],[0.0212503,-1.005223,-1.003'
    '398],[0.1148292,-1.104625,-1.011019],[0.02188326,-0.9977035,-1.002248],[-0.004366241,-1.002226,-0.9991077]],"mu"'
    ':[[17.01323,12.06251,0.1538412,0.4044477,0.1217726,19.25775,0.0,3.271412,-1.235848,1.183644,3.211014,3.211014,-0'
    '.05625308,3.297571,3.211014,3.220999,3.242086,3.285011,8.34996,1.183644,3.211014,3.211014,2.375124,0.144825,-0.3'
    '659405,3.804402,0.1496405,-0.3755742,0.9983549,0.1527529,-4.162614,2.556417,0.728446,2.926766,2.563077],[9.59071'
    '8,7.331239,0.2620599,0.6385693,0.1242792,7.197486,0.02303256,7.681748,3.271521,-1.236858,1.769261,3.065536,4.575'
    '323,-0.1634593,3.284975,3.008688,3.137442,3.212151,3.269187,8.364542,1.769261,3.065536,4.575323,2.323654,0.20424'
    '14,-0.2976749,3.749503,0.2114292,-0.3072684,0.9493118,0.2150882,-3.452563,1.90938,0.7959751,2.215705,2.100243],['
    '5.585563,4.28537,0.3950512,0.7288075,0.1062709,2.992324,0.1031498,5.214122,3.258006,-1.236093,2.053869,2.679844,'
    '4.956185,-0.3918236,3.208961,2.473688,2.785871,3.07558,3.19138,8.3792,2.053869,2.679844,4.956185,2.057011,0.2337'
    '155,-0.2662905,3.452032,0.2413362,-0.2862408,0.7960033,0.2456203,-3.305836,1.771613,0.7879916,2.069743,2.070072]'
    ',[3.67575,2.717503,0.5211481,0.7568896,0.09573689,1.57517,0.2062012,3.316622,3.199187,-1.216357,2.189292,2.29255'
    '7,4.929845,-0.5503585,3.088502,1.958017,2.292557,2.882674,3.091694,8.382176,2.189292,2.292557,4.929845,1.615856,'
    '0.2640227,-0.184342,2.908135,0.2722644,-0.2306093,0.6520559,0.2778993,-3.270269,1.750872,0.7749366,2.053911,2.09'
    '3568],[2.735639,1.883349,0.6277931,0.7523868,0.09279963,1.025235,0.2998648,2.065216,3.085424,-1.17571,2.261931,2'
    '.006104,4.845704,-0.5995113,2.936826,1.644333,1.913927,2.671516,2.979998,8.374633,2.261931,2.006104,4.845704,1.2'
    '34146,0.3143704,0.003640776,2.314111,0.3294922,-0.07893381,0.5868999,0.3379718,-3.143444,1.669045,0.7791245,1.96'
    '7734,2.036834],[1.8531,1.119068,0.7227695,0.7259786,0.09541139,0.4329803,0.3727263,1.655719,2.912689,-1.107029,2'
    '.316533,1.846008,4.85037,-0.5676029,2.772665,1.545139,1.666241,2.470348,2.861241,8.363649,2.316533,1.846008,4.85'
    '037,1.118561,0.397646,0.2829093,1.95198,0.4223028,0.2026721,0.6142239,0.4389065,-2.910887,1.5173,0.800033,1.8038'
    '58,1.890473],[1.487162,0.6632375,0.8198941,0.6829387,0.09923596,0.3149251,0.4403616,0.5768829,2.736516,-1.031989'
    ',2.389321,1.834006,5.017076,-0.4998196,2.641199,1.609731,1.657404,2.322287,2.753392,8.344165,2.389321,1.834006,5'
    '.017076,1.232481,0.5177348,0.6281942,1.943342,0.5472972,0.5967031,0.6706396,0.568836,-2.648338,1.355184,0.82556,'
    '1.616348,1.715098],[1.3262,0.3687756,0.8953884,0.6326457,0.1050338,0.2755844,0.486574,0.2830404,2.605685,-0.9746'
    '266,2.488484,1.912885,5.320991,-0.4286294,2.56574,1.732333,1.786667,2.247908,2.674519,8.347356,2.488484,1.912885'
    ',5.320991,1.462083,0.6440025,0.9609096,2.190406,0.6743671,0.975493,0.7651599,0.7047338,-2.411476,1.200252,0.8479'
    '568,1.436849,1.553034],[1.274826,0.1402771,0.9616196,0.5688599,0.1127184,0.2688812,0.513814,0.1790659,2.487354,-'
    '0.9202128,2.710327,2.076784,5.935518,-0.3426357,2.519811,1.949528,1.991951,2.216328,2.592533,8.369929,2.710327,2'
    '.076784,5.935518,1.805291,0.8020096,1.341813,2.6501,0.8367055,1.395199,0.8901685,0.8683837,-2.143022,1.025521,0.'
    '8726953,1.222809,1.365945],[1.261074,0.08153846,0.9778121,0.5519118,0.1123058,0.2657169,0.5199833,0.09023417,2.4'
    '72545,-0.9137505,2.896528,2.14367,6.335237,-0.3145918,2.503539,2.036716,2.071456,2.203975,2.554426,8.404047,2.89'
    '6528,2.14367,6.335237,1.9306,0.8507764,1.463016,2.816417,0.8851992,1.518344,0.9296924,0.9173467,-2.095614,0.9968'
    '11,0.876055,1.181864,1.334836],[1.225935,0.04926209,0.9871587,0.547118,0.1085529,0.2540897,0.5282603,0.09794144,'
    '2.458683,-0.9093228,3.178157,2.175427,6.841743,-0.2961208,2.471548,2.086811,2.116587,2.175427,2.501789,8.478206,'
    '3.178157,2.175427,6.841743,2.029427,0.8812762,1.557823,2.938346,0.9148114,1.614355,0.9546777,0.9470064,-2.065839'
    ',0.9753456,0.8777886,1.156516,1.324079],[1.140961,0.02561758,0.993245,0.5544417,0.1011725,0.2266322,0.5441555,0.'
    '09839703,2.486987,-0.9217046,3.601227,2.219325,7.561978,-0.2784283,2.472411,2.170636,2.181455,2.193378,2.443822,'
    '8.606383,3.601227,2.219325,7.561978,2.111902,0.9028602,1.647586,3.062205,0.9363087,1.707251,0.9684938,0.9643945,'
    '-2.078785,0.9784151,0.8774889,1.15708,1.333674],[1.080022,0.02045777,0.9946937,0.5681899,0.09242571,0.208544,0.5'
    '604931,0.05531594,2.518795,-0.9323902,3.920704,2.268703,8.124846,-0.2653551,2.511741,2.239489,2.257011,2.251425,'
    '2.415295,8.715098,3.920704,2.268703,8.124846,2.162362,0.9123309,1.723877,3.148193,0.9453001,1.775575,0.9744839,0'
    '.9717603,-2.124342,1.008293,0.8719897,1.191952,1.366475]],"sd":[[4.32136,2.595226,0.1381232,0.2160661,0.04460978'
    ',9.160203,1.0,1.830719,0.8713655,0.7337208,2.95495,2.95495,2.358779,0.8365721,2.95495,1.628795,1.061615,0.831333'
    '2,1.890378,0.7337208,2.95495,2.95495,0.154943,0.08242061,0.1343767,0.3903261,0.05030095,0.09284893,0.04052631,0.'
    '3597491,2.869985,3.068856,0.4447611,2.986518,2.027694],[4.814344,3.619674,0.2413769,0.2012573,0.0863888,6.379486'
    ',0.1500069,4.557631,1.823445,0.8696107,0.6802028,1.99849,2.939642,1.566411,0.8166059,2.769266,1.55535,1.029374,0'
    '.8009767,1.85419,0.6802028,1.99849,2.939642,0.1684565,0.1028418,0.1914267,0.3923324,0.06595001,0.1370549,0.21936'
    '02,0.4108835,2.678359,2.555998,0.4029873,2.651885,1.998791],[4.083953,3.229321,0.3217661,0.245662,0.1439958,3.83'
    '8416,0.3041545,4.03525,1.811961,0.8654024,0.6017988,1.436753,2.656022,1.120811,0.780221,2.391332,1.344418,0.9560'
    '872,0.7400298,1.804554,0.6017988,1.436753,2.656022,0.2722237,0.1403046,0.3149704,0.417972,0.09682261,0.2545843,0'
    '.4029665,0.4304544,2.605298,2.295145,0.4087308,2.479255,2.025065],[3.419297,2.709332,0.354599,0.2946961,0.181851'
    '8,2.5754,0.4045766,3.272076,1.82009,0.8648479,0.567955,1.138167,2.453997,0.880722,0.7651137,2.06363,1.138167,0.9'
    '124151,0.7131342,1.775218,0.567955,1.138167,2.453997,0.4404105,0.201601,0.4920104,0.4948775,0.1493634,0.4311842,'
    '0.4763182,0.4479635,2.610376,2.204202,0.4176241,2.421827,2.055555],[2.986645,2.343937,0.3579294,0.3383653,0.2103'
    '245,1.934006,0.4581986,2.610649,1.845931,0.8697956,0.5908647,1.027516,2.413052,0.7572661,0.7863594,1.911539,1.11'
    '1159,0.9114508,0.7239671,1.764282,0.5908647,1.027516,2.413052,0.609901,0.2803946,0.7098486,0.666409,0.2251537,0.'
    '6659065,0.4923905,0.4730189,2.609447,2.110174,0.4148367,2.358727,2.062775],[1.869145,1.515751,0.3392259,0.385327'
    '8,0.2386728,0.573977,0.4835302,2.392695,1.870163,0.8709083,0.6668568,1.068215,2.548197,0.7009967,0.8435169,1.956'
    '367,1.191325,0.9606094,0.7708086,1.772395,0.6668568,1.068215,2.548197,0.7556042,0.3490138,0.9358322,0.932943,0.3'
    '043573,0.9206079,0.4867781,0.4962535,2.588385,2.018149,0.3999752,2.277442,2.040716],[1.681413,1.175001,0.2944011'
    ',0.428692,0.2657523,0.4756481,0.4964305,1.096286,1.900447,0.8763529,0.7872265,1.209391,2.835688,0.682876,0.93759'
    '37,2.12284,1.370149,1.062767,0.8469476,1.782595,0.7872265,1.209391,2.835688,0.8599082,0.3824978,1.122184,1.17788'
    '4,0.3540122,1.111468,0.469981,0.4952389,2.55698,1.95333,0.3794874,2.192284,1.999652],[1.628049,0.884823,0.235633'
    '7,0.4611571,0.2876492,0.4539686,0.4998197,0.7492151,1.935875,0.887121,0.903034,1.345903,3.148693,0.6798502,1.035'
    '854,2.308523,1.542944,1.176445,0.9209577,1.785309,0.903034,1.345903,3.148693,0.874761,0.3667036,1.21897,1.271171'
    ',0.3497389,1.18066,0.4238989,0.4561623,2.487429,1.81416,0.3590628,2.07063,1.949346],[1.636129,0.5539535,0.146215'
    '4,0.4854802,0.3083635,0.4574399,0.4998091,0.6127354,1.965601,0.8970625,1.039058,1.445359,3.501519,0.6543962,1.15'
    '5459,2.52704,1.691157,1.315092,1.010516,1.766876,1.039058,1.445359,3.501519,0.7377848,0.2679214,1.230454,1.12532'
    '6,0.2518331,1.14455,0.3126796,0.3380731,2.379636,1.65601,0.333314,1.9033,1.872934],[1.631307,0.4166696,0.1109338'
    ',0.4910545,0.3107348,0.4538637,0.4996005,0.4331948,1.973993,0.9006995,1.094645,1.435454,3.625151,0.6162037,1.195'
    '259,2.587078,1.715614,1.361659,1.037769,1.72787,1.094645,1.435454,3.625151,0.6523713,0.2166874,1.213547,1.024041'
    ',0.1988753,1.118207,0.2556647,0.2753574,2.361464,1.631016,0.3295188,1.869082,1.860873],[1.600787,0.3278273,0.084'
    '3014,0.4937772,0.3079399,0.4388628,0.4992007,0.4505516,1.976056,0.8996796,1.141642,1.389699,3.728496,0.5536907,1'
    '.219859,2.618171,1.722465,1.389699,1.054142,1.644122,1.141642,1.389699,3.728496,0.5855021,0.1835242,1.212075,0.9'
    '736193,0.169085,1.11546,0.20801,0.2240206,2.342171,1.594606,0.3275299,1.829806,1.856154],[1.524573,0.2356493,0.0'
    '6044602,0.4947696,0.300137,0.4059181,0.4980465,0.4646856,1.989224,0.9089414,1.166769,1.321654,3.809237,0.4864143'
    ',1.211411,2.673007,1.739282,1.384946,1.077351,1.515803,1.166769,1.321654,3.809237,0.524758,0.1561906,1.204915,0.'
    '9268382,0.1496916,1.116639,0.1746813,0.1853045,2.337264,1.578076,0.3278752,1.814048,1.858595],[1.473179,0.220160'
    '1,0.05493597,0.4936661,0.2885323,0.380889,0.4963271,0.3404354,1.992402,0.9095507,1.16655,1.280817,3.86139,0.4477'
    '918,1.19449,2.721179,1.76356,1.381808,1.105252,1.423329,1.16655,1.280817,3.86139,0.4788791,0.1418871,1.212887,0.'
    '8748084,0.1360729,1.124578,0.1576864,0.1656569,2.360106,1.600691,0.3341013,1.839946,1.878322]],"top":{"comps":["'
    'ptr","vote","src"],"cols":["ptr_run","ptr_hits","ptr_miss16","ptr_acc64","ptr_since","ptr_age","ptr_share","ptr_'
    'tcnt","ptr_ntok","ptr_seedlen","ptr_nrec","ptr_margin","ptr_score","ptr_how0","ptr_how1","ptr_how2","ptr_how3","'
    'ptr_how4","ptr_how5","hist16_ret_c","hist16_ret_acc","hist16_ptr_c","hist16_ptr_acc","hist16_either","hist64_ret'
    '_c","hist64_ret_acc","hist64_ptr_c","hist64_ptr_acc","hist64_either","histseg_ret_c","histseg_ret_acc","histseg_'
    'ptr_c","histseg_ptr_acc","histseg_either","hist2_ret_r32","hist2_ret_rseg","hist2_ptr_r32","hist2_ptr_rseg","mem'
    '_acc_h8","mem_n_h8","mem_acc_p8","mem_hit_p8","mem_acc_h32","mem_n_h32","mem_acc_p32","mem_hit_p32","mem_acc_h12'
    '8","mem_n_h128","mem_acc_p128","mem_hit_p128","mem_r_h32","mem_streak","mem_since_wrong","mem_since_hit","mem_pr'
    'ev_corr","mem_prev_hit","mem_ema90","mem_ema98","mem_nlong32","mem_nlong16","mem_acc_doc","mem_lenl","mem_lenr",'
    '"mem_cont","mem_cont_run","mem_cont_corr_run","mem_npos","mem_ncand","src_n","src_purity","src_lbest","src_lbest'
    '_l2","src_nsrc","src_since","src_newlen","src_n2","src_acc_h8","src_acc_h32","src_acc_doc","src_nhit","src_strea'
    'k","src_prev_corr","src_r_h32","ret_logn","ret_purity","ret_logl","ind_has","ind_hp","ptr_agree","both_has","bot'
    'h_lstar","both_hs","lm_ent","lm_mx","tp_ptr_v","tp_ptr_in","tp_ptr_gap","tp_vote_v","tp_vote_in","tp_vote_gap","'
    'tp_mem_v","tp_mem_in","tp_mem_gap","su_ptr_log","su_ptr_mean","su_ptr_max","su_ptr_ment","su_src_log","su_src_me'
    'an","su_src_max","su_src_ment","su_mem_log","su_mem_mean","su_mem_max","su_mem_ment","chain_logo","chain_logn","'
    'chain_purity","chain_logd","llr_ret_32","llr_ret_seg","llr_ptr_32","llr_ptr_seg"],"W":[[-6.783671,-8.452305,-8.4'
    '55154],[-0.5556663,0.2264874,-0.3531102],[-0.4385264,-0.04193966,0.1053908],[0.02855436,0.2068655,0.3180149],[0.'
    '5249112,-0.04398061,-0.137628],[-0.190898,0.2056954,-0.170343],[1.046622,0.2852554,0.01226461],[0.9977064,0.2553'
    '735,0.1658963],[0.5232357,0.7785775,0.1107928],[-0.109735,-0.1544522,0.2656268],[-0.1532343,0.1986519,-0.0485563'
    '3],[-0.4334316,-0.2520903,-0.5696823],[-0.1964862,-0.4173418,0.191629],[0.2127287,0.07547363,0.2054669],[-0.0418'
    '8354,0.07800951,-0.02646772],[0.1295597,0.1465288,-0.02491554],[0.003521236,-0.2174566,0.05135288],[-0.1440275,-'
    '0.005050406,-0.1460912],[0.0260057,0.04383667,-0.05142366],[-0.2502902,0.07283014,0.03776997],[-0.1351316,-0.003'
    '764728,0.003878121],[0.149593,-0.1251604,0.03538472],[0.482207,0.3964147,0.1438987],[0.04159973,0.256259,-0.1615'
    '726],[-0.3704642,0.1213692,-0.001015606],[-0.2598263,0.1602515,-0.3157183],[0.01488802,-0.08748925,0.01964692],['
    '0.5310374,0.3934535,0.40122],[0.1433293,0.0809099,0.1365832],[-0.07562029,-0.1906432,-0.2010844],[0.01200374,-0.'
    '008417948,-0.3666753],[-0.1257011,-0.07135457,-0.03177221],[0.5675872,0.3139511,0.2884495],[0.001905314,0.002280'
    '572,-0.1701659],[0.01431287,0.07323494,-0.0545039],[0.0594482,0.2000387,-0.1208133],[0.3020709,-0.09019452,-0.24'
    '67381],[-0.4203726,0.03104628,-0.004692123],[0.001905314,0.002280572,-0.1701659],[-0.05992399,-0.08130256,0.0690'
    '9573],[-0.08843346,-0.2046437,0.1238777],[0.1715739,-0.02783145,-0.1654419],[-0.1799295,-0.08054339,-0.004221643'
    '],[-0.06603706,0.03481828,0.2186303],[-0.1686186,0.1995036,0.1120413],[0.03939527,-0.1179811,-0.341114],[-0.0154'
    '7276,0.08722469,0.2787301],[0.15484,-0.195874,0.1237677],[0.05385197,-0.1364051,-0.1049628],[-0.0758302,-0.03262'
    '182,0.004189642],[-0.04007702,0.07094655,-0.04188129],[0.2923222,0.1539745,-0.1486367],[0.08115018,-0.2723431,-0'
    '.3591979],[-0.06342363,-0.001212092,0.1863795],[-0.01480678,0.07273418,-0.2833407],[0.01482377,0.05982909,-0.205'
    '7124],[0.002500845,0.02153561,0.1599929],[-0.07504934,-0.03282743,0.07917074],[-0.2793949,-0.03670177,0.2133405]'
    ',[-0.02136881,-0.1136819,0.009022183],[-0.1091035,-0.1299879,-0.1658569],[-0.3624988,0.1008641,0.1414813],[-0.10'
    '89108,-0.05770311,-0.2375918],[-0.1477849,-0.08653506,-0.2452642],[0.08412222,0.1049138,-0.2827399],[-0.02184526'
    ',0.008262801,-0.03402193],[-0.1920765,0.00709117,0.4345974],[-0.3243016,-0.05020843,-0.5455166],[-0.07627318,-0.'
    '280542,-0.005400456],[0.1593397,-0.07703009,0.9045468],[0.2330794,-0.1143617,1.139524],[-0.09269302,0.1403483,-0'
    '.06718113],[-0.1016927,0.05269931,0.7526933],[-0.07272568,-0.09877568,-0.01981393],[-0.1047208,-0.1662628,-0.049'
    '76593],[-0.033996,-0.333543,-0.19056],[-0.1253176,-0.03090443,-0.4493815],[0.1535243,-0.06310913,0.2339684],[0.0'
    '3844962,0.005552168,0.1395373],[0.07322251,0.1127563,0.410689],[0.1570151,-0.01006767,0.8640438],[-0.09967792,0.'
    '1694252,0.2902494],[-0.07551851,0.01411949,0.3088291],[-0.1082448,-0.3355964,0.2889786],[-0.04904965,0.00808167,'
    '0.1089799],[0.1874442,-0.01712428,0.3017282],[-0.09157105,-0.04973519,-0.1699559],[0.07208941,0.1085325,0.222522'
    '4],[0.01667118,-0.01065399,-0.007882212],[-0.1122729,-0.09250452,-0.1549744],[0.07208941,0.1085325,0.2225224],[-'
    '0.09157105,-0.04973519,-0.1699559],[4.452398e-05,0.4576128,0.03047498],[0.456633,0.7706266,0.02571622],[0.457803'
    '4,0.2120692,0.02121105],[0.6766315,0.02275129,0.2523197],[0.5607075,-0.06013484,-0.02222583],[-0.6157875,0.03241'
    '736,-0.1366793],[0.147712,0.10631,0.02874644],[0.05822821,-0.187127,0.003737417],[-0.04274325,-0.05735365,0.1062'
    '129],[-0.3245835,0.02599779,-0.01447711],[-0.03649299,0.2141975,0.1049812],[-0.2704851,0.05595322,-0.06257801],['
    '-0.2227328,0.1270246,0.2052669],[0.4099274,-0.05720064,0.1599839],[0.2714633,0.1929204,-0.362552],[-0.1179909,-0'
    '.1178712,0.01601473],[-0.06966257,-0.07483555,1.259315],[-0.03777979,-0.123772,0.06485288],[-0.07336886,-0.00398'
    '1246,0.2251119],[-0.1311628,0.1841415,-0.1320502],[0.06650874,-0.03281426,0.2942671],[-0.1853475,-0.102944,-0.30'
    '39945],[0.113424,-0.03578138,0.07269126],[0.01092296,0.113501,-0.02437701],[-0.1841372,-0.3157426,-1.216695],[-0'
    '.154171,-0.1731328,0.37107],[0.04692882,-0.06370689,0.09501502],[0.09040135,-0.06087944,-0.6210358],[0.0810508,-'
    '0.09284982,0.07516476],[0.2101478,0.6405104,0.7129493],[0.3994512,0.5809366,0.08479922],[-0.04513772,0.09631885,'
    '-0.0262911]],"mu":[0.9465571,2.252345,1.387345,0.569372,0.8808334,1.406062,0.2761387,0.8826586,2.023592,1.974279'
    ',0.9249795,0.4106734,0.3402532,0.1154892,0.1870312,0.346909,0.001321445,0.03303896,0.01515559,0.6937144,0.216100'
    '8,0.8307763,0.1677062,0.1156362,1.371303,0.2773161,1.595158,0.1699111,0.09182364,3.524108,0.3374236,3.875527,0.1'
    '758669,0.07257547,0.2254847,0.3110419,0.1687479,0.1758669,0.319048,2.104793,0.09254865,0.1970942,0.3424927,3.041'
    '684,0.07443316,0.1442767,0.348799,3.463757,0.06167597,0.1201966,0.3173914,0.1911647,2.065082,1.741855,0.1051582,'
    '0.2391872,0.08724523,0.06267364,0.2144928,0.4250836,0.3474065,0.7604482,0.7591509,0.09252945,0.1676262,0.1645282'
    ',6.001597,0.4447951,0.7289668,0.3901751,2.574149,0.9000032,4.937626,4.146515,2.837219,0.9930464,0.2788894,0.2802'
    '49,0.2795572,3.962516,0.2623623,0.1653844,0.2549425,0.3304861,0.1868258,0.7261421,0.2400276,0.6989454,0.1853164,'
    '0.2400276,0.7261421,0.5394693,3.155593,-1.204437,-3.825071,0.2941474,3.026051,-3.674075,0.3206527,2.875055,-0.72'
    '00655,0.1886356,0.4543387,0.7783155,0.6558335,1.624153,-0.2192355,0.9675551,1.018094,2.006961,-0.3952025,0.65859'
    '52,0.4622731,1.374676,-0.1349074,1.992174,3.756649,0.5350541,2.854679,-0.3058333,-1.289003,-1.121757,-3.040511],'
    '"sd":[1.508095,1.65903,1.806774,0.3961811,1.514144,1.583502,0.3333577,0.7879523,1.597942,1.297755,1.099967,1.240'
    '058,0.9951244,0.3196114,0.3899365,0.4759865,0.03632766,0.1787383,0.1221716,1.05981,0.3156123,1.158412,0.2738905,'
    '0.2246424,1.432077,0.2800176,1.528774,0.2300796,0.1856516,1.848484,0.2140385,1.92092,0.1851664,0.1343039,0.28840'
    '08,0.2155811,0.2523852,0.1851664,0.2136272,0.3069468,0.195555,0.2279383,0.2037526,0.7166681,0.1705174,0.1823374,'
    '0.2004177,1.078014,0.1389675,0.145662,0.2068047,0.7079783,1.499443,1.476556,0.3067571,0.4265873,0.1889123,0.1394'
    '607,0.8991131,1.180745,0.1963659,1.460004,1.456289,0.2897719,0.6654846,0.6619749,1.143108,1.210375,1.459528,0.44'
    '17794,5.336534,1.145194,2.912622,2.706828,0.9873643,1.708515,0.225373,0.2115427,0.1936233,1.770653,0.7737356,0.3'
    '715271,0.2118035,1.008602,0.3672336,1.338869,0.4270999,0.4587164,0.388554,0.4270999,1.338869,0.4984397,1.862598,'
    '0.8821905,3.67277,0.4556586,3.396529,3.639406,0.4667275,3.369243,1.816523,0.3912189,1.393727,1.250885,1.14174,2.'
    '867108,0.634401,1.115546,1.313649,2.717584,0.8328118,1.244708,0.9474902,2.80609,0.3914858,0.816216,3.886539,0.36'
    '90696,3.113906,1.189213,2.598557,1.569045,2.832158]},"fit":{"what":"CPU placeholder (the CPU proof\'s fit): repla'
    'ced by fits on our model, one per step count, from FIT dev runs (tools/stream_retrieval/README.md)","proxy":"llm'
    '.c GPT-2 124M on CPU, not our model: P_0 a CPLM-like copy mixture with its constants fitted on training position'
    's; the side output its entropy, max log p and top-32 log-probs","total_steps":1050,"freeze":287882714,"source":"'
    'the CPU proof: the hooked timed loader replaying the record schedule (978 scheduled + 72 = 1050 steps, 8 ranks, '
    'the real shards) into the helper, its FIT dump (ranks 0-3)","data":"the run\'s own last 16 batches (steps 1034-10'
    "49) of that 1050-step run, queried by the helper's FIT path against the memory as it stood before step 1034 (ent"
    'ry 287,882,714)","ranks":[0,1,2,3],"positions":1048576,"matched":1.0,"batches":16,"mode":"lm+su2+tp+hi","parts":'
    '"P1 + P2 + P3","top_comps":["ptr","vote","src"],"in_sample_mnat":57.153,"note":"CPU proof, P1 + P2 + P3 (P2 at 0'
    ".75x table sizes and 2^26 buckets for this VM's memory: the val rows equal the default sizes', the FIT rows diff"
    'er at 3 P1 records and 21 P3 rows of 1,048,576 positions)","val_mnat_cpu_proxy":"34.91 [29.49, 40.99] millinats '
    'over the CPLM-like base on the first 1,048,576 val positions (research A+B+C, 2-fold on val: 34.16 deployable fo'
    'rm, 34.27); P2\'s increment +10.25 [9.40, 11.11]"}}]'
)
# GATE_V2_LOW_END
_as_list = lambda spec: spec if isinstance(spec, list) else [spec]
if GATE_V2_JSON:
    GATE_V2 = _as_list(json.loads(GATE_V2_JSON))
if GATE_V2_LOW_JSON:
    GATE_V2_LOW = _as_list(json.loads(GATE_V2_LOW_JSON))
