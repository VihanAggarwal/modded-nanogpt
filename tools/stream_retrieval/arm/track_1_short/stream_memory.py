"""Stream-only retrieval at the final validation (STREAM_RETRIEVAL=1; off by default).

At the final validation, the probability p of each val token (CPLM with the canonical mask) is mixed with a
next-token distribution read from an exact-match memory: q = (1 - lambda) p + lambda C/N.

- The memory holds only the tokens this run trains on. Rank 0's loader passes the document spans that
  Shard.next_batch computes for all ranks (data.py, `on_spans`) to a helper process, which reads those token
  ranges from the shard files and indexes them as the batches are fetched: on the clock, in parallel with
  training. Only the timed loader is tapped; the warmup loader is not.
- At the last step rank 0 sends GO. The helper reads the val shard and computes one row (a, b) per val
  position, all before the clock stops: a = 1 - lambda and b = lambda C/N, where the walk over the memory, the
  counts N, C, M and the level L* are described in stream_memory.c. Each rank copies its slice to the device.
- The mixing (`mix`) runs in the untimed eval loop, like CPLM's own mixture. Training, the model and the token
  streams are untouched: the same weights are scored with and without the mixture in one run (`val_loss_lm` is
  logged next to `val_loss`), so one run measures the gain exactly.

Why the result is a valid probability model: r_t(v) = |{j in S : next_j = v}| / N sums to 1 over the vocabulary
and is a function of the memory and val[<= t] only; lambda_t depends on (N, M, L*), none of which reads
val[t + 1]. So sum_v q_t(v) = (1 - lambda) sum_v p_t(v) + lambda <= 1, since CPLM's p sums to <= 1 (its LM branch
is renormalized over the canonical tokens; its copy branch's mass on masked tokens is dropped). Retrieved mass on a
masked token is dropped too, never renormalized. The row evaluates r_t at the realised token, the same operation as
a cross-entropy gather. Forward-only, nothing learned from val.

Where the gain comes from (CPU proxies on the real 1050-step stream, tools/stream_retrieval/README.md): it is
concentrated in val documents that share long verbatim passages with trained documents, mostly web boilerplate
(site navigation, "most read" sidebars, templates). The top 1% of val documents carry 30-47% of it, the top 5%
65-92%; positions matched at 32 tokens are 0.3-0.4% of val and carry 31-37%.

The gate's 4 constants (W) are hyperparameters, never fitted on val (provenance at W). STREAM_RETRIEVAL_FIT=<path>
(dev runs only, untimed) dumps the features and our model's per-token NLL on 64 training batches past the run's
stream (never trained on, not in the memory), and tools/stream_retrieval/fit_gate.py refits W on them.

Credits. No code is taken from another PR; the design builds on two of them:
- PR #367 (Herman Brunborg): exact-match retrieval of continuations from training data on this track, and its
  StreamIndex (exact_match/src/stream.rs), whose memory and row rule this one follows: the stream of every rank's
  documents (inputs plus the last target), step by step, each followed by a STOP; positions keyed on their last 6
  tokens and resolved against the most recent occurrences; the deepest match level reached; the row from the next
  tokens of the occurrences that match at least that deep; its (length, count, top-token share) as the features.
- PR #380 (Deven): mixing CPLM's probability at the output with the retrieved count share, P = (1 - lam) P +
  lam C / N, under a sigmoid gate on the match order and log2 N, fitted on training positions (its ChainFit; here
  one link instead of a chain, fitted off the clock: STREAM_RETRIEVAL_FIT).
- kNN-LM (Khandelwal et al., 2020) and Infini-gram (Liu et al., 2024) for retrieval/n-gram interpolation; the
  LZ77/zlib hash chain for the index.
This branch's part: restricting the memory to this run's consumed stream and using it at the final validation (in
place of #367's 103-shard SlotIndex and #380's corpus counts), the hash-chain index and C helper filled from the
loader's spans on the clock, the single longest-match link with its 4-constant gate, and the tests. Nothing from
#381.
"""
import atexit
import fcntl
import glob
import hashlib
import itertools
import mmap
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

ENABLED = os.environ.get("STREAM_RETRIEVAL") == "1"
SOURCE = Path(__file__).resolve().with_name("stream_memory.c")
# lambda = sigmoid(W . [1, log2 N, M/N, log2 L*]). Provenance, never val: tools/stream_retrieval/fit_gate.py on held-out
# TRAINING data, the helper's own FIT rows for the 16 training batches past the 1050-step stream (4 ranks x 262,144
# positions of shard 4, never in the memory), scored by a CPU proxy LM never trained on FineWeb (OpenAI's GPT-2 124M);
# 56.0 millinats held out there (2-fold by chunk). That continuation is more retrieval-friendly than val, so its gain
# overstates val's; on the design-phase val proxies these constants are within 0.8 millinats of ones fitted on val.
# Refit on our own model with STREAM_RETRIEVAL_FIT (module docstring).
W = (-15.8, 0.957, 7.412, 2.528)
HASH_BITS = 29                  # 2^29 buckets, 2 GB of heads
QUERY_THREADS = 32              # at most; never more than the node's CPUs
CFLAGS = ("-O2", "-std=c11", "-pthread")
FIT_BATCHES = 64                # STREAM_RETRIEVAL_FIT: training batches past the stream (8.4M tokens at batch 8)
COLLECT_TIMEOUT_S = 60.0

# The rows file: header words (u64) as stream_memory.c's H_* enum, then f32 [world][val_steps][chunk][2]. These
# constants mirror the C file's (tools/stream_retrieval/test_stream_retrieval.py checks that they match).
H_MAGIC, H_STATE, H_ERROR, H_WORLD, H_VAL_STEPS, H_CHUNK, H_STEPS, H_ENTRIES, H_INSERTED, H_HITS, H_CHITS, \
    H_QUERIED, H_T_READY, H_T_GO, H_T_DONE, H_INSERT_NS, H_QUERY_NS, H_BYTES_READ, H_LEVEL0 = range(19)
LEVELS = (6, 8, 12, 16, 24, 32)
H_FIT_POSITIONS = H_LEVEL0 + len(LEVELS)
HEADER_MAGIC = 0x314D454D52545353
ST_STARTING, ST_READY, ST_DONE, ST_FIT_DONE, ST_ERROR = 0, 1, 2, 3, 14
ERRMSG_OFFSET, CHECKSUM_OFFSET, ROWS_OFFSET = 512, 1024, 4096
MSG_STEP, MSG_GO, MSG_FIT_STEP, MSG_FIT_GO = 1, 2, 3, 4
F_SETPIPE_SZ = 1031
PIPE_BYTES = 1 << 20


def build_helper() -> Path:
    """Compile stream_memory.c (cached by its hash). cc is on every node: Triton needs it too."""
    source = SOURCE.read_bytes()
    out = Path(tempfile.gettempdir()) / "stream_memory_build" / f"stream_memory_{hashlib.sha256(source).hexdigest()[:16]}"
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


def mix(nll: Tensor, rows: Tensor) -> Tensor:
    """Per-token NLL of q = a p + b, with p = exp(-nll) - 1e-9 (CPLM's NLL is -log(p + 1e-9)), where the memory
    matched; elsewhere (a, b) = (1, 0) and the NLL is returned bit-identical."""
    a, b = rows.unbind(-1)
    hit = (a != 1) | (b != 0)
    q = a * (torch.exp(-nll) - 1e-9).clamp_min(0) + b
    return torch.where(hit, -torch.log(q + 1e-9), nll)


def _sync():
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


class StreamMemory:
    """The memory's handle on every rank: rank 0 owns the helper and feeds it; every rank maps the rows.

    world: ranks whose spans each step message carries (the loader's world size) and the rows' layout.
    """

    def __init__(self, *, train_files: list[str], val_file: str, total_steps: int, stream_tokens: int, world: int,
                 rank: int, master: bool, val_tokens: int, chunk: int, device, print0=None, threads: int | None = None,
                 hash_bits: int = HASH_BITS, weights=W, fit_path: str | None = None,
                 dump_path: str | None = None, readlog_path: str | None = None, features_path: str | None = None,
                 rows_dir: str | None = None):
        assert val_tokens % (world * chunk) == 0, "val_tokens must be a whole number of world x chunk steps"
        self.world, self.rank, self.master, self.chunk = world, rank, master, chunk
        self.val_steps = val_tokens // (world * chunk)
        self.total_steps = total_steps
        self.device = torch.device(device)
        self.print0 = print0 or (lambda s, console=False: None)
        self.fit_path = fit_path
        self.sent = self.fit_sent = self.max_lag = 0
        self.fitting = False
        self.proc = self.fd = None
        self.waited_ms = 0.0
        nbytes = ROWS_OFFSET + world * self.val_steps * chunk * 8
        path = None
        if master:
            rows_dir = rows_dir or ("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir())
            fd, path = tempfile.mkstemp(prefix="stream_rows_", dir=rows_dir)
            self._path = path
            atexit.register(self.close)
            os.ftruncate(fd, nbytes)
            os.close(fd)
            # Allocation only: the stream (cap entries reserved, the expected ones prefaulted) and the heads.
            cap = 2 * stream_tokens + 2  # a span holds >= 1 token, so separators <= tokens
            prefault = min(cap, int(1.02 * stream_tokens) + (1 << 20))
            threads = threads or min(QUERY_THREADS, os.cpu_count() or 1)
            args = [f"rows={path}", f"val={val_file}", f"world={world}", f"val_tokens={val_tokens}", f"chunk={chunk}",
                    f"steps={total_steps}", f"cap={cap}", f"prefault={prefault}", f"hash_bits={hash_bits}",
                    f"threads={threads}", "w=" + ",".join(map(repr, map(float, weights)))]
            args += [f"{k}={v}" for k, v in (("fit", fit_path), ("dump", dump_path), ("readlog", readlog_path),
                                             ("features", features_path)) if v]
            self.proc = subprocess.Popen([str(build_helper()), *args, "--", *train_files], stdin=subprocess.PIPE)
            self.fd = self.proc.stdin.fileno()
            try:
                fcntl.fcntl(self.fd, F_SETPIPE_SZ, PIPE_BYTES)
            except OSError:
                pass  # the default 64 KB still holds ~16 steps of spans
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            box = [path]
            dist.broadcast_object_list(box, src=0)
            path = box[0]
        with open(path, "r+b") as f:
            self.map = mmap.mmap(f.fileno(), nbytes)
        self.hdr = np.frombuffer(self.map, dtype=np.uint64, count=64)
        if master:
            self._wait(ST_READY, timeout=300.0)
            self.print0(f"stream retrieval: helper ready ({stream_tokens} stream tokens expected, {threads} query threads, "
                        f"2^{hash_bits} buckets, W={tuple(weights)})", console=True)
        _sync()
        if master:
            os.unlink(path)  # the mappings stay
            self._path = None
        all_rows = np.frombuffer(self.map, dtype=np.float32, offset=ROWS_OFFSET).reshape(world, self.val_steps, chunk, 2)
        self.own_rows = all_rows[rank] if rank < world else None
        self.all_rows = all_rows
        self.host = torch.empty((self.val_steps, chunk, 2), dtype=torch.float32,
                                pin_memory=self.device.type == "cuda")

    @classmethod
    def for_run(cls, *, train_pattern: str, val_pattern: str, step_batch_sizes: list[int], world: int, rank: int,
                master: bool, val_tokens: int, chunk: int, device, print0=None):
        """The trainer's memory: the shards in data.py's order, sized from the schedule (each rank's spans hold
        its batch tokens + 1), STREAM_RETRIEVAL_FIT read from the environment."""
        local = int(os.environ.get("LOCAL_WORLD_SIZE", world))
        if local != world:
            raise RuntimeError(f"STREAM_RETRIEVAL needs one node: world size {world}, local world size {local}")
        train_files = sorted(glob.glob(train_pattern))
        val_file = sorted(glob.glob(val_pattern))[0]
        stream_tokens = sum(batch_size + world for batch_size in step_batch_sizes)
        return cls(train_files=train_files, val_file=val_file, total_steps=len(step_batch_sizes),
                   stream_tokens=stream_tokens, world=world, rank=rank, master=master, val_tokens=val_tokens,
                   chunk=chunk, device=device, print0=print0, fit_path=os.environ.get("STREAM_RETRIEVAL_FIT") or None)

    # ---------------------------------------------------------------- rank 0: the messages

    def _write(self, words: np.ndarray):
        view = memoryview(words.astype(np.uint32, copy=False).tobytes())
        while view:
            n = os.write(self.fd, view)
            view = view[n:]

    def on_spans(self, file_idx: int, starts: list, ends: list):
        """The loader's tap (rank 0): one fetched step's spans for every rank, from shard file `file_idx`. Runs on the
        thread that fetches (rank 0's main thread): one message, packed straight from the loader's lists."""
        counts = [len(s) for s in starts]
        n, head_words = sum(counts), 4 + len(counts)
        msg = np.empty(head_words + 2 * n, dtype=np.uint32)
        if self.fitting:
            msg[:4], self.fit_sent = (MSG_FIT_STEP, self.fit_sent, file_idx, len(starts)), self.fit_sent + 1
        else:
            msg[:4], self.sent = (MSG_STEP, self.sent, file_idx, len(starts)), self.sent + 1
        msg[4:head_words] = counts
        msg[head_words::2] = np.fromiter(itertools.chain.from_iterable(starts), dtype=np.int64, count=n)
        msg[head_words + 1::2] = np.fromiter(itertools.chain.from_iterable(ends), dtype=np.int64, count=n)
        self._write(msg)
        if not self.fitting:
            self.max_lag = max(self.max_lag, self.sent - int(self.hdr[H_STEPS]))

    def go(self):
        """Rank 0, at the last step: the memory is complete; read val and compute the rows."""
        self._write(np.array([MSG_GO, self.sent], dtype=np.uint32))

    # ---------------------------------------------------------------- every rank

    def _wait(self, state: int, timeout: float):
        t0 = time.perf_counter()
        while (current := int(self.hdr[H_STATE])) != state:
            if current == ST_ERROR:
                msg = bytes(self.map[ERRMSG_OFFSET:ERRMSG_OFFSET + 256]).split(b"\0")[0].decode(errors="replace")
                raise RuntimeError(f"stream retrieval helper failed: {msg}")
            if self.proc is not None and self.proc.poll() is not None and int(self.hdr[H_STATE]) != state:
                raise RuntimeError(f"stream retrieval helper exited ({self.proc.returncode}) before state {state}")
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError(f"stream retrieval: no state {state} after {timeout:.1f} s (state {current})")
            time.sleep(1e-4)
        return 1000 * (time.perf_counter() - t0)

    def collect(self) -> Tensor:
        """This rank's rows [val_steps, chunk, 2] on its device, once the helper is done. Raises if it failed."""
        self.waited_ms = self._wait(ST_DONE, COLLECT_TIMEOUT_S)
        np.copyto(self.host.numpy(), self.own_rows)
        if self.device.type == "cuda":
            return self.host.to(self.device, non_blocking=True)
        return self.host.clone()

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
        h = [int(v) for v in self.hdr[:H_FIT_POSITIONS + 1]]
        q = max(h[H_QUERIED], 1)
        levels = " ".join(f"{lv}:{h[H_LEVEL0 + k] / q:.4f}" for k, lv in enumerate(LEVELS))
        return (f"stream retrieval: memory {h[H_ENTRIES] - 1} entries ({h[H_INSERTED]} indexed) from {h[H_STEPS]} steps, "
                f"insert {h[H_INSERT_NS] / 1e9:.2f} s ({h[H_INSERT_NS] / max(h[H_INSERTED], 1):.1f} ns/entry), max lag "
                f"{self.max_lag} steps; GO->rows {(h[H_T_DONE] - h[H_T_GO]) / 1e6:.1f} ms (queries {h[H_QUERY_NS] / 1e6:.1f} "
                f"ms); waited {self.waited_ms:.1f} ms at collect; hit {h[H_HITS] / q:.4f}, C>0 {h[H_CHITS] / q:.4f}, "
                f"L* {levels}")

    # ---------------------------------------------------------------- STREAM_RETRIEVAL_FIT (dev, untimed)

    def fit_go(self):
        self._write(np.array([MSG_FIT_GO, self.fit_sent], dtype=np.uint32))

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if getattr(self, "_path", None):
            try:
                os.unlink(self._path)
            except OSError:
                pass
            self._path = None


def pseudo_val_batches(train_batches: list, chunk: int, staging):
    """Val-shaped batches (chunk tokens each, documents from their BOS, as the val loader builds them) from
    consecutive training batches of this rank: the same compiled eval forward and helper query apply."""
    from track_1_short.data import BOS_ID, Batch, cu_seqlens_rows
    from track_1_short.ngram_table import ngram_row_ids
    per = train_batches[0].inputs_cpu.size
    assert chunk % per == 0 and len(train_batches) * per % chunk == 0, "training batches do not tile the chunks"
    group = chunk // per
    out = []
    for k in range(0, len(train_batches), group):
        part = train_batches[k:k + group]
        inputs = torch.from_numpy(np.concatenate([b.inputs_cpu for b in part]))
        targets = torch.cat([b.targets_cpu for b in part])
        starts = torch.nonzero(inputs == BOS_ID)[:, 0]
        cum = torch.full((cu_seqlens_rows(chunk),), chunk)
        cum[0] = 0
        cum[1:len(starts) + 1] = starts
        cum = cum.to(torch.int32)
        ngram_ids = ngram_row_ids(inputs)
        dev = staging.upload(inputs, targets, cum, ngram_ids)
        out.append(Batch(*dev, ngram_ids_cpu=ngram_ids.numpy(), targets_cpu=targets, inputs_cpu=inputs.numpy()))
    return out


def fit_dump(memory: StreamMemory, loader, forward, chunk: int, staging, rank: int, print0):
    """STREAM_RETRIEVAL_FIT (dev runs only, after the final validation, untimed): FIT_BATCHES more training batches
    (never trained on, not in the memory) become val-shaped chunks; the helper writes their features
    (N, C, M, L*) to <path> and every rank writes its NLL from the eval forward to <path>.rank<r>.npy.
    tools/stream_retrieval/fit_gate.py fits W on them."""
    memory.fitting = True
    batches = [loader.send(None) for _ in range(FIT_BATCHES)]
    pseudo = pseudo_val_batches(batches, chunk, staging)
    if memory.master:
        memory.fit_go()
    nll = np.stack([forward(batch).float().cpu().numpy() for batch in pseudo])
    np.save(f"{memory.fit_path}.rank{rank}.npy", nll)
    if memory.master:
        memory._wait(ST_FIT_DONE, timeout=600.0)
        print0(f"stream retrieval fit: {nll.size} positions per rank; features in {memory.fit_path}, NLL in "
               f"{memory.fit_path}.rank*.npy; fit with tools/stream_retrieval/fit_gate.py {memory.fit_path} "
               f"--world {memory.world} --chunk {chunk}", console=True)
    _sync()
