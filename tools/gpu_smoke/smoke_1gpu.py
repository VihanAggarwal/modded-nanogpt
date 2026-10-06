"""Single-GPU smoke test of this branch's systems patches (Colab works): the parts CPU tests cannot see.

The speedrun itself needs 8xH100 in one node. This checks, on any one CUDA GPU, that the patched code is
correct under real CUDA and measures the host costs the patches remove:

  A. PyTorch's pinned host cache: a fresh 256 MB-class pinned allocation (cudaHostAlloc) vs a reused block,
     i.e. what the final validation's val-shard read paid vs now pays once the training loader is closed.
  B. The loader with real pinned staging and CUDA events: every batch of the first steps, the threaded first
     fetch (loader thread with set_device) and the threaded val reads are identical to the record's loader
     (upstream 4ea6b93, read from git) on the main thread.
  C. The canonical-mask builder: memfd + cudaHostRegister + spawned builder + collect into a CUDA tensor
     (NCCL broadcast, 1 rank) gives the record's mask; start() vs os.fork() of this CUDA process.

Run from the repo root (needs git history containing 4ea6b93 for B, and network for the tokenizer):
    pip install tiktoken numpy && python tools/gpu_smoke/smoke_1gpu.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RECORD_COMMIT = "4ea6b93"
results = []


def report(name: str, ok: bool, detail: str = ""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)


def ms(t0: float) -> float:
    return 1000 * (time.perf_counter() - t0)


def make_shards(directory: Path, n_train: int = 2, n_tokens: int = 100_000_000):
    """FineWeb-format shards (256 int32 header + uint16 tokens) with lognormal document lengths."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, seed in [("fineweb_val_000000.bin", 1000)] + [(f"fineweb_train_{i:06d}.bin", i) for i in range(1, n_train + 1)]:
        path = directory / name
        if path.exists():
            continue
        rng = np.random.default_rng(seed)
        tokens = rng.integers(0, 50256, size=n_tokens, dtype=np.uint16)
        starts = np.cumsum(np.maximum(2, rng.lognormal(6.4, 1.0, size=n_tokens // 300).astype(np.int64)))
        tokens[0] = 50256
        tokens[starts[starts < n_tokens]] = 50256
        header = np.zeros(256, dtype=np.int32)
        header[:3] = (20240520, 1, n_tokens)
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())


def record_module(path: str):
    source = subprocess.run(["git", "show", f"{RECORD_COMMIT}:{path}"], cwd=ROOT, capture_output=True, text=True)
    if source.returncode:
        return None
    spec = importlib.util.spec_from_loader(f"record_{Path(path).stem}", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source.stdout, f"record_{Path(path).name}", "exec"), module.__dict__)
    return module


def same_batch(a, b) -> bool:
    for x, y in zip(a, b):
        if isinstance(x, torch.Tensor):
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x.cpu(), y.cpu()):
                return False
        elif x.dtype != y.dtype or not np.array_equal(x, y):
            return False
    return True


def section_a_pinned_cache():
    n = 100_000_000  # a shard's uint16 tokens: 200 MB, a 256 MB block in the pinned cache
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    block = torch.empty(n, dtype=torch.uint16, pin_memory=True)
    fresh = ms(t0)
    ptr = block.data_ptr()
    del block
    t0 = time.perf_counter()
    again = torch.empty(n, dtype=torch.uint16, pin_memory=True)
    reused = ms(t0)
    report("A1. pinned cache reuses a freed 256 MB block", again.data_ptr() == ptr,
           f"fresh cudaHostAlloc {fresh:.1f} ms, reused {reused:.3f} ms (the val read's allocation, before/after)")
    del again
    # A fresh pinned allocation on another thread, while this thread launches kernels: does the driver
    # hold up the launches (a second 256 MB-class block would be fresh: one is cached now, so 512 MB)?
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(50):
        x @ x
    torch.cuda.synchronize()
    done = []
    worker = ThreadPoolExecutor(1)
    alloc = worker.submit(lambda: (torch.empty(200_000_000, dtype=torch.uint16, pin_memory=True), done.append(1))[0])
    gaps, last = [], time.perf_counter()
    while not done:
        x.add_(0)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
    block = alloc.result()
    worker.shutdown()
    torch.cuda.synchronize()
    report("A2. launches during a fresh 512 MB pinned allocation on another thread", True,
           f"{len(gaps)} launches, longest gap {1000 * max(gaps):.1f} ms (a stall here is what the val read paid)")
    del block, x


def section_b_loader(data_dir: Path, device: torch.device):
    from track_1_short import data as branch
    from track_1_short.config import (LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT,
                                      Hyperparameters)
    from track_1_short.perf.pinned_batches import PinnedBatchStaging
    from track_1_short.schedule import TrainingSchedule
    record = record_module("track_1_short/data.py")
    if record is None:
        report("B. loader", False, f"git history lacks {RECORD_COMMIT}: clone the fork without --depth")
        return
    real_rank, real_world = dist.get_rank, dist.get_world_size
    dist.get_rank, dist.get_world_size = (lambda: 0), (lambda: 8)  # rank 0's view of an 8-rank run
    # Held to its intended stream: the record's loader must not outrun its BOS scan (a paced run never does;
    # this loop fetches back to back). This branch's loader waits for the full index by itself.
    switch = record.Shard._maybe_switch

    def waiting_switch(shard):
        shard._ready.wait()
        switch(shard)

    record.Shard._maybe_switch = waiting_switch
    try:
        args = Hyperparameters()
        schedule = TrainingSchedule(TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations,
                                    device=device, cooldown_frac=LR_COOLDOWN_FRAC, split_embed_stage=SPLIT_EMBED_STAGE,
                                    ws_post_yarn_ext=WS_POST_YARN_EXT)
        max_tokens = max(s.batch_size for s in TRAINING_STAGES) // 8
        staging = [PinnedBatchStaging(max(max_tokens, args.val_batch_size // 8), 2048, device) for _ in range(2)]
        train = str(data_dir / "fineweb_train_*.bin")
        first = TRAINING_STAGES[0]
        steps = 300  # stage 0, past the partial index's span (~45 steps of real FineWeb)
        rec = record.ScheduledBatches(record.distributed_data_generator(train, first.batch_size, first.train_max_seq_len,
                                                                       staging[0]), schedule, steps=range(steps))
        new = branch.ScheduledBatches(branch.distributed_data_generator(train, first.batch_size, first.train_max_seq_len,
                                                                        staging[1]), schedule, steps=range(steps))
        exhausted = [0]
        next_batch = branch.Shard._next_batch

        def counting_next_batch(shard, *a):
            try:
                return next_batch(shard, *a)
            except branch.PartialIndexExhausted:
                exhausted[0] += 1
                raise

        branch.Shard._next_batch = counting_next_batch
        loader_thread = ThreadPoolExecutor(1, initializer=torch.cuda.set_device, initargs=(device,))
        t0 = time.perf_counter()
        first_record = rec.peek(0)
        record_first = ms(t0)
        x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        t0 = time.perf_counter()
        first_new = loader_thread.submit(new.peek, 0)
        for _ in range(100):  # the GPU busy on the main thread's stream while the loader thread uploads
            x @ x
        first_new = first_new.result()
        new_first = ms(t0)
        shard = new.loader.gi_frame.f_locals["shard"]
        shard._ready.wait()
        scan = ms(t0)
        torch.cuda.synchronize()
        del x
        ok = same_batch(first_record, first_new)
        for step in range(steps):
            ok &= same_batch(rec.take(step), new.take(step))
        torch.cuda.synchronize()
        branch.Shard._next_batch = next_batch
        report("B1. training batches identical (record loader on the main thread vs this branch, first fetch on "
               "the loader thread)", ok, f"first fetch {record_first:.1f} ms -> {new_first:.1f} ms; first shard's "
               f"tail read + full scan done {scan:.0f} ms after the fetch started; partial index outrun "
               f"{exhausted[0]}x (back-to-back fetching; a paced run: 0)")
        frame = new.loader.gi_frame.f_locals
        train_blocks = {frame["tokens"].data_ptr(), frame["next_shard_getter"]().tokens.data_ptr()}
        new.close()
        val_it = branch.distributed_data_generator(str(data_dir / "fineweb_val_*.bin"), args.val_batch_size, -1,
                                                   staging[1], align_to_bos=False)
        next(val_it)
        report("B2. after close(), the val shard's read reuses a training shard's pinned block",
               val_it.gi_frame.f_locals["tokens"].data_ptr() in train_blocks)
        val_it.close()

        def read_val(module, stage):
            it = module.distributed_data_generator(str(data_dir / "fineweb_val_*.bin"), args.val_batch_size, -1, stage,
                                                   align_to_bos=False)
            return [next(it) for _ in range(args.val_tokens // args.val_batch_size)]

        val_record = read_val(record, staging[0])
        t0 = time.perf_counter()
        val_new = loader_thread.submit(read_val, branch, staging[1]).result()
        torch.cuda.synchronize()
        report("B3. val batches identical (read on the loader thread)",
               all(same_batch(a, b) for a, b in zip(val_record, val_new)), f"threaded val read {ms(t0):.1f} ms")
        loader_thread.shutdown()
    finally:
        dist.get_rank, dist.get_world_size = real_rank, real_world


def section_c_canonical_mask(device: torch.device):
    from track_1_short.canonical_mask import BackgroundCanonicalMask, build_canonical_mask
    vocab = 50304
    logs = []
    builder = BackgroundCanonicalMask(vocab, owner=True, print0=lambda s, console=False: logs.append(s))
    report("C1. memfd buffer page-locked (cudaHostRegister)", builder.pinned, "; ".join(logs))
    ballast = np.ones(4 << 30, dtype=np.uint8)  # this CUDA process grows, as the trainer has by t0
    t0 = time.perf_counter()
    builder.start()
    spawn = ms(t0)
    started = t0
    t0 = time.perf_counter()
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    fork = ms(t0)
    os.waitpid(pid, 0)
    del ballast
    builder.wait()
    built = ms(started)
    out = torch.zeros(vocab, vocab // 8, dtype=torch.uint8, device=device)
    t0 = time.perf_counter()
    builder.collect(out)
    torch.cuda.synchronize()
    collect = ms(t0)
    expected = build_canonical_mask(vocab)
    report("C2. spawned build collected to the GPU equals the inline build",
           np.array_equal(out.cpu().numpy(), expected) and not any("WARNING" in s for s in logs),
           f"start() {spawn:.2f} ms vs os.fork() {fork:.2f} ms (4 GB host + CUDA context); spawn to built "
           f"{built / 1000:.1f} s; collect {collect:.1f} ms")


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    thp = Path("/sys/kernel/mm/transparent_hugepage/enabled")
    print(f"{torch.cuda.get_device_name(device)} | torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"{os.cpu_count()} CPUs | THP {thp.read_text().strip() if thp.exists() else '?'}", flush=True)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")
    dist.init_process_group("cuda:nccl,cpu:gloo", rank=0, world_size=1, device_id=device)
    data_dir = Path(os.environ.get("SMOKE_DATA", Path(tempfile.gettempdir()) / "speedrun_smoke_shards"))
    print(f"synthetic shards in {data_dir} ...", flush=True)
    make_shards(data_dir)
    section_a_pinned_cache()
    section_b_loader(data_dir, device)
    section_c_canonical_mask(device)
    dist.destroy_process_group()
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
