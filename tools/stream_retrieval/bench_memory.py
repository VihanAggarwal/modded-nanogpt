"""Benchmark of the stream-only retrieval helper (not a test): replays the timed loader of a run's schedule through
the tap exactly as the trainer's rank 0 does (8 ranks' spans per step, in fetch order), sends GO, collects rank 0's
rows, and reports the helper's insertion cost per entry (and P2's insertion CPU and backlog), the GO->rows time and
the query cost per position per thread (P1 + P3, + P2 with --low), the hit rates and rank 0's collect time.

It needs the hooked loader, so it runs in the streamret arm (or any stack tree with apply_overlay.sh applied):
  ARM=$(bash tools/stream_retrieval/make_streamret_arm.sh)
  python $ARM/tools/stream_retrieval/bench_memory.py DATA_DIR [--scheduled 978] [--threads N] [--low] [--no-pointer]
DATA_DIR holds fineweb_train_*.bin and fineweb_val_000000.bin (real shards for meaningful hit rates). The helper needs
~10 GiB of host RAM for the whole val (the memory, its query arenas, the rows file, staging: stream_memory.host_bytes),
--low ~15 GiB more (P2's tables, ~13 GiB, and its rows).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
if not (REPO / "track_1_short/stream_memory.py").exists():
    sys.exit(f"{REPO} is the stack, without the retrieval code: run this script from the streamret arm "
             "(bash tools/stream_retrieval/make_streamret_arm.sh prints it)")
sys.path.insert(0, str(REPO))
from track_1_short import data, stream_memory  # noqa: E402
from track_1_short.config import (LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, TRAINING_STAGES, WS_POST_YARN_EXT,  # noqa: E402
                                  Hyperparameters)
from track_1_short.schedule import TrainingSchedule  # noqa: E402


class HostStaging:
    def upload(self, *tensors):
        return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("--scheduled", type=int, default=978, help="NUM_SCHEDULED_ITERATIONS")
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--low", action="store_true", help="with P2's exact low-order tables (STREAM_RETRIEVAL_LOW=1)")
    parser.add_argument("--no-pointer", action="store_true", help="without P3 (STREAM_RETRIEVAL_POINTER=0)")
    parser.add_argument("--dump", default=None, help="write the memory's stream (u16, spans and 0xFFFF separators)")
    args = parser.parse_args()
    real_empty = torch.empty
    data.torch.empty = lambda *a, pin_memory=False, **k: real_empty(*a, **k)  # no CUDA needed
    data.ngram_row_ids = lambda x: x  # the n-gram ids play no part in the stream
    dist.get_rank, dist.get_world_size = (lambda: 0), (lambda: args.world)
    hp = Hyperparameters()
    schedule = TrainingSchedule(TRAINING_STAGES, args.scheduled, hp.num_extension_iterations, device="cpu",
                                cooldown_frac=LR_COOLDOWN_FRAC, split_embed_stage=SPLIT_EMBED_STAGE,
                                ws_post_yarn_ext=WS_POST_YARN_EXT)
    sizes = [schedule.lookup(s)[0].batch_size for s in range(schedule.total_steps)]
    train = os.path.join(args.data_dir, "fineweb_train_*.bin")
    t0 = time.perf_counter()
    memory = stream_memory.StreamMemory(
        train_files=sorted(__import__("glob").glob(train)), val_file=os.path.join(args.data_dir, "fineweb_val_000000.bin"),
        total_steps=schedule.total_steps, stream_tokens=sum(b + args.world for b in sizes), world=args.world, rank=0,
        master=True, val_tokens=hp.val_tokens, chunk=hp.val_batch_size // args.world, device="cpu", threads=args.threads,
        low=args.low, pointer=not args.no_pointer, dump_path=args.dump,
        print0=lambda s, console=False: print(s))
    print(f"helper ready in {time.perf_counter() - t0:.1f} s (compile, allocation, prefault)")
    first = TRAINING_STAGES[0]
    loader = data.ScheduledBatches(data.distributed_data_generator(train, first.batch_size, first.train_max_seq_len,
                                                                   HostStaging(), on_spans=memory.on_spans),
                                   schedule, steps=range(schedule.total_steps))
    t0 = time.perf_counter()
    for step in range(schedule.total_steps):
        loader.take(step)
    loader.close()
    t_replay = time.perf_counter() - t0
    memory.go()
    memory.collect()
    h = [int(v) for v in memory.hdr[:stream_memory.H_PTR_ROW_BYTES + 1]]
    ins = h[stream_memory.H_INSERT_NS] / max(h[stream_memory.H_INSERTED], 1)
    qry = h[stream_memory.H_QUERY_NS] * args.threads / max(h[stream_memory.H_QUERIED], 1)
    print(memory.stats())
    parts = "P1" + (" + P2" if args.low else "") + ("" if args.no_pointer else " + P3")
    print(f"replay of {schedule.total_steps} steps {t_replay:.1f} s; insertion {ins:.1f} ns/entry; queries ({parts}) "
          f"{qry:.0f} ns/position/thread on {args.threads} threads; rank 0's collect {memory.collect_ms:.0f} ms")
    memory.close()


if __name__ == "__main__":
    main()
