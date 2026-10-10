"""Benchmark of the exact low-order tables (STREAM_RETRIEVAL_LOW=1, track_1_short/stream_lowtables.c) on a real stream.
Not a test: it replays a trained stream (u16 tokens, SEP 0xFFFF after every span, e.g. a STREAM_RETRIEVAL dump=) in
step blocks through the tables, as the helper does, then queries the val shard's first positions, as at GO.

    python tools/stream_retrieval/bench_lowtables.py STREAM.u16 fineweb_val_000000.bin [--groups 1,2,3:4:5]
        [--steps 1050] [--tokens N] [--threads 8] [--pace-ns 0] [--query 10485760] [--query-threads 32]
        [--check-tokens 0] [--tier-from 4] [--json out.json]

Per group of orders (one table set at a time, so a box with less RAM than the full set can measure every order):
the tables' bytes, the create time (allocation and the owners' first touch, before the clock), the insertion's wall
time, its thread-busy and thread-CPU time per position and order, the largest backlog when blocks arrive at
--pace-ns per token (the trainer's rate; 0: as fast as possible), the finish latency after the last block (what GO
waits for), every order's contexts, pair entries and fullest partition, and the query time per val position.
--check-tokens N: also checks the rows of the first 200k val positions against brute_force_rows on the stream's
first N tokens (exact equality, every field)."""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "arm/track_1_short"))
import stream_lowtables as L  # noqa: E402


def vm(key):
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    return 0


def step_blocks(stream, steps):
    seps = np.flatnonzero(stream == L.SEP)
    targets = np.linspace(0, stream.size, steps + 1)[1:-1]
    cuts = np.unique(seps[np.minimum(np.searchsorted(seps, targets), seps.size - 1)] + 1)
    return np.split(stream, cuts)


def run_group(orders, blocks, positions, val, args):
    out = {"orders": list(orders)}
    rss0 = vm("VmRSS")
    t0 = time.perf_counter()
    tables = L.LowTables(orders, threads=args.threads, expected_positions=positions, prefault=True,
                         tier_from=args.tier_from)
    out["create_s"] = time.perf_counter() - t0
    out["rss_tables_bytes"] = vm("VmRSS") - rss0
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    pend, ntok = 0, 0
    t0 = time.perf_counter()
    for b in blocks:
        if args.pace_ns:
            due = t0 + 1e-9 * args.pace_ns * ntok
            while (now := time.perf_counter()) < due:
                time.sleep(min(due - now, 0.002))
        tables.insert_block(b)
        ntok += b.size
        pend = max(pend, tables.stats()["pending"])
    t_last = time.perf_counter()
    tables.finish()
    t_done = time.perf_counter()
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    s = tables.stats()
    npos = sum(o["positions"] for o in s["orders"].values())
    out.update(insert_wall_s=t_done - t0, finish_after_last_block_ms=1e3 * (t_done - t_last), max_pending_blocks=pend,
               busy_s=s["busy_s"], cpu_s=s["cpu_s"], process_cpu_s=(ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime),
               position_orders=npos, busy_ns_per_position_order=1e9 * s["busy_s"] / npos,
               cpu_ns_per_position_order=1e9 * s["cpu_s"] / npos, table_bytes=s["bytes"], per_order=s["orders"])
    if args.pace_ns:
        out["pace_ns_per_token"] = args.pace_ns
    if args.query:
        x, y = val[:-1], val[1:]
        rows = np.empty((x.size, len(orders)), dtype=L.ROW)
        rows.view(np.uint8)[...] = 0  # touch the pages now: the helper's rows file is mapped before GO
        t0 = time.perf_counter()
        tables.query_rows(x, y, 262144, threads=args.query_threads, out=rows)
        tq = time.perf_counter() - t0
        out.update(query_positions=int(x.size), query_s=tq, query_threads=args.query_threads,
                   query_ns_per_position_thread=1e9 * tq * args.query_threads / x.size,
                   hit=[float((rows["N"][:, i] > 0).mean()) for i in range(len(orders))],
                   c_pos=[float((rows["C"][:, i] > 0).mean()) for i in range(len(orders))])
        del rows
    tables.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("val")
    ap.add_argument("--groups", default="1,2,3:4:5")
    ap.add_argument("--steps", type=int, default=1050, help="step blocks for the whole stream (fewer for a prefix)")
    ap.add_argument("--tokens", type=int, default=0, help="use the stream's first N tokens (whole spans)")
    ap.add_argument("--threads", type=int, default=L.THREADS)
    ap.add_argument("--pace-ns", type=float, default=0.0)
    ap.add_argument("--query", type=int, default=10485760)
    ap.add_argument("--query-threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--check-tokens", type=int, default=0)
    ap.add_argument("--tier-from", type=int, default=None, help="orders >= this have the index tier (default 4; 9: none)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    full = np.memmap(args.stream, dtype=np.uint16, mode="r")
    stream = np.array(full[:args.tokens] if args.tokens else full)
    stream = stream[:np.flatnonzero(stream == L.SEP)[-1] + 1]
    steps = max(1, round(args.steps * stream.size / full.size))
    blocks = step_blocks(stream, steps)
    positions = int(stream.size)
    val = np.fromfile(args.val, dtype=np.uint16, offset=1024, count=args.query + 1) if args.query else None
    res = {"stream_tokens": int(stream.size), "steps": len(blocks), "threads": args.threads, "groups": []}
    print(f"stream {stream.size} tokens in {len(blocks)} step blocks; {args.threads} insertion threads", flush=True)
    for g in args.groups.split(":"):
        orders = tuple(int(k) for k in g.split(","))
        r = run_group(orders, blocks, positions, val, args)
        res["groups"].append(r)
        print(f"orders {orders}: tables {r['table_bytes'] / 1e9:.2f} GB (rss +{r['rss_tables_bytes'] / 1e9:.2f}), create "
              f"{r['create_s']:.2f} s; insert wall {r['insert_wall_s']:.2f} s, busy {r['busy_s']:.1f} s, cpu {r['cpu_s']:.1f} s = "
              f"{r['cpu_ns_per_position_order']:.1f} thread-ns per position and order; max backlog {r['max_pending_blocks']} "
              f"blocks; finish {r['finish_after_last_block_ms']:.1f} ms after the last block", flush=True)
        for k, o in r["per_order"].items():
            tier = (f"index tier: {o['contexts']} contexts (fullest partition {o['idx_maxload']:.3f}), {o['promoted']} "
                    f"seen twice or more in stats slots ({o['ctx_maxload']:.3f})" if o["tiered"] else
                    f"{o['contexts']} contexts (fullest partition {o['ctx_maxload']:.3f})")
            print(f"  order {k}: {o['positions']} positions, {tier}, {o['pairs']} pair entries ({o['pair_maxload']:.3f}), "
                  f"{o['bytes'] / 1e9:.2f} GB; defaults {L.default_entries(k, positions)} / "
                  f"{L.default_entries(k, positions, 'pairs')} / {L.default_entries(k, positions, 'promoted')}", flush=True)
        if args.query:
            print(f"  query {r['query_positions']} val positions on {r['query_threads']} threads: {r['query_s']:.2f} s = "
                  f"{r['query_ns_per_position_thread']:.0f} ns per position per thread; hit {np.round(r['hit'], 4).tolist()}, "
                  f"C>0 {np.round(r['c_pos'], 4).tolist()}", flush=True)
    if args.check_tokens:
        pre = stream[:args.check_tokens]
        pre = pre[:np.flatnonzero(pre == L.SEP)[-1] + 1]
        v = np.fromfile(args.val, dtype=np.uint16, offset=1024, count=200001)
        orders = tuple(sorted({int(k) for g in args.groups.split(":") for k in g.split(",")}))
        t0 = time.perf_counter()
        with L.LowTables(orders, threads=args.threads, expected_positions=pre.size, tier_from=args.tier_from) as t:
            for b in step_blocks(pre, max(1, round(args.steps * pre.size / full.size))):
                t.insert_block(b)
            t.finish()
            rows = t.query_rows(v[:-1], v[1:], 262144)
        ref = L.brute_force_rows(pre, v[:-1], v[1:], 262144, orders)
        same = {f: bool((rows[f] == ref[f]).all()) for f in L.FIELDS}
        res["check"] = {"tokens": int(pre.size), "positions": 200000, "equal": same}
        print(f"check on the first {pre.size} stream tokens, 200000 val positions, orders {orders}: every field equal "
              f"{all(same.values())} {same} ({time.perf_counter() - t0:.0f} s)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
