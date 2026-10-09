# The stream-only retrieval gate

Exact-match retrieval is the only idea found that could take the record past -10 s. PR #367
(Herman Brunborg) reports 21.6 s, and #380 (Deven), built on it, reports 9.65 s. Both build their validation
memory from all 103 train shards, though the run trains on about 2. That is a grey zone under rule 1. A
memory of only the tokens the run trained on is the same kind of thing as the record's 84.6M-row n-gram
table. The open question is how much of the gain it keeps.

`run_gate.sh` answers that with one GPU session. It runs #367 as shipped, 3 seeds at 610 steps, with
`gate_on_367.patch`. After the normal, timed final validation, the patch re-scores the same weights,
untimed:

- **stream**: validation rows from a SlotIndex over exactly the token segments the run trained on, which
  rank 0 records from the loader's documents as batches are fetched;
- **none**: no retrieval rows.

`gate_report.py` compares those with the shipped (103-shard) line. Means are in millinats:

| | stream - shipped | none - stream |
|---|---|---|
| go | <= 100 | >= 40 |
| kill | >= 150 | or < 20 |

GO means building the stream-only memory on the CPLM stack: the path to about -10 s. KILL means about -7 s is the
realistic target without retrieval.

## How this relates to `STREAM_RETRIEVAL` (the stream-only retrieval arm)

The stream-only memory is now built on our own stack as a separate arm, behind `STREAM_RETRIEVAL=1`
(`tools/stream_retrieval/`: an overlay of `track_1_short/stream_memory.py`, `stream_memory.c` and the trainer hooks,
which `make_streamret_arm.sh` puts on the stack; `tools/RULES_CHECK.md`). It takes no code from #367, #380 or #381,
but its memory and row rule follow #367's StreamIndex (Herman Brunborg): the same stream layout (every rank's
documents step by step, a STOP after each), the 6-token key, the most recent occurrences, the deepest level reached,
and (length, count, top share) as features. Its output mixture, (1 - lam) p + lam C/N under a sigmoid gate on
(order, log2 N) fitted on training positions, is #380's (Deven), with one link instead of a chain and the fit off the
clock. What differs: no training-time hints, no model change, the memory restricted to the run's consumed stream and
used at the final validation (instead of #367's 103-shard SlotIndex or #380's corpus counts), a hash-chain index
built by a C helper on the clock.

So this gate is no longer on the critical path. It measures how much of #367's feature model survives a stream-only
memory, which says how far a stream-only memory is from #367/#380's claims. Whether `STREAM_RETRIEVAL` pays is
measured directly: every run with the flag logs `val_loss_lm` (the same weights without the mixture) next to
`val_loss`, so one 1050-step dev run gives the gain on our model, paired (design section 10, G1). The CPU proxies
bracket it at 15-32 millinats (~2-4 s); `tools/record_attempt` pilots it only with `STREAMRET_CUTS`, cuts chosen from
that measured gain (its rule 4). The gate's `stream` index (a SlotIndex over the segments rank 0 records) and
`STREAM_RETRIEVAL`'s memory hold the same tokens: every rank's trained spans of every timed step.

## Ask the maintainers first

Whether either kind of memory is allowed is the maintainers' call. Asking costs nothing. A draft for the
discussion thread or #367:

> Rules question on retrieval at validation. #367/#380 index all 103 train shards for the validation
> pass, though a run trains on ~3. Would a memory built only from the tokens the run actually trained on
> (exact-match rows or counts, built and queried on the clock, forward-only at validation) be within the
> rules, the same way the hashed n-gram table is? And is the 103-shard version allowed?
>
> For our implementation specifically: (1) the memory is filled from the loader's own document spans as
> batches are fetched, and queried for all val positions before the clock stops; (2) a helper process is
> spawned and allocates its arrays before t0 (no data touched), or we can move the spawn to t0; (3) the
> helper is ~600 lines of C (C11, pthreads); (4) the mixture's 4 gate constants are tuned on training
> batches past the run's stream (never on val). Are these acceptable?
>
> One fact to weigh: on CPU proxies the gain is concentrated in val documents that share long verbatim
> passages with documents the run trained on, mostly web boilerplate. The top 1% of val documents carry
> 30-47% of it and the top 5% 65-92%; positions matched at 32 tokens are 0.3-0.4% of val and carry 31-37%.
> The largest single contributor in the first 1M val tokens is a local-news site's "most read" sidebar
> (a 276-token run of 24+-token matches, 10% of that proxy's whole gain). No val token is in the memory;
> these passages occur in the training stream itself. (Memory layout and row rule after #367's StreamIndex;
> output mixture after #380.)
