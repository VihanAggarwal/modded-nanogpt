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

## Ask the maintainers first

Whether either kind of memory is allowed is the maintainers' call. Asking costs nothing. A draft for the
discussion thread or #367:

> Rules question on retrieval at validation. #367/#380 index all 103 train shards for the validation
> pass, though a run trains on ~2. Would a memory built only from the tokens the run actually trained on
> (exact-match rows or counts, built and queried on the clock, forward-only at validation) be within the
> rules, the same way the hashed n-gram table is? And is the 103-shard version allowed?
