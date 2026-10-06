# Rules check for this branch

The rules (README "Rules"), and how each change on this branch meets them. Every new idea is held to
the same checklist before it lands.

| | rule | this branch |
|---|---|---|
| 1 | Do not modify the train or validation data pipelines; the streams of tokens must not change | Every batch is byte-identical to upstream `4ea6b93`'s: `tools/tests/test_token_stream_identical.py` replays the whole 1194-step schedule and the validation through both loaders, on two ranks. Only the loader's mechanics change (when and on which thread a shard is read, how the BOS index is computed), as records #33 (asynchronous batch fetch) and #360 (parallel shard loader) did. One difference only shows up in a race that a paced run never reaches (batches outrunning a shard's BOS scan). The record's loader would then silently skip the rest of the shard; this branch waits for the scan and yields the intended stream. |
| 2 | Mean val <= 3.28 at p < 0.01, waived for systems-only changes | Systems-only: no file on the ML path changes (model, optimizer, kernels, CUDA graphs, schedules, sampled softmax, n-gram table, tail averaging, training manager). The canonical mask is byte-identical (`tools/tests/test_canonical_mask_spawn.py`). The A/B report still prints both arms' val statistics. |
| 3 | No extra `torch._inductor.config` or `torch.compile` flags | None added. Mechanical check: `git diff 4ea6b93 -- train_gpt.py track_1_short/ \| grep -E "_inductor\|torch.compile\|dynamo.config"` finds nothing. |
| 4 | Faster than the prior record on the same hardware | Not yet measured; needs 8xH100. `tools/speedrun_ab/ab_bench.py` runs the interleaved A/B. Baseline: master (`4ea6b93`), as recent PRs #378/#379 used. The original #360 trainer (`f9b6266`) is about 0.3 s faster than master and should be reported too. |
| D1 | Readability: the gain must justify the code | Core diff +308/-190 lines, about 120 of them the mask build moved verbatim into its own torch-free module. Each change is its own commit and can be dropped alone. |
| D2 | Do not consume the 0.001-0.002 loss buffer unless it beats a plain step cut | No ML change, so the buffer is untouched. |

## What must stay on the clock

The repo's convention is: the first shard's read, the val shard's read, the canonical-mask build, the
prefix-table build and the validation's n-gram pulls all count. On this branch they still do. Before
t0 there are only allocations and thread/process setup, the same kind the record already does before the
clock: pinned staging, the prep thread, `cudaHostRegister` of the mask buffer. That includes the mask
builder's interpreter check. The builder process itself is spawned on the clock, at step 25.

## Checklist for new ideas

1. Same token stream into training and validation, batch for batch? (Different batch size, sequence
   length or attention structure is allowed; a different tokenizer, data order, filtering or extra
   data is not.)
2. Is the evaluation still a valid probability model over the 10,485,760 val tokens? No untimed
   backward passes (the test-time-training ruling), and nothing learned from val data.
3. Is all work that the result depends on inside the timed region, warmup aside? (Warmup must run on
   the record's terms: compile and capture, then reset the state.)
4. No new compile/inductor flags.
5. If the ML changes: enough runs for p < 0.01 on mean val <= 3.28, all runs counted. Does it beat a
   plain step cut at equal loss?
6. Is the code proportionate to the gain?
