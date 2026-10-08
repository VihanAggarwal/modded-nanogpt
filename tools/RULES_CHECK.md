# Rules check for this branch

The rules (README "Rules"), and how this branch meets them. The branch is two layers:

- **Systems layer** (`4ea6b93..35bed11`, plus the pre-clock barrier and the shard-load fix after the merges):
  speedups that change no ML. Submitted alone on `4ea6b93`, the rule-2 waiver applies.
- **ML stack** (the merges of #375 and #379): incorporated open PRs, credited. This changes the model, the loss,
  the n-gram hashing and the schedule (`NUM_SCHEDULED_ITERATIONS=978`, 1050 trained steps). **It is not
  waived.** It needs its own pool of runs at p < 0.01 on mean val <= 3.28, all runs counted. #379's own pool
  does not cover this combination. `tools/speedrun_ab/sweep_stack.sh` gives the first numbers.

| | rule | systems layer | ML stack |
|---|---|---|---|
| 1 | Do not modify the train or validation data pipelines; the streams of tokens must not change | Every batch is byte-identical to upstream `4ea6b93`'s. `tools/tests/test_token_stream_identical.py` replays the record's schedule (1122 scheduled, 1194 trained steps) and the stack's (978, 1050) plus the validation through both loaders, on two ranks. Only the loader's mechanics change (when and on which thread a shard is read, how the BOS index is computed), as records #33 (asynchronous batch fetch) and #360 (parallel shard loader) did. One difference shows up only in a race a paced run never reaches (batches outrunning a shard's BOS scan): the record's loader would silently skip the rest of the shard, while this one waits for the scan and yields the intended stream. | Unchanged token stream. #375 normalizes ids only inside the n-gram hashes; embeddings, targets and data keep raw GPT-2 ids. #379 changes the output distribution only. |
| 2 | Mean val <= 3.28 at p < 0.01, waived for systems-only changes | No file on the ML path changes (model, optimizer, kernels, CUDA graphs, schedules, sampled softmax, n-gram table, tail averaging, training manager). The canonical mask is byte-identical (`tools/tests/test_canonical_mask_spawn.py`). | Needs its own p < 0.01 pool (see above). |
| 3 | No extra `torch._inductor.config` or `torch.compile` flags | None. Check: `git diff 4ea6b93 -- train_gpt.py track_1_short/ \| grep -E "_inductor\|torch.compile\|dynamo.config"` | None. |
| 4 | Faster than the prior record on the same hardware | Not yet measured; needs 8xH100 (`tools/speedrun_ab/ab_bench.py`, interleaved). Baseline: master (`4ea6b93`), as #378/#379 used. Also report the original #360 trainer (`f9b6266`), about 0.3 s faster than master. | Same. |
| D1 | Readability: the gain must justify the code | +318/-194 lines over upstream (net +124), about 120 of the added lines the mask build moved verbatim into its own torch-free module. | #379 brings ~750 lines of Triton plus ~30 environment knobs; maintainers will want the knobs stripped to its shipped configuration. |
| D2 | Do not consume the 0.001-0.002 loss buffer unless it beats a plain step cut | No ML change, so the buffer is untouched. | The step count must keep the buffer (>= ~2 millinats under 3.28 on the pool mean). |

## Tried and removed: Canon layers

Canon layers (Allen-Zhu 2025) were added behind a flag (commit 3ae6b37) and screened on the one-GPU proxy: their
gain on a record-like base reversed between 1200 and 3600 steps (`tools/proxy/README.md`, Results). The trainer code,
its tests and its sweep were removed again, so no off-by-default Canon code ships in the logged source.

## What must stay on the clock

The repo's convention: the first shard's read, the val shard's read, the canonical-mask build, the prefix-table
build and the validation's n-gram pulls all count. On this branch they still do.

Before t0 there are only allocations and thread or process setup, the same kind the record already does before
the clock: pinned staging, the prep thread, `cudaHostRegister` of the mask buffer, and the mask builder's
interpreter check. Every rank waits at a barrier before starting its clock. Without it, rank 0, which alone does
that mask setup and whose clock is the one reported, would start late, and the other ranks' first loader work
would fall outside its timed window. The builder process itself is spawned on the clock, at step 25.

## Open items in the incorporated PRs

- #375 builds its token-normalization map (`token_norm.py`, ~45-70 ms) at import, before the clock. The record's
  convention puts tokenizer-derived tables on the clock (the prefix table, the canonical mask); only the
  tokenizer's own download and parse happen before it. A reviewer may ask for the map to move onto the clock.
- #379's eval mixture uses `-log(p + 1e-9)`, normalized to within 5e-5 nats (disclosed in #379). Its 8192-token
  eval copy band was chosen on val, like any other eval hyperparameter.

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
