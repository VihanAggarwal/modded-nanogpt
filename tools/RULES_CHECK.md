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

## Candidate: Canon layers (`CANON_LAYERS`, off by default)

Canon layers (Allen-Zhu 2025, Physics of Language Models Part 4.1): a causal depthwise conv over 4 tokens, plus a
residual, on each sublayer input. Here: `track_1_short/model/gpt.py` (one parameter, `canon_layer_taps`, three small
functions, one call per site) and its optimizer entry in `track_1_short/training.py`. The proxy screen
(`tools/proxy/`, 1200 steps of 16K tokens) measured -207 millinats alone and -48 / -92 on top of a stack of record
techniques. The record already has smear, the n-gram table, value embeddings and the partial key offset, which
cover part of the same short-range mixing, so expect much less here. **Nothing below has run on a GPU.**

**Proxy round 3 says no** (`tools/proxy/README.md`, Results): on a record-like base Canon gains -24.8 ± 6.0
millinats at 1200 steps but loses +16.3 ± 7.4 at 3600 (the change, +41 ± 5, is p = 0.004). The record trains on far
more tokens than either, so the flag stays off and `sweep_canon.sh` is not worth 8xH100 time unless a form whose
gain holds at longer horizons turns up.

With the flag unset nothing changes. A one-off CPU check against the commit before the patch (`e441d6d`) found the
same parameters, RNG stream, optimizer tables (CPLM on and off), eval loss, gradients and traced eval graphs.
`tools/tests/test_canon_layers.py` covers the flag on against off, and the layers themselves. Its forward and gradient
checks run with CPLM off (the copy head has no CPU stand-ins); with CPLM on, the record's default, only the optimizer
tables are checked. The run log records the code, not the environment, so a `CANON_LAYERS` run's log reads like a
flag-off one: hard-code the chosen setting before any pool.

| | rule | Canon layers |
|---|---|---|
| 1 | Token stream | Unchanged. `data.py` is untouched; only the optional BOS mask reads the tokens, inside the forward. |
| 2 | p < 0.01 | An ML change: it needs its own pool at the chosen setting, all runs counted. `tools/speedrun_ab/sweep_canon.sh` only picks the setting. |
| 3 | No compile flags | None. The conv is plain shift-and-sum torch, compiled at the default settings (`F.conv1d` would stay an extern kernel). Each shift is a roll and a mask, which the backward recomputes; a CPU test checks that the compiled site keeps no fp32 copy of the shifted rows. How inductor fuses it on a GPU is not known yet. |
| 4 | Faster | Not measured. Expected with stock inductor, for 16 sites: ~1 s per run (~6 millinats at 164 ms each). The conv reads neighbouring rows of the norm's output, so it cannot join the norm's per-row kernel, and the taps' gradient is a sum over tokens: about 3 extra passes over the rows per site. ~1.7 s if every op got its own kernel; ~0.04 s only with hand-written kernels that fold it into the norm and quantize. Measure with `PROFILE_STEPS` (see `sweep_canon.sh`). |
| D1 | Readability | About +90 lines in the two files, a third of them comments. There are five knobs (sites, `_K`, `_NORM`, `_BOS_MASK`, `_LR_MUL`); the sweep has an arm for each but `_K`, which the proxy screens. A submission should keep only the winning setting. |
| D2 | Loss buffer | The gain has to buy a step cut, not eat the buffer: the sweep's `canonAC950` arm runs 28 fewer scheduled steps. |

Still to check on a GPU, before any pool:
- At init, the compiled fp8 training step matches the flag-off run within rounding, with CPLM on.
- Warmup capture, the self-check and the address census pass.
- The cost per step.
- Under the default `post`, per site and against flag off: max |input| and the share of entries the static fp8
  scales clamp (past 28 in MLP inputs, past 8 in attention inputs; never NaN). A filter of gain a = |1 + w0| + sum
  |wj| can clamp MLP entries once a channel holds (28 / a)^2 / 768 of its row's energy, and attention entries past
  8 / a.
  The bf16 validation forward does not clamp, so these widen the gap between training and validation.
  `CANON_LAYERS_NORM=renorm` keeps each row's RMS, so the MLP bound stays exact.

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
