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

## Stream-only retrieval (`STREAM_RETRIEVAL=1`, a separate arm)

The stack's own tree (`train_gpt.py`, `track_1_short/`) carries none of this code
(`tools/tests/test_stream_retrieval_arm.py::test_the_stack_carries_no_retrieval_code`), so a stack record's source is
only the stack. The retrieval is an overlay in `tools/stream_retrieval/arm/`: `track_1_short/stream_memory.py` (the
design, in its docstring), `stream_memory.c` (the helper process) and `hooks.patch` (the trainer's hooks in
`train_gpt.py`, `data.py`, `run_log.py`). `apply_overlay.sh` puts it on a stack tree and `make_streamret_arm.sh` builds
the record attempt's `streamret` arm (HEAD plus the overlay, one commit). In that arm, at the final validation the CPLM
probability p of each val token is mixed with a next-token distribution read from an exact-match memory of the tokens
this run trained on: q = (1 - lambda) p + lambda C/N. With the flag unset the arm's trainer does exactly what the stack
does: no compile, no process, no tap, no rows, and the validation loop computes the same values
(`test_trainer_touches_the_memory_only_behind_the_flag`).

| | rule | how stream retrieval meets it |
|---|---|---|
| 1 | Train and val token streams unchanged | The tap (`on_spans` in `data.py`) reads the span lists `Shard.next_batch` already computes, after each successful call; every batch stays byte-identical (`test_the_tap_leaves_every_batch_unchanged`: the stack's whole 1050-step schedule with and without the tap, ranks 0 and 5). The memory is a verbatim copy of tokens this run trained on: every rank's (inputs + last target) of every fetched timed batch and nothing else (`test_memory_is_exactly_every_ranks_trained_tokens`, 8 ranks across shard switches); the helper reads exactly those byte ranges plus the val prefix, never an unread shard byte, never another corpus (`test_no_val_or_unread_shard_token_enters_the_memory` checks its read log). Val tokens are never inserted. Only the timed loader is tapped, never the warmup's. Unlike the record's hashed n-gram table (a learned embedding: hashed rows trained by Adam), it is a nonparametric store that returns exact continuations of 6-32-token matches, kNN-LM style; whether that is acceptable is maintainers' question 1 below, not something the n-gram table settles. |
| eval | A valid probability model of the 10,485,760 val tokens; no untimed backward; nothing learned from val | r_t(v) = (count of v among the retrieved next tokens) / N sums to 1 over the vocabulary, never puts mass on BOS, and depends on the memory and val[<= t] only; lambda_t depends on (N, M, L*) only, never on val[t + 1]. So sum_v q_t(v) = (1 - lambda) sum_v p_t(v) + lambda <= 1 (+ V x 1e-9, the slack below). CPLM's p sums to <= 1: its LM branch is renormalized over the canonical tokens (masked logits at -60 before the logsumexp, `gpt.py`), and only its copy branch's mass on masked tokens is dropped; the retrieval's lambda r mass on a masked token is dropped too, never renormalized, which only lowers the sum. The row evaluates r_t at the realised token, as a cross-entropy gather does. Forward-only. Tests: causality (randomizing val[t + 2:] leaves rows <= t bit-identical; changing val[t + 1] changes only C at t), normalisation (every possible target of one context: sum_y (a p(y) + b_y) <= 1 + 1e-6, = 1 when sum p = 1), exactness against a brute-force reference. The 1e-9 slack is #379's, disclosed the same way. |
| eval | The 4 gate constants | Hyperparameters, never fitted on val (provenance below). `STREAM_RETRIEVAL_FIT` refits them on our model from training batches past the run's stream, untimed, in a dev run only. |
| clock | Everything the result depends on is timed | Insertion runs during training, as batches are fetched (one helper core; measured 25 ns per entry on this VM, 7 s for the whole 290M-token stream, against a budget of ~119 ns per token; ~40 ns without huge pages). GO at the last step: the helper reads val (~14 ms here, the only serial part) and queries all 10,485,760 positions (3.2 CPU-s measured here, so ~0.1 s on 32 threads, estimated; the val checks run on a thread of their own meanwhile) while the existing tail runs; each rank waits for the rows and copies them to its GPU before `torch.cuda.synchronize()` stops the clock. That tail is ~0.2 s in ANVIL2's logs and ~0.4 s in #379's, and unmeasured on our stack, so some exposure is possible: the dev run's `waited ... at collect` measures it. Before t0 only: compiling the helper (`cc -O2`), spawning it, its allocation and prefault (~3.7 GB of host RAM), and mapping the 84 MB rows file on every rank, the same kind of setup as the canonical mask's buffer. The mixing arithmetic is part of the untimed eval forward, like CPLM's own mixture. A checksum of each rank's val chunks is verified after the clock stops (a check, not an input). The loader tap costs ~58 us per step on rank 0's main thread (61 ms over the 1050 steps, replayed here; `Shard.next_batch` itself is ~370 us); the dev run's step_avg against the stack's measures whether it shows. |
| 2 | p < 0.01 on mean val <= 3.28 | Needs its own pool (the record attempt's `streamret` arm with `STREAMRET_CUTS`, rule 4 of its pre-registered rule). Training is unchanged, so every run logs `val_loss_lm` (the same weights unmixed) next to `val_loss`: the gain at a step count is measured paired, in-run. |
| 3 | No compile or inductor flags | None. |
| 4 | Faster than the prior record | Unmeasured: no GPU run yet. CPU proxies put the gain at 15-32 millinats at 1050 steps, about -2 to -4 s as a step cut (design note, section 9). |
| D1 | Readability | Only in the streamret arm: `stream_memory.c` ~600 lines with comments (C11 + pthreads, no new dependency), `stream_memory.py` ~350 (about 90 of them the dev-only `STREAM_RETRIEVAL_FIT` path in C and Python), hooks ~60 lines in `train_gpt.py`, `data.py` and `run_log.py` (which then logs the C source too). A C file is new to the repo. |

**Where the gain comes from** (CPU proxies, the helper's own val features on the real 1050-step stream and the
shipped W; llm.c GPT-2 124M on the first 1M val tokens / OpenAI GPT-2 on the first 512k): from val documents that share
long verbatim passages with trained documents, mostly web boilerplate. The top 1% of val documents carry 46.7% / 29.9%
of the gain, the top 5% 91.7% / 65.4%. Positions matched at L* = 32 are 0.39% / 0.32% of val and carry 37.3% / 30.8% of
it; at 96.5% of them every retrieved continuation is the target. The largest single document (val doc 137, 886 tokens)
is a local-news site's "most read" sidebar: 276 consecutive tokens at L* >= 24, 10.2% of the llm.c proxy's whole gain.
Others among the top are a Bible site's interface text, a dumpster-rental template and an Android permission list.
No val token is in the memory: these passages are in the training stream itself. This is the fact maintainers would
weigh in question 1, so the maintainers draft and the record README state it.

Measured on CPU (`tools/stream_retrieval/bench_memory.py` on real shards 1-4 and val; GPU-side cost unmeasured): the
1050-step stream is 289,983,448 entries (286,465,921 indexed); it reaches into shard 4 (the design note said shards
1-3). 7.55% of val positions match a 6-token context. The helper's rows agree with the design-phase Rust probe on
1,048,574 of the first 1,048,576 val positions (2 differ by hash-collision walk limits).

**Gate constants** `W = (-15.8, 0.957, 7.412, 2.528)`: fitted with `tools/stream_retrieval/fit_gate.py` on held-out
training data only: the helper's own FIT rows for the 16 training batches that follow the 1050-step stream (4 ranks x
262,144 positions in shard 4, never in the memory), scored by OpenAI's GPT-2 124M (trained on WebText, never on
FineWeb). The design note's first constants had been fitted on the first 1M val tokens with a proxy model; these
replace them. On those val proxies the training-fitted constants are within 0.8 millinats of val-fitted ones (OpenAI
GPT-2 33.0 vs 32.8, llm.c 14.6 vs 15.4 millinats). The training continuation itself is more retrieval-friendly than val
(56 vs ~33 millinats for the same proxy), so a FIT run's gain is not an estimate of the val gain: `val_loss_lm -
val_loss` is.

**Credits.** No code is taken from another PR, but the design builds on two. #367 (Herman Brunborg): exact-match
retrieval from training data on this track, and its StreamIndex (`exact_match/src/stream.rs`), whose stream layout
(every rank's documents, inputs plus the last target, step by step, a STOP after each), 6-token key, most-recent
occurrences, deepest-level rule and (length, count, top share) features this memory follows. #380 (Deven): mixing
CPLM's p at the output with the retrieved count share, (1 - lam) p + lam C/N, under a sigmoid gate on (order, log2 N)
fitted on training positions (`STREAM_RETRIEVAL_FIT` is that fit moved off the clock). Also kNN-LM, Infini-gram and the
LZ77/zlib hash chain. This branch's part: restricting the memory to the run's own consumed stream and using it at the
final validation (in place of #367's 103-shard SlotIndex and #380's corpus counts), the hash-chain index and C helper
fed from the loader's spans on the clock, the single longest-match link and its 4-constant gate, the tests. Nothing
from #381.

**Questions for the maintainers** (add to the draft in `tools/retrieval_gate/README.md`):

1. Is a memory restricted to the run's own trained tokens, built and queried on the clock, acceptable, given that its
   gain comes mostly from val passages that also occur verbatim in trained documents (web boilerplate; numbers
   above)?
2. Is spawning the helper before t0 acceptable (compile and allocation only)? Otherwise it can spawn at t0 and
   prefault during the first steps, off the critical path (insertion is ~5x faster than consumption).
3. Is a C helper acceptable in the record's source?
4. Is tuning the 4 gate constants on held-out training-data continuation acceptable?

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
