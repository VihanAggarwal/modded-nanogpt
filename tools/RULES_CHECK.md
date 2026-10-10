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

## Stream-only retrieval (`STREAM_RETRIEVAL=1`, a separate arm; v2)

The stack's own tree (`train_gpt.py`, `track_1_short/`) carries none of this code
(`tools/tests/test_stream_retrieval_arm.py::test_the_stack_carries_no_retrieval_code`), so a stack record's source is
only the stack. The retrieval is an overlay in `tools/stream_retrieval/arm/`: `track_1_short/stream_memory.py` (the
design, in its docstring), the helper `stream_memory.c` with its parts `stream_lowtables.c` (P2) and
`stream_pointer.c` (P3), and `hooks.patch` (the trainer's hooks in `train_gpt.py`, `data.py`, `model/gpt.py`,
`run_log.py`). `apply_overlay.sh` puts it on a stack tree and `make_streamret_arm.sh` builds the record attempt's
`streamret` arm (HEAD plus the overlay, one commit). In that arm, at the final validation the CPLM probability p of
each val token is mixed with next-token distributions from memories of the tokens this run trained on: P1, the
hash-chain memory's match records (levels 6-32); P2 (`STREAM_RETRIEVAL_LOW=1`, off by default), exact count tables of
orders 1-5; P3, a pointer beam over the memory, its vote and a copy from source documents of the memory. A
stick-breaking chain of gated Kneser-Ney components over the orders and a softmax over [chain, pointer, vote, source
copy] mix them with p (`tools/stream_retrieval/README.md`). With the flag unset the arm's trainer does exactly what the
stack does: no compile, no process, no tap, no rows, no side-output buffers, and the validation computes the same
values (`test_trainer_touches_the_memory_only_behind_the_flag`,
`test_the_models_eval_forward_reports_lm_features_only_with_the_buffers`).

| | rule | how stream retrieval meets it |
|---|---|---|
| 1 | Train and val token streams unchanged | The tap (`on_spans` in `data.py`) reads the span lists `Shard.next_batch` already computes, after each successful call; every batch stays byte-identical (`test_the_tap_leaves_every_batch_unchanged`: the stack's whole 1050-step schedule with and without the tap, ranks 0 and 5). The memory is a verbatim copy of tokens this run trained on: every rank's (inputs + last target) of every fetched timed batch and nothing else (`test_memory_is_exactly_every_ranks_trained_tokens`, 8 ranks across shard switches). P2's tables are filled from the same appended tokens, P3 reads only the memory; the helper reads exactly those byte ranges plus the val prefix, never an unread shard byte, never another corpus, and with val sharing no token with the memory no part has a row (`test_no_val_or_unread_shard_token_enters_any_part` checks its read log with P2 and P3 on). Val tokens are never inserted. Only the timed loader is tapped, never the warmup's. Unlike the record's hashed n-gram table (a learned embedding: hashed rows trained by Adam), these are nonparametric stores that return exact continuations and exact counts, kNN-LM / n-gram style; whether that is acceptable is maintainers' question 1 below. |
| eval | A valid probability model of the 10,485,760 val tokens; no untimed backward; nothing learned from val | Every component sums to <= 1 over the vocabulary: each KN component to exactly 1 (sum_v max(C_v - d(C_v), 0) is its denominator, 0 < d1 <= d2 <= d3 < 1), the pointer is one token, the vote's shares sum to 1, the source copy's counts sum to N (less where its distribution is truncated); none puts mass on BOS or the separator. Each is a function of the memory, val[<= t] and the outcomes of EARLIER positions of t's segment (P3's doc-state reads t's target only after row t is written). Every gate (the chain's sigmoids, the top softmax) reads only target-independent features: counts, the model's entropy / max log p / log p of the predicted tokens at t, the model's NLL of positions < t, causal histories over positions < t. Stick-breaking and the softmax are convex combinations, so sum_v q_t(v) = sum_v p_t(v) <= 1 (+ V x 1e-9, the slack below); CPLM's p sums to <= 1, and retrieved mass on a masked token is dropped, never renormalized. The eval reads q at the realised token, as a cross-entropy gather does: C_L(y) from P1's candidate list on the GPU, P2's C(y) (the one target-dependent field, computed by the helper at the realised token because the tables are too large to ship; `test_no_part_reads_the_target_before_its_row`: changing val[t + 1] changes only C(y) at t), the pointer's / vote's / source's share of y. Forward-only. Tests: brute-force exactness of P1, P2 and P3, causality (randomizing val[t + 1:] leaves every row <= t bit-identical but P2's C at t; the gate features at t do not move with t's target or NLL), normalization (`test_the_mixture_sums_to_one_over_the_vocabulary`: the whole mixture, chain over P2's orders and the memory's levels plus the top softmax, enumerated over every possible target, sums to 1 within 1e-6). The 1e-9 slack is #379's, disclosed the same way. |
| eval | The gate's constants (`GATE_V2` 691 fitted, `GATE_V2_LOW` 891, plus their features' standardization; per step count) | Never fitted on val, and never on data the run does not consume: `STREAM_RETRIEVAL_FIT` dumps the helper's rows of the run's OWN last 16 timed batches, queried against the memory as it stood before the first of them was inserted (P1's walk skips newer entries, P3 reads only below that point, P2 holds those steps' table insertions until their rows are queried; `test_stream_fit_freeze.py` checks every part against a helper whose memory physically ends at the freeze, with a control that shows a leak would change the rows), and the model's eval outputs there; `fit_gate_v2.py` fits on that dump (dev runs, untimed). The constants are per step count: the dump's header records the run's trained steps, each constants block holds one spec per step count, and a run uses the one fitted at its own. The loader is deterministic, so a FIT dev run at a record leg's `NUM_SCHEDULED_ITERATIONS` has exactly the leg's own last 16 batches (a 1050-step run's last batches are tokens no shorter leg trains on: the batch sizes and document cuts differ); the record attempt's preflight refuses a cut without a spec fitted on our model at that step count (`Gate.check_record`), and the record README quotes the spec's provenance. Not the leg's own: the model outputs, which come from the dev run's model (the same code and step count, another run). The shipped specs are the CPU proof's, fitted at 1050 steps on the replayed run's own last 16 batches with the llm.c GPT-2 124M proxy's outputs and marked as a proxy's placeholder. v1's constants were fitted on 16 training batches past the run's stream, which the run never consumes; the reviews asked for this change. On the proxy, fitting on the run's own last batches gives 24.65 millinats on val (P1 + P3; 34.91 with P2) against 23.90 (34.09) for a 2-fold fit on val itself: it has twice the rows to fit. Whether ~1k constants fitted on the run's own data count as hyperparameters is question 4 below. |
| clock | Everything the result depends on is timed | Insertion runs during training, as batches are fetched: P1 on one helper thread (~20-23 ns per entry measured here, 6-6.5 s for the whole 290M-token stream, against ~119 ns per token of training); P2 on 8 threads (~111 ns per position and order on average: ~150-185 CPU-s, ~4 cores on average and 5-6 over the last ~480 steps, as its tables fill; this 4-vCPU VM could not keep pace, so G1 must show the step time unchanged and P2's backlog over the last 100 steps small). GO at the last step: the helper reads val (the only serial part) and queries all 10,485,760 positions with P1 + P3 (2.98-3.24 s on 4 threads measured here, ~1.2 us per position per thread, ~12.5 CPU-s) while P2's insertion threads finish (every step's block in, or an error), then P2's rows (~0.4 us per position and thread, ~4.5 CPU-s), on up to 96 threads while the existing tail runs: ~0.13-0.18 s without P2 and ~0.2 s with it, estimated, plus any P2 backlog the P1 / P3 queries do not hide, against a tail of ~0.2-0.4 s (unmeasured on our stack; the dev run's `waited ... at collect` measures the exposure). The longest val segment (31.5k positions, ~60-80 ms on one thread) is a floor; the blocks are handed out longest first. Each rank then copies its rows to its GPU (P1 17 MB, P3 162 MB, P2 131 MB per rank; 40-130 ms per rank here for P1 + P3) before `torch.cuda.synchronize()` stops the clock. Before t0 only: compiling the helper (`cc -O2`), spawning it, its allocation and prefault (~5.3 GB of host RAM with the query arenas the rows are written into, ~13 GiB more for P2's tables, and the rows file's pages in /dev/shm; ~10 GiB in all without P2, ~25 GiB with it, `stream_memory.host_bytes`), pinned staging buffers, mapping the rows file on every rank, and the model's side-output buffers: the same kind of setup as the canonical mask's buffer. The mixing arithmetic, including the LM features of the eval forward, is part of the untimed eval, like CPLM's own mixture. A checksum of each rank's val chunks is verified after the clock stops (a check, not an input). The loader tap only queues the spans (~2-13 us on rank 0's main thread; packing them, ~40-60 us per step, runs on a writer thread); the dev run's step_avg against the stack's measures whether it shows. |
| 2 | p < 0.01 on mean val <= 3.28 | Needs its own pool (the record attempt's `streamret` arm with `STREAMRET_CUTS`, rule 4 of its pre-registered rule; `STREAMRET_LOW=1` for P2). Training is unchanged, so every run logs `val_loss_lm` (the same weights unmixed) next to `val_loss`: the gain at a step count is measured paired, in-run. |
| 3 | No compile or inductor flags | None. The eval forward's side output is traced by the existing `torch.compile(fullgraph=True)`. |
| 4 | Faster than the prior record | Unmeasured: no GPU run yet. On the CPU proxy with the shipped placeholder constants: 24.65 millinats on val (P1 + P3) and 34.91 with P2 (+10.25 paired), about -3.4 s and -4.8 s as step cuts at ~0.137 s per millinat, before the on-clock cost above. |
| D1 | Readability | Only in the streamret arm: three C files (~3,500 lines with comments; C11 + pthreads, no new dependency), `stream_memory.py` ~1,900 lines (about 390 of them the two shipped constants blocks, which grow by ~200 lines per fitted step count; 400 the gate's
features), two Python bindings (~680 lines), hooks ~100 lines in `train_gpt.py`, `data.py`, `model/gpt.py` and `run_log.py` (which then logs the C sources too): ~6,100 lines against the stack's ~10,000. A C file is new to the repo. Some of it serves only the parts' own exactness tests and benchmarks (the ctypes bindings and brute-force references in `stream_lowtables.py` / `stream_pointer.py`, `sp_run` and the bench entry points of `stream_pointer.c`, `lt_query_rows` / `lt_census` / `lt_digest` of `stream_lowtables.c`); moving it out of `track_1_short` would shrink the logged source by ~700 lines without changing behaviour (not done yet). |

**Where the gain comes from** (the CPU proxy, the first 1,048,576 val positions): from val documents that share long
verbatim passages with trained documents, mostly web boilerplate. With P1 + P3 the top 1% of val documents carry 40%
of the gain and the top 5% 76%; with P2 (34.9 millinats) 29% and 60%, since the exact low orders touch every
position. In v1's analysis (one longest-match link) the largest single document (val doc
137, 886 tokens) was a local-news site's "most read" sidebar: 276 consecutive tokens at L* >= 24, 10.2% of the
proxy's whole gain; others among the top were a Bible site's interface text, a dumpster-rental template and an
Android permission list. No val token is in any memory: these passages are in the training stream itself. This is the
fact maintainers would weigh in question 1, so the maintainers draft and the record README state it. The research
also found that about a third of the gain exists because the trained stream is the shard next to val in file order
(near-duplicate density, `rg_check1`), which a maintainer may weigh too.

**Credits.** No code is taken from another PR, but the design builds on two. #367 (Herman Brunborg): exact-match
retrieval from training data on this track, and its StreamIndex (`exact_match/src/stream.rs`), whose stream layout
(every rank's documents, inputs plus the last target, step by step, a STOP after each), 6-token key, most-recent
occurrences, match levels and (length, count, top share) features this memory follows. #380 (Deven): mixing CPLM's p
at the output with retrieved count distributions under sigmoid gates fitted on training positions, chained over the
match orders; P2 (exact counts at low orders, a gated chain over orders, fitted on training positions) is #380's
recipe, credited as such (no #380 code: P2 ports this branch's research tools), and was the user's call to include
behind its own flag. Also kNN-LM, Infini-gram, interpolated Kneser-Ney and the LZ77/zlib hash chain. This branch's
part: restricting the memories to the run's own consumed stream and using them at the final validation (in place of
#367's 103-shard SlotIndex and #380's corpus counts), the hash-chain index and C helper fed from the loader's spans on
the clock, the records and their GPU-side gated chain with model-aware features, the pointer beam / vote / source copy
/ doc-state, the fit on the run's own last batches, the tests. Nothing from #381.

**Questions for the maintainers** (add to the draft in `tools/retrieval_gate/README.md`):

1. Are memories restricted to the run's own trained tokens, built and queried on the clock, acceptable, given that
   most of their gain comes from val passages that also occur verbatim in trained documents (web boilerplate;
   numbers above)?
2. Is spawning the helper before t0 acceptable (compile and allocation only)? Otherwise it can spawn at t0 and
   prefault during the first steps, off the critical path.
3. Is a C helper acceptable in the record's source, and with `STREAM_RETRIEVAL_LOW=1` its ~13 GiB of tables (~15 GiB
   with its rows and staging) and ~4-6 cores of insertion during training (the "run-away CPU farm" concern raised on
   #367)? P2 is off by default for this reason.
4. Is fitting the gate's ~1k constants on the run's own consumed batches (dev runs at the record's step count, untimed,
   never val; the model outputs are the dev run's) acceptable as hyperparameter tuning? The fallback is to fit them on
   the clock, as #380 does.

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
