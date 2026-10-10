# Stream-only retrieval v2: a separate arm on top of the stack

At the final validation, the CPLM probability p of each val token is mixed with next-token distributions read from
memories of the tokens this run trained on, and nothing else (`STREAM_RETRIEVAL=1`, off by default). The stack's own
tree carries none of this code, so a stack record's source is only the stack; the retrieval lives here as an overlay,
and the record attempt runs it as its own arm (`streamret`).

## What it computes

A helper process (`stream_memory.c`), spawned by rank 0 before the clock, is fed by rank 0's timed loader: every
fetched step's document spans for all 8 ranks. It reads those token ranges from the shards and indexes them on the
clock, while training runs. At the last step rank 0 sends GO; before the clock stops the helper reads val and writes,
per val position, the rows of three parts, which every rank copies to its GPU:

| part | what | built | rows |
|---|---|---|---|
| P1 records (always) | a hash chain over 6-token contexts; per val position up to 32 verified candidates (next token, match length), and per match level 6, 7, 8, 10, 12, 16, 24, 32 the counts N, D, M, n1, n2 and the top token | insertion during training (1 thread) | 176 B per matched position (7.4% of val) |
| P2 low orders (`STREAM_RETRIEVAL_LOW=1`, off) | exact count tables of orders 1-5 (`stream_lowtables.c`): per order N, C(y), M, D, n1, n2, top of the context in t's segment | 8 insertion threads during training; any backlog left at GO drains while the P1 / P3 queries run, then P2's own pass | 100 B per val position |
| P3 pointer (on) | per segment, sequentially (`stream_pointer.c`): a pointer beam of alignment hypotheses over the memory, its vote, a copy from up to 8 source documents of the memory, and doc-state counters of earlier positions' outcomes | nothing during training; at GO from P1's candidates, whole segments per query block, longest first | 380 B per active position (33.4% of val) |

In the untimed eval (`StreamEval`, the same code at the FIT dump and in `fit_gate_v2.py`), a stick-breaking chain over
the orders ascending (P2's 1-5 when on, then the memory's levels), P_i = (1 - lam_i) P_{i-1} + lam_i r_i with r_i a
modified-Kneser-Ney distribution of order i's next tokens, then a softmax over [chain, pointer, vote, source copy]
where P3 has a row. Every gate reads target-independent features only: the counts, the model's entropy, max log p and
log p of the predicted tokens (the eval forward's side output, `lm_features`), the model's surprisal over the matched
context, and causal histories of EARLIER positions of the segment (hits, correct predictions, log-likelihood ratios).
Positions where nothing matched keep p bit-identical; the same weights are scored with and without the mixture in one
run (`val_loss_lm` next to `val_loss`), so every run measures its gain exactly.

The design, the validity argument and the credits are in `stream_memory.py`'s docstring; the rules argument is in
`tools/RULES_CHECK.md`.

## Flags (all off by default; the record attempt sets them per arm)

| variable | effect |
|---|---|
| `STREAM_RETRIEVAL=1` | the retrieval: helper, tap, P1 + P3 rows, the mixture at the final validation (shipped gate `GATE_V2`, the spec fitted at the run's step count) |
| `STREAM_RETRIEVAL_LOW=1` | adds P2 and uses `GATE_V2_LOW`. Off by default: ~13 GiB of host RAM for its tables (~15 GiB with its rows region and every rank's staging), allocated and touched before the clock; ~150-185 CPU-s of insertion during training on 8 threads, about 4 cores on average and 5-6 over the last ~480 steps (the tables fill, and those steps carry the largest batches), with little slack on 8 threads; at GO the backlog's drain overlaps the P1 / P3 queries, then ~0.4 us per val position and thread. That is the CPU farm the maintainers objected to on #367, so it is the user's call, after a dev run shows the step time unchanged and the late backlog small. Worth ~10 millinats on the CPU proxy (table below). |
| `STREAM_RETRIEVAL_POINTER=0` | drops P3 (an ablation; needs a gate fitted without it, `STREAM_RETRIEVAL_GATE`) |
| `STREAM_RETRIEVAL_FIT=<dir>` | dev runs: after the final validation (untimed), the helper writes the rows of the run's own last `STREAM_RETRIEVAL_FIT_K` (16) timed batches, queried against the memory as it stood before the first of them was inserted, and the run's trained steps in the dump's header; every rank writes the model's eval outputs there: `fit_gate_v2.py <dir>` fits the gate on them, as the spec for that step count. K times the final per-rank batch must be whole eval chunks (16, 32, ... on the record schedule; checked at startup). Without P2 a FIT run times exactly as a plain one; with P2 the last 16 steps' low-order insertion waits until they have all arrived (their rows are queried against the tables before them), so its GO->rows is longer: time the arm with a run without FIT |
| `STREAM_RETRIEVAL_GATE=<json>` | dev runs: the gate's constants from a `fit_gate_v2.py --out` file instead of the shipped ones |
| `STREAM_RETRIEVAL_THREADS=<n>` | the helper's query threads (default min(96, the CPUs this process may use - 16): its affinity mask, capped by a cgroup CPU quota) |

Every run picks the spec of its gate fitted at its own trained steps (the nearest one otherwise, flagged in the log:
"NOT this run's ... steps"), logs where it comes from (`Gate.describe`), and checks it against its parts before
anything is built (`Gate.for_run`: chain orders, P3's top level and its columns), so a mismatch fails at startup, not
after 35 s of training. A record attempt's preflight requires, for every cut, a spec fitted at that cut's step count on
our model, not a proxy's placeholder (`Gate.check_record`).

## Files

| file | what |
|---|---|
| `arm/track_1_short/stream_memory.py` | The Python side: the helper's handle (`StreamMemory`: spawn, tap, GO, `collect`, FIT), the rows' parsers, the gate (`Gate`, `chain_features`, `top_block`, `mix_v2`, `chunk_mix`, `StreamEval`), the FIT dump and the shipped constants `GATE_V2` / `GATE_V2_LOW`. Its docstring has the design. |
| `arm/track_1_short/stream_memory.c` | The helper: the memory, its insertion on the clock, the val and FIT queries; it `#include`s the two parts below (one binary, built by `build_helper()`, cached by the three sources' hash). |
| `arm/track_1_short/stream_lowtables.{c,py}` | P2: the exact low-order tables and their ctypes binding (tests, benchmarks) with a brute-force reference. |
| `arm/track_1_short/stream_pointer.{c,py}` | P3: the pointer beam, vote, source copy and doc-state; the Python side has the row layout, the components' gathers and the gate's feature columns. |
| `arm/hooks.patch` | The trainer's hooks: `train_gpt.py` (the gate and the side-output buffers, create, tap the timed loader, GO, collect, `StreamEval`, the FIT dump), `data.py` (`on_spans`), `model/gpt.py` (the eval forward's side output: the LM's entropy, max log p and top-32 log-probs at the gate's query tokens, only when the buffers are set), `run_log.py` (log the C sources). |
| `apply_overlay.sh DIR` | Puts the overlay on a stack tree (refuses a stale patch or a second application). |
| `make_streamret_arm.sh [WORK]` | The record attempt's `streamret` arm: a worktree of HEAD plus the overlay, one commit with a fixed identity and date, reused while HEAD and the overlay are unchanged. |
| `fit_gate_v2.py DIR` | Fits the gate (one end-to-end L-BFGS fit of every order's gate weights and KN discounts and the top softmax, as rg/combo/joint.py) on a FIT dump; `--heldout` adds a 2-fold-by-document held-out gain, `--out` writes JSON, `--module` puts them into the constants block (`GATE_V2`, or `GATE_V2_LOW` for a chain with P2's orders) of the overlay's `stream_memory.py` as the spec for the dump's step count (replacing one fitted at the same step count; a fit on the run's own model replaces every placeholder); `--proxy TEXT` marks a fit on a proxy's outputs (a record refuses it). |
| `test_stream_retrieval.py` | The arm's tests (they need the hooked tree): the memory is exactly the trained stream, no leakage into any part, the tap leaves every batch unchanged, the flags and the hooks' guards, failures, and an end-to-end run on 2 gloo ranks with P2, P3 and the FIT dump. |
| `test_stream_memory_v2.py`, `test_stream_integration.py`, `test_stream_fit_freeze.py`, `test_stream_lowtables.py`, `test_stream_pointer.py` | Standalone (the overlay loaded by path): P1's records against a brute-force walk, the gate's features against the research code, normalization and causality, the constants' selection by step count; P2 and P3 inside the helper against their references (and in FIT, and across threads and batching), the whole mixture summing to 1 over the vocabulary, the eval through the model's side output, the fit; the FIT rows of every part equal to the rows of a helper whose memory physically ends at the freeze (with a control that would catch a leak), and no state crossing a chunk start; each part's own exactness tests. `tools/tests/test_stream_retrieval_arm.py` builds a copy of the working tree with the overlay and runs all of them there, and checks that the stack carries no retrieval code. |
| `bench_memory.py` | Not a test, run from the arm: replays the timed loader of the record schedule through the tap, as rank 0 does, then GO and rank 0's collect; prints the insertion and query costs and the hit rates (`--low` for P2). |
| `bench_lowtables.py`, `stream_pointer_bench.c` | P2's and P3's own benchmarks on the real stream. |

## On the GPU node, in order (each step decides the next)

```bash
ARM=$(bash tools/stream_retrieval/make_streamret_arm.sh)   # ../record_work/streamret
cd "$ARM" && python data/cached_fineweb10B.py 9
# Add STREAM_RETRIEVAL_LOW=1 to every run below for P2 (and STREAMRET_LOW=1 to the attempt).
# G1: one 1050-step dev run with the FIT dump. Its log, after the final val_loss line: the gain with the nearest
#   shipped constants (the CPU placeholders) and the costs:
#   step:1050 stream_retrieval val_loss_lm:... val_loss_mixed:... gain:...mnat
#   stream retrieval: memory ... max lag ... GO->rows ... waited ... at collect, copied in ... [P2 tables ...]
STREAM_RETRIEVAL=1 STREAM_RETRIEVAL_FIT=$PWD/fit ./run.sh
# Fit the gate on this run's own last 16 batches and this model's outputs there (untimed, CPU, minutes):
python tools/stream_retrieval/fit_gate_v2.py $PWD/fit --heldout --out $PWD/fit/gate.json
# G2: the gain with the refit constants (the same code; only the constants change): its gain line is g.
STREAM_RETRIEVAL=1 STREAM_RETRIEVAL_GATE=$PWD/fit/gate.json ./run.sh
# The cuts from g (tools/record_attempt/README.md): safe s = 978 - 4 x (g - 3), bold s = 978 - 4 x (g - 0.5),
# rounded to 10. G3: a FIT dev run at each cut. The loader is deterministic, so its last 16 batches are the record
# legs' own last 16 batches at that step count (a 1050-step run's are batches no shorter leg ever trains on).
for s in <safe> <bold>; do NUM_SCHEDULED_ITERATIONS=$s STREAM_RETRIEVAL=1 STREAM_RETRIEVAL_FIT=$PWD/fit$s ./run.sh; done
# Hardcode one spec per cut, from the stack checkout (the overlay is the source of the arm), and commit:
cd - && for s in <safe> <bold>; do
    python tools/stream_retrieval/fit_gate_v2.py "$ARM/fit$s" --heldout --module --note "G3 dev run at $s, <date>, <node>"
done
git commit -am "stream retrieval: the gate fitted at each cut on the model's own last batches (G3)"
# The record attempt: its preflight refuses a cut without a spec fitted on our model at that step count.
STREAMRET_CUTS=<safe>,<bold> bash tools/record_attempt/run.sh
```

Decision on G2: under 10 millinats, drop it; 10-37, keep it as one part of a -5 s stack; 37 or more, -5 s alone.
Check in G2's log: step_avg within ~0.05 ms of the stack's (with P2 above all: its insertion runs on 8 threads during
training), `max lag` under ~5 steps (and P2's backlog over the last 100 steps a few steps: it drains at GO, on the
clock), `waited` at collect a few ms (the rows ready before the clock stops), `copied in` (the rows' copy to the GPU,
on the clock) and the helper's huge pages (insertion is ~2x slower without them). If `waited` is large, GO could be
sent as soon as rank 0's loader has queued the last step's spans (1-8 steps before the last step's validation),
overlapping the queries with the last training steps; not done, since it reads val while training runs and its CPU
use beside the trainer is unmeasured.

## Measured here on CPU (4 vCPUs shared with other jobs; nothing has run on a GPU)

On the real 1050-step stream (289,983,448 entries): the hooked timed loader replaying the record schedule (978 + 72
steps, 8 ranks, the real shards) into the helper, as the trainer's rank 0 does; its FIT dump is that run's own last
16 batches (`tools/stream_retrieval` CPU proof):

- Insertion (P1): ~20-23 ns per entry (6-6.5 s for the stream), against the ~119 ns per token at which training
  consumes it. P2: 150-200 CPU-s for the whole stream on 4 threads (~111 ns per position and order on average, two
  random DRAM lines per update; per token it doubles at orders 4-5 as the tables fill); 13.2 GiB of tables at the
  default sizes (orders 1-3 6.4, orders 4-5 6.9 with their index tier); this VM could not keep pace with training on
  4 vCPUs (the 8xH100 host must, which G1 shows: its stats line gives P2's backlog over the last 100 steps).
- GO to rows, P1 + P3, the record's whole val (10,485,760 positions, 8 ranks x 5 steps): 2.98-3.24 s on 4 threads,
  ~1.2 us per position per thread, ~12.5 CPU-s (4.56-4.70 s before each query thread wrote its rows into an arena
  touched before the clock instead of fresh buffers per block: the same rows, A/B on this box). P2 adds ~0.4-0.45 us
  per position and thread at the default table sizes (~4.5 CPU-s), after its backlog drains, which now overlaps the P1
  / P3 queries (here, unpaced: drained 1.37 s after GO, of which 0.56 s past the P1 queries). On 96 query threads that
  is ~0.13-0.18 s without P2 and ~0.2 s with it if it scales (estimated: linear to 70% efficiency), on the clock,
  against an existing tail of ~0.2-0.4 s (unmeasured on our stack: G1's `waited` measures the exposure). The floor is
  the longest val segment, which one thread runs whole (P3 is sequential within a segment): 31.5k positions, ~60-80 ms;
  the blocks are handed out longest first, so it starts first.
- Rows: P1 7.6% of val positions matched (8.8 candidates each), P3 active at 34.0%, P2 orders 1-5 matched at 100 / 94
  / 68 / 36 / 16%. Per rank 17 MB (P1) + 162 MB (P3) (+ 131 MB with P2), copied to the GPU on the clock: read with
  `preadv` from the rows file (whose pages the helper allocates before the clock) into pinned buffers on 8 threads,
  40-130 ms per rank here for P1 + P3 (one rank at a time, including a host copy the GPU path does not make).
- Host RAM (`stream_memory.host_bytes`, all before the clock): ~10 GiB without P2 (the memory 3.7, the query arenas
  1.7, the rows file 2.1, every rank's pinned staging 2.7), ~25 GiB with P2 (its tables 13.2, its rows region 1.0,
  its staging 1.0 more). The record attempt's preflight checks it, with an allowance for the trainers.
- The tap: packing a step's ~470 spans costs ~40-60 us, on a writer thread; the loader's thread only queues the lists
  (~2-3 us; ~13 us median in a synthetic test where the writer is packing at the same time, as they share the GIL).
- Exactness: P2's rows equal a sort-based brute force and P3's equal `stream_pointer.run` on the brute-force walk's
  candidates (toy data, `test_stream_integration.py`); on the real stream P1's records equal the research's
  deployable counts at all but 2 of 1,048,576 positions (hash collisions under the 128-visit limit), and P3's rows
  through the bench driver equal the research tools' (`test_stream_pointer.py`). The helper's own P3 rows differ from
  the research convention in one detail: its memory positions are stream index + 1 (`tok[0]` is a separator), which
  shifts the source copy's 512-entry bins, so ~13k rows' source fields and ~375 active positions differ on the first
  1,048,576 val positions; the gain is the same (24.65 millinats either way).

## The gain on the CPU proxy (the shipped placeholder constants)

The shipped specs are the CPU proof's fits by `fit_gate_v2.py`, exactly as a dev run would: on the FIT dump of the
replayed 1050-step run above (ranks 0-3: 1,048,576 positions of its own last 16 batches, queried against the memory
before them), with the llm.c GPT-2 124M proxy as the model (its NLL replaced by a CPLM-like copy mixture whose
constants are fitted on training positions; its entropy, max log p and top-32 log-probs as the side output). With P2
the run used 0.75x table sizes and 2^26 buckets (this VM's 13.4 GiB memory limit): its val rows equal the default
sizes', its FIT rows differ at 3 P1 records and 21 P3 rows. Applied unchanged (as `Gate.for_run` loads them) to the
first 1,048,576 val positions through `chunk_mix` (what `StreamEval` runs), over that base, 90% document-bootstrap
interval:

| configuration | constants | val gain | research (2-fold on val) |
|---|---|---|---|
| P1 + P3 (default) | `GATE_V2` (691) | **24.65** [19.41, 30.64] | 24.10 |
| P1 + P2 + P3 (`STREAM_RETRIEVAL_LOW=1`) | `GATE_V2_LOW` (891) | **34.91** [29.49, 40.99] | 34.16 (deployable form; 34.27 with exact orders >= 6) |
| v1 (one longest-match link, 4 constants) | | 14.60 | 15.17 |

In-sample on the FIT positions: 46.4 (P1 + P3) and 57.2 (P1 + P2 + P3) millinats; those positions sit next to the
stream, so in-sample numbers overstate val. Fitting on the run's own last batches is worth slightly more than the
research's 2-fold fit on val: the same code fitted 2-fold on val gives 23.90 and 34.09, and the own-batch fit adds
+0.76 [0.47, 1.15] and +0.81 [0.51, 1.23] (paired; it has twice as many rows to fit, as in the research's own
train-fit, +0.65). These are a proxy's numbers: the dev runs refit the constants on our model at each step count
(G1, G3) and measure the gain on it (G2).

Where it comes from (P1 + P3 on the proxy): the top 1% of val documents carry 40% of the gain and the top 5% 76%,
from val documents that share long verbatim passages with trained documents, mostly web boilerplate (v1's analysis: a
local-news site's "most read" sidebar, a Bible site's interface text, a dumpster-rental template, an Android
permission list). With P2 the gain spreads over every position: the top 1% / 5% of val documents carry 29% / 60% of
the 34.9 millinats, and P2's own increment is 10.25 [9.40, 11.11] millinats (paired). No val token is in any memory:
these passages are in the training stream itself. The maintainers question (`tools/retrieval_gate/README.md`) and the
record README say so.

## Credits

No code from another PR. The memory's layout and row rule follow PR #367's StreamIndex (Herman Brunborg,
`exact_match/src/stream.rs`): every rank's documents step by step with a STOP after each, the 6-token key, the most
recent occurrences, the match levels, the next tokens of the occurrences at least that deep, and (length, count, top
share) as features. The output mixture, CPLM's p mixed with retrieved count distributions under sigmoid gates fitted
on training positions and chained over the match orders, is PR #380's (Deven); P2, exact counts at low orders feeding
a gated chain over orders fitted on training positions, is #380's recipe too (no #380 code: P2 ports this branch's
research tools). kNN-LM (Khandelwal et al., 2020), Infini-gram (Liu et al., 2024), interpolated Kneser-Ney (Chen &
Goodman, 1998), the LZ77/zlib hash chain. This branch's part: the memories restricted to the run's own consumed stream
and used at the final validation, the hash-chain index and C helper fed from the loader's spans on the clock, the
records and their GPU-side gated chain with model-aware features, the pointer beam, vote, source copy and doc-state
(P3), the fit on the run's own last batches at each step count, and the tests. Nothing from #381.
