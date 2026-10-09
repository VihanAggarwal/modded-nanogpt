# Stream-only retrieval: a separate arm on top of the stack

At the final validation, the CPLM probability p of each val token is mixed with a next-token distribution read from
an exact-match memory of the tokens this run trained on: q = (1 - lambda) p + lambda C/N (`STREAM_RETRIEVAL=1`). The
stack's own tree carries none of this code, so a stack record's source is only the stack. The retrieval lives here as
an overlay:

| file | what |
|---|---|
| `arm/track_1_short/stream_memory.py` | The Python side; its docstring has the design, the rules argument and the credits. |
| `arm/track_1_short/stream_memory.c` | The helper process: the memory, its insertion on the clock, the val queries. |
| `arm/hooks.patch` | The trainer's hooks: `train_gpt.py` (create, tap the timed loader, GO, collect, mix), `data.py` (`on_spans`), `run_log.py` (log the C source). |
| `apply_overlay.sh DIR` | Puts the overlay on a stack tree (refuses a stale patch or a second application). |
| `make_streamret_arm.sh [WORK]` | The record attempt's `streamret` arm: a worktree of HEAD plus the overlay, one commit with a fixed identity and date, reused while HEAD and the overlay are unchanged. |
| `test_stream_retrieval.py` | The CPU tests (26). They need the arm's tree; `tools/tests/test_stream_retrieval_arm.py` builds a copy of the working tree with the overlay and runs them there, and checks that the stack carries no retrieval code. |
| `bench_memory.py` | Not a test, run from the arm. Replays the timed loader of the record schedule through the tap, as rank 0 does, then GO; prints the helper's insertion ns per entry and query ns per position per thread and the hit rates. `--features`/`--dump` write the val features and the stream. |
| `fit_gate.py` | Fits the gate's 4 constants on a `STREAM_RETRIEVAL_FIT` dump (training batches past the run's stream, never val) and prints the line for the overlay's `stream_memory.py`. The gain it prints is on that continuation, which is more retrieval-friendly than val (56 vs ~33 millinats for the same CPU proxy): it is not an estimate of the val gain. |

The rules argument is in `tools/RULES_CHECK.md`. The record attempt pilots the arm only when given `STREAMRET_CUTS`
(`tools/record_attempt`, rule 4).

## On the GPU node, in order (each step decides the next)

```bash
ARM=$(bash tools/stream_retrieval/make_streamret_arm.sh)   # ../record_work/streamret
cd "$ARM" && python data/cached_fineweb10B.py 9
# G0 + G1: one 1050-step dev run. Its log has, after the final val_loss line:
#   step:1050 stream_retrieval val_loss_lm:... val_loss_mixed:... gain:...mnat   (the gain, paired, same weights)
#   stream retrieval: memory ... max lag ... GO->rows ... waited ... at collect ...
STREAM_RETRIEVAL=1 ./run.sh
# G1b: refit the gate on our model (untimed, after the final validation: 64 more training batches)
STREAM_RETRIEVAL=1 STREAM_RETRIEVAL_FIT=$PWD/fit ./run.sh
python tools/stream_retrieval/fit_gate.py $PWD/fit --world 8 --chunk 262144
# G2: the record attempt, from the stack checkout, with cuts from G1's gain g (tools/record_attempt/README.md):
#   safe s = 978 - 4 x (g - 3), bold s = 978 - 4 x (g - 0.5), rounded to 10
STREAMRET_CUTS=<safe>,<bold> bash tools/record_attempt/run.sh
```

Decision on G1 (design section 10): under 10 millinats, drop it; 10-37, keep it as one part of a -5 s stack;
37 or more, -5 s alone. Check in the same log: step_avg within ~0.05 ms of the stack's, `max lag` under ~5 steps,
`waited` at collect a few ms (the rows ready before the clock stops).

The shipped constants came from the helper's FIT path on CPU: the 16 training batches after the 1050-step stream,
scored by OpenAI's GPT-2 124M (never trained on FineWeb): `W = (-15.8, 0.957, 7.412, 2.528)`.

## Measured here on CPU (4 vCPUs, real shards 1-4 and val; nothing has run on a GPU)

- The 1050-step stream: 289,983,448 entries, 286,465,921 indexed; it reaches into shard 4. Insertion 25 ns per entry
  (7.1 s for the whole stream; ~38 ns per entry without huge pages, still ~3x under the ~119 ns per token at which
  training consumes it). 7.55% of val positions matched.
- GO to rows: 807 ms on 4 query threads. Of that, only the val read is serial (13.5 ms); the queries take ~300 ns per
  position per thread (3.2 CPU-s, so ~0.1 s on 32 threads), and the val checks (separator scan, chunk checksums) run
  on a thread of their own meanwhile. Rows bit-identical to the previous helper's.
- The loader tap: ~58 us per step on rank 0's main thread (61 ms over the run), against ~370 us for `Shard.next_batch`.

## Where the gain comes from

The helper's own val features on the real 1050-step stream with the shipped W, scored by two CPU proxies (llm.c's
GPT-2 124M on the first 1M val tokens / OpenAI GPT-2 on the first 512k):

| | llm.c proxy | OpenAI GPT-2 |
|---|---|---|
| top 1% of val documents | 46.7% of the gain | 29.9% |
| top 5% of val documents | 91.7% | 65.4% |
| positions at L* = 32 (0.39% / 0.32% of val) | 37.3% | 30.8% |
| of those, every retrieved continuation is the target | 96.5% | 96.9% |

The top documents are web boilerplate that also occurs in trained documents: a local-news site's "most read" sidebar
(val doc 137: 276 consecutive tokens at L* >= 24, 10.2% of the llm.c proxy's whole gain), a Bible site's interface
text, a dumpster-rental template, an Android permission list. No val token is in the memory: these passages are in
the training stream itself. The maintainers question (`tools/retrieval_gate/README.md`) and the record README say so.

## Credits

No code from another PR. The memory's layout and row rule follow PR #367's StreamIndex (Herman Brunborg,
`exact_match/src/stream.rs`): every rank's documents step by step with a STOP after each, the 6-token key, the most
recent occurrences, the deepest level reached, the row from the next tokens of the occurrences at least that deep, and
(length, count, top share) as features. The output mixture, (1 - lam) p + lam C/N under a sigmoid gate on
(order, log2 N) fitted on training positions, is PR #380's (Deven), here with one link and the fit off the clock.
kNN-LM (Khandelwal et al., 2020), Infini-gram (Liu et al., 2024), the LZ77/zlib hash chain. This branch's part: the
memory restricted to the run's consumed stream and used at the final validation, the hash-chain index and C helper fed
from the loader's spans on the clock, the single longest-match link with its 4-constant gate, the tests. Nothing from
#381.
