# A/B timing on one 8xH100 node

Rule 4 asks that a new record be faster than the prior one on the same hardware. This runs both
interleaved (ABBA) on one node, so drift lands on both arms, and keeps every leg.

## Setup (once per node)

Follow the record README's requirements: driver >= 580, CUDA 13 runtime, `torch==2.10.0+cu128`
(not a nightly), and the FA3 kernel from the HF hub (`hf auth login`, or the offline pre-seed).

```bash
git clone https://github.com/KellerJordan/modded-nanogpt base && cd base && git checkout 4ea6b93 && cd ..
git clone <this fork> cand && cd cand && git checkout <branch> && cd ..
pip install -r cand/requirements.txt
(cd base && python data/cached_fineweb10B.py 9) && ln -s "$PWD/base/data/fineweb10B" cand/data/fineweb10B
```

## Run

```bash
python cand/tools/speedrun_ab/ab_bench.py --arm baseline=base --arm candidate=cand --legs 10 \
    --out ab_runs/$(date +%m%d_%H%M)
```

- Warm caches (the default): each arm reuses its compile caches after its first leg, about 1.5 minutes
  per leg, so 10 legs per arm take about 30 minutes. That is the right setting for a systems A/B: about 10
  pairs resolve ~0.05 s (#379 measured the record at 40.575 +/- 0.024 s, n=13).
- `--cold` gives every leg empty caches, as the ANVIL2 certification did. Each leg then takes about 8
  minutes, 7 of them compiling.
- `--dry-run` prints the plan and the preflight checks (GPU count, driver, torch build, shards).
- The ledger (`ledger.jsonl`), every leg's stdout and run log, and `report.txt` land in `--out`.
  `ab_stats.py --baseline 'ab_runs/X/baseline/*.txt' --candidate 'ab_runs/X/candidate/*.txt'` recomputes the report.

## Reading the report

- **rule 4**: the wall-time difference, with a 95% CI and a one-sided Welch p-value.
- **rule 2**: each arm's one-sided t-test of mean val < 3.28. A systems-only change is exempt, and its
  val should match the baseline's: check the two-sided val p-value.
- **interval table**: mean ms per logged 25-step interval and for the final validation. Every interval
  after the first has an sd of about 1-3 ms, so a change shows up in the intervals it touched. The
  systems patches on this branch target steps `0-25` (startup) and the `...-val` row (final validation).

## Sweeps of one checkout

With more than two arms, `ab_bench.py` reports one row per arm against the first. Arms may share a directory
and differ only by `--arm-env NAME:KEY=VALUE` (repeat it for several variables).

- `sweep_stack.sh`: the stack (systems patches + #375 + #379) against master and #379 alone, at 978, 963 and 950
  scheduled steps.
- `sweep_canon.sh`: Canon layers (`CANON_LAYERS`, `track_1_short/model/gpt.py`; off by default) against the stack
  with the flag off (`stack978`): sites A+C, A only, C only, A+C at 950 steps, and A+C with `CANON_LAYERS_NORM=renorm`,
  with 3x the taps' lr (`CANON_LAYERS_LR_MUL=3`) or with `CANON_LAYERS_BOS_MASK=1`.
  At equal steps, `d wall ms` is the layers' cost per run; `d adj ms` nets cost against the val change at 164 ms per
  millinat. The script's comments give the `PROFILE_STEPS` runs that split the cost per step by kernel. None of
  these arms has run on a GPU yet. Any arm that wins changes the ML, so it needs its own p < 0.01 pool
  (`tools/RULES_CHECK.md`).
