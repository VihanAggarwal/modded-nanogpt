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
- Re-running the same command with the same `--out` resumes it: legs in the ledger never run again, and a different
  plan for that `--out` is refused (`plan.json`). A leg cut short by a kill is moved to `interrupted/` and run again.
- `--arm-legs NAME=N` gives one arm its own count (the record attempt's 12 + 6 + 6 pool), `--max-crashes 2` stops
  after two crashed legs in a row of one arm (`--droppable NAME` drops that arm's remaining legs instead), and
  `--leg-timeout S` kills a hung leg's whole process group (it counts as a crash unless its log has the final val).
  Every leg's processes carry `AB_BENCH_OUT=<--out>` in their environment; whatever of a leg is still running after
  it (torchrun's workers run in sessions of their own, or the bench was `kill -9`ed) is killed before the next leg.
- `tools/record_attempt/run.sh` drives all of this for a full record attempt in one command.

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
