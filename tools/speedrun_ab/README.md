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
python cand/tools/speedrun_ab/ab_bench.py --arm baseline=base --arm candidate=cand --legs 10 --cold \
    --out ab_runs/$(date +%m%d_%H%M)
```

- Each leg is one full `torchrun` run of about 8 minutes cold (7 minutes of compile), so 10 legs per arm take about 2.7 hours.
  Without `--cold` the arms reuse their compile caches and a leg takes about 1.5 minutes.
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
