# A track-1 record attempt in one command (one 8xH100 node)

`run.sh` takes a fresh 8xH100 SXM node to a record folder ready for a PR, with a short verdict. It runs the protocol
the merged records use: an interleaved same-node pool, a same-node baseline of the merged record (master `4ea6b93`),
the open PR this one builds on (#379) measured alongside, every leg counted, and
`scipy.stats.ttest_1samp(vals, 3.28, alternative='less')`.

## What to rent

8x H100 **SXM** 80GB (not PCIe or NVL), NVIDIA driver >= 580, >= 50 GB of disk where the clone goes (RunPod and
others size it when you rent), >= 64 GB of RAM, and an image with `git`, `tmux` and `python3` (any version from 3.10;
`run.sh` builds its own 3.11+ environment). Everything else (torch, the data, the kernel) it installs itself.

## The command

```bash
git clone -b claude/nanogpt-optimization-n49ur2 https://github.com/VihanAggarwal/modded-nanogpt
cd modded-nanogpt && tmux new -s rec 'bash tools/record_attempt/run.sh; exec bash'
```

- tmux keeps the run alive if SSH drops: reconnect and `tmux attach -t rec`. `; exec bash` keeps the window open
  after the run ends, so the VERDICT block and any error stay on screen.
- If you are already inside tmux (vast.ai opens every SSH session in tmux), run `bash tools/record_attempt/run.sh`
  directly. Without tmux: `setsid nohup bash tools/record_attempt/run.sh > run.out 2>&1 &`, then `tail -f run.out`.
- Everything printed is also saved in `../record_work/runs/console_*.log`, and the verdict in
  `../record_work/runs/verdict.txt` (`cat` it to see it again).
- `bash tools/record_attempt/run.sh --dry-run` prints the checks and the full plan (every leg in order) and installs,
  downloads and runs nothing.

## What it does

1. **Setup and preflight**, stopping with what to do at the first problem: 8 H100 SXM GPUs, driver >= 580, idle
   GPUs (no processes and no memory in use), enough free disk for `WORK`, the Inductor cache (`/tmp`) and the
   Triton and Hugging Face caches (`~`), summed per disk, >= 64 GB free RAM; upstream master still `4ea6b93` (a new
   record would make the comparison stale); a venv in `WORK` from Python 3.11+ (the stack starts its canonical-mask
   builder with `python -P`; `run.sh` picks `python3.12`, `python3` or `python3.11`, and on a node with only an
   older python, such as Ubuntu 22.04's 3.10, builds a 3.12 venv with `uv`) with `torch==2.10.0+cu128` (not a
   nightly) and `requirements.txt` (`tokenizers==0.23.2` for #375, `kernels==0.16.1`, `huggingface-hub==1.29.0`) plus
   scipy; `libcudart.so.13` for the FA3 kernel (system CUDA 13, else the `nvidia-cuda-runtime==13.4.92` wheel on
   `LD_LIBRARY_PATH`) and CUDA headers for nvrtc (`CUDA_HOME`); the data once (9 train shards + val) in `WORK/data`.
   Then `preflight.py` runs, off the books, what a leg needs: a Triton kernel compiled and launched on `cuda:0` (it
   needs gcc and `Python.h`), an NCCL all_reduce over the 8 GPUs, the pinned FA3 kernel (`devenpzak/flash-attn3-12864`
   @ `64c1e6d1`, sha256-checked as `load_flash_attn3` does), #375's token-normalization map, tiktoken's GPT-2 files,
   and the stack's canonical-mask builder; it writes `environment.txt`. Once FA3 loads from the local cache, every
   leg runs with `HF_HUB_OFFLINE=1`, so no counted leg depends on the Hub answering. No HF token is needed: if the
   Hub refuses the anonymous kernels request (the 401 in the ANVIL2 README), the cache is pre-seeded from the model
   repo. Trainer knobs inherited from the shell (`NUM_SCHEDULED_ITERATIONS`, `TRAIN_SEED`, `CPLM_*`,
   `TORCHINDUCTOR_*`, ...) are unset, so every leg runs each arm's defaults.
2. **Arms**: `master` (worktree of `4ea6b937`), `pr379` (worktree of #379's head `c44cc41e`), `stack` (this checkout:
   the systems patches + #375 + #379), and `mlonly` (#379 + #375 on master without the systems patches, built by
   `make_mlonly_arm.sh` from the same pinned heads even if an author has pushed since; if the build fails the attempt
   goes on without it and says so).
3. **Phase A, smoke**: one leg of the stack at 978 scheduled steps; if it crashes, one leg of `mlonly`.
4. **Phase B, pilot**: 3 legs per arm, interleaved: master, pr379, the candidate at 978 and at 963 scheduled steps
   (`NUM_SCHEDULED_ITERATIONS=963`), and mlonly at 978.
5. **Phase C, decision** by the rule below, written to `runs/PREREGISTRATION.txt` before the first leg.
6. **Phase D, certification**: a fresh interleaved pool, candidate n=12, master n=6, #379 n=6.
7. **Report**: `records/track_1_short/<date>_<name>/` with `README.md` (tables, deltas, changes, credits, rules
   checklist, disclosures, hardware, reproduction), `this_pr/`, `baseline/`, `baseline_pr379/`, `pilot/`, `smoke/`,
   `ledgers/`, `decision.json`, `earlier_attempts/` (if any), `statistics.py` (recomputes every number from the
   logs), and the verdict in `runs/verdict.txt`. A leg that died before writing its run log still counts: its folder
   gets a `crashed_*.txt` stand-in holding its stdout.

The pre-registered rule, as `attempt.py` writes and applies it (with the default leg counts):

```
PRE-REGISTERED DECISION RULE (fixed before the first leg; applied by tools/record_attempt/attempt.py)
1. Viable: a candidate (stack = this branch; mlonly = #379 + #375 without the systems patches) is viable
   only if every leg it ran (smoke and pilot, at both step counts) finished with a final validation.
2. Candidate: stack, unless it is not viable, or its pilot train time at 978 steps exceeds mlonly's by
   more than twice the Welch standard error of the difference; then mlonly, if viable. Neither: stop.
3. Step count: 963 scheduled steps (1035 trained) only if all 3 pilot legs of the candidate at 963
   finished with a mean final val <= 3.2765 (>= 3.5 millinats under 3.28); otherwise 978 (1050 trained).
4. Certification: a fresh interleaved pool of the candidate (n=12) with master (n=6) and
   #379 (n=6). Every leg is kept and counted, and only this pool enters the p-value; smoke and
   pilot legs are reported, never pooled.
```

If the stack fails its smoke leg, the pilot runs mlonly at 978 and 963 in its place.

## Time and cost

40 legs (1 smoke + 15 pilot + 24 certification) at ~2-3 min each with warm compile caches, 4 first-time compiles
at ~7 min (one per arm's source), ~15 min of setup (venv, ~2 GB of data, the kernel): **about 2-2.7 hours, $30-90 at
$15-32 per node-hour.** These are estimates: no leg of this trainer has been timed yet (nothing here has run on a
GPU), and a hung leg adds up to `LEG_TIMEOUT` (30 min). `run.sh` prints its estimate first, and the time of the smoke
leg once it is done. `COLD=1` (empty caches every leg, the ANVIL2 convention) adds ~7 min per leg, about 5 more hours.

## Resuming and stopping

- Re-running `bash tools/record_attempt/run.sh` resumes. A leg in a phase's ledger (`WORK/runs/<phase>/ledger.jsonl`)
  never runs again, crashed or not. A leg that was running when the attempt was killed is kept if its log reached
  the final validation; otherwise what it left goes to `interrupted/` and it runs again. Interruptions are disclosed.
  Whatever of a killed leg is still running is killed before the next leg starts.
- `runs/attempt.json` fixes the settings, the arms, the rule and a hash of every arm's source at the first start.
  A resume with any of them changed is refused.
- Two crashed legs in a row of master or #379 stop the attempt (exit 3, `runs/verdict.txt` says why): the node or
  the environment is broken. Send `send_back.tar.gz` and release the node rather than keep paying; re-running
  resumes only if the cause is fixed without touching the code. A candidate's arm that crashes twice in a row in the
  pilot only loses its remaining legs (it is no longer viable). In the certification pool, two in a row of any arm
  stop it.
- If no candidate finishes its smoke leg, the attempt is over (its smoke legs count, so a re-run stops there again).
  After fixing the node, start a new attempt, below.

## What to send back

`WORK/send_back.tar.gz` (default `../record_work/send_back.tar.gz`), written on every exit, a failed setup check
included: every attempt's ledgers, every leg's log and stdout, the console logs, the decisions, the verdicts and the
record folders. Copy it off the node before releasing it (`scp node:<path>/send_back.tar.gz .`), and paste the
VERDICT block from the end of the output (also in `runs/verdict.txt`).

## What not to do

- Do not delete, edit or re-run anything under `WORK/runs`. A crashed leg deleted from a ledger is a dropped run.
- Do not change the code, the branch or any knob (`LEGS_*`, `COLD`) between phases or before a resume. If something
  must change, start a separate attempt in a new directory directly inside `WORK`:
  `RUNS=../record_work/runs_2 bash tools/record_attempt/run.sh` (it reuses the venv, data and arms). Its record
  README discloses every other attempt in `WORK` with its outcome and copies its logs into `earlier_attempts/`, and
  `send_back.tar.gz` carries both.
- Do not run anything else on the GPUs while it runs, and do not use a torch nightly.
- Do not repeat the attempt to get a better pool. A second attempt is disclosed next to the first.
- Before the PR: if the decision is 963 steps, change the `NUM_SCHEDULED_ITERATIONS` default in `train_gpt.py` to
  963 (the README discloses that this is the only difference from the logged source), as #379 did. If mlonly was
  certified, the PR's code is `WORK/mlonly` (master + #375 + #379, no systems patches), not this branch; the verdict's
  `PR source` line says which.

## Knobs

`WORK` (default `../record_work`), `DATA`, `RUNS` (default `WORK/runs`: the attempt; must be directly inside
`WORK`), `VENV_PYTHON` (the python >= 3.11 the venv is built from), `PYTHON` (use this python >= 3.11 instead of a
venv), `LEGS_PILOT=3`, `LEGS_CERT=12`, `LEGS_CERT_BASE=6`, `COLD=1`, `MAX_CRASHES=2`, `LEG_TIMEOUT=1800` (seconds
before a hung leg is killed, compile included; it counts as a crash), `RECORD_NAME`, `ALLOW_NEW_MASTER=1`,
`ALLOW_PR_DRIFT=1` (build mlonly from the PRs' current heads instead of the pins). For tests:
`SETUP=0` (no node setup), `LEG_COMMAND`, `STACK_DIR`, `MASTER_DIR`, `PR379_DIR`, `MAKE_MLONLY`, `RECORDS_DIR`.

## Files

- `run.sh`: setup, preflight, arms, then `attempt.py`.
- `preflight.py`: the Python-side checks (Python 3.11+, pinned packages, GPUs, headers, C compiler and `Python.h`,
  a Triton kernel, NCCL over 8 GPUs, FA3, tokenizer, the canonical-mask builder) and `environment.txt`.
- `attempt.py`: phases A-D through `tools/speedrun_ab/ab_bench.py` (one `--out` per phase), the rule, the verdict.
- `make_record.py`: the record folder and its README. `record_stats.py`: the statistics (the folder's `statistics.py`).
- `make_mlonly_arm.sh`: the ML-only arm.
- `test_record_attempt.py`: CPU tests: the statistics on #379's own logs, the rule, and `run.sh` end to end with a
  fake trainer (both step counts, the mlonly fallback, a crash stop and a second attempt, a kill and resume, a leg
  that dies before its log, a failed setup check).

Nothing here has run on a GPU. The node-specific parts (the checks against `nvidia-smi`, the venv, uv and wheel
installs, the Triton, NCCL and FA3 checks, `libcudart.so.13` and `CUDA_HOME` discovery) are exercised only by
`--dry-run` on CPU.
