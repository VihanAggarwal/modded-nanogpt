"""Interleaved A/B timing runs of two speedrun checkouts on one 8xH100 node (rule 4: same hardware).

Each arm is a directory holding a train_gpt.py (a git worktree of the baseline record, one of the
candidate). Legs alternate in an ABBA pattern so slow drift of the node (thermals, a noisy neighbour,
page cache) lands on both arms equally. Every leg is kept, crashed or not: all runs count.

  python tools/speedrun_ab/ab_bench.py --arm baseline=../base --arm candidate=. --legs 10 --cold \
      --out ab_runs/$(date +%m%d_%H%M)

--cold gives every leg empty Inductor/Triton caches (the ANVIL2 certification convention), so the
untimed compile reruns each leg; without it legs share the warm caches of their arm.
The final report is tools/speedrun_ab/ab_stats.py over the two arms' logs (also runnable alone).

Resuming: re-running the same command with the same --out continues the ledger. Legs already in it are never
run again; a leg that was running when the bench stopped is kept if its log reached the final validation, and
otherwise moved to interrupted/ (listed in interrupted.jsonl) and run again. A different plan (arms, order,
environment, command) for an --out that holds a ledger is refused: its legs count, so they are not re-planned.
--max-crashes N stops the bench after N crashed legs in a row of one arm (an arm named by --droppable instead
loses its remaining legs and the bench goes on). --leg-timeout kills a leg that hangs (it counts as a crash).
Each leg runs in its own process group, which is killed if the bench is interrupted or terminated. Every leg's
processes carry AB_BENCH_OUT=<--out> in their environment: whatever of a leg escaped its group (torchrun starts its
workers in sessions of their own), or outlived a bench that was SIGKILLed, is killed after the leg and when the bench
resumes, so an orphan never shares the GPUs with the next leg.
"""
import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_stats  # noqa: E402

LOG_PATH_RE = re.compile(r"^(logs/[0-9a-f-]+\.txt)\s*$")
DEFAULT_COMMAND = "torchrun --standalone --nproc_per_node={nproc} train_gpt.py"
MARKER = "AB_BENCH_OUT"   # in every leg process's environment: the --out it belongs to
STOP_GRACE_S = 45         # SIGTERM to SIGKILL; torchrun gives its workers 30 s after a SIGTERM before it kills them


def leg_order(arms: list[str], legs: int | dict[str, int]) -> list[str]:
    """ABBA ABBA ... for two arms (round-robin, alternating direction, for more): `legs` runs per arm.

    With a count per arm (a dict), the legs come in gcd(counts) rounds. In each round an arm with k legs per
    round appears k times, spread evenly over the round; the direction alternates as above.
    """
    counts = legs if isinstance(legs, dict) else dict.fromkeys(arms, legs)
    rounds = math.gcd(*(counts[a] for a in arms))
    if rounds == 0:
        return []
    one_round = [a for _, _, a in sorted(((j + 0.5) / (counts[a] // rounds), i, a)
                                          for i, a in enumerate(arms) for j in range(counts[a] // rounds))]
    order = []
    for i in range(rounds):
        order += one_round if i % 2 == 0 else one_round[::-1]
    return order


def preflight(arm_dirs: dict[str, Path], data_path: str | None) -> list[str]:
    """Advisory checks for what the record README says bites: GPU count, driver, torch build, data."""
    notes = []
    smi = shutil.which("nvidia-smi")
    if smi is None:
        notes.append("nvidia-smi not found: is this a GPU node?")
    else:
        out = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        if len(out) != 8 or not all("H100" in line for line in out):
            notes.append(f"expected 8 H100s, nvidia-smi lists {out}")
        drivers = {line.split(",")[-1].strip() for line in out}
        if any(int(d.split(".")[0]) < 580 for d in drivers if d.split(".")[0].isdigit()):
            notes.append(f"driver {drivers} < 580: the record README measured a uniform ~35% slowdown below 580")
    try:
        import torch
        if torch.__version__ != "2.10.0+cu128":
            notes.append(f"torch {torch.__version__}, the record pins 2.10.0+cu128 (nightlies produced NaNs)")
    except ImportError:
        notes.append("torch is not importable from this python")
    for name, arm in arm_dirs.items():
        base = Path(data_path) if data_path else arm
        shards = sorted((base / "data/fineweb10B").glob("fineweb_*_*.bin"))
        if not any("val" in s.name for s in shards) or sum("train" in s.name for s in shards) < 4:
            notes.append(f"{name}: expected the val shard and >= 4 train shards under {base / 'data/fineweb10B'} "
                         "(python data/cached_fineweb10B.py 9)")
    return notes


def read_ledger(path: Path) -> list[dict]:
    """The legs a ledger holds, in the order they finished."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def logged_path(stdout_path: Path) -> str | None:
    """The run log the trainer announced on its stdout (logs/<uuid>.txt, relative to the arm)."""
    with open(stdout_path, errors="replace") as f:
        return next((m.group(1) for line in f if (m := LOG_PATH_RE.match(line))), None)


def salvage(index: int, arm: str, arm_dir: Path, out: Path) -> dict | None:
    """A leg with a stdout but no ledger entry was running when the bench stopped. If its log reached the final
    validation the run finished: keep it (`recovered`). Otherwise move what it left to interrupted/, note it in
    interrupted.jsonl and return None, so the leg runs again."""
    stdout_path = out / arm / f"leg{index:03d}.stdout"
    if not stdout_path.exists():
        return None
    log_rel = logged_path(stdout_path)
    log = arm_dir / log_rel if log_rel is not None and (arm_dir / log_rel).exists() else None
    run = ab_stats.parse_log(str(log)) if log is not None else None
    if run is not None:
        kept = out / arm / log.name
        shutil.copy(log, kept)
        return dict(leg=index, arm=arm, exit_code=None, started=None, seconds=None, stdout=str(stdout_path),
                    log=str(kept), wall_ms=run.wall_ms, val_loss=run.val_loss, recovered=True)
    dest = out / "interrupted"
    dest.mkdir(exist_ok=True)
    tag = f"{arm}_leg{index:03d}_{int(time.time())}"
    note = dict(leg=index, arm=arm, stdout=str(dest / f"{tag}.stdout"), log=None, found=time.time())
    shutil.move(stdout_path, note["stdout"])
    if log is not None:
        note["log"] = str(dest / f"{tag}_{log.name}")
        shutil.copy(log, note["log"])
    with open(out / "interrupted.jsonl", "a") as f:
        f.write(json.dumps(note) + "\n")
    print(f"leg {index:3d} {arm:>10}: was interrupted before its final validation; kept in {dest}, running it again")
    return None


def terminate(signum, frame):
    """SIGTERM/SIGHUP: leave through run_leg's cleanup (once; repeats are ignored while it runs)."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    sys.exit(128 + signum)


def stop_group(proc: subprocess.Popen):
    """End a leg's process group (torchrun and its workers): SIGTERM, then SIGKILL after a grace period."""
    for sig, grace in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=grace)
            if sig == signal.SIGTERM:
                os.killpg(proc.pid, signal.SIGKILL)  # stragglers of the group, if any
        except ProcessLookupError:
            break
        except subprocess.TimeoutExpired:
            continue
        break


def kill_marked(out: Path) -> list[int]:
    """SIGKILL every process whose environment marks it as a leg of `out` (Linux /proc); return their pids."""
    marker = f"{MARKER}={out}".encode()
    killed = []
    for environ in Path("/proc").glob("[0-9]*/environ"):
        pid = int(environ.parent.name)
        try:
            if pid != os.getpid() and marker in environ.read_bytes().split(b"\0"):
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
        except (OSError, ValueError):  # gone, or another user's
            continue
    return killed


def run_leg(index: int, arm: str, arm_dir: Path, command: str, env: dict[str, str], out: Path, cold: bool,
            timeout: float | None = None) -> dict:
    leg_env = {**os.environ, **env, MARKER: str(out)}
    cache_root = None
    if cold:
        cache_root = out / "caches" / f"leg{index:03d}"
        shutil.rmtree(cache_root, ignore_errors=True)  # a leg run again after an interruption starts cold too
        leg_env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor")
        leg_env["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    stdout_path = out / arm / f"leg{index:03d}.stdout"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    timed_out = False
    with open(stdout_path, "w") as sink:
        proc = subprocess.Popen(shlex.split(command), cwd=arm_dir, env=leg_env, stdout=sink, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:  # also on an interrupt: no leg outlives the bench
            stop_group(proc)
            kill_marked(out)
    record = dict(leg=index, arm=arm, exit_code=proc.returncode, started=started, seconds=round(time.time() - started, 1),
                  stdout=str(stdout_path), log=None, wall_ms=None, val_loss=None)
    if timed_out:
        record["timed_out"] = True
    log_rel = logged_path(stdout_path)
    if log_rel is not None and (arm_dir / log_rel).exists():
        kept = out / arm / Path(log_rel).name
        shutil.copy(arm_dir / log_rel, kept)
        record["log"] = str(kept)
        run = ab_stats.parse_log(str(kept))
        if run is not None:
            record.update(wall_ms=run.wall_ms, val_loss=run.val_loss)
    if cache_root is not None:
        shutil.rmtree(cache_root, ignore_errors=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                        help="an arm and its checkout; pass twice, baseline first")
    parser.add_argument("--legs", type=int, default=10, help="runs per arm")
    parser.add_argument("--arm-legs", action="append", default=[], metavar="NAME=N",
                        help="runs for one arm instead of --legs (interleaved in gcd-sized rounds)")
    parser.add_argument("--out", required=True, help="directory for stdout, logs and the ledger")
    parser.add_argument("--cold", action="store_true", help="empty compile caches for every leg")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra environment for every leg")
    parser.add_argument("--arm-env", action="append", default=[], metavar="NAME:KEY=VALUE",
                        help="environment for one arm's legs (e.g. a step count); arms may share a directory")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--command", default=DEFAULT_COMMAND, help="leg command, run in the arm's directory")
    parser.add_argument("--data-path", default=None, help="DATA_PATH both arms read (default: each arm's own data/)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the preflight notes only")
    parser.add_argument("--max-crashes", type=int, default=0,
                        help="stop after this many crashed legs in a row of one arm (0: never; exit code 3)")
    parser.add_argument("--droppable", action="append", default=[], metavar="NAME",
                        help="with --max-crashes: this arm's remaining legs are skipped instead of stopping the bench")
    parser.add_argument("--leg-timeout", type=float, default=0, help="seconds before a leg is killed (0: never)")
    args = parser.parse_args()

    arm_dirs = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arm_dirs[name] = Path(path).resolve()
        assert (arm_dirs[name] / "train_gpt.py").exists() or args.command != DEFAULT_COMMAND, f"no train_gpt.py in {path}"
    env = dict(kv.split("=", 1) for kv in args.env)
    arm_env = {name: {} for name in arm_dirs}
    for spec in args.arm_env:
        name, _, kv = spec.partition(":")
        assert name in arm_env, f"--arm-env for unknown arm {name}"
        key, _, value = kv.partition("=")
        arm_env[name][key] = value
    if args.data_path:
        env["DATA_PATH"] = str(Path(args.data_path).resolve())
    command = args.command.format(nproc=args.nproc)
    legs = dict.fromkeys(arm_dirs, args.legs)
    for spec in args.arm_legs:
        name, _, n = spec.partition("=")
        assert name in legs, f"--arm-legs for unknown arm {name}"
        legs[name] = int(n)
    order = leg_order(list(arm_dirs), legs)

    for note in preflight(arm_dirs, args.data_path):
        print(f"PREFLIGHT: {note}")
    print(f"{len(order)} legs: {' '.join(order)}\ncommand: {command}\nenv: {env}  cold caches: {args.cold}")
    for name, extra in arm_env.items():
        if extra:
            print(f"  {name}: {extra}")
    if args.dry_run:
        return

    out = Path(args.out).resolve()
    plan = dict(order=order, arms={name: str(path) for name, path in arm_dirs.items()}, env=env, arm_env=arm_env,
                command=command, cold=args.cold)
    ledger_path, plan_path = out / "ledger.jsonl", out / "plan.json"
    done = {}
    if ledger_path.exists() or plan_path.exists():
        if not plan_path.exists() or json.loads(plan_path.read_text()) != plan:
            raise SystemExit(f"{out} holds the ledger of a different plan (see its plan.json). Its legs count, so "
                             "they are neither re-run nor re-planned: use a fresh --out.")
        done = {r["leg"]: r for r in read_ledger(ledger_path)}
        print(f"resuming {out}: {len(done)} of {len(order)} legs are in the ledger and will not run again")
    finished = lambda r: r.get("val_loss") is not None
    streak = dict.fromkeys(arm_dirs, 0)  # crashed legs in a row, per arm
    for index in sorted(done):
        arm = done[index]["arm"]
        streak[arm] = 0 if finished(done[index]) else streak[arm] + 1
    dropped = {arm for arm in args.droppable if args.max_crashes and streak[arm] >= args.max_crashes}
    for arm, n in streak.items():
        if args.max_crashes and n >= args.max_crashes:
            print(f"NOTE: {arm} crashed its last {n} legs in the ledger; "
                  + ("it stays dropped" if arm in dropped else "one more crash stops the bench again"))

    out.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=1) + "\n")
    if orphans := kill_marked(out):
        print(f"killed {len(orphans)} processes left over from a leg of an earlier run of this bench: {orphans}")
        time.sleep(2)  # let the driver release their GPU memory
    for sig in (signal.SIGTERM, signal.SIGHUP):  # unwind through run_leg's cleanup instead of dying in place
        if signal.getsignal(sig) is not signal.SIG_IGN:  # under nohup, a hangup stays ignored
            signal.signal(sig, terminate)
    with open(ledger_path, "a") as ledger:
        for index, arm in enumerate(order):
            if index in done or arm in dropped:
                continue
            record = (salvage(index, arm, arm_dirs[arm], out)
                      or run_leg(index, arm, arm_dirs[arm], command, {**env, **arm_env[arm]}, out, args.cold,
                                 args.leg_timeout or None))
            ledger.write(json.dumps(record) + "\n")
            ledger.flush()
            print(f"leg {index:3d} {arm:>10}: exit {record['exit_code']}{' (TIMED OUT)' * bool(record.get('timed_out'))}"
                  f"  wall {record['wall_ms']} ms  "
                  f"val {record['val_loss']}  ({record['seconds']} s with compile)", flush=True)
            streak[arm] = 0 if finished(record) else streak[arm] + 1
            if args.max_crashes and streak[arm] >= args.max_crashes:
                if arm not in args.droppable:
                    print(f"STOPPED: {arm} crashed {streak[arm]} legs in a row (last: {record['stdout']}). Every leg so "
                          "far is in the ledger; fix the cause (not the code) and re-run the same command to resume.")
                    sys.exit(3)
                dropped.add(arm)
                print(f"DROPPED {arm}: {streak[arm]} crashed legs in a row; its remaining legs are skipped")

    names = list(arm_dirs)
    if len(names) > 2:
        report = ab_stats.report_many({name: ab_stats.load(str(out / name / "*.txt"))[0] for name in names})
        (out / "report.txt").write_text(report + "\n")
        print(report)
    elif len(names) == 2:
        baseline, _ = ab_stats.load(str(out / names[0] / "*.txt"))
        candidate, _ = ab_stats.load(str(out / names[1] / "*.txt"))
        if len(baseline) >= 2 and len(candidate) >= 2:
            report = ab_stats.report(baseline, candidate)
            (out / "report.txt").write_text(report + "\n")
            print(report)


if __name__ == "__main__":
    main()
