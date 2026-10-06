"""Interleaved A/B timing runs of two speedrun checkouts on one 8xH100 node (rule 4: same hardware).

Each arm is a directory holding a train_gpt.py (a git worktree of the baseline record, one of the
candidate). Legs alternate in an ABBA pattern so slow drift of the node (thermals, a noisy neighbour,
page cache) lands on both arms equally. Every leg is kept, crashed or not: all runs count.

  python tools/speedrun_ab/ab_bench.py --arm baseline=../base --arm candidate=. --legs 10 --cold \
      --out ab_runs/$(date +%m%d_%H%M)

--cold gives every leg empty Inductor/Triton caches (the ANVIL2 certification convention), so the
untimed compile reruns each leg; without it legs share the warm caches of their arm.
The final report is tools/speedrun_ab/ab_stats.py over the two arms' logs (also runnable alone).
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_stats  # noqa: E402

LOG_PATH_RE = re.compile(r"^(logs/[0-9a-f-]+\.txt)\s*$")
DEFAULT_COMMAND = "torchrun --standalone --nproc_per_node={nproc} train_gpt.py"


def leg_order(arms: list[str], legs: int) -> list[str]:
    """ABBA ABBA ... for two arms (round-robin, alternating direction, for more): `legs` runs per arm."""
    order = []
    for i in range(legs):
        order += arms if i % 2 == 0 else arms[::-1]
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


def run_leg(index: int, arm: str, arm_dir: Path, command: str, env: dict[str, str], out: Path, cold: bool) -> dict:
    leg_env = dict(os.environ, **env)
    cache_root = None
    if cold:
        cache_root = out / "caches" / f"leg{index:03d}"
        leg_env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor")
        leg_env["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    stdout_path = out / arm / f"leg{index:03d}.stdout"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(stdout_path, "w") as sink:
        proc = subprocess.run(shlex.split(command), cwd=arm_dir, env=leg_env, stdout=sink, stderr=subprocess.STDOUT)
    record = dict(leg=index, arm=arm, exit_code=proc.returncode, started=started, seconds=round(time.time() - started, 1),
                  stdout=str(stdout_path), log=None, wall_ms=None, val_loss=None)
    with open(stdout_path, errors="replace") as f:
        log_rel = next((m.group(1) for line in f if (m := LOG_PATH_RE.match(line))), None)
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
    parser.add_argument("--out", required=True, help="directory for stdout, logs and the ledger")
    parser.add_argument("--cold", action="store_true", help="empty compile caches for every leg")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra environment for every leg")
    parser.add_argument("--arm-env", action="append", default=[], metavar="NAME:KEY=VALUE",
                        help="environment for one arm's legs (e.g. a step count); arms may share a directory")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--command", default=DEFAULT_COMMAND, help="leg command, run in the arm's directory")
    parser.add_argument("--data-path", default=None, help="DATA_PATH both arms read (default: each arm's own data/)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the preflight notes only")
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
    order = leg_order(list(arm_dirs), args.legs)

    for note in preflight(arm_dirs, args.data_path):
        print(f"PREFLIGHT: {note}")
    print(f"{len(order)} legs: {' '.join(order)}\ncommand: {command}\nenv: {env}  cold caches: {args.cold}")
    for name, extra in arm_env.items():
        if extra:
            print(f"  {name}: {extra}")
    if args.dry_run:
        return

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ledger.jsonl", "a") as ledger:
        for index, arm in enumerate(order):
            record = run_leg(index, arm, arm_dirs[arm], command, {**env, **arm_env[arm]}, out, args.cold)
            ledger.write(json.dumps(record) + "\n")
            ledger.flush()
            print(f"leg {index:3d} {arm:>10}: exit {record['exit_code']}  wall {record['wall_ms']} ms  "
                  f"val {record['val_loss']}  ({record['seconds']} s with compile)", flush=True)

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
