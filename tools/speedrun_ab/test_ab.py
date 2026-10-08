"""CPU tests for the A/B tooling: run `python -m pytest tools/speedrun_ab -q`."""
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_bench  # noqa: E402
import ab_stats  # noqa: E402

RECORD = Path(__file__).resolve().parents[2] / "records/track_1_short/2026-08-30_ANVIL2"

FAKE_TRAINER = '''
import os, random, sys, uuid
random.seed(os.environ.get("FAKE_SEED", "0") + str(uuid.uuid4()))
step_ms = float(os.environ["FAKE_STEP_MS"])
crash = os.environ.get("FAKE_CRASH") == "1"
path = f"logs/{uuid.uuid4()}.txt"
os.makedirs("logs", exist_ok=True)
print(path)
t = 0.0
with open(path, "w") as f:
    for step in range(25, 101, 25):
        t += 25 * step_ms + random.gauss(0, 1)
        print(f"step:{step}/100 train_time:{t:.0f}ms step_avg:{t/step:.2f}ms", file=f)
    if crash:
        sys.exit(3)
    print(f"step:100/100 val_loss:{random.gauss(3.277, 0.001):.4f} train_time:{t + 200:.0f}ms step_avg:1ms", file=f)
'''


def test_stats_reproduce_published_record_numbers():
    """ANVIL2's statistics.md: baseline (record #89) 73.889 +/- 0.137 s, val 3.2782777778 (n=9)."""
    runs, bad = ab_stats.load(str(RECORD / "baseline/*.txt"))
    assert not bad and len(runs) == 9
    walls = [r.wall_ms / 1000 for r in runs]
    assert abs(sum(walls) / 9 - 73.889) < 5e-4
    assert abs(sum(r.val_loss for r in runs) / 9 - 3.2782777778) < 1e-9


def test_t_distribution_against_known_values():
    # Two-sided 95% critical values from standard t tables.
    for df, crit in [(1, 12.706), (5, 2.571), (17, 2.110), (30, 2.042)]:
        assert abs(ab_stats.t_ppf(0.975, df) - crit) < 1e-3
    assert abs(ab_stats.t_cdf(0.0, 7) - 0.5) < 1e-12
    # Normal limit.
    assert abs(ab_stats.t_cdf(-1.959964, 1e7) - 0.025) < 1e-5


def test_leg_order_is_abba():
    assert ab_bench.leg_order(["a", "b"], 3) == ["a", "b", "b", "a", "a", "b"]


@pytest.mark.parametrize("cold", [False, True])
def test_bench_end_to_end_with_fake_trainer(tmp_path, cold):
    arms = {}
    for name in ("base", "cand"):
        arms[name] = tmp_path / name
        arms[name].mkdir()
        (arms[name] / "fake.py").write_text(FAKE_TRAINER)
    out = tmp_path / "out"
    # The candidate's fake steps are 1 ms faster: 100 ms faster per run.
    for name, step_ms in (("base", "10"), ("cand", "9")):
        (arms[name] / "run.sh").write_text(f"FAKE_STEP_MS={step_ms} {sys.executable} fake.py\n")
    cmd = [sys.executable, str(Path(ab_bench.__file__)), "--arm", f"base={arms['base']}", "--arm", f"cand={arms['cand']}",
           "--legs", "4", "--out", str(out), "--command", "bash run.sh"] + (["--cold"] if cold else [])
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in ledger] == ["base", "cand", "cand", "base", "base", "cand", "cand", "base"]
    assert all(r["exit_code"] == 0 and r["val_loss"] is not None for r in ledger)
    report = (out / "report.txt").read_text()
    delta_ms = float(report.split("rule 4: wall candidate - baseline = ")[1].split(" ms")[0])
    assert abs(delta_ms + 100) < 6, report
    assert not (out / "caches").exists() or not any((out / "caches").iterdir())


def test_bench_keeps_crashed_legs(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()
    (arm / "fake.py").write_text(FAKE_TRAINER)
    (arm / "run.sh").write_text(f"FAKE_STEP_MS=10 FAKE_CRASH=1 {sys.executable} fake.py\n")
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(Path(ab_bench.__file__)), "--arm", f"a={arm}", "--legs", "2", "--out", str(out),
                    "--command", "bash run.sh"], check=True, capture_output=True)
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 2 and all(r["exit_code"] == 3 and r["val_loss"] is None and r["log"] for r in ledger)
    runs, unfinished = ab_stats.load(str(out / "a/*.txt"))
    assert runs == [] and len(unfinished) == 2


def test_interval_table_localizes_a_gain():
    random.seed(1)

    def fake_run(first_interval_ms):
        r = ab_stats.Run("x", 100, 0, 3.277, {})
        t = 0
        for step, dt in zip((25, 50, 75, 100), (first_interval_ms, 400, 400, 400)):
            t += dt + random.gauss(0, 1)
            r.step_ms[step] = round(t)
        r.wall_ms = round(t + 220)
        return r

    base = [fake_run(600) for _ in range(6)]
    cand = [fake_run(450) for _ in range(6)]
    report = ab_stats.report(base, cand)
    first = next(line for line in report.splitlines() if line.strip().startswith("0-25"))
    assert float(first.split()[-1]) < -140
    assert "100-val" in report


def test_sweep_arms_share_a_checkout_with_their_own_env(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()
    (arm / "fake.py").write_text(FAKE_TRAINER)
    (arm / "run.sh").write_text(f"{sys.executable} fake.py\n")
    out = tmp_path / "out"
    cmd = [sys.executable, str(Path(ab_bench.__file__)), "--legs", "3", "--out", str(out), "--command", "bash run.sh"]
    for name, step_ms in (("base", "10"), ("faster", "9"), ("slower", "11")):
        cmd += ["--arm", f"{name}={arm}", "--arm-env", f"{name}:FAKE_STEP_MS={step_ms}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = {line.split()[0]: line.split() for line in (out / "report.txt").read_text().splitlines()
            if line.split() and line.split()[0] in ("base", "faster", "slower")}
    d_wall = {name: float(row[6]) for name, row in rows.items()}
    assert d_wall["base"] == 0 and abs(d_wall["faster"] + 100) < 6 and abs(d_wall["slower"] - 100) < 6, rows


def test_adjusted_wall_prices_val_at_the_record_rate():
    run = lambda wall_s, val: ab_stats.Run("x", 100, int(wall_s * 1000), val, {})
    # 2 millinats above the target cost 2 * 164 ms.
    assert abs(ab_stats.adjusted_wall([run(40.0, 3.2795)]) - 40.328) < 1e-9


def test_leg_order_with_a_count_per_arm():
    # The certification pool's shape: the candidate twice per round, master and #379 once, direction alternating.
    order = ab_bench.leg_order(["m", "p", "c"], {"m": 6, "p": 6, "c": 12})
    assert order[:8] == ["c", "m", "p", "c", "c", "p", "m", "c"]
    assert {a: order.count(a) for a in "mpc"} == {"m": 6, "p": 6, "c": 12}
    assert ab_bench.leg_order(["a", "b"], {"a": 3, "b": 3}) == ab_bench.leg_order(["a", "b"], 3)


def bench(out, arm, *extra, legs=3):
    cmd = [sys.executable, str(Path(ab_bench.__file__)), "--arm", f"a={arm}", "--arm", f"b={arm}", "--legs", str(legs),
           "--out", str(out), "--command", "bash run.sh", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def fake_arm(tmp_path, script="FAKE_STEP_MS=10 {py} fake.py\n"):
    arm = tmp_path / "arm"
    arm.mkdir(exist_ok=True)
    (arm / "fake.py").write_text(FAKE_TRAINER)
    (arm / "run.sh").write_text(script.format(py=sys.executable))
    return arm


def test_bench_resumes_without_rerunning_a_leg(tmp_path):
    arm, out = fake_arm(tmp_path), tmp_path / "out"
    assert bench(out, arm, legs=2).returncode == 0
    first = (out / "ledger.jsonl").read_text()
    # A leg that was running when the bench was killed: its stdout and a log that never reached the final validation.
    lines = first.splitlines()
    (out / "ledger.jsonl").write_text("\n".join(lines[:3]) + "\n")
    partial = arm / "logs/0123abcd.txt"
    partial.write_text("step:25/100 train_time:250ms step_avg:10ms\n")
    (out / "a/leg003.stdout").write_text("logs/0123abcd.txt\n")
    result = bench(out, arm, legs=2)
    assert result.returncode == 0, result.stderr
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert [r["leg"] for r in ledger] == [0, 1, 2, 3] and ledger[:3] == [json.loads(l) for l in lines[:3]]
    (note,) = [json.loads(line) for line in (out / "interrupted.jsonl").read_text().splitlines()]
    assert note["leg"] == 3 and Path(note["log"]).read_text() == partial.read_text()
    # Nothing left to run; a different plan for the same --out is refused.
    assert "4 of 4 legs" in bench(out, arm, legs=2).stdout
    assert (out / "ledger.jsonl").read_text().count("\n") == 4
    refused = bench(out, arm, legs=3)
    assert refused.returncode != 0 and "different plan" in refused.stderr


def test_bench_keeps_a_leg_that_finished_after_the_bench_died(tmp_path):
    arm, out = fake_arm(tmp_path), tmp_path / "out"
    assert bench(out, arm, legs=1).returncode == 0
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    (out / "ledger.jsonl").write_text(json.dumps(ledger[0]) + "\n")  # leg 1 finished but never reached the ledger
    assert bench(out, arm, legs=1).returncode == 0
    resumed = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert resumed[1]["recovered"] and resumed[1]["val_loss"] == ledger[1]["val_loss"]
    assert not (out / "interrupted.jsonl").exists()


def test_bench_stops_after_crashes_in_a_row_unless_the_arm_is_droppable(tmp_path):
    arm = fake_arm(tmp_path, "FAKE_STEP_MS=10 FAKE_CRASH=$CRASH {py} fake.py\n")
    cmd = lambda out, *extra: [sys.executable, str(Path(ab_bench.__file__)), "--arm", f"ok={arm}", "--arm", f"bad={arm}",
                               "--arm-env", "bad:CRASH=1", "--legs", "4", "--out", str(out), "--command", "bash run.sh",
                               "--max-crashes", "2", *extra]
    stopped = subprocess.run(cmd(tmp_path / "stop"), capture_output=True, text=True)
    assert stopped.returncode == 3 and "STOPPED: bad crashed 2 legs in a row" in stopped.stdout
    ledger = [json.loads(line) for line in (tmp_path / "stop/ledger.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in ledger] == ["ok", "bad", "bad"]
    dropped = subprocess.run(cmd(tmp_path / "drop", "--droppable", "bad"), capture_output=True, text=True)
    assert dropped.returncode == 0 and "DROPPED bad" in dropped.stdout
    ledger = [json.loads(line) for line in (tmp_path / "drop/ledger.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in ledger] == ["ok", "bad", "bad", "ok", "ok", "ok"]


def test_bench_kills_a_hung_leg_and_its_process_group(tmp_path):
    arm = fake_arm(tmp_path, "sleep 60 & echo $! > child.pid; FAKE_STEP_MS=10 {py} fake.py; sleep 60\n")
    out = tmp_path / "out"
    started = time.time()
    result = bench(out, arm, "--leg-timeout", "1", legs=1)
    assert result.returncode == 0 and time.time() - started < 30, result.stderr
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert all(r["timed_out"] and r["exit_code"] != 0 for r in ledger)
    child = int((arm / "child.pid").read_text())
    time.sleep(0.2)
    assert not Path(f"/proc/{child}").exists() or "Z" in Path(f"/proc/{child}/stat").read_text().split()[2]


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_bench_resume_kills_a_leg_orphaned_by_a_killed_bench(tmp_path):
    # SIGKILL (OOM killer, kill -9) skips the bench's cleanup, and the leg, in its own session, keeps running.
    arm, out = fake_arm(tmp_path, "echo $$ > leg.pid; exec sleep 600\n"), tmp_path / "out"
    cmd = [sys.executable, str(Path(ab_bench.__file__)), "--arm", f"a={arm}", "--arm", f"b={arm}", "--legs", "1",
           "--out", str(out), "--command", "bash run.sh"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
    deadline = time.time() + 30
    while not (arm / "leg.pid").exists() or not (arm / "leg.pid").read_text().strip():
        assert time.time() < deadline
        time.sleep(0.05)
    proc.kill()
    proc.wait()
    orphan = int((arm / "leg.pid").read_text())
    assert alive(orphan)
    (arm / "run.sh").write_text(f"FAKE_STEP_MS=10 {sys.executable} fake.py\n")
    result = bench(out, arm, legs=1)
    assert result.returncode == 0 and f"left over from a leg of an earlier run of this bench: [{orphan}]" in result.stdout
    time.sleep(0.2)
    assert not alive(orphan)
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert [r["leg"] for r in ledger] == [0, 1] and all(r["val_loss"] is not None for r in ledger)


def test_bench_under_nohup_survives_a_hangup(tmp_path):
    arm, out = fake_arm(tmp_path, "touch started; sleep 1; FAKE_STEP_MS=10 {py} fake.py\n"), tmp_path / "out"
    cmd = [sys.executable, str(Path(ab_bench.__file__)), "--arm", f"a={arm}", "--arm", f"b={arm}", "--legs", "1",
           "--out", str(out), "--command", "bash run.sh"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, start_new_session=True,
                            preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
    deadline = time.time() + 30
    while not (arm / "started").exists():
        assert time.time() < deadline
        time.sleep(0.05)
    os.killpg(proc.pid, signal.SIGHUP)  # what an interactive bash sends its jobs when the SSH session drops
    assert proc.wait(timeout=60) == 0
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 2 and all(r["val_loss"] is not None for r in ledger)
