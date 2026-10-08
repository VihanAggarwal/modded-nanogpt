"""CPU tests for the record attempt: `python -m pytest tools/record_attempt -q`.

The end-to-end tests run run.sh itself (SETUP=0 skips the node setup) with a fake trainer in place of torchrun: it
writes a run log in the real format, with a train time and final val set per arm in the arm's fake.json, and the
step count taken from NUM_SCHEDULED_ITERATIONS as the real trainer does.
"""
import collections
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools/speedrun_ab"))
import attempt  # noqa: E402
import preflight  # noqa: E402
import record_stats  # noqa: E402

CPLM = REPO / "records/track_1_short/2026-10_CPLM"

FAKE_TRAINER = r'''
import json, os, random, sys, time, uuid
cfg = json.load(open("fake.json"))
n = 0
if counter := os.environ.get("FAKE_COUNTER"):
    n = (int(open(counter).read()) if os.path.exists(counter) else 0) + 1
    open(counter, "w").write(str(n))
rng = random.Random(f"{cfg['name']}-{n}")
if n and str(n) == os.environ.get("FAKE_EARLY_CRASH_AT"):  # dies before the run log exists (imports, NCCL, FA3)
    sys.exit("ImportError: fake crash before the run log")
sched = int(os.environ.get("NUM_SCHEDULED_ITERATIONS", cfg["default_sched"]))
steps = sched + 72
path = f"logs/{uuid.uuid4()}.txt"
os.makedirs("logs", exist_ok=True)
print(path, flush=True)
with open(path, "w") as f:
    f.write(open("train_gpt.py").read() + "=" * 100 + "\n")
    f.write("Running Python 3.12.3 (main, Aug 14 2025) [GCC 13.3.0]\nRunning PyTorch 2.10.0+cu128 compiled for CUDA 12.8\n"
            "Running Triton version 3.6.0\n| NVIDIA-SMI 580.126.09    Driver Version: 580.126.09     CUDA Version: 13.0     |\n")
    for gpu in range(8):
        f.write(f"|   {gpu}  NVIDIA H100 80GB HBM3          On  |   00000000:19:00.0 Off |                    0 |\n")
    t = 0.0
    for step in range(25, steps, 25):
        t += 25 * cfg["ms_per_step"]
        f.write(f"step:{step}/{steps} train_time:{t:.0f}ms step_avg:{t / step:.2f}ms\n")
    f.flush()
    if n and str(n) == os.environ.get("FAKE_HANG_AT") and os.path.exists(os.environ["FAKE_HANG_ARMED"]):
        open(os.environ["FAKE_HANGING"], "w").close()
        time.sleep(600)
    if cfg.get("crash"):
        sys.exit("Traceback (most recent call last): fake crash")
    wall = steps * cfg["ms_per_step"] + rng.gauss(0, 30)
    val = cfg["val978"] + cfg.get("val_per_step", 0) * (978 - sched) + rng.gauss(0, cfg.get("val_sd", 0.0008))
    f.write(f"step:{steps}/{steps} val_loss:{val:.4f} train_time:{wall:.0f}ms step_avg:{wall / steps:.2f}ms\n")
'''

# Per arm: the default scheduled steps (master 1122 -> 1194 trained), ms per step, val at 978 scheduled steps and
# its change per step removed (0.25 millinats, the CPLM README's rate).
ARMS = {
    "master": dict(default_sched=1122, ms_per_step=34.0, val978=3.2765, val_per_step=0.0),
    "pr379": dict(default_sched=978, ms_per_step=34.3, val978=3.2769, val_per_step=0.00025),
    "stack": dict(default_sched=978, ms_per_step=33.8, val978=3.2700, val_per_step=0.00025),
    "mlonly": dict(default_sched=978, ms_per_step=34.3, val978=3.2700, val_per_step=0.00025),
}


def make_node(tmp_path: Path, **overrides) -> dict:
    """Fake arms, a fake mlonly builder and run.sh's environment."""
    fake = tmp_path / "fake_trainer.py"
    fake.write_text(FAKE_TRAINER)
    dirs = {}
    for name, cfg in ARMS.items():
        cfg = dict(cfg, name=name, **overrides.get(name, {}))
        dirs[name] = tmp_path / "arms" / name
        (dirs[name] / "track_1_short").mkdir(parents=True)
        default = "978" if name != "master" else "1122"
        (dirs[name] / "train_gpt.py").write_text(
            f'# fake {name}\nfor _k, _v in (("NUM_SCHEDULED_ITERATIONS", "{default}"),):\n    pass\n')
        (dirs[name] / "track_1_short/__init__.py").write_text("")
        (dirs[name] / "fake.json").write_text(json.dumps(cfg))
    builder = tmp_path / "make_mlonly.sh"
    builder.write_text(overrides.get("builder", f'echo "building" >&2\necho "{dirs["mlonly"]}"\n'))
    work = tmp_path / "work"
    env = dict(os.environ, SETUP="0", WORK=str(work), DATA=str(tmp_path / "data"), PYTHON=sys.executable,
               LEG_COMMAND=f"{sys.executable} {fake}", STACK_DIR=str(dirs["stack"]), MASTER_DIR=str(dirs["master"]),
               PR379_DIR=str(dirs["pr379"]), MAKE_MLONLY=str(builder), RECORDS_DIR=str(tmp_path / "records"),
               FAKE_COUNTER=str(tmp_path / "counter"))
    for key in ("NUM_SCHEDULED_ITERATIONS", "TRAIN_SEED", "DATA_PATH"):
        env.pop(key, None)
    return dict(env=env, dirs=dirs, work=work, runs=work / "runs", records=tmp_path / "records",
                counter=tmp_path / "counter")


def run_sh(node: dict, *args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(HERE / "run.sh"), *args], env=node["env"], capture_output=True, text=True,
                          timeout=600, **kw)


def ledger(node: dict, phase: str) -> list[dict]:
    path = node["runs"] / phase / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def record_folder(node: dict) -> Path:
    (folder,) = [p for p in node["records"].iterdir() if p.is_dir()]
    return folder


# ---------------------------------------------------------------- statistics on real logs

def test_stats_reproduce_the_cplm_record():
    """#379's README: n=8 at 36.009 s and val 3.27686 (p = 5.1e-4) against 13 same-node master runs at 40.575 s."""
    this_pr, baseline = record_stats.load(str(CPLM / "this_pr")), record_stats.load(str(CPLM / "baseline"))
    s, b = record_stats.summarize(this_pr), record_stats.summarize(baseline)
    assert (s["n"], s["finished"], b["n"], b["finished"]) == (8, 8, 13, 13)
    assert abs(s["val_mean"] - 3.27686) < 5e-6 and abs(s["p"] - 5.1e-4) / 5.1e-4 < 0.02
    assert abs(s["wall_mean"] - 36.009) < 5e-4 and abs(b["wall_mean"] - 40.575) < 5e-4
    assert s["steps"] == [1050] and b["steps"] == [1194]
    c = record_stats.compare(this_pr, baseline)
    assert abs(c["wall"] + 4.566) < 1e-3 and abs(c["wall_pct"] + 11.25) < 0.01
    assert abs(c["val_mnat"] - 0.33) < 0.01 and abs(c["p_val"] - 0.66) < 0.01  # README: Welch p = 0.66
    report = record_stats.report(str(CPLM))
    assert "p=0.000512" in report and "PASS" in report


def test_stats_flag_crashes_high_runs_and_outliers(tmp_path):
    for i, (val, wall) in enumerate([(3.2770, 36000), (3.2772, 36010), (3.2810, 35990), (3.2771, 37500), (None, 0)]):
        lines = [f"step:1025/1050 train_time:{wall - 500}ms step_avg:1ms"]
        if val is not None:
            lines.append(f"step:1050/1050 val_loss:{val:.4f} train_time:{wall}ms step_avg:1ms")
        (tmp_path / f"{i}.txt").write_text("\n".join(lines) + "\n")
    flags = record_stats.flags("arm", record_stats.load(str(tmp_path)))
    assert any("4.txt has no final validation" in f for f in flags)
    assert any("2.txt ended above 3.28" in f for f in flags)
    assert any("3.txt is an outlier in train time" in f for f in flags)


# ---------------------------------------------------------------- the pre-registered rule

def rec(arm, val=3.2760, wall=36.0):
    return dict(arm=arm, val_loss=val, wall_ms=None if val is None else wall * 1000)


PILOT = ["master", "pr379", "stack978", "stack963", "mlonly978"]


def pilot_ledger(stack963_val=3.2760, stack_wall=35.5, mlonly_wall=36.0, crash=None):
    out = []
    for i in range(3):
        for arm in PILOT:
            val = stack963_val if arm == "stack963" else 3.2760
            wall = {"stack978": stack_wall, "stack963": stack_wall - 0.5, "mlonly978": mlonly_wall}.get(arm, 36.0) + 0.01 * i
            out.append(rec(arm, None if (arm, i) == crash else val, wall))
    return out


@pytest.mark.parametrize("kwargs, smoke, expected", [
    (dict(), {"stack": True}, ("stack", 963)),
    (dict(stack963_val=3.2765), {"stack": True}, ("stack", 963)),            # the boundary is inclusive
    (dict(stack963_val=3.2766), {"stack": True}, ("stack", 978)),
    (dict(stack_wall=36.5), {"stack": True}, ("mlonly", 978)),               # slower beyond 2 se: mlonly, no 963 legs
    (dict(stack_wall=36.005), {"stack": True}, ("stack", 963)),              # within noise: stack stays
    (dict(crash=("stack963", 1)), {"stack": True}, ("mlonly", 978)),         # any stack crash: not viable
    (dict(crash=("mlonly978", 0)), {"stack": True}, ("stack", 963)),
])
def test_decision_rule(kwargs, smoke, expected):
    order = [a for i in range(3) for a in (PILOT if i % 2 == 0 else PILOT[::-1])]
    d = attempt.decide(order, pilot_ledger(**kwargs), smoke)
    assert (d["candidate"], d["steps"]) == expected, d["reasons"]
    assert d["arm"] == f"{expected[0]}{expected[1]}"


def test_decision_rule_without_a_viable_candidate():
    pilot = ["master", "pr379", "mlonly978", "mlonly963"]
    ledger = [rec(a, None if a == "mlonly963" else 3.276) for a in pilot]
    d = attempt.decide(pilot, ledger, {"stack": False, "mlonly": True})
    assert d["candidate"] is None and any("stack: not viable" in r for r in d["reasons"])


def test_readme_carries_the_rule_verbatim():
    assert attempt.rule_text(3, 12, 6) in (HERE / "README.md").read_text()


# ---------------------------------------------------------------- run.sh end to end

def test_dry_run_prints_the_plan_and_touches_nothing(tmp_path):
    node = make_node(tmp_path)
    result = run_sh(node, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "40 legs" in result.stdout and "PRE-REGISTERED DECISION RULE" in result.stdout
    assert "D certify: 24 legs" in result.stdout and "B pilot:   15 legs" in result.stdout
    assert not node["work"].exists() and not node["counter"].exists()


def test_end_to_end_stack_at_963(tmp_path):
    node = make_node(tmp_path)
    result = run_sh(node)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    decision = json.loads((node["runs"] / "decision.json").read_text())
    assert (decision["candidate"], decision["steps"]) == ("stack", 963)
    assert len(ledger(node, "smoke_stack")) == 1 and not (node["runs"] / "smoke_mlonly").exists()
    pilot = ledger(node, "pilot")
    assert collections.Counter(r["arm"] for r in pilot) == dict.fromkeys(PILOT, 3)
    cert = ledger(node, "cert")
    assert collections.Counter(r["arm"] for r in cert) == {"stack963": 12, "master": 6, "pr379": 6}
    assert all(r["val_loss"] is not None for r in pilot + cert)
    assert int(node["counter"].read_text()) == 1 + 15 + 24

    folder = record_folder(node)
    assert folder.name.endswith("_CPLMNormHashesSystems")
    counts = {sub: len(list((folder / sub).glob("*.txt"))) for sub in ("this_pr", "baseline", "baseline_pr379")}
    assert counts == {"this_pr": 12, "baseline": 6, "baseline_pr379": 6}
    assert {p.name: len(list(p.glob("*.txt"))) for p in (folder / "pilot").iterdir()} == dict.fromkeys(PILOT, 3)
    assert len(list((folder / "smoke/stack").glob("*.txt"))) == 1
    assert all("step:1035/1035 val_loss" in p.read_text() for p in (folder / "this_pr").glob("*.txt"))
    assert all("step:1194/1194 val_loss" in p.read_text() for p in (folder / "baseline").glob("*.txt"))

    # statistics.py recomputes the README's numbers from the logs alone, run from the folder.
    stats = subprocess.run([sys.executable, "statistics.py"], cwd=folder, capture_output=True, text=True)
    assert stats.returncode == 0, stats.stderr
    assert "PASS" in stats.stdout and "this_pr" in stats.stdout and "PILOT" in stats.stdout
    s = record_stats.summarize(record_stats.load(str(folder / "this_pr")))
    readme = (folder / "README.md").read_text()
    assert f"{s['wall_mean']:.3f} ± {s['wall_sd']:.3f} s" in readme and f"{s['val_mean']:.5f}" in readme
    prereg = (folder / "PREREGISTRATION.txt").read_text()
    assert attempt.rule_text(3, 12, 6) in prereg and attempt.rule_text(3, 12, 6).rstrip() in readme
    for text in ("@NathanGodey and @yoavartzi", "Daniel Monroe", "963 scheduled steps (1035 trained)",
                 "must become 963", "READY FOR A PR: yes", "Systems patches", "No leg was interrupted",
                 "driver 580.126.09", "8x NVIDIA H100 80GB HBM3"):
        assert text in readme, text
    verdict = (node["runs"] / "verdict.txt").read_text()
    assert "rule 2 (mean val < 3.28 at p < 0.01): PASS" in verdict and "faster than master (rule 4): YES" in verdict
    assert ("faster than #379 (the increment over #379: #375 + the systems patches, with the 963-step cut): YES"
            in verdict)
    assert f"PR source: the stack checkout {node['dirs']['stack']}" in verdict
    # The systems patches alone (pilot only), and every run of the shipped configuration (a check, not the claim).
    assert "Systems patches alone (pilot only, not pooled: stack978 vs mlonly978" in readme
    assert "All 15 runs of the shipped configuration (`this_pr/` + `pilot/stack963/`; 15 finished)" in readme
    assert "all runs of the shipped configuration (this_pr + pilot/stack963): n=15 finished=15" in stats.stdout
    names = tarfile.open(node["work"] / "send_back.tar.gz").getnames()
    assert "runs/verdict.txt" in names and f"{folder.name}/README.md" in names


def test_end_to_end_978_without_the_mlonly_arm(tmp_path):
    # #375 buys less here: 963 steps would leave the pilot mean above 3.2765.
    node = make_node(tmp_path, stack=dict(val978=3.2740), builder='echo "no network" >&2\nexit 7\n')
    result = run_sh(node)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "no mlonly arm" in result.stdout
    decision = json.loads((node["runs"] / "decision.json").read_text())
    assert (decision["candidate"], decision["steps"]) == ("stack", 978)
    assert collections.Counter(r["arm"] for r in ledger(node, "pilot")) == dict.fromkeys(PILOT[:4], 3)
    assert collections.Counter(r["arm"] for r in ledger(node, "cert")) == {"stack978": 12, "master": 6, "pr379": 6}
    readme = (record_folder(node) / "README.md").read_text()
    assert "fallback arm was unavailable (make_mlonly_arm.sh failed (exit 7)" in readme
    assert "That equals the default in `train_gpt.py`" in readme and "978 scheduled steps (1050 trained)" in readme


def test_end_to_end_falls_back_to_mlonly_when_the_stack_crashes(tmp_path):
    node = make_node(tmp_path, stack=dict(crash=True))
    result = run_sh(node)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert [r["val_loss"] for r in ledger(node, "smoke_stack")] == [None]
    assert len(ledger(node, "smoke_mlonly")) == 1
    pilot = collections.Counter(r["arm"] for r in ledger(node, "pilot"))
    assert pilot == {"master": 3, "pr379": 3, "mlonly978": 3, "mlonly963": 3}
    decision = json.loads((node["runs"] / "decision.json").read_text())
    assert (decision["candidate"], decision["steps"]) == ("mlonly", 963)
    folder = record_folder(node)
    assert folder.name.endswith("_CPLMNormHashes")
    readme = (folder / "README.md").read_text()
    assert "No systems patches" in readme and "smoke_stack stack leg 0" in readme
    # The PR's code is the ML-only arm, which no branch holds: the README and verdict say how to rebuild it.
    assert "ARM=$(bash tools/record_attempt/make_mlonly_arm.sh)" in readme and "git clone -b HEAD" not in readme
    assert f"PR source: {node['dirs']['mlonly']}" in (node["runs"] / "verdict.txt").read_text()
    assert (folder / "crashes/smoke_stack_stack_leg000.stdout").exists()
    assert "fake crash" in (folder / "crashes/smoke_stack_stack_leg000.stdout").read_text()


def test_two_crashes_in_a_row_of_master_stop_the_attempt(tmp_path):
    node = make_node(tmp_path, master=dict(crash=True))
    result = run_sh(node)
    assert result.returncode == attempt.STOPPED, result.stdout[-3000:]
    pilot = ledger(node, "pilot")
    assert [r["arm"] for r in pilot if r["val_loss"] is None] == ["master", "master"]
    assert len(pilot) == 10  # master's second leg is the tenth in the pilot's order
    assert "STOPPED" in (node["runs"] / "verdict.txt").read_text()
    assert not (node["runs"] / "cert").exists() and not node["records"].exists()

    # Once the node is fixed, a second attempt in its own RUNS reuses the rest and discloses the first.
    cfg = json.loads((node["dirs"]["master"] / "fake.json").read_text())
    (node["dirs"]["master"] / "fake.json").write_text(json.dumps(dict(cfg, crash=False)))
    node["env"]["RUNS"] = str(node["work"] / "runs_2")
    second = run_sh(node)
    assert second.returncode == 0, second.stdout[-3000:]
    folder = record_folder(node)
    readme = (folder / "README.md").read_text()
    assert ("Another attempt on this node** (`runs/`, not pooled; its ledgers and logs are in `earlier_attempts/runs/`): "
            "STOPPED: phase pilot stopped") in readme and "It never reached certification." in readme
    assert len(list((folder / "earlier_attempts/runs/pilot/master").glob("*.txt"))) == 2
    names = tarfile.open(node["work"] / "send_back.tar.gz").getnames()
    assert "runs/verdict.txt" in names and "runs_2/verdict.txt" in names and f"{folder.name}/README.md" in names


def test_resume_after_interruption_never_reruns_a_counted_leg(tmp_path):
    node = make_node(tmp_path)
    armed, hanging = tmp_path / "armed", tmp_path / "hanging"
    armed.touch()
    # Invocation 6 is the pilot's fifth leg (after the smoke leg): it hangs mid-run, and the whole run is killed.
    node["env"].update(FAKE_HANG_AT="6", FAKE_HANG_ARMED=str(armed), FAKE_HANGING=str(hanging))
    with open(tmp_path / "first.out", "w") as out:
        proc = subprocess.Popen(["bash", str(HERE / "run.sh")], env=node["env"], stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        deadline = time.time() + 120
        while not hanging.exists():
            assert proc.poll() is None and time.time() < deadline, (tmp_path / "first.out").read_text()[-3000:]
            time.sleep(0.05)
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    before = ledger(node, "pilot")
    assert [r["leg"] for r in before] == [0, 1, 2, 3]
    armed.unlink()

    result = run_sh(node)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "resuming" in result.stdout
    assert len(ledger(node, "smoke_stack")) == 1
    pilot = ledger(node, "pilot")
    assert sorted(r["leg"] for r in pilot) == list(range(15)) and pilot[:4] == before
    (interrupted,) = [json.loads(l) for l in (node["runs"] / "pilot/interrupted.jsonl").read_text().splitlines()]
    assert (interrupted["leg"], interrupted["arm"]) == (4, "mlonly978") and Path(interrupted["log"]).exists()
    assert int(node["counter"].read_text()) == 1 + 15 + 24 + 1  # the interrupted leg ran twice, nothing else did
    folder = record_folder(node)
    readme = (folder / "README.md").read_text()
    assert "1 leg was cut short" in readme and "pilot mlonly978 leg 4" in readme
    assert len(list((folder / "interrupted/pilot").iterdir())) == 2  # its stdout and its partial log

    # A finished attempt re-runs nothing; it only rebuilds the folder.
    again = run_sh(node)
    assert again.returncode == 0 and int(node["counter"].read_text()) == 41
    assert record_folder(node) == folder

    # Changed code is refused: every leg must run the source the attempt started with.
    train_gpt = node["dirs"]["stack"] / "train_gpt.py"
    train_gpt.write_text(train_gpt.read_text() + "# edited\n")
    changed = run_sh(node)
    assert changed.returncode != 0 and "source of arm stack changed" in changed.stdout + changed.stderr


@pytest.mark.parametrize("crash_at, arm, ready", [(17, "stack963", "no"), (18, "master", "yes")])
def test_a_certification_leg_that_dies_before_its_log_still_counts(tmp_path, crash_at, arm, ready):
    # Invocation 17 is the first certification leg (the candidate's), 18 the second (master's).
    node = make_node(tmp_path)
    node["env"]["FAKE_EARLY_CRASH_AT"] = str(crash_at)
    result = run_sh(node)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    (crashed,) = [r for r in ledger(node, "cert") if r["val_loss"] is None]
    assert (crashed["arm"], crashed["log"]) == (arm, None)
    folder = record_folder(node)
    sub = {"stack963": "this_pr", "master": "baseline"}[arm]
    stand_in = folder / sub / f"crashed_cert_{arm}_leg{crashed['leg']:03d}.txt"
    assert "fake crash before the run log" in stand_in.read_text()
    runs = record_stats.load(str(folder / sub))
    n = {"this_pr": 12, "baseline": 6}[sub]
    assert (len(runs), sum(r.val is not None for r in runs)) == (n, n - 1)
    readme = (folder / "README.md").read_text()
    stats = subprocess.run([sys.executable, "statistics.py"], cwd=folder, capture_output=True, text=True).stdout
    verdict = (node["runs"] / "verdict.txt").read_text()
    assert f"READY FOR A PR: {ready}" in verdict
    if arm == "stack963":
        assert "| **this PR** | 12 (11) |" in readme and "(11 finished of 12 runs)" in readme
        assert "over 11 finished runs (1 crashed)" in stats
    else:
        assert "| **this PR** | 12 (12) |" in readme and "baseline legs that crashed" in verdict
        assert f"baseline: {stand_in.name} has no final validation" in stats


def test_a_second_attempt_on_the_same_day_discloses_the_first_and_sends_both(tmp_path):
    node = make_node(tmp_path)
    assert run_sh(node).returncode == 0
    first = record_folder(node)
    node["env"]["RUNS"] = str(node["work"] / "runs_2")
    second = run_sh(node)
    assert second.returncode == 0, second.stdout[-3000:]
    (folder,) = [p for p in node["records"].iterdir() if p != first]
    assert folder.name == first.name + "_runs_2"  # the first attempt's folder keeps its name and its content
    readme = (folder / "README.md").read_text()
    assert "Another attempt on this node** (`runs/`, not pooled" in readme
    assert "finished: rule 2 (mean val < 3.28 at p < 0.01): PASS" in readme and "READY FOR A PR: yes" in readme
    assert "Its certification pool (stack963): 12 runs, 12 finished, mean val 3.27" in readme
    assert len(list((folder / "earlier_attempts/runs/this_pr").glob("*.txt"))) == 12
    stats = subprocess.run([sys.executable, "statistics.py"], cwd=folder, capture_output=True, text=True).stdout
    assert "EARLIER ATTEMPT runs (disclosed, not pooled)" in stats
    names = set(tarfile.open(node["work"] / "send_back.tar.gz").getnames())
    assert {"runs/verdict.txt", "runs_2/verdict.txt", f"{first.name}/README.md", f"{folder.name}/README.md"} <= names


def test_runs_outside_work_is_refused(tmp_path):
    node = make_node(tmp_path)
    node["env"]["RUNS"] = str(tmp_path / "elsewhere")
    result = run_sh(node)
    assert result.returncode == 1 and "must be a directory directly inside WORK" in result.stderr


@pytest.mark.skipif(shutil.which("nvidia-smi") is not None, reason="needs a node without nvidia-smi")
def test_a_failed_setup_check_still_writes_send_back(tmp_path):
    node = make_node(tmp_path)
    node["env"]["SETUP"] = "1"
    result = run_sh(node)
    assert result.returncode == 1 and "nvidia-smi not found" in result.stdout + result.stderr
    names = tarfile.open(node["work"] / "send_back.tar.gz").getnames()
    assert any(n.startswith("runs/console_") for n in names) and "Send back:" in result.stdout


def test_preflight_checks_that_run_on_cpu(tmp_path, capsys):
    preflight.check_python()
    if shutil.which("gcc") or shutil.which("clang"):
        preflight.check_toolchain()
    # The stack's canonical-mask builder starts with `python -P` (3.11+) and needs tiktoken's GPT-2 files.
    try:
        import tiktoken
        tiktoken.get_encoding("gpt2")
    except Exception:  # noqa: BLE001 - no tiktoken or no network here
        pytest.skip("tiktoken's GPT-2 files are not available")
    preflight.check_mask_builder(REPO)
    assert "canonical-mask builder starts" in capsys.readouterr().out


def test_an_explicit_record_name_that_is_taken_is_refused_before_the_first_leg(tmp_path):
    node = make_node(tmp_path)
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    (node["records"] / f"{date}_Taken").mkdir(parents=True)
    node["env"]["RECORD_NAME"] = "Taken"
    result = run_sh(node)
    assert result.returncode != 0 and "choose another RECORD_NAME" in result.stdout + result.stderr
    assert not node["counter"].exists()
