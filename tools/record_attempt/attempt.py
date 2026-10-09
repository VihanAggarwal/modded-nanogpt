"""The phases of a track-1 record attempt on one 8xH100 node. tools/record_attempt/run.sh sets the node up and calls
this; see the README next to it.

  A smoke     one leg of the stack (and of the ML-only arm if the stack fails): which candidate runs at all.
  B pilot     interleaved: master, #379, the candidate at 978 and 963 scheduled steps (+ mlonly at 978), and, only
              when the attempt is given --streamret-cuts, the stack with stream retrieval (streamret) at those cuts.
  C decision  the pre-registered rule below picks the candidate and its step count from the pilot.
  D certify   a fresh interleaved pool: the candidate, master and #379. Only this pool enters the p-value.

Every leg runs through tools/speedrun_ab/ab_bench.py, one --out per phase under --runs. Its ledgers make the attempt
resumable: re-running picks up where it stopped and never re-runs a leg it recorded. The attempt's settings, arms,
rule and the source of every arm are fixed in attempt.json at the first start, and checked at every phase.
At the end it writes the record folder (make_record.py) and a verdict.
"""
import argparse
import datetime
import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
AB_BENCH = REPO / "tools/speedrun_ab/ab_bench.py"
sys.path.insert(0, str(REPO / "tools/speedrun_ab"))
import ab_bench  # noqa: E402
import ab_stats  # noqa: E402

FAMILIES = ("stack", "mlonly")       # candidates in order of preference
STEP_OPTIONS = (978, 963)            # scheduled steps; +72 growth and extension steps are trained on top
GROWTH_STEPS = 72
GATE_963 = 3.2765                    # >= 3.5 millinats under 3.28
# The stack plus stream-only retrieval (tools/stream_retrieval/make_streamret_arm.sh builds that arm), run with
# STREAM_RETRIEVAL=1. Its gain only pays as a step cut, so it has no default cuts: they come from a measured gain
# (one STREAM_RETRIEVAL=1 dev run's `gain:` line, tools/stream_retrieval/README.md) and are passed as --streamret-cuts.
RETRIEVAL = "streamret"
MIN_PER_LEG, MIN_PER_COMPILE, MIN_SETUP = (2, 3), 7, 15   # per warm leg: a range, never measured for this trainer
STOPPED = 3                          # ab_bench's exit code after --max-crashes


def parse_cuts(text: str | None) -> tuple[int, ...]:
    """--streamret-cuts '940,900' -> (900, 940): scheduled step counts below the stack's 978, ascending."""
    cuts = tuple(sorted({int(c) for c in (text or "").replace(" ", "").split(",") if c}))
    if any(not 0 < c < STEP_OPTIONS[0] for c in cuts):
        raise SystemExit(f"--streamret-cuts {text}: each cut is a scheduled step count below {STEP_OPTIONS[0]}")
    return cuts


def rule_text(legs_pilot: int, legs_cert: int, legs_cert_base: int, cuts: tuple[int, ...] = ()) -> str:
    retrieval = bool(cuts)
    step4 = f"""\
4. Stream retrieval: streamret (the stack plus stream retrieval, STREAM_RETRIEVAL=1) runs {legs_pilot} pilot legs at
   each of {" and ".join(map(str, cuts))} scheduled steps (cuts set from a dev run's measured gain). Its step count is
   the lowest of these whose {legs_pilot} legs all finished with a mean final val <= {GATE_963}; if none, it is not
   eligible. It replaces the candidate of rules 2-3 only if all its pilot legs finished and its pilot train time at
   that step count is below the candidate's (at the step count of rule 3) by more than twice the Welch standard
   error of the difference.
""" if retrieval else ""
    return f"""\
PRE-REGISTERED DECISION RULE (fixed before the first leg; applied by tools/record_attempt/attempt.py)
1. Viable: a candidate (stack = this branch; mlonly = #379 + #375 without the systems patches) is viable
   only if every leg it ran (smoke and pilot, at both step counts) finished with a final validation.
2. Candidate: stack, unless it is not viable, or its pilot train time at 978 steps exceeds mlonly's by
   more than twice the Welch standard error of the difference; then mlonly, if viable. Neither: stop.
3. Step count: 963 scheduled steps (1035 trained) only if all {legs_pilot} pilot legs of the candidate at 963
   finished with a mean final val <= {GATE_963} (>= 3.5 millinats under 3.28); otherwise 978 (1050 trained).
{step4}{5 if retrieval else 4}. Certification: a fresh interleaved pool of the candidate (n={legs_cert}) with master (n={legs_cert_base}) and
   #379 (n={legs_cert_base}). Every leg is kept and counted, and only this pool enters the p-value; smoke and
   pilot legs are reported, never pooled.
"""


def family_of(arm: str) -> str:
    """stack963 -> stack, streamret900 -> streamret; master and pr379 are their own."""
    stem = arm.rstrip("0123456789")
    return stem if stem in FAMILIES + (RETRIEVAL,) else arm


def finished(record: dict) -> bool:
    return record.get("val_loss") is not None


def source_digest(arm_dir: Path) -> str:
    """sha256 of what a run log embeds: train_gpt.py and every module (and C source) of track_1_short/
    (run_log.read_source)."""
    h = hashlib.sha256()
    package = arm_dir / "track_1_short"
    for path in [arm_dir / "train_gpt.py"] + sorted([*package.rglob("*.py"), *package.rglob("*.c")]):
        if path.exists():
            h.update(str(path.relative_to(arm_dir)).encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def pilot_arms(families: list[str], cuts: tuple[int, ...] = ()) -> list[str]:
    """The pilot's arms for the viable families in `families` (the first is the primary candidate), and the stream
    retrieval arms at `cuts` when the attempt has them."""
    primary = families[0]
    arms = ["master", "pr379"] + [f"{primary}{steps}" for steps in STEP_OPTIONS]
    return arms + [f"{f}978" for f in families[1:]] + [f"{RETRIEVAL}{c}" for c in cuts]


def cert_legs(candidate_arm: str, legs_cert: int, legs_cert_base: int) -> dict[str, int]:
    return {"master": legs_cert_base, "pr379": legs_cert_base, candidate_arm: legs_cert}


def decide(pilot_order: list[str], ledger: list[dict], smoke_ok: dict[str, bool]) -> dict:
    """The pre-registered rule, applied to the pilot's plan and ledger."""
    by_arm: dict[str, list[dict]] = {}
    for record in ledger:
        by_arm.setdefault(record["arm"], []).append(record)
    planned = {arm: pilot_order.count(arm) for arm in pilot_order}
    reasons = []

    def viable(family: str) -> bool:
        if not smoke_ok.get(family, True):
            reasons.append(f"{family}: not viable (its smoke leg did not finish)")
            return False
        arms = [arm for arm in planned if family_of(arm) == family]
        if not arms:
            return False
        for arm in arms:
            done = by_arm.get(arm, [])
            if len(done) < planned[arm] or not all(map(finished, done)):
                reasons.append(f"{family}: not viable ({arm}: {sum(map(finished, done))} of {planned[arm]} pilot legs "
                               "finished)")
                return False
        return True

    walls = {arm: [r["wall_ms"] / 1000 for r in rs if finished(r)] for arm, rs in by_arm.items()}
    vals = {arm: [r["val_loss"] for r in rs if finished(r)] for arm, rs in by_arm.items()}
    candidate = None
    if viable("stack"):
        candidate = "stack"
        if viable("mlonly") and min(len(walls["stack978"]), len(walls["mlonly978"])) >= 2:
            diff, _, _, se = ab_stats.welch(walls["stack978"], walls["mlonly978"])
            slower = diff > 2 * se
            reasons.append(f"stack978 - mlonly978 train time {diff:+.3f} s, 2 se = {2 * se:.3f} s: "
                           + ("stack is slower beyond noise, so mlonly" if slower else "stack stays"))
            candidate = "mlonly" if slower else "stack"
    elif viable("mlonly"):
        candidate = "mlonly"
    if candidate is None:
        reasons.append("no viable candidate: stop before certification")
        return dict(candidate=None, steps=None, arm=None, reasons=reasons)
    steps = 978
    arm963 = f"{candidate}963"
    if arm963 in planned:
        mean = sum(vals[arm963]) / len(vals[arm963])
        steps = 963 if mean <= GATE_963 else 978
        reasons.append(f"{arm963}: mean pilot val {mean:.5f} over {len(vals[arm963])} legs "
                       f"{'<=' if steps == 963 else '>'} {GATE_963}: {steps} scheduled steps")
    else:
        reasons.append(f"{candidate} has no pilot legs at 963 steps: 978 scheduled steps")
    base = f"{candidate}{steps}"
    retrieval_arms = [arm for arm in planned if family_of(arm) == RETRIEVAL]
    if retrieval_arms:
        cut = None
        if viable(RETRIEVAL):
            for arm in sorted(retrieval_arms, key=lambda a: int(a[len(RETRIEVAL):])):
                mean = sum(vals[arm]) / len(vals[arm])
                reasons.append(f"{arm}: mean pilot val {mean:.5f} over {len(vals[arm])} legs "
                               f"{'<=' if mean <= GATE_963 else '>'} {GATE_963}")
                if mean <= GATE_963:
                    cut = arm
                    break
        if cut is None:
            reasons.append(f"{RETRIEVAL}: not eligible (no step cut passed): {base} stays")
        elif min(len(walls[cut]), len(walls[base])) < 2:
            reasons.append(f"{RETRIEVAL}: fewer than 2 finished legs to compare with {base}: {base} stays")
        else:
            diff, _, _, se = ab_stats.welch(walls[cut], walls[base])
            better = diff < -2 * se
            reasons.append(f"{cut} - {base} train time {diff:+.3f} s, 2 se = {2 * se:.3f} s: "
                           + (f"faster beyond noise, so {cut}" if better else f"not faster beyond noise, so {base} stays"))
            if better:
                return dict(candidate=RETRIEVAL, steps=int(cut[len(RETRIEVAL):]), arm=cut, reasons=reasons, base=base)
    return dict(candidate=candidate, steps=steps, arm=base, reasons=reasons)


class Attempt:
    def __init__(self, args):
        self.args = args
        self.runs = Path(args.runs).resolve()
        self.arms, self.given = {}, {}
        for spec in args.arm:
            name, _, path = spec.partition("=")
            self.arms[name], self.given[name] = Path(path).resolve(), path
        missing = {"master", "pr379", "stack"} - set(self.arms)
        assert not missing, f"missing --arm for {sorted(missing)}"
        self.cuts = parse_cuts(args.streamret_cuts)
        self.retrieval = RETRIEVAL in self.arms
        if self.retrieval != bool(self.cuts):
            raise SystemExit("the streamret arm and --streamret-cuts go together (STREAMRET_CUTS in run.sh)")
        self.settings = dict(legs_pilot=args.legs_pilot, legs_cert=args.legs_cert, legs_cert_base=args.legs_cert_base,
                             cold=args.cold, command=args.command, max_crashes=args.max_crashes, data=args.data)
        if self.cuts:
            self.settings["streamret_cuts"] = list(self.cuts)
        self.rule = rule_text(args.legs_pilot, args.legs_cert, args.legs_cert_base, self.cuts)

    # ------------------------------------------------------------------ plan

    def plan(self) -> str:
        s = self.settings
        families = [f for f in FAMILIES if f in self.arms]
        pilot = ab_bench.leg_order(pilot_arms(families, self.cuts), s["legs_pilot"])
        cert = ab_bench.leg_order(["master", "pr379", "CAND"], cert_legs("CAND", s["legs_cert"], s["legs_cert_base"]))
        legs = 1 + len(pilot) + len(cert)
        compiles = 2 + len(families) + self.retrieval
        minutes = [legs * per_leg + compiles * MIN_PER_COMPILE + MIN_SETUP + legs * MIN_PER_COMPILE * s["cold"]
                   for per_leg in MIN_PER_LEG]
        lines = [
            "PLAN",
            "  A smoke:   1 leg of stack at 978 steps (+1 of mlonly only if the stack leg fails)",
            f"  B pilot:   {len(pilot)} legs, interleaved: {' '.join(pilot)}",
            "  C decide:  the rule below picks the candidate (CAND) and its step count",
            f"  D certify: {len(cert)} legs, interleaved: {' '.join(cert)}",
            f"  total: {legs} legs at ~{MIN_PER_LEG[0]}-{MIN_PER_LEG[1]} min each ({'cold' if s['cold'] else 'warm'} "
            f"compile caches), {compiles} first-time compiles at ~{MIN_PER_COMPILE} min, ~{MIN_SETUP} min of setup: "
            f"~{minutes[0] / 60:.1f}-{minutes[1] / 60:.1f} h, ${15 * minutes[0] / 60:.0f}-{32 * minutes[1] / 60:.0f} at "
            "$15-32 per node-hour (an estimate: no leg of this trainer has been timed yet; a hung leg adds up to the "
            "leg timeout)",
            "  arms: " + ", ".join(f"{name}={path}" for name, path in self.given.items()),
        ]
        if "mlonly" not in self.arms:
            lines.append("  (no mlonly arm: the fallback candidate is unavailable)")
        if self.retrieval:
            lines.append(f"  streamret = the stack plus stream retrieval (make_streamret_arm.sh), STREAM_RETRIEVAL=1, pilot "
                         f"at {' and '.join(map(str, self.cuts))} scheduled steps (rule 4)")
        return "\n".join(lines) + "\n\n" + self.rule

    # ------------------------------------------------------------------ state

    def open_state(self) -> dict:
        """attempt.json: written at the first start, then the contract every later phase and resume is held to."""
        path = self.runs / "attempt.json"
        sources = {name: source_digest(path_) for name, path_ in self.arms.items()}
        if not path.exists():
            self.runs.mkdir(parents=True, exist_ok=True)
            now = datetime.datetime.now(datetime.timezone.utc)
            state = dict(created=now.isoformat(timespec="seconds"), date=now.strftime("%Y-%m-%d"), rule=self.rule,
                         settings=self.settings, arms={k: str(v) for k, v in self.arms.items()}, sources=sources,
                         notes=list(self.args.note))
            path.write_text(json.dumps(state, indent=1) + "\n")
            (self.runs / "PREREGISTRATION.txt").write_text(f"written {state['created']}, before the first leg\n\n"
                                                           + self.rule)
            return state
        state = json.loads(path.read_text())
        problems = [f"{key} changed: {state[key]} -> {now}" for key, now in
                    (("rule", self.rule), ("settings", self.settings),
                     ("arms", {k: str(v) for k, v in self.arms.items()})) if state[key] != now]
        problems += [f"the source of arm {name} changed since the attempt started" for name in state["sources"]
                     if sources.get(name) != state["sources"][name]]
        if problems:
            raise SystemExit("This attempt (" + str(path) + ") was started with other settings, arms or code:\n  "
                             + "\n  ".join(problems) + "\nEvery leg so far counts under the original ones: restore "
                             "them, or start a separate attempt (RUNS=<a new directory>; disclose both).")
        new_notes = [n for n in self.args.note if n not in state["notes"]]
        if new_notes:
            state["notes"] += new_notes
            path.write_text(json.dumps(state, indent=1) + "\n")
        return state

    def check_sources(self, state: dict):
        changed = [name for name, digest in state["sources"].items() if source_digest(self.arms[name]) != digest]
        if changed:
            raise SystemExit(f"STOPPED: the source of {changed} changed during the attempt. Every leg must run the code "
                             "the attempt started with: restore it and re-run to resume.")

    # ------------------------------------------------------------------ legs

    def bench(self, phase: str, arms: dict[str, str], legs: dict[str, int], droppable=()) -> list[dict]:
        """Run (or resume) one phase's legs with ab_bench.py; return its ledger. arms: name -> family."""
        s = self.settings
        cmd = [sys.executable, str(AB_BENCH), "--out", str(self.runs / phase), "--legs", "1",
               "--max-crashes", str(s["max_crashes"]), "--leg-timeout", str(self.args.leg_timeout)]
        for name, family in arms.items():
            cmd += ["--arm", f"{name}={self.arms[family]}", "--arm-legs", f"{name}={legs[name]}"]
            if family in FAMILIES + (RETRIEVAL,):  # explicit step count, whatever the arm's default
                cmd += ["--arm-env", f"{name}:NUM_SCHEDULED_ITERATIONS={name[len(family):] or STEP_OPTIONS[0]}"]
            if family == RETRIEVAL:
                cmd += ["--arm-env", f"{name}:STREAM_RETRIEVAL=1"]
        cmd += [arg for name in droppable for arg in ("--droppable", name)]
        if s["data"]:
            cmd += ["--data-path", s["data"]]
        if s["cold"]:
            cmd.append("--cold")
        if s["command"]:
            cmd += ["--command", s["command"]]
        print(f"\n===== {phase}: {sum(legs.values())} legs of {', '.join(arms)}", flush=True)
        proc = subprocess.Popen(cmd)
        try:
            code = proc.wait()
        except KeyboardInterrupt:  # ab_bench got the same SIGINT: it stops its leg's process group, then exits
            proc.wait()
            raise
        except SystemExit:  # this process was terminated: pass it on, so the leg is stopped too
            proc.terminate()
            proc.wait()
            raise
        if code == STOPPED:
            self.stop(f"phase {phase} stopped: an arm crashed {s['max_crashes']} legs in a row (see the log above and "
                      f"{self.runs / phase}). The node or its environment is probably broken. Send back "
                      "send_back.tar.gz (next to the runs directory) and release the node rather than keep paying. If "
                      "you know the cause and can fix it without touching the code, re-running run.sh resumes this "
                      "attempt: the crashed legs stay counted.")
        if code != 0:
            raise SystemExit(f"ab_bench failed in phase {phase} (exit {code})")
        return ab_bench.read_ledger(self.runs / phase / "ledger.jsonl")

    def next_runs(self) -> Path:
        """A fresh RUNS directory next to this one, for a separate attempt."""
        return next(p for i in range(2, 100) if not (p := self.runs.parent / f"runs_{i}").exists())

    def stop(self, why: str):
        text = f"STOPPED: {why}\n"
        (self.runs / "verdict.txt").write_text(text)
        print("\n" + text, flush=True)
        raise SystemExit(STOPPED)

    # ------------------------------------------------------------------ phases

    def smoke(self) -> dict[str, bool]:
        ok = {}
        for family in (f for f in FAMILIES if f in self.arms):
            ledger = self.bench(f"smoke_{family}", {family: family}, {family: 1})
            ok[family] = bool(ledger) and all(map(finished, ledger))
            print(f"smoke {family}: {'finished' if ok[family] else 'did NOT finish'}"
                  + (f" (val {ledger[0]['val_loss']}, {ledger[0]['wall_ms']} ms)" if ok[family] else ""), flush=True)
            if ok[family]:
                if ledger[0].get("seconds"):  # the first leg compiles; later legs of an arm reuse its caches
                    print(f"  this leg took {ledger[0]['seconds'] / 60:.1f} min with its first compile", flush=True)
                break
        return ok

    def decision(self, smoke_ok: dict[str, bool]) -> dict:
        families = [f for f in FAMILIES if f in self.arms and smoke_ok.get(f, True)]
        if not families:
            self.stop("no candidate finished its smoke leg (see the smoke_* folders and their stdout); nothing to "
                      "certify. This attempt is over: its smoke legs count, so re-running it stops here again. Send back "
                      "send_back.tar.gz. Once the cause is fixed (often the node: a missing C compiler or Python.h for "
                      "Triton, NCCL, the driver), start a new attempt in a new RUNS directory inside WORK, e.g. "
                      f"RUNS={self.next_runs()} bash tools/record_attempt/run.sh; its record README discloses "
                      "this one.")
        # streamret is the stack plus the retrieval overlay: it runs only if the stack's smoke leg finished.
        arms = pilot_arms(families, self.cuts if smoke_ok.get("stack", True) else ())
        legs = dict.fromkeys(arms, self.settings["legs_pilot"])
        # Master and #379 crashing in a row means the node is broken: stop. A candidate's arm only loses its legs.
        ledger = self.bench("pilot", {arm: family_of(arm) for arm in arms}, legs,
                            droppable=[a for a in arms if family_of(a) in FAMILIES + (RETRIEVAL,)])
        path = self.runs / "decision.json"
        decided = decide(ab_bench.leg_order(arms, legs), ledger, smoke_ok)
        decided["smoke"] = smoke_ok
        if path.exists():
            stored = json.loads(path.read_text())
            if {k: stored[k] for k in ("candidate", "steps")} != {k: decided[k] for k in ("candidate", "steps")}:
                raise SystemExit(f"{path} says {stored['arm']} but the pilot ledger now gives {decided['arm']}: "
                                 "the pilot changed after the decision; investigate before certifying anything")
            decided = stored
        else:
            path.write_text(json.dumps(decided, indent=1) + "\n")
        print("\nDECISION (pre-registered rule)\n  " + "\n  ".join(decided["reasons"]), flush=True)
        if decided["candidate"] is None:
            self.stop("the pilot left no viable candidate: " + "; ".join(decided["reasons"]))
        return decided

    def run(self) -> int:
        print(self.plan(), flush=True)
        if self.args.dry_run:
            return 0
        state = self.open_state()
        if self.args.record_name:  # an explicit name is never suffixed (make_record.py): refuse a clash now, not at the end
            folder = Path(self.args.records_dir) / f"{state['date']}_{self.args.record_name}"
            ours = self.runs / "record_folder.txt"
            if folder.exists() and not (ours.exists() and ours.read_text().strip() == str(folder)):
                raise SystemExit(f"{folder} exists and this attempt did not write it: choose another RECORD_NAME")
        self.check_sources(state)
        smoke_ok = self.smoke()
        self.check_sources(state)
        decided = self.decision(smoke_ok)
        self.check_sources(state)
        cand = decided["arm"]
        legs = cert_legs(cand, self.settings["legs_cert"], self.settings["legs_cert_base"])
        self.bench("cert", {"master": "master", "pr379": "pr379", cand: decided["candidate"]}, legs)
        self.check_sources(state)

        import make_record  # needs numpy and scipy, which the node's environment has by now
        folder, verdict = make_record.build(self.runs, state, decided, self.arms, Path(self.args.records_dir),
                                            self.args.record_name, Path(self.args.environment) if self.args.environment
                                            else None)
        (self.runs / "verdict.txt").write_text(verdict + f"\nrecord folder: {folder}\n")
        print("\n" + verdict + f"\nrecord folder: {folder}", flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, help="the attempt's directory: ledgers, logs, decision, verdict")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                        help="master, pr379, stack and (optionally) mlonly checkouts, and streamret (with --streamret-cuts)")
    parser.add_argument("--streamret-cuts", default=None, metavar="S1,S2",
                        help="the streamret arm's scheduled step counts (from a dev run's measured gain)")
    parser.add_argument("--data", default=None, help="DATA_PATH of every leg")
    parser.add_argument("--legs-pilot", type=int, default=3)
    parser.add_argument("--legs-cert", type=int, default=12, help="certification legs of the candidate")
    parser.add_argument("--legs-cert-base", type=int, default=6, help="certification legs of master and of #379, each")
    parser.add_argument("--cold", action="store_true", help="empty compile caches for every leg")
    parser.add_argument("--command", default=None, help="leg command (default: ab_bench's torchrun command)")
    parser.add_argument("--max-crashes", type=int, default=2, help="crashed legs in a row of one arm that stop a phase")
    parser.add_argument("--leg-timeout", type=float, default=1800,
                        help="seconds before a hung leg is killed and counted as a crash (compile included)")
    parser.add_argument("--records-dir", default=str(REPO / "records/track_1_short"))
    parser.add_argument("--record-name", default=None, help="the record folder's name after its date")
    parser.add_argument("--environment", default=None, help="the node's environment report, for the record README")
    parser.add_argument("--note", action="append", default=[], help="a disclosure for the record README")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the rule only")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(sig) is not signal.SIG_IGN:  # under nohup, a hangup stays ignored
            signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))
    sys.exit(Attempt(parser.parse_args()).run())


if __name__ == "__main__":
    main()
