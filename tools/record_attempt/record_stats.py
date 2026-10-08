"""Every number of a track-1 record folder, recomputed from its run logs alone.

tools/record_attempt copies this file into the record folder as statistics.py. (Its source is named record_stats.py
because a statistics.py next to tools that import the standard library's `statistics` would shadow it; this file
imports no module of that name, so it runs from the record folder.)

  python statistics.py         # the record folder this file is in
  python statistics.py DIR     # another folder laid out the same way

Folder layout: this_pr/ (the certification pool), baseline/ (master, interleaved with it), baseline_pr379/ (the open
PR this one builds on, interleaved too), pilot/<arm>/ and smoke/<arm>/ (development legs, reported, never pooled),
decision.json (the pre-registered rule's outcome), earlier_attempts/<name>/ (other attempts, laid out the same way).
The p-value is the maintainers' scipy.stats.ttest_1samp(vals, 3.28, alternative='less'). Every log counts: a log
without a final validation (including the crashed_*.txt stand-in of a leg that died before writing its log) is
reported as a crash, never dropped.
"""
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import scipy.stats

GATE = 3.28
# ANVIL2's exchange rate between wall time and val (tools/speedrun_ab/ab_stats.py): each millinat of mean val above
# 3.2775 costs 164 ms. It prices arms that land at different losses (e.g. different step counts) at equal loss.
MS_PER_MILLINAT = 164.0
TARGET_VAL = 3.2775
CERTIFICATION = ("this_pr", "baseline", "baseline_pr379")
FINAL_RE = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms")
STEP_RE = re.compile(r"^step:\d+/(\d+) ")


@dataclass
class Run:
    path: str
    steps: int | None     # trained steps, the N of the step:S/N lines
    wall: float | None    # seconds: train_time at the final validation
    val: float | None     # final val loss; None if the run never reached the final validation


def parse_log(path: str) -> Run:
    steps = wall = val = None
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("step:"):
                continue
            if m := STEP_RE.match(line):
                steps = int(m[1])
            if (m := FINAL_RE.match(line)) and m[1] == m[2]:
                val, wall = float(m[3]), int(m[4]) / 1000
    return Run(path, steps, wall, val)


def load(directory: str) -> list[Run]:
    return [parse_log(p) for p in sorted(glob.glob(os.path.join(directory, "*.txt")))]


def _sd(xs) -> float:
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else math.nan


def _outliers(runs: list[Run], attr: str, floor: float) -> list[Run]:
    """Runs farther from the arm's median than 5 robust sd (1.4826 MAD), and than `floor`."""
    if len(runs) < 3:
        return []
    xs = np.array([getattr(r, attr) for r in runs])
    median = np.median(xs)
    limit = max(5 * 1.4826 * np.median(np.abs(xs - median)), floor)
    return [r for r, x in zip(runs, xs) if abs(x - median) > limit]


def summarize(runs: list[Run]) -> dict:
    done = [r for r in runs if r.val is not None]
    walls, vals = [r.wall for r in done], [r.val for r in done]
    out = dict(n=len(runs), finished=len(done), crashed=[r.path for r in runs if r.val is None],
               steps=sorted({r.steps for r in runs if r.steps is not None}),
               above_gate=[r.path for r in done if r.val > GATE],
               outliers=[(r.path, "train time") for r in _outliers(done, "wall", 0.25)]
               + [(r.path, "val") for r in _outliers(done, "val", 0.005)],
               wall_mean=math.nan, wall_sd=math.nan, val_mean=math.nan, val_sd=math.nan, t=math.nan, p=math.nan)
    if done:
        out.update(wall_mean=float(np.mean(walls)), wall_sd=_sd(walls), val_mean=float(np.mean(vals)), val_sd=_sd(vals))
    if len(done) > 1:
        test = scipy.stats.ttest_1samp(vals, GATE, alternative="less")
        out.update(t=float(test.statistic), p=float(test.pvalue))
    return out


def _welch(a: list[float], b: list[float]) -> tuple[float, float, float, float]:
    """mean(a) - mean(b), its standard error, Welch's df and t."""
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    se = math.sqrt(va + vb)
    diff = float(np.mean(a) - np.mean(b))
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se else float(len(a) + len(b) - 2)
    return diff, se, df, (diff / se if se else math.copysign(math.inf, diff) if diff else 0.0)


def adjusted(run: Run) -> float:
    """Train time at equal loss: + 164 ms per millinat of val above 3.2775."""
    return run.wall + MS_PER_MILLINAT / 1000 * (run.val - TARGET_VAL) * 1000


def compare(cand: list[Run], base: list[Run]) -> dict | None:
    """Candidate minus baseline over the finished runs: train time (one-sided p that the candidate is faster), val
    (millinats, two-sided p) and the loss-adjusted train time. None with fewer than 2 finished runs a side."""
    a, b = [r for r in cand if r.val is not None], [r for r in base if r.val is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    wall, wall_se, df, t = _welch([r.wall for r in a], [r.wall for r in b])
    val, val_se, val_df, val_t = _welch([r.val for r in a], [r.val for r in b])
    adj, adj_se, adj_df, adj_t = _welch([adjusted(r) for r in a], [adjusted(r) for r in b])
    return dict(wall=wall, wall_se=wall_se, df=df, t=t, p_faster=float(scipy.stats.t.cdf(t, df)),
                wall_pct=100 * wall / float(np.mean([r.wall for r in b])),
                val_mnat=1000 * val, val_se_mnat=1000 * val_se, p_val=float(2 * scipy.stats.t.sf(abs(val_t), val_df)),
                adj=adj, adj_se=adj_se, p_adj_faster=float(scipy.stats.t.cdf(adj_t, adj_df)))


def flags(name: str, runs: list[Run]) -> list[str]:
    s = summarize(runs)
    out = [f"{name}: {os.path.basename(p)} has no final validation (crashed or cut short); it still counts"
           for p in s["crashed"]]
    out += [f"{name}: {os.path.basename(p)} ended above {GATE}" for p in s["above_gate"]]
    out += [f"{name}: {os.path.basename(p)} is an outlier in {what} (kept)" for p, what in s["outliers"]]
    if len(s["steps"]) > 1:
        out.append(f"{name}: mixed step counts {s['steps']} (an environment override leaked into some legs?)")
    return out


def table(arms: dict[str, list[Run]]) -> list[str]:
    lines = [f"{'arm':<16} {'n':>3} {'done':>4} {'steps':>9} {'train time (s)':>19} {'final val':>21} "
             f"{'p(val<3.28)':>11} {'>3.28':>5}"]
    for name, runs in arms.items():
        s = summarize(runs)
        steps = "/".join(map(str, s["steps"])) or "-"
        lines.append(f"{name:<16} {s['n']:>3} {s['finished']:>4} {steps:>9} {s['wall_mean']:9.3f} +/- {s['wall_sd']:5.3f} "
                     f"{s['val_mean']:9.5f} +/- {s['val_sd']:7.5f} {s['p']:11.2g} {len(s['above_gate']):>5}")
    return lines


def describe_comparison(cand_name: str, base_name: str, c: dict | None) -> list[str]:
    if c is None:
        return [f"{cand_name} vs {base_name}: fewer than 2 finished runs on a side"]
    return [f"{cand_name} vs {base_name}: train time {c['wall']:+.3f} s (se {c['wall_se']:.3f}, {c['wall_pct']:+.2f} %, "
            f"Welch t={c['t']:.2f} df={c['df']:.1f}, one-sided p(faster)={c['p_faster']:.2g})",
            f"    val {c['val_mnat']:+.2f} millinats (se {c['val_se_mnat']:.2f}, two-sided p={c['p_val']:.2g}); "
            f"at equal loss ({MS_PER_MILLINAT:.0f} ms/millinat) {c['adj']:+.3f} s (se {c['adj_se']:.3f}, "
            f"one-sided p={c['p_adj_faster']:.2g})"]


def subdirs(folder: str, name: str) -> dict[str, list[Run]]:
    return {os.path.basename(d): load(d) for d in sorted(glob.glob(os.path.join(folder, name, "*"))) if os.path.isdir(d)}


def shipped(folder: str) -> dict[str, list[Run]]:
    """Every run of the configuration the certification pool tested: this_pr/, the pilot legs of the same arm, and
    the candidate's smoke leg if it ran the same step count. A sensitivity check only: the pilot fed the decision."""
    path = os.path.join(folder, "decision.json")
    decision = json.load(open(path)) if os.path.exists(path) else {}
    if not decision.get("arm") or not os.path.isdir(os.path.join(folder, "this_pr")):
        return {}
    cert = load(os.path.join(folder, "this_pr"))
    steps = {r.steps for r in cert if r.val is not None}
    parts = {"this_pr": cert, f"pilot/{decision['arm']}": load(os.path.join(folder, "pilot", decision["arm"])),
             f"smoke/{decision['candidate']}": [r for r in load(os.path.join(folder, "smoke", decision["candidate"]))
                                                if r.steps in steps]}
    return {name: runs for name, runs in parts.items() if runs}


def report(folder: str) -> str:
    cert = {name: load(os.path.join(folder, name)) for name in CERTIFICATION if os.path.isdir(os.path.join(folder, name))}
    lines = ["CERTIFICATION POOL (interleaved on one node; every leg counted)"] + table(cert)
    for base in ("baseline", "baseline_pr379"):
        if "this_pr" in cert and base in cert:
            lines += describe_comparison("this_pr", base, compare(cert["this_pr"], cert[base]))
    if "this_pr" in cert:
        s = summarize(cert["this_pr"])
        lines.append(f"rule 2: ttest_1samp(vals, {GATE}, alternative='less'): t={s['t']:.2f} p={s['p']:.3g} over "
                     f"{s['finished']} finished runs ({len(s['crashed'])} crashed): {'PASS' if s['p'] < 0.01 else 'FAIL'}")
    if parts := shipped(folder):
        s = summarize([r for runs in parts.values() for r in runs])
        lines.append(f"all runs of the shipped configuration ({' + '.join(parts)}): n={s['n']} finished={s['finished']} "
                     f"mean val {s['val_mean']:.5f} p={s['p']:.3g} (a check, not the claim: the pilot fed the decision)")
    for phase, title in (("pilot", "PILOT (picks the candidate and step count; not pooled)"), ("smoke", "SMOKE (not pooled)")):
        arms = subdirs(folder, phase)
        if arms:
            lines += ["", title] + table(arms)
            if phase == "pilot":
                for name in arms:
                    for base in ("master", "pr379"):
                        if name not in ("master", "pr379") and base in arms:
                            lines += describe_comparison(name, base, compare(arms[name], arms[base]))
    found = [f for prefix, arms in (("", cert), ("pilot/", subdirs(folder, "pilot")), ("smoke/", subdirs(folder, "smoke")))
             for name, runs in arms.items() for f in flags(prefix + name, runs)]
    lines += ["", "FLAGS"] + (found or ["none"])
    for other in sorted(glob.glob(os.path.join(folder, "earlier_attempts", "*"))):
        arms = {name: load(os.path.join(other, name)) for name in CERTIFICATION if os.path.isdir(os.path.join(other, name))}
        lines += ["", f"EARLIER ATTEMPT {os.path.basename(other)} (disclosed, not pooled)"]
        lines += table(arms) if arms else ["  no certification pool"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))))
