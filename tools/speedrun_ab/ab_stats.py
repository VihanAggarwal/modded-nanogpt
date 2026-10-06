"""Statistics for a speedrun A/B comparison: wall time, final val loss, and where the time moved.

Reads the trainer's run logs (logs/<uuid>.txt; the lines `step:S/N train_time:Tms` and the final
`step:N/N val_loss:V train_time:Tms`) for two arms and reports what the rules ask for:

  rule 2  one-sided t-test that the arm's mean final val loss is below the 3.28 gate (p < 0.01);
          waived for systems-only changes, which should instead show val indistinguishable from
          the baseline (the two-sided Welch test on val below).
  rule 4  faster than the prior record on the same hardware: Welch's t-test on wall time,
          one-sided (candidate < baseline), with a 95% confidence interval on the difference.

The per-interval table splits the wall into the logged 25-step intervals plus the final
validation, so a gain (or a regression) can be pinned to the part of the run it came from.

Dependency-free (no scipy on a stock speedrun node): the Student-t CDF is computed from the
regularized incomplete beta function (Numerical Recipes' continued fraction).

Usage:
    python ab_stats.py --baseline 'runs/baseline/*.txt' --candidate 'runs/candidate/*.txt'
"""
import argparse
import glob
import math
import re
import statistics
from dataclasses import dataclass, field

GATE = 3.28
# Loss-adjusted wall: the record's own exchange rate between val loss and wall time (ANVIL2 README: 164 ms per
# millinat, the marginal cost of buying val with extra steps), and the val a record is kept at (~2.5 millinats
# below the gate). Compares variants that land at different losses: wall + rate * (val - target).
MS_PER_MILLINAT = 164.0
TARGET_VAL = 3.2775
STEP_RE = re.compile(r"^step:(\d+)/(\d+) (?:val_loss:([\d.]+) )?train_time:(\d+)ms")


@dataclass
class Run:
    path: str
    total_steps: int
    wall_ms: int                      # train_time at the final validation
    val_loss: float
    step_ms: dict[int, int] = field(default_factory=dict)  # cumulative train_time at logged steps


def parse_log(path: str) -> Run | None:
    """The run's final validation and its cumulative step timings; None if the run did not finish."""
    step_ms, final = {}, None
    total = None
    with open(path, errors="replace") as f:
        for line in f:
            m = STEP_RE.match(line)
            if not m:
                continue
            step, total = int(m.group(1)), int(m.group(2))
            if m.group(3) is not None:
                if step == total:
                    final = (int(m.group(4)), float(m.group(3)))
            else:
                step_ms[step] = int(m.group(4))
    if final is None:
        return None
    return Run(path, total, final[0], final[1], step_ms)


# ---------------------------------------------------------------- Student t, no scipy

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with df degrees of freedom."""
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Inverse of t_cdf by bisection (only used for confidence intervals)."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def one_sample_less(xs: list[float], mu: float) -> tuple[float, float]:
    """One-sided one-sample t-test of mean(xs) < mu: (t, p)."""
    n = len(xs)
    gap, sd = statistics.mean(xs) - mu, statistics.stdev(xs)
    if sd == 0:  # identical runs (e.g. seeded): the sign of the gap decides
        return math.copysign(math.inf, gap), 0.0 if gap < 0 else 1.0
    t = gap / (sd / math.sqrt(n))
    return t, t_cdf(t, n - 1)


def welch(a: list[float], b: list[float]) -> tuple[float, float, float, float]:
    """Welch's test of mean(a) - mean(b): (diff, t, df, standard error)."""
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    se = math.sqrt(va + vb)
    diff = statistics.mean(a) - statistics.mean(b)
    if se == 0:  # both arms constant
        return diff, (math.copysign(math.inf, diff) if diff else 0.0), float(len(a) + len(b) - 2), 0.0
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return diff, diff / se, df, se


# ---------------------------------------------------------------- report

def load(pattern: str) -> tuple[list[Run], list[str]]:
    runs, unfinished = [], []
    for path in sorted(glob.glob(pattern)):
        run = parse_log(path)
        (runs.append(run) if run is not None else unfinished.append(path))
    return runs, unfinished


def interval_means(runs: list[Run]) -> dict[tuple[int, int | str], float]:
    """Mean ms of each logged interval (prev logged step -> step) over the runs, plus the final
    validation's on-clock section as (last logged step, "val")."""
    common = sorted(set.intersection(*(set(r.step_ms) for r in runs)))
    out, prev = {}, 0
    for step in common:
        out[(prev, step)] = statistics.mean(r.step_ms[step] - r.step_ms.get(prev, 0) for r in runs)
        prev = step
    if prev:
        out[(prev, "val")] = statistics.mean(r.wall_ms - r.step_ms[prev] for r in runs)
    return out


def describe(name: str, runs: list[Run]) -> list[str]:
    walls = [r.wall_ms / 1000 for r in runs]
    vals = [r.val_loss for r in runs]
    lines = [f"{name}: n={len(runs)}  wall {statistics.mean(walls):.3f} +/- {statistics.stdev(walls):.3f} s  "
             f"val {statistics.mean(vals):.5f} +/- {statistics.stdev(vals):.5f}  "
             f"(runs above {GATE}: {sum(v > GATE for v in vals)})"]
    t, p = one_sample_less(vals, GATE)
    lines.append(f"  rule 2: one-sided t vs {GATE}: t={t:.2f}  p={p:.3g}  {'PASS' if p < 0.01 else 'FAIL'} (p < 0.01)")
    return lines


def report(baseline: list[Run], candidate: list[Run]) -> str:
    lines = describe("baseline ", baseline) + describe("candidate", candidate)
    wb = [r.wall_ms / 1000 for r in baseline]
    wc = [r.wall_ms / 1000 for r in candidate]
    diff, t, df, se = welch(wc, wb)
    half = t_ppf(0.975, df) * se
    lines.append(f"rule 4: wall candidate - baseline = {diff * 1000:+.1f} ms  (95% CI {1000 * (diff - half):+.1f} .. "
                 f"{1000 * (diff + half):+.1f} ms, {100 * diff / statistics.mean(wb):+.2f}%)  "
                 f"Welch t={t:.2f} df={df:.1f}  one-sided p(candidate faster)={t_cdf(t, df):.3g}")
    vdiff, vt, vdf, _ = welch([r.val_loss for r in candidate], [r.val_loss for r in baseline])
    p_two = 2 * min(t_cdf(vt, vdf), 1 - t_cdf(vt, vdf))
    lines.append(f"val candidate - baseline = {vdiff * 1000:+.2f} millinats  (Welch two-sided p={p_two:.3g}; a systems-only "
                 f"change should not move it)")
    # Intervals only line up between arms that train the same number of steps.
    if {r.total_steps for r in baseline} == {r.total_steps for r in candidate} and len({r.total_steps for r in baseline}) == 1:
        ib, ic = interval_means(baseline), interval_means(candidate)
        lines.append("")
        lines.append("interval        baseline_ms  candidate_ms   delta_ms")
        for key in (k for k in ib if k in ic):
            label = f"{key[0]}-{key[1]}"
            lines.append(f"{label:>13}  {ib[key]:11.1f}  {ic[key]:12.1f}  {ic[key] - ib[key]:+9.1f}")
    else:
        lines.append("(per-interval table skipped: the arms train different step counts)")
    return "\n".join(lines)


def adjusted_wall(runs: list[Run]) -> float:
    """Mean wall (s) at TARGET_VAL: each millinat of val above it costs MS_PER_MILLINAT."""
    val = statistics.mean(r.val_loss for r in runs)
    return statistics.mean(r.wall_ms for r in runs) / 1000 + MS_PER_MILLINAT / 1000 * (val - TARGET_VAL) * 1000


def report_many(arms: dict[str, list[Run]]) -> str:
    """One row per arm against the first: the sweep view (variants, step counts)."""
    names = list(arms)
    base = arms[names[0]]
    lines = [f"{'arm':>24} {'n':>3} {'steps':>6} {'wall s':>15} {'val':>17} {'p(val<3.28)':>11} {'d wall ms':>10} "
             f"{'adj wall s':>10} {'d adj ms':>9}",
             "(adj wall = wall + 164 ms per millinat of mean val above 3.2775: variants compared at equal loss)"]
    for name in names:
        runs = arms[name]
        if not runs:
            lines.append(f"{name:>24}   0  (no finished run)")
            continue
        walls = [r.wall_ms / 1000 for r in runs]
        vals = [r.val_loss for r in runs]
        steps = "/".join(sorted({str(r.total_steps) for r in runs}))
        sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else float("nan")
        p = one_sample_less(vals, GATE)[1] if len(vals) > 1 else float("nan")
        d_wall = 1000 * (statistics.mean(walls) - statistics.mean(r.wall_ms / 1000 for r in base)) if base else float("nan")
        adj = adjusted_wall(runs)
        d_adj = 1000 * (adj - adjusted_wall(base)) if base else float("nan")
        lines.append(f"{name:>24} {len(runs):>3} {steps:>6} {statistics.mean(walls):7.3f}+/-{sd(walls):5.3f} "
                     f"{statistics.mean(vals):.5f}+/-{sd(vals):.5f} {p:11.3g} {d_wall:+10.0f} {adj:10.3f} {d_adj:+9.0f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", help="glob of the baseline arm's run logs")
    parser.add_argument("--candidate", help="glob of the candidate arm's run logs")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=GLOB",
                        help="sweep mode: one per arm, the first is the baseline")
    args = parser.parse_args()
    if args.arm:
        arms, bad = {}, []
        for spec in args.arm:
            name, _, pattern = spec.partition("=")
            arms[name], unfinished = load(pattern)
            bad += unfinished
        for path in bad:
            print(f"NOTE: no final validation in {path} (crashed or unfinished run); all runs count, so investigate it")
        print(report_many(arms))
        return
    baseline, b_bad = load(args.baseline)
    candidate, c_bad = load(args.candidate)
    for path in b_bad + c_bad:
        print(f"NOTE: no final validation in {path} (crashed or unfinished run); all runs count, so investigate it")
    if len(baseline) < 2 or len(candidate) < 2:
        raise SystemExit("need at least two finished runs per arm")
    print(report(baseline, candidate))


if __name__ == "__main__":
    main()
