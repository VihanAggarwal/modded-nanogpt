"""Read the stream-only retrieval gate's run logs and decide go / kill.

Each gated run (RETRIEVAL_GATE=1, gate_on_367.patch) prints its shipped final validation (the 103-shard index) and,
from the same weights, `gate val_loss[stream]` (an index of only the tokens the run trained on) and
`gate val_loss[none]` (no retrieval rows). Thresholds (millinats, means over seeds), from the vetted plan:
  go    stream - shipped <= 100  and  none - stream >= 40
  kill  stream - shipped >= 150  or   none - stream <  20
"""
import re
import statistics
import sys

FINAL = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+)")
GATE = re.compile(r"^gate val_loss\[(\w+)\]:([\d.]+)")


def parse(path: str) -> dict | None:
    out = {}
    for line in open(path, errors="replace"):
        if (m := FINAL.match(line)) and m.group(1) == m.group(2):
            out["shipped"] = float(m.group(3))
        elif m := GATE.match(line):
            out[m.group(1)] = float(m.group(2))
    return out if {"shipped", "stream", "none"} <= set(out) else None


def decide(runs: list[dict]) -> tuple[float, float, str]:
    lost = 1000 * statistics.mean(r["stream"] - r["shipped"] for r in runs)
    kept = 1000 * statistics.mean(r["none"] - r["stream"] for r in runs)
    if lost >= 150 or kept < 20:
        verdict = "KILL: the stream-only memory keeps too little; plan for about -7 s without it"
    elif lost <= 100 and kept >= 40:
        verdict = "GO: build the stream-only memory on the CPLM stack"
    else:
        verdict = "GREY: run more seeds, or price the extra steps the loss would cost"
    return lost, kept, verdict


def main(paths: list[str]):
    runs = [r for r in map(parse, paths) if r]
    if not runs:
        raise SystemExit("no gated runs found (need the shipped val line and both gate lines)")
    for r in runs:
        print(f"shipped {r['shipped']:.4f}  stream {r['stream']:.4f}  none {r['none']:.4f}")
    lost, kept, verdict = decide(runs)
    print(f"n={len(runs)}  stream - shipped = {lost:+.1f} mnat  none - stream = {kept:+.1f} mnat\n{verdict}")


if __name__ == "__main__":
    main(sys.argv[1:])
