#!/usr/bin/env bash
# Phase-1 sweep on one 8xH100 node: does the stack (systems patches + #378 + #379) beat its parts, and at
# what step count? Run from a checkout of this branch:  bash tools/speedrun_ab/sweep_stack.sh
# Warm caches; about 4 legs x 6 arms x ~2 min plus one compile per arm: ~1.5 h. LEGS=6 for tighter numbers.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-$HERE/../sweep_work}
LEGS=${LEGS:-4}
UPSTREAM=https://github.com/KellerJordan/modded-nanogpt
mkdir -p "$WORK"

cd "$HERE"
git fetch -q "$UPSTREAM" master:upstream-master pull/378/head:pr378 pull/379/head:pr379
for ref in upstream-master pr378 pr379; do
    [ -d "$WORK/$ref" ] || git worktree add -q --detach "$WORK/$ref" "$ref"
done
# One copy of the data for every arm (the first 9 train shards and the val shard).
DATA=${DATA:-$WORK/data}
if [ ! -f "$DATA/data/fineweb10B/fineweb_val_000000.bin" ]; then
    mkdir -p "$DATA/data" && cp "$HERE/data/cached_fineweb10B.py" "$DATA/data/"
    (cd "$DATA" && python data/cached_fineweb10B.py 9)
fi

python "$HERE/tools/speedrun_ab/ab_bench.py" --legs "$LEGS" --data-path "$DATA" --out "$WORK/runs_$(date +%m%d_%H%M)" \
    --arm master="$WORK/upstream-master" \
    --arm pr378="$WORK/pr378" \
    --arm pr379="$WORK/pr379" \
    --arm stack978="$HERE" \
    --arm stack950="$HERE" --arm-env stack950:NUM_SCHEDULED_ITERATIONS=950 \
    --arm stack920="$HERE" --arm-env stack920:NUM_SCHEDULED_ITERATIONS=920
# Read report.txt: 'd adj ms' is each arm's time against master at equal loss (164 ms per millinat).
# stack978 vs pr379 says whether #378 and the systems patches still add on top of CPLM; the stack's
# step-count rows locate the cheapest step count that keeps mean val clearly below 3.28.
