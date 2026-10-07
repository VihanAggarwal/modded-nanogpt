#!/usr/bin/env bash
# Canon layers (CANON_LAYERS, track_1_short/model/gpt.py) on one 8xH100 node: which sites pay for themselves, and
# can they buy a step cut? Every arm is this checkout; only the environment differs. Run from a checkout of this
# branch:  bash tools/speedrun_ab/sweep_canon.sh
# Warm caches; 4 legs x 8 arms x ~2 min plus one compile per arm: ~2 h. LEGS=6 for tighter numbers.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-$HERE/../sweep_work}
LEGS=${LEGS:-4}
unset CANON_LAYERS CANON_LAYERS_K CANON_LAYERS_NORM CANON_LAYERS_BOS_MASK CANON_LAYERS_LR_MUL \
    NUM_SCHEDULED_ITERATIONS  # arms set their own
mkdir -p "$WORK"

# One copy of the data for every arm (the first 9 train shards and the val shard).
DATA=${DATA:-$WORK/data}
if [ ! -f "$DATA/data/fineweb10B/fineweb_val_000000.bin" ]; then
    mkdir -p "$DATA/data" && cp "$HERE/data/cached_fineweb10B.py" "$DATA/data/"
    (cd "$DATA" && python data/cached_fineweb10B.py 9)
fi

python "$HERE/tools/speedrun_ab/ab_bench.py" --legs "$LEGS" --data-path "$DATA" --out "$WORK/canon_$(date +%m%d_%H%M)" \
    --arm stack978="$HERE" \
    --arm canonAC="$HERE" --arm-env canonAC:CANON_LAYERS=AC \
    --arm canonA="$HERE" --arm-env canonA:CANON_LAYERS=A \
    --arm canonC="$HERE" --arm-env canonC:CANON_LAYERS=C \
    --arm canonAC950="$HERE" --arm-env canonAC950:CANON_LAYERS=AC --arm-env canonAC950:NUM_SCHEDULED_ITERATIONS=950 \
    --arm canonACrenorm="$HERE" --arm-env canonACrenorm:CANON_LAYERS=AC --arm-env canonACrenorm:CANON_LAYERS_NORM=renorm \
    --arm canonAClr3="$HERE" --arm-env canonAClr3:CANON_LAYERS=AC --arm-env canonAClr3:CANON_LAYERS_LR_MUL=3 \
    --arm canonACbos="$HERE" --arm-env canonACbos:CANON_LAYERS=AC --arm-env canonACbos:CANON_LAYERS_BOS_MASK=1
# Read report.txt against stack978 (the flag off). At 978 steps, 'd wall ms' is what the Canon layers cost per run
# and the val column what they buy; 'd adj ms' nets the two at 164 ms per millinat. canonAC950 is the step cut that
# would cash a gain in (28 fewer scheduled steps, ~0.9 s). Arms other than stack978 change the ML: a record needs
# its own p < 0.01 pool at the chosen setting, all runs counted (tools/RULES_CHECK.md); these legs only pick it.
# canonAClr3 gives the taps 3x the Adam lr (the proxy's was 2.5x ours, with ~2.3x the updates): a null canonAC
# next to a winning canonAClr3 means the taps were undertrained. canonACbos stops the taps at document starts.
#
# The per-step cost, kernel by kernel (two extra runs, not counted in the sweep: profiling perturbs the
# profiled steps). PROFILE_STEPS=a+b profiles those steps on rank 0 (train_gpt.py). At 978 steps, 100 and 101 are in
# stage 0 (16K tokens per rank) and 700 and 701 in stage 2 (49K); odd steps also run the Adam update:
#   export DATA_PATH="$DATA" PROFILE_STEPS=100+101+700+701 && cd "$HERE"
#   torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee profile_off.txt
#   CANON_LAYERS=AC torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee profile_canonAC.txt
#   grep PROFILE profile_off.txt profile_canonAC.txt
# (The run log, logs/<run id>.txt, does not record CANON_LAYERS: keep the console output under the setting's name.)
# The cost per step is the difference in 'PROFILE step S: total device self time' between the two runs (times the
# steps of that stage, for the cost per run); the kernels behind it are the ones new or slower in the second list
# (inductor names its Triton kernels after the ops they fuse). Expect each site to add kernels of its own: the conv
# reads neighbouring rows of the norm's output, so it cannot join the norm's per-row kernel, and the taps' gradient
# is a sum over tokens (~1 s per run for 16 sites, tools/RULES_CHECK.md). Nothing here has run on a GPU yet.
