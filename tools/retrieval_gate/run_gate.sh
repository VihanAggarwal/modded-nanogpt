#!/usr/bin/env bash
# The stream-only retrieval gate on one 8xH100 node: does exact-match retrieval keep its value when the validation
# index holds only the tokens the run trained on (rule-safe), instead of all 103 train shards (PR #367 as shipped)?
# Needs ~21 GB of shards, a Rust toolchain (installed if missing), #367's host RAM (its 103-shard index), and about
# 1-2 h with downloads and three seeds. Run from a checkout of this branch:  bash tools/retrieval_gate/run_gate.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-$HERE/../gate_work}
SEEDS=${SEEDS:-"1 2 3"}
mkdir -p "$WORK"
cd "$HERE"
git fetch -q https://github.com/KellerJordan/modded-nanogpt pull/380/head:pr380  # carries #367's commit ab5a8d2
[ -d "$WORK/pr367" ] || git worktree add -q --detach "$WORK/pr367" ab5a8d2
cd "$WORK/pr367"
if git apply --check "$HERE/tools/retrieval_gate/gate_on_367.patch" 2>/dev/null; then
    git apply "$HERE/tools/retrieval_gate/gate_on_367.patch"
fi
if ! command -v cargo >/dev/null; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"
fi
pip install -q -r requirements.txt ./exact_match
python data/cached_fineweb10B.py 103  # #367's validation index covers all 103 train shards
for seed in $SEEDS; do
    RETRIEVAL_GATE=1 TRAIN_SEED=$seed torchrun --standalone --nproc_per_node=8 train_gpt.py
done
python "$HERE/tools/retrieval_gate/gate_report.py" logs/*.txt
