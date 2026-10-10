#!/usr/bin/env bash
# Puts stream-only retrieval on top of a stack tree: copies arm/track_1_short/stream_{memory,lowtables,pointer}.{c,py}
# (the helper and its P2 / P3 parts, and their Python sides) into DIR/track_1_short and applies arm/hooks.patch (the
# trainer's hooks in train_gpt.py, track_1_short/data.py, track_1_short/run_log.py and track_1_short/model/gpt.py).
#   bash tools/stream_retrieval/apply_overlay.sh DIR
# The stack itself (this branch's train_gpt.py and track_1_short/) never carries the retrieval code: a stack record's
# source is only the stack. make_streamret_arm.sh builds the record attempt's streamret arm with this; the tests build
# a copy of the working tree with it.
set -euo pipefail
ARM_SRC=$(cd "$(dirname "$0")/arm" && pwd)
DIR=${1:?usage: apply_overlay.sh DIR}
DIR=$(cd "$DIR" && pwd)
die() { echo "apply_overlay: $*" >&2; exit 1; }
[ -f "$DIR/train_gpt.py" ] && [ -d "$DIR/track_1_short" ] || die "$DIR holds no stack tree (train_gpt.py, track_1_short/)"
[ ! -e "$DIR/track_1_short/stream_memory.py" ] || die "$DIR already has the stream-retrieval code"
# git apply works in a worktree and in a plain directory alike; --check first, so a stale patch changes nothing.
(cd "$DIR" && git apply --check "$ARM_SRC/hooks.patch") \
    || die "arm/hooks.patch does not apply to $DIR's train_gpt.py / track_1_short: the stack moved; regenerate the patch"
(cd "$DIR" && git apply "$ARM_SRC/hooks.patch")
for part in stream_memory stream_lowtables stream_pointer; do
    cp "$ARM_SRC/track_1_short/$part.c" "$ARM_SRC/track_1_short/$part.py" "$DIR/track_1_short/"
done
