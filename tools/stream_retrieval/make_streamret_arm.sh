#!/usr/bin/env bash
# The record attempt's streamret arm: this branch's stack (HEAD) with stream-only retrieval on top
# (apply_overlay.sh: track_1_short/stream_memory.{c,py} and the trainer hooks in arm/hooks.patch), committed as one
# commit with a fixed identity and date, so the same HEAD gives the same arm commit on every node. The stack's own
# checkout never carries the retrieval code, so a stack record's source is only the stack. Run from a checkout:
#   ARM=$(bash tools/stream_retrieval/make_streamret_arm.sh [WORK])    # WORK defaults to ../record_work
# Prints the arm's directory ($WORK/streamret) on stdout, progress on stderr. Idempotent: an arm already built from
# this HEAD and overlay is reused; otherwise it is rebuilt in place (untracked files such as logs/ are kept).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${1:-${WORK:-$HERE/../record_work}}
mkdir -p "$WORK" && WORK=$(cd "$WORK" && pwd)
ARM=$WORK/streamret
say() { echo "make_streamret_arm: $*" >&2; }
die() { say "$*"; exit 1; }

BASE=$(git -C "$HERE" rev-parse HEAD)
# The arm is HEAD plus the committed overlay: uncommitted changes to either would not be in it.
dirty=$(git -C "$HERE" status --porcelain --untracked-files=all -- train_gpt.py track_1_short tools/stream_retrieval/arm \
        tools/stream_retrieval/apply_overlay.sh | grep -v '__pycache__' || true)
[ -z "$dirty" ] || die "uncommitted changes to the stack or the overlay (commit them first):
$dirty"
OVERLAY=$(cd "$HERE/tools/stream_retrieval" && cat arm/hooks.patch arm/track_1_short/stream_memory.c \
          arm/track_1_short/stream_memory.py apply_overlay.sh | sha256sum | cut -c1-16)
MESSAGE="Stream-only retrieval on top of ${BASE:0:12} (overlay $OVERLAY)"

git -C "$HERE" worktree prune
if [ "$(git -C "$ARM" rev-parse HEAD^ 2>/dev/null)" = "$BASE" ] && [ "$(git -C "$ARM" log -1 --format=%s 2>/dev/null)" = "$MESSAGE" ] \
        && git -C "$ARM" diff --quiet HEAD; then
    say "reusing $ARM"
else
    if [ -e "$ARM" ]; then
        [ -e "$ARM/.git" ] || die "$ARM exists and is not a git worktree: move it away"
        git -C "$ARM" diff --quiet HEAD || die "$ARM has modified tracked files: discard them (git -C $ARM checkout -- .) and rerun"
        git -C "$ARM" checkout -q --detach "$BASE"
        rm -f "$ARM/track_1_short/stream_memory.c" "$ARM/track_1_short/stream_memory.py"  # untracked at BASE
    else
        git -C "$HERE" worktree add -q --detach "$ARM" "$BASE"
    fi
    bash "$HERE/tools/stream_retrieval/apply_overlay.sh" "$ARM" || die "the overlay does not apply to ${BASE:0:12}"
    git -C "$ARM" add train_gpt.py track_1_short
    GIT_AUTHOR_DATE=2026-10-09T00:00:00Z GIT_COMMITTER_DATE=2026-10-09T00:00:00Z git -C "$ARM" \
        -c user.name=make_streamret_arm -c user.email=make_streamret_arm@localhost -c commit.gpgsign=false \
        -c core.hooksPath=/dev/null commit -q -m "$MESSAGE" >&2
fi
say "${BASE:0:8} + stream retrieval (overlay $OVERLAY) = $(git -C "$ARM" rev-parse --short=8 HEAD)"
echo "$ARM"
