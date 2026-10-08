#!/usr/bin/env bash
# The ML-only fallback arm: upstream master + PR #375 + PR #379 at pinned heads, with none of this branch's systems
# patches (they have never run on a GPU). If the fork's stack crashes or is slower on the node, this is the record
# candidate. Run from a checkout of this branch:
#   ARM=$(bash tools/record_attempt/make_mlonly_arm.sh [WORK])    # WORK defaults to ../record_work
# Prints the arm's directory ($WORK/mlonly) on stdout, progress on stderr. Idempotent: an intact arm is reused, a
# damaged one rebuilt in place (untracked files such as logs/ are kept). The arm reads its data from DATA_PATH
# (ab_bench.py --data-path).
# Both merges are clean, so there is no resolution patch. A fixed identity and date make the merge commits, not
# only the tree, the same on every node. The arm differs from this branch's train_gpt.py, track_1_short/ and
# requirements.txt by exactly the systems commits e38a6fc 422b636 90284e0 d8ab27c 4b65282 03adbef and the Canon
# flag 3ae6b37 (off by default): their diffs on those paths applied to the arm, with ee6cd98's resolution of
# train_gpt.py, give this branch's files byte for byte.
# The arm is always built from the pins, which are what this branch's stack merged and what run.sh's master and #379
# arms run: a moved head (an author's push to an open PR, a new master) is reported and otherwise ignored, so the
# arm, the stack and the same-node baselines keep matching. ALLOW_PR_DRIFT=1 builds from the fetched heads instead
# (no tree check). Offline, the pinned commits already in this checkout (they are in its history) are used.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${1:-${WORK:-$HERE/../record_work}}
mkdir -p "$WORK" && WORK=$(cd "$WORK" && pwd)
ARM=$WORK/mlonly
UPSTREAM=${UPSTREAM:-https://github.com/KellerJordan/modded-nanogpt}
MASTER=4ea6b937337a4889b8cfe3f38a93d120048d8f71  # record #360 (ANVIL2), #373's track_1_short refactor, README
PR375=85ee5d30d99aabea4b42419a472990bc5dac861d   # token-normalized n-gram hashes (Daniel Monroe)
PR379=c44cc41e1219d3c7f8b0b0e64b02a9ac39f64d85   # CPLM copy-sink pointer LM, 978 scheduled steps (NathanGodey, yoavartzi)
TREE=a67ca04d0f91a36c278701f1b798ed7b88affeee    # the arm's tree at these pins
say() { echo "make_mlonly_arm: $*" >&2; }
die() { say "$*"; exit 1; }

if git -C "$HERE" fetch -q --no-tags "$UPSTREAM" +refs/heads/master:refs/mlonly/master \
        +refs/pull/375/head:refs/mlonly/pr375 +refs/pull/379/head:refs/mlonly/pr379; then
    heads=$(git -C "$HERE" rev-parse refs/mlonly/master refs/mlonly/pr375 refs/mlonly/pr379 | tr '\n' ' ')
    if [ "$heads" != "$MASTER $PR375 $PR379 " ]; then
        say "WARNING: upstream moved: pinned master #375 #379 = $MASTER $PR375 $PR379"
        say "                                          fetched = ${heads% }"
        if [ "${ALLOW_PR_DRIFT:-0}" = 1 ]; then
            say "ALLOW_PR_DRIFT=1: building from the fetched heads"
            read -r MASTER PR375 PR379 <<< "$heads"
            TREE=""
        else
            say "building from the pins (what this branch merged and the baselines run); ALLOW_PR_DRIFT=1 uses the new heads"
        fi
    fi
else
    say "WARNING: fetch from $UPSTREAM failed; using the pinned commits in $HERE (drift not checked)"
fi
for c in $MASTER $PR375 $PR379; do
    git -C "$HERE" cat-file -e "$c^{commit}" 2>/dev/null || die "$c is not in $HERE (and could not be fetched from $UPSTREAM)"
done

merge() {  # merge <commit> <message>
    GIT_AUTHOR_DATE=2026-10-08T00:00:00Z GIT_COMMITTER_DATE=2026-10-08T00:00:00Z git -C "$ARM" \
        -c user.name=make_mlonly_arm -c user.email=make_mlonly_arm@localhost -c commit.gpgsign=false \
        -c core.hooksPath=/dev/null merge -q --no-ff -m "$2" "$1" >&2 \
        || die "merging $1 into $ARM conflicts (the pinned heads merge cleanly): resolve it there and re-pin"
}

git -C "$HERE" worktree prune
if [ "$(git -C "$ARM" rev-parse HEAD^1^1 HEAD^1^2 HEAD^2 2>/dev/null | tr '\n' ' ')" = "$MASTER $PR375 $PR379 " ] \
        && git -C "$ARM" diff --quiet HEAD; then
    say "reusing $ARM"
else
    if [ -e "$ARM" ]; then
        [ -e "$ARM/.git" ] || die "$ARM exists and is not a git worktree: move it away"
        git -C "$ARM" merge --abort 2>/dev/null || true
        git -C "$ARM" diff --quiet HEAD || die "$ARM has modified tracked files: discard them (git -C $ARM checkout -- .) and rerun"
        git -C "$ARM" checkout -q --detach "$MASTER"
    else
        git -C "$HERE" worktree add -q --detach "$ARM" "$MASTER"
    fi
    merge "$PR375" "Merge PR #375 (token-normalized n-gram hashes, Daniel Monroe) at ${PR375:0:8}"
    merge "$PR379" "Merge PR #379 (CPLM copy-sink pointer LM, NathanGodey & yoavartzi) at ${PR379:0:8}"
fi

tree=$(git -C "$ARM" rev-parse "HEAD^{tree}")
[ -z "$TREE" ] || [ "$tree" = "$TREE" ] || die "$ARM has tree $tree, pinned $TREE: the merge differs on this git ($(git --version))"
say "master ${MASTER:0:8} + #375 ${PR375:0:8} + #379 ${PR379:0:8} = $(git -C "$ARM" rev-parse --short=8 HEAD) (tree ${tree:0:8})"
echo "$ARM"
