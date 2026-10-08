#!/usr/bin/env bash
# A track-1 record attempt on one fresh 8xH100 node, in one command (tools/record_attempt/README.md):
#   git clone -b claude/nanogpt-optimization-n49ur2 https://github.com/VihanAggarwal/modded-nanogpt
#   cd modded-nanogpt && tmux new -s rec 'bash tools/record_attempt/run.sh; exec bash'
# --dry-run prints the checks and the full plan, and installs, downloads and runs nothing. Re-running resumes: a leg
# in a ledger never runs again. Knobs: WORK (default ../record_work), DATA, VENV_PYTHON (the python the venv is built
# from; default python3.12, python3 or python3.11, else a uv-built 3.12), PYTHON (use this python, no venv),
# RUNS (default WORK/runs; a new directory directly in WORK is a new attempt that reuses the venv, data and arms),
# LEGS_PILOT=3, LEGS_CERT=12, LEGS_CERT_BASE=6, COLD=1, MAX_CRASHES=2, LEG_TIMEOUT=1800 (s), RECORD_NAME,
# ALLOW_NEW_MASTER=1, ALLOW_PR_DRIFT=1 (make_mlonly_arm.sh). For tests: SETUP=0 skips the node setup; LEG_COMMAND,
# STACK_DIR, MASTER_DIR, PR379_DIR, MAKE_MLONLY, RECORDS_DIR.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
TOOLS=$HERE/tools/record_attempt
DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg (see --help)" >&2; exit 2 ;;
    esac
done

UPSTREAM=https://github.com/KellerJordan/modded-nanogpt
MASTER_SHA=4ea6b937337a4889b8cfe3f38a93d120048d8f71  # record #92 (ANVIL2): upstream master
PR379_SHA=c44cc41e1219d3c7f8b0b0e64b02a9ac39f64d85   # PR #379 (CPLM), the head this branch merged
WORK=$(realpath -m "${WORK:-$(dirname "$HERE")/record_work}")
DATA=$(realpath -m "${DATA:-$WORK/data}")
RUNS=$(realpath -m "${RUNS:-$WORK/runs}")
SETUP=${SETUP:-1}
STACK_DIR=${STACK_DIR:-$HERE}
MAKE_MLONLY=${MAKE_MLONLY:-$TOOLS/make_mlonly_arm.sh}
PY=${PYTHON:-python3}

say() { printf '%s\n' "$@"; }  # one line per argument
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
# A failed check stops the attempt; under --dry-run it is reported and the plan still prints.
check() { if [ "$DRY_RUN" = 1 ]; then say "  WOULD STOP HERE: $*"; else die "$*"; fi; }
act() { if [ "$DRY_RUN" = 1 ]; then say "  would run: $*"; else "$@"; fi; }
py_at_least() { command -v "$1" >/dev/null 2>&1 && "$1" -c "import sys; sys.exit(sys.version_info < ($2,))" 2>/dev/null; }

# Every attempt lives directly in WORK, where send_back.tar.gz and the record README's disclosure look for them.
[ "$(dirname "$RUNS")" = "$WORK" ] \
    || die "RUNS=$RUNS must be a directory directly inside WORK=$WORK, e.g. RUNS=$WORK/runs_2"
py_at_least "$PY" 3,10 || die "no python 3.10+ to run this script ($PY): apt-get install -y python3 python3-venv," \
    "or set PYTHON"

package() {  # every attempt in WORK (ledgers, logs, console logs, decisions, verdicts) and their record folders
    local d f parts=()
    for d in "$WORK"/*/; do
        d=${d%/}
        if [ -f "$d/attempt.json" ] || [ "$d" = "$RUNS" ]; then parts+=(-C "$WORK" "$(basename "$d")"); fi
        f=$(cat "$d/record_folder.txt" 2>/dev/null || true)
        if [ -n "$f" ] && [ -d "$f" ]; then parts+=(-C "$(dirname "$f")" "$(basename "$f")"); fi
    done
    say ""
    if [ -f "$RUNS/verdict.txt" ]; then say "Verdict: $RUNS/verdict.txt (cat it to see it again)."; fi
    if tar czf "$WORK/send_back.tar.gz" --exclude=caches "${parts[@]}"; then
        say "Send back: $WORK/send_back.tar.gz (every attempt's ledgers, logs, console log and verdict, and the" \
            "record folder if one was written). Copy it off the node before releasing it. This console's log: $log"
    else
        say "WARNING: could not write $WORK/send_back.tar.gz: send $RUNS instead. This console's log: $log"
    fi
}
if [ "$DRY_RUN" = 0 ]; then
    mkdir -p "$RUNS"
    log=$RUNS/console_$(date +%m%d_%H%M%S).log
    exec > >(trap '' INT TERM HUP; exec tee -a "$log") 2>&1  # tee outlives a Ctrl-C, so the last lines still land
    trap package EXIT  # from here on, every exit (a failed check too) writes send_back.tar.gz
    say "console log: $log"
    if [ -z "${TMUX:-}${STY:-}" ] && [ -t 0 ]; then
        say "WARNING: not inside tmux or screen, so an SSH drop kills this run. Ctrl-C now and start it as" \
            "  tmux new -s rec 'bash tools/record_attempt/run.sh; exec bash'   (re-running resumes)"
    fi
fi

# Every leg runs each arm's own defaults: no trainer knob, seed or compile setting may leak in from this shell (rule 3).
leaked=$(compgen -e | grep -E '^(CPLM|FA3_|TORCHINDUCTOR_|TORCHDYNAMO_|TORCH_COMPILE|INDUCTOR_)' \
    | grep -vE '_CACHE_DIR$' || true)
for v in ALLOW_4_GPUS ALL_SHORT COPY_TRUE_DOCS DATA_PATH LR_COOLDOWN_FRAC NGRAM_CACHE_MIN_ROWS NO_MTP NO_PREFIX \
         NUM_SCHEDULED_ITERATIONS PROFILE_STEPS PROFILE_TOP TRAIN_SEED VAL_EVERY WS_SCALE KX_STEPS KX_SEED; do
    if [ -n "${!v+x}" ]; then leaked="$leaked $v"; fi
done
if [ -n "${leaked// /}" ]; then say "unsetting trainer settings inherited from this shell:$leaked"; unset $leaked; fi

ATTEMPT=(--runs "$RUNS" --data "$DATA" --legs-pilot "${LEGS_PILOT:-3}" --legs-cert "${LEGS_CERT:-12}"
         --legs-cert-base "${LEGS_CERT_BASE:-6}" --max-crashes "${MAX_CRASHES:-2}"
         --records-dir "${RECORDS_DIR:-$HERE/records/track_1_short}" --leg-timeout "${LEG_TIMEOUT:-1800}")
if [ "${COLD:-0}" = 1 ]; then ATTEMPT+=(--cold); fi
if [ -n "${RECORD_NAME:-}" ]; then ATTEMPT+=(--record-name "$RECORD_NAME"); fi

say "===== record attempt: $STACK_DIR ($(git -C "$STACK_DIR" rev-parse --short HEAD 2>/dev/null || echo 'not git'))"
PLANNED=(--arm master="${MASTER_DIR:-$WORK/arms/master}" --arm pr379="${PR379_DIR:-$WORK/arms/pr379}" --arm stack="$STACK_DIR")
if ! grep -qs '^none:' "$RUNS/mlonly_arm.txt"; then PLANNED+=(--arm mlonly="(built by make_mlonly_arm.sh)"); fi
"$PY" "$TOOLS/attempt.py" --dry-run "${ATTEMPT[@]}" "${PLANNED[@]}"

# Free space per filesystem, summed over what lands on it: path, GB, what.
declare -A NEED=() FOR=()
need() {
    local p=$1 mnt
    while [ ! -d "$p" ]; do p=$(dirname "$p"); done
    mnt=$(df -P "$p" | awk 'NR==2 {print $6}')
    NEED[$mnt]=$(( ${NEED[$mnt]:-0} + $2 )); FOR[$mnt]="${FOR[$mnt]:-}, $3"
}

if [ "$SETUP" = 1 ]; then
    say "" "===== node checks"
    if ! command -v nvidia-smi >/dev/null; then
        check "nvidia-smi not found: this needs an 8xH100 node with the NVIDIA driver"
    else
        gpus=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true)
        say "$gpus" | sort | uniq -c | sed 's/^/  /'
        [ "$(grep -c H100 <<<"$gpus")" = 8 ] || check "expected 8 H100s; nvidia-smi lists the above"
        if grep -Eiq 'PCIe|NVL' <<<"$gpus"; then check "H100 PCIe/NVL, not SXM: records are timed on H100 SXM"; fi
        driver=$(head -1 <<<"$gpus" | cut -d, -f2 | tr -d ' ')
        [ "${driver%%.*}" -ge 580 ] 2>/dev/null \
            || check "driver $driver < 580: every step runs ~35% slower, silently (ANVIL2 README). Use another node"
        # Inside most containers nvidia-smi lists no processes, so memory in use counts as busy too.
        busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader || true)
        used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
               | awk -F', *' '$2 > 1024 {printf " GPU%s:%sMiB", $1, $2}' || true)
        [ -z "$busy$used" ] || check "the GPUs are busy (a leftover leg?):${busy:+ $busy}${used:+ memory in use on$used}." \
            "Stop it (pkill -f train_gpt.py), wait until nvidia-smi shows ~0 MiB used on every GPU, then re-run"
        if ! nvidia-smi topo -m 2>/dev/null | grep -q NV18; then say "  note: no NV18 links in nvidia-smi topo -m"; fi
    fi
    need "$WORK" 30 "WORK=$WORK (venv, data, logs)"
    need "${TORCHINDUCTOR_CACHE_DIR:-/tmp}" 10 "the Inductor cache (${TORCHINDUCTOR_CACHE_DIR:-/tmp})"
    need "${HOME:-/root}" 3 "the Triton and Hugging Face caches in ${HOME:-/root}"
    for mnt in "${!NEED[@]}"; do
        have=$(df -Pk "$mnt" | awk 'NR==2 {print int($4 / 1048576)}')
        [ "$have" -ge "${NEED[$mnt]}" ] || check "only $have GB free on $mnt, which holds ${FOR[$mnt]#, }: it needs" \
            "~${NEED[$mnt]} GB. Free some, or put WORK on a bigger disk: WORK=/path/with/space bash tools/record_attempt/run.sh"
    done
    mem=$(awk '/^MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    [ "$mem" -ge 64 ] || check "only $mem GB of RAM available (want >= 64)"
    say "  disk and memory checked ($mem GB RAM available)"

    say "" "===== upstream"
    remote=$(timeout 60 git ls-remote "$UPSTREAM" refs/heads/master refs/pull/379/head 2>/dev/null || true)
    remote_master=$(awk '$2 == "refs/heads/master" {print $1}' <<<"$remote")
    remote_379=$(awk '$2 == "refs/pull/379/head" {print $1}' <<<"$remote")
    if [ -z "$remote_master" ]; then
        say "  WARNING: could not reach $UPSTREAM to check that master is still ${MASTER_SHA:0:7}"
    elif [ "$remote_master" != "$MASTER_SHA" ] && [ "${ALLOW_NEW_MASTER:-0}" != 1 ]; then
        check "upstream master moved to ${remote_master:0:7} (a new record?) since this branch was built on" \
              "${MASTER_SHA:0:7}: a record must beat the current one. Rebase first, or ALLOW_NEW_MASTER=1 to measure anyway"
    else
        say "  master is ${MASTER_SHA:0:7}"
    fi
    if [ -n "$remote_379" ] && [ "$remote_379" != "$PR379_SHA" ]; then
        say "  WARNING: PR #379 moved to ${remote_379:0:7}; this branch merged ${PR379_SHA:0:7}, which the pr379 arm runs"
    fi

    say "" "===== python environment (3.11+: the stack starts a helper with python -P)"
    if [ -z "${PYTHON:-}" ]; then
        if [ ! -x "$WORK/venv/bin/python" ]; then
            base=${VENV_PYTHON:-}
            if [ -z "$base" ]; then
                for c in python3.12 python3 python3.11; do if py_at_least "$c" 3,11; then base=$c; break; fi; done
            fi
            if [ -n "$base" ]; then
                say "  a venv in $WORK/venv from $base ($("$base" --version 2>&1 || true))"
                act "$base" -m venv "$WORK/venv" || check "$base -m venv failed: apt-get install -y python3-venv" \
                    "(python3.12-venv for python3.12), or set VENV_PYTHON to another python >= 3.11"
            else
                say "  no python >= 3.11 on PATH ($("$PY" --version 2>&1)): building a Python 3.12 venv with uv"
                uv=$WORK/uv/bin/uv
                if [ ! -x "$uv" ]; then
                    act "$PY" -m venv "$WORK/uv" && act "$WORK/uv/bin/pip" install -q --no-cache-dir uv \
                        || act sh -c "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$WORK/uv/bin UV_NO_MODIFY_PATH=1 sh" \
                        || check "could not install uv: install python3.12 (or 3.11) and its venv package, or set VENV_PYTHON"
                    if [ ! -x "$uv" ] && [ -x "$WORK/uv/bin/bin/uv" ]; then uv=$WORK/uv/bin/bin/uv; fi  # older installers
                fi
                act "$uv" venv --seed --python 3.12 "$WORK/venv" \
                    || check "uv could not build a Python 3.12 venv: install python3.12 and python3.12-venv, or set VENV_PYTHON"
            fi
        fi
        PY=$WORK/venv/bin/python
    fi
    if [ "$DRY_RUN" = 0 ]; then
        py_at_least "$PY" 3,11 || die "$PY is Python $("$PY" --version 2>&1): the stack needs 3.11+." \
            "Remove the venv (rm -rf $WORK/venv) and re-run, or set VENV_PYTHON (or PYTHON) to a python >= 3.11"
    fi
    if ! "$PY" -c 'import torch, sys; sys.exit(torch.__version__ != "2.10.0+cu128")' 2>/dev/null; then
        say "  installing torch 2.10.0+cu128 (~3 GB with its CUDA libraries; a few minutes)"
        act "$PY" -m pip install -q --no-cache-dir --upgrade pip
        act "$PY" -m pip install --no-cache-dir --progress-bar off torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    fi
    act "$PY" -m pip install -q --no-cache-dir -r "$HERE/requirements.txt" scipy

    say "" "===== CUDA runtime 13 (the FA3 kernel links libcudart.so.13) and headers (nvrtc)"
    if [ "$DRY_RUN" = 1 ]; then
        say "  would find libcudart.so.13 (system CUDA 13, else pip install nvidia-cuda-runtime==13.4.92) and CUDA headers"
    elif ! ldconfig -p 2>/dev/null | grep 'libcudart\.so\.13' >/dev/null; then  # grep -q would SIGPIPE ldconfig
        cudart=$(ls /usr/local/cuda*/targets/x86_64-linux/lib/libcudart.so.13 /usr/local/cuda*/lib64/libcudart.so.13 \
                 2>/dev/null | head -1 || true)
        if [ -z "$cudart" ]; then
            act "$PY" -m pip install -q --no-cache-dir nvidia-cuda-runtime==13.4.92
            cudart=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)/nvidia/cu13/lib/libcudart.so.13
        fi
        [ -f "$cudart" ] || check "no libcudart.so.13: install cuda-cudart-13 (ANVIL2 README, item 4)"
        export LD_LIBRARY_PATH=$(dirname "$cudart")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        say "  libcudart.so.13: $cudart"
    fi
    if [ ! -f "${CUDA_HOME:-/usr/local/cuda}/include/cuda_bf16.h" ]; then
        site=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)
        for c in /usr/local/cuda /usr/local/cuda-12* /usr/local/cuda-13* "$site/nvidia/cuda_runtime"; do
            if [ -f "$c/include/cuda_bf16.h" ] && [ -f "$c/include/math_constants.h" ]; then export CUDA_HOME=$c; break; fi
        done
    fi
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    say "  CUDA_HOME=$CUDA_HOME"

    say "" "===== data: 9 train shards + val (once, shared by every arm)"
    if [ ! -f "$DATA/data/fineweb10B/fineweb_train_000009.bin" ] || [ ! -f "$DATA/data/fineweb10B/fineweb_val_000000.bin" ]; then
        act mkdir -p "$DATA/data"
        act cp "$HERE/data/cached_fineweb10B.py" "$DATA/data/"
        if [ "$DRY_RUN" = 0 ]; then (cd "$DATA" && "$PY" data/cached_fineweb10B.py 9); else say "  would download into $DATA"; fi
    fi
    say "  $DATA/data/fineweb10B"

    say "" "===== preflight: packages, GPUs, headers, Triton, NCCL, FA3 kernel, tokenizer, mask builder"
    if [ "$DRY_RUN" = 0 ]; then
        "$PY" "$TOOLS/preflight.py" --stack "$STACK_DIR" --runs "$RUNS" || die "preflight failed (above)"
        source "$RUNS/preflight.env"
        ATTEMPT+=(--environment "$RUNS/environment.txt")
    else
        say "  would run: $PY $TOOLS/preflight.py --stack $STACK_DIR --runs $RUNS"
    fi
fi

say "" "===== arms"
worktree() {  # name sha -> a pinned, unmodified worktree of upstream
    local dir=$WORK/arms/$1
    if [ ! -d "$dir" ]; then
        if [ "$DRY_RUN" = 1 ]; then say "  would add a worktree of ${2:0:7} at $dir"; return; fi
        git -C "$HERE" cat-file -e "$2^{commit}" 2>/dev/null || git -C "$HERE" fetch -q "$UPSTREAM" master pull/379/head
        git -C "$HERE" worktree add -q --detach "$dir" "$2"
    fi
    [ "$(git -C "$dir" rev-parse HEAD)" = "$2" ] || check "$dir is not at $2"
    [ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ] \
        || check "$dir has local changes: git worktree remove --force $dir, then re-run"
}
NOTES=()
if [ -z "${MASTER_DIR:-}" ]; then worktree master "$MASTER_SHA"; MASTER_DIR=$WORK/arms/master; fi
if [ -z "${PR379_DIR:-}" ]; then worktree pr379 "$PR379_SHA"; PR379_DIR=$WORK/arms/pr379; fi
if [ -n "$(git -C "$STACK_DIR" status --porcelain --untracked-files=no -- train_gpt.py track_1_short 2>/dev/null)" ]; then
    say "  WARNING: uncommitted changes in the stack's trainer; the logs embed them, but the PR must match"
    NOTES+=("The stack checkout had uncommitted changes to the trainer when the attempt ran; the logs hold that source.")
fi
# The ML-only arm (#379 + #375 without the systems patches) is the fallback candidate. Built once; the choice sticks
# once a leg has run (attempt.json fixes the arms), but a failed build is retried until then.
if grep -qs '^none:' "$RUNS/mlonly_arm.txt" && [ ! -f "$RUNS/attempt.json" ]; then rm "$RUNS/mlonly_arm.txt"; fi
MLONLY_DIR=
if [ -f "$RUNS/mlonly_arm.txt" ]; then
    MLONLY_DIR=$(cat "$RUNS/mlonly_arm.txt")
elif [ "$DRY_RUN" = 1 ]; then
    say "  would build the mlonly arm: bash $MAKE_MLONLY $WORK"
elif [ ! -f "$MAKE_MLONLY" ]; then
    echo "none: $MAKE_MLONLY does not exist" > "$RUNS/mlonly_arm.txt"
else
    rc=0
    bash "$MAKE_MLONLY" "$WORK" > "$RUNS/mlonly_build.out" 2> "$RUNS/mlonly_build.err" || rc=$?
    sed 's/^/  /' "$RUNS/mlonly_build.err"
    built=$(tail -n 1 "$RUNS/mlonly_build.out")
    if [ "$rc" != 0 ]; then
        echo "none: make_mlonly_arm.sh failed (exit $rc): $(tail -n 3 "$RUNS/mlonly_build.err" | tr '\n' ' ')" > "$RUNS/mlonly_arm.txt"
    elif [ -f "$built/train_gpt.py" ]; then
        echo "$built" > "$RUNS/mlonly_arm.txt"
    else
        echo "none: make_mlonly_arm.sh printed '$built', which holds no train_gpt.py" > "$RUNS/mlonly_arm.txt"
    fi
    MLONLY_DIR=$(cat "$RUNS/mlonly_arm.txt")
fi
ARMS=(--arm master="$MASTER_DIR" --arm pr379="$PR379_DIR" --arm stack="$STACK_DIR")
if [ -n "$MLONLY_DIR" ] && [ "${MLONLY_DIR#none: }" = "$MLONLY_DIR" ]; then
    ARMS+=(--arm mlonly="$MLONLY_DIR")
elif [ -n "$MLONLY_DIR" ]; then
    say "  WARNING: no mlonly arm (${MLONLY_DIR#none: }); continuing with the stack as the only candidate"
    NOTES+=("The ML-only fallback arm was unavailable (${MLONLY_DIR#none: }), so the stack was the only candidate.")
fi
for arm in "${ARMS[@]}"; do [ "$arm" = --arm ] || say "  $arm"; done
for note in "${NOTES[@]+"${NOTES[@]}"}"; do ATTEMPT+=(--note "$note"); done
# Every leg runs under this environment's python (the same one that installed the requirements).
ATTEMPT+=(--command "${LEG_COMMAND:-$PY -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py}")

if [ "$DRY_RUN" = 1 ]; then
    say "" "dry run: nothing was installed, downloaded or run. The plan above is what bash tools/record_attempt/run.sh does."
    exit 0
fi

say "" "===== attempt"
"$PY" "$TOOLS/attempt.py" "${ATTEMPT[@]}" "${ARMS[@]}"
