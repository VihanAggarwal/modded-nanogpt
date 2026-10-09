"""The Python side of run.sh's preflight: the pinned stack, the GPUs, the CUDA headers, Triton's C toolchain, NCCL
across the 8 GPUs, the FA3 kernel, the tokenizer and the stack's canonical-mask builder.

  python preflight.py --stack DIR --runs DIR

Each check runs what a leg runs, off the books, so a broken node stops here instead of crashing counted legs. Stops
(exit 1) at the first problem with what to do about it. Writes RUNS/environment.txt (the node, for the record README)
and RUNS/preflight.env, which run.sh sources: `export HF_HUB_OFFLINE=1` once the FA3 kernel is in the local cache and
loads from it, so no leg depends on the Hub answering (ANVIL2 README, "five things that will bite you": the Hub can
refuse the anonymous kernels request, and then the cache is pre-seeded from the model repo).
"""
import argparse
import datetime
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

PINS = {"torch": "2.10.0+cu128", "tokenizers": "0.23.2", "kernels": "0.16.1", "huggingface-hub": "1.29.0"}
REPORTED = ("triton", "numpy", "scipy", "tiktoken")

# Loads the kernel exactly as track_1_short/model/attention.py does and checks the binary's sha256.
LOAD_FA3 = """
import hashlib, pathlib, sys
from kernels import get_kernel
repo, revision, digest = sys.argv[1:4]
fa3 = get_kernel(repo, revision=revision, trust_remote_code=True)
binary = next(pathlib.Path(fa3.__file__).parent.glob("_flash_attn3_cuda_*.so"))
assert hashlib.sha256(binary.read_bytes()).hexdigest() == digest, f"unexpected FA3 binary {binary}"
print(binary)
"""
# Triton compiles each kernel's launcher with the C compiler against Python.h at run time; the trainer launches
# @triton.jit kernels of its own (cplm_copy.py, perf/kernels/). @triton.jit reads its source, so this is a file.
CHECK_TRITON = """
import torch, triton, triton.language as tl

@triton.jit
def add_one(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) + 1, mask=offs < n)

x = torch.zeros(1000, device="cuda")
add_one[(4,)](x, 1000, BLOCK=256)
assert x.sum().item() == 1000, x.sum().item()
"""
# One all_reduce over the 8 GPUs, launched the way every leg is (NCCL, /dev/shm, the rendezvous).
CHECK_NCCL = """
import os, torch, torch.distributed as dist
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
x = torch.ones(1 << 20, device=device)
dist.all_reduce(x)
assert x[0].item() == dist.get_world_size() == 8, x[0].item()
dist.destroy_process_group()
"""
# #375's normalization map asserts its own sha256 (the tokenizers pin); tiktoken caches the GPT-2 BPE files.
CHECK_TOKENIZER = """
import sys
sys.path.insert(0, sys.argv[1])
import tiktoken
tiktoken.get_encoding("gpt2")
import track_1_short.token_norm
"""


def fail(message: str):
    print(f"PREFLIGHT FAILED: {message}", flush=True)
    sys.exit(1)


def ok(message: str):
    print(f"  ok: {message}", flush=True)


def tail(text: str, n: int = 12) -> str:
    return "\n    ".join(text.strip().splitlines()[-n:])


def run_check(what: str, cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run one check's command; a hang stops the preflight with what hung instead of a traceback."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=600, **kw)
    except subprocess.TimeoutExpired:
        script = next(Path(arg).name for arg in cmd if arg.endswith(".py"))
        fail(f"{what} hung for 10 minutes (stop what is left of it: pkill -f {script}; the GPUs, NCCL or the network "
             "are not healthy on this node)")


def check_python():
    if sys.version_info < (3, 11):
        fail(f"{sys.executable} is Python {sys.version.split()[0]}: the stack starts its canonical-mask builder with "
             "`python -P`, which needs 3.11 or newer. Remove the venv in WORK and re-run (run.sh picks python3.12 or "
             "python3.11, or builds a 3.12 venv with uv), or set VENV_PYTHON to a python >= 3.11")
    ok(f"Python {sys.version.split()[0]}")


def check_versions() -> dict[str, str]:
    import torch
    found = {name: importlib.metadata.version(name) for name in PINS if name != "torch"}
    found["torch"] = torch.__version__
    wrong = {name: v for name, v in found.items() if v != PINS[name]}
    if wrong:
        fail(f"pinned packages differ: {wrong}, want {PINS}. A torch nightly or another build can train with "
             "different numerics (ANVIL2 saw NaNs): pip install torch==2.10.0 --index-url "
             "https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt")
    for name in REPORTED:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "scipy":
                fail("scipy is missing (the record's statistics.py needs it): pip install scipy")
    ok("packages " + ", ".join(f"{k} {v}" for k, v in found.items()))
    return found


def check_gpus():
    import torch
    if not torch.cuda.is_available():
        fail("torch sees no CUDA device (driver, container --gpus, or a CPU-only torch?)")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    caps = {torch.cuda.get_device_capability(i) for i in range(len(names))}
    if len(names) != 8 or not all("H100" in n for n in names) or caps != {(9, 0)}:
        fail(f"torch sees {names} (capability {caps}); the record needs 8 H100s")
    ok(f"8x {names[0]}")


def check_headers():
    home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    missing = [h for h in ("cuda_bf16.h", "cuda_fp16.h", "math_constants.h") if not Path(home, "include", h).exists()]
    if missing:
        fail(f"CUDA_HOME={home} lacks include/{missing}: the training CE kernel compiles with nvrtc at import and "
             "needs them. Point CUDA_HOME at a CUDA toolkit (or the pip nvidia/cuda_runtime directory)")
    ok(f"CUDA headers under {home}")


def check_toolchain():
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        fail("no C compiler: Triton builds its kernel launchers with gcc at run time, so every leg would crash. "
             "apt-get install -y gcc (or build-essential), then re-run")
    scheme = sysconfig.get_default_scheme()
    include = sysconfig.get_paths(scheme="posix_prefix" if scheme == "posix_local" else scheme)["include"]
    if not Path(include, "Python.h").exists():
        fail(f"no Python.h in {include}: Triton's launchers include it, so every leg would crash. apt-get install -y "
             f"python{sys.version_info.major}.{sys.version_info.minor}-dev (the headers of {sys.executable}), then re-run")
    ok(f"C compiler {cc}, Python.h in {include}")


def check_triton(runs: Path):
    with tempfile.TemporaryDirectory(dir=runs) as tmp:  # an empty cache: the launcher really compiles
        script = Path(tmp, "check_triton.py")
        script.write_text(CHECK_TRITON)
        result = run_check("the Triton check", [sys.executable, str(script)],
                           env=dict(os.environ, TRITON_CACHE_DIR=str(Path(tmp, "cache"))))
    if result.returncode:
        fail("a one-line Triton kernel does not compile or run on cuda:0 (the trainer's own kernels would fail the "
             f"same way):\n    {tail(result.stderr)}")
    ok("a Triton kernel compiles and runs on cuda:0")


def check_nccl(runs: Path):
    with tempfile.TemporaryDirectory(dir=runs) as tmp:
        script = Path(tmp, "check_nccl.py")
        script.write_text(CHECK_NCCL)
        result = run_check("the NCCL check", [sys.executable, "-m", "torch.distributed.run", "--standalone",
                                              "--nproc_per_node=8", str(script)])
    if result.returncode:
        fail("an NCCL all_reduce over the 8 GPUs fails (NCCL, /dev/shm size, or the GPUs are busy):\n    "
             + tail(result.stdout + result.stderr, 20))
    ok("NCCL all_reduce across 8 GPUs (torch.distributed.run, as every leg launches)")


def check_fa3(stack: Path, runs: Path):
    source = (stack / "track_1_short/model/attention.py").read_text()
    pin = {k: re.search(rf'^{k} = "([^"]+)"', source, re.M).group(1) for k in ("FA3_REPO", "FA3_REVISION", "FA3_BINARY_SHA256")}
    args = [sys.executable, "-c", LOAD_FA3, pin["FA3_REPO"], pin["FA3_REVISION"], pin["FA3_BINARY_SHA256"]]
    offline = dict(os.environ, HF_HUB_OFFLINE="1")
    first = subprocess.run(args, capture_output=True, text=True)
    if first.returncode == 0:
        ok(f"FA3 {pin['FA3_REPO']}@{pin['FA3_REVISION'][:8]} loads, sha256 matches (online)")
    else:
        # kernels>=0.16 can answer 401 for this public repo: fetch it as a model repo and run offline (ANVIL2 README).
        print("  FA3 did not load from the Hub; pre-seeding the kernel cache and retrying offline", flush=True)
        from huggingface_hub import constants, snapshot_download
        try:
            snapshot_download(pin["FA3_REPO"], revision=pin["FA3_REVISION"])
        except Exception as err:  # noqa: BLE001 - report whatever the Hub said
            print(f"  snapshot_download failed: {err}", flush=True)
        owner, name = pin["FA3_REPO"].split("/")
        hub = Path(constants.HF_HUB_CACHE)
        src, dst = hub / f"models--{owner}--{name}", hub / f"kernels--{owner}--{name}"
        if src.exists():
            shutil.copytree(src, dst, symlinks=True, copy_function=os.link, dirs_exist_ok=True)
    # Online, every rank of every leg would ask the Hub API (status, variant list, snapshot) at import, with no
    # fallback: one refused or failed request would crash a counted leg. From the verified cache, none does.
    second = subprocess.run(args, capture_output=True, text=True, env=offline)
    if second.returncode != 0:
        err = (second.stderr or first.stderr).strip().splitlines()[-12:]
        if first.returncode == 0:
            print("  WARNING: FA3 loads online but not from the local cache, so every leg loads it from the Hub:\n    "
                  + "\n    ".join(err), flush=True)
            return
        hint = ("libcudart.so.13 is not on the loader path: the kernel links the CUDA 13 runtime (pip install "
                "nvidia-cuda-runtime==13.4.92 and add .../site-packages/nvidia/cu13/lib to LD_LIBRARY_PATH)"
                if "libcudart.so.13" in "\n".join(err) else
                "if the Hub refused the request (no token is normally needed), log in with any free account: "
                f"{Path(sys.executable).parent / 'hf'} auth login, then re-run")
        fail("the FA3 kernel does not load:\n    " + "\n    ".join(err) + f"\n  {hint}")
    (runs / "preflight.env").write_text("export HF_HUB_OFFLINE=1\n")
    ok("FA3 loads offline from the verified cache, sha256 matches: every leg runs with HF_HUB_OFFLINE=1")


def check_tokenizer(stack: Path):
    result = subprocess.run([sys.executable, "-c", CHECK_TOKENIZER, str(stack)], capture_output=True, text=True)
    if result.returncode != 0:
        fail("the tokenizer check failed (tiktoken's GPT-2 files, or #375's normalization map, which needs "
             "tokenizers==0.23.2):\n    " + "\n    ".join(result.stderr.strip().splitlines()[-6:]))
    ok("tiktoken gpt2 cached; #375's token normalization map matches its sha256")


def check_stream_helper(stack: Path):
    """The streamret arm (STREAMRET_CUTS) compiles its C helper with the node's C compiler before its clock; a missing
    compiler or a build error would crash every one of its legs. Builds it exactly as the trainer does
    (stream_memory.build_helper, from the overlay the arm is made of; the build is cached by the source's hash)."""
    module = stack / "tools/stream_retrieval/arm/track_1_short/stream_memory.py"
    spec = importlib.util.spec_from_file_location("stream_memory_overlay", module)
    stream_memory = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(stream_memory)
        helper = stream_memory.build_helper()
    except Exception as e:  # noqa: BLE001 - any failure here would crash every streamret leg
        fail(f"the stream-retrieval helper does not build ({module}): {tail(str(e))}")
    ok(f"the stream-retrieval helper builds ({helper})")


def check_mask_builder(stack: Path):
    """The stack starts its canonical-mask builder as `python -P canonical_mask_build.py` and checks it before the
    clock; a failure there crashes the leg, so the stack would lose its smoke leg to an environment problem."""
    script = stack / "track_1_short/canonical_mask_build.py"
    if not script.exists():
        return
    result = run_check("the canonical-mask builder check", [sys.executable, "-P", str(script), "--check"])
    if result.returncode:
        fail(f"the stack's canonical-mask builder does not start ({sys.executable} -P {script} --check):\n    "
             + tail(result.stderr))
    ok("the stack's canonical-mask builder starts (python -P)")


def environment_report(found: dict[str, str]) -> str:
    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return "(unavailable)"
    cpu = next((line.split(":", 1)[1].strip() for line in run("lscpu").splitlines() if line.startswith("Model name")), "?")
    mem = next((line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal")), "0")
    return "\n".join([
        f"date (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}",
        f"kernel: {run('uname', '-r')}",
        f"CPU: {cpu}, {os.cpu_count()} logical cores; memory {int(mem) // 2**20} GiB",
        f"python: {sys.version.split()[0]} ({sys.executable})",
        "packages: " + ", ".join(f"{k} {v}" for k, v in sorted(found.items())),
        f"CUDA_HOME: {os.environ.get('CUDA_HOME', '/usr/local/cuda')}",
        f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}",
        "",
        "GPUs (name, driver, power limit W, max SM clock MHz, persistence):",
        run("nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm,persistence_mode", "--format=csv,noheader"),
        "",
        "topology (nvidia-smi topo -m):",
        run("nvidia-smi", "topo", "-m"),
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True, help="the stack checkout (FA3 pin and #375's map are read from it)")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--stream-helper", action="store_true", help="the attempt has the streamret arm")
    args = parser.parse_args()
    stack, runs = Path(args.stack).resolve(), Path(args.runs).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "preflight.env").write_text("")
    check_python()
    found = check_versions()
    check_gpus()
    check_headers()
    check_toolchain()
    check_triton(runs)
    check_nccl(runs)
    check_fa3(stack, runs)
    check_tokenizer(stack)
    check_mask_builder(stack)
    if args.stream_helper:
        check_stream_helper(stack)
    (runs / "environment.txt").write_text(environment_report(found))
    ok(f"environment report in {runs / 'environment.txt'}")


if __name__ == "__main__":
    main()
