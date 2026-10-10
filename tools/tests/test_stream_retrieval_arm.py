"""The stream-retrieval overlay (tools/stream_retrieval): the stack carries none of its code, the overlay applies to
the stack, its own tests pass inside the resulting tree, and make_streamret_arm.sh builds and reuses the arm.

The retrieval's tests (tools/stream_retrieval/test_*.py: test_stream_retrieval.py imports track_1_short.stream_memory
and the hooked data.py / model/gpt.py, which exist only in the arm; the others load the overlay by path) run in a
subprocess inside a copy of this working tree with the overlay applied; their output is printed (pytest -s shows it).
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "tools/stream_retrieval"
RETRIEVAL_WORDS = re.compile(r"stream_memory|stream_lowtables|stream_pointer|STREAM_RETRIEVAL|on_spans|stream_rows|stream_lm")
PARTS = ("stream_memory", "stream_lowtables", "stream_pointer")


def stack_tree(dst: Path) -> Path:
    """A copy of this working tree's stack (train_gpt.py, track_1_short/) and tools/stream_retrieval/."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    dst.mkdir(parents=True)
    shutil.copy2(ROOT / "train_gpt.py", dst / "train_gpt.py")
    shutil.copytree(ROOT / "track_1_short", dst / "track_1_short", ignore=ignore)
    shutil.copytree(OVERLAY, dst / "tools/stream_retrieval", ignore=ignore)
    return dst


def overlay(tree: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(tree / "tools/stream_retrieval/apply_overlay.sh"), str(tree)],
                          capture_output=True, text=True)


def test_the_stack_carries_no_retrieval_code():
    """A stack record's source (train_gpt.py and track_1_short/, what run_log embeds) holds nothing of the retrieval."""
    files = [ROOT / "train_gpt.py", *sorted((ROOT / "track_1_short").rglob("*.py")),
             *sorted((ROOT / "track_1_short").rglob("*.c"))]
    hits = [f"{p.relative_to(ROOT)}:{i + 1}" for p in files for i, line in enumerate(p.read_text().splitlines())
            if RETRIEVAL_WORDS.search(line)]
    assert not hits, hits
    assert not [p for part in PARTS for p in (ROOT / "track_1_short").glob(f"{part}*")]


def test_the_overlay_applies_once_and_only_adds_the_retrieval(tmp_path):
    tree = stack_tree(tmp_path / "arm")
    before = {p: p.read_text() for p in [tree / "train_gpt.py", *(tree / "track_1_short").rglob("*.py")]}
    done = overlay(tree)
    assert done.returncode == 0, done.stderr
    assert all((tree / f"track_1_short/{part}.{ext}").exists() for part in PARTS for ext in ("c", "py"))
    changed = sorted(str(p.relative_to(tree)) for p, text in before.items() if p.read_text() != text)
    assert changed == ["track_1_short/data.py", "track_1_short/model/gpt.py", "track_1_short/run_log.py", "train_gpt.py"]
    again = overlay(tree)
    assert again.returncode != 0 and "already has the stream-retrieval code" in again.stderr


def test_the_retrieval_tests_pass_in_the_arm(tmp_path):
    """Every test file of the overlay (the arm's, and the standalone ones of the helper, its parts and the gate)."""
    tree = stack_tree(tmp_path / "arm")
    assert overlay(tree).returncode == 0
    env = {k: v for k, v in os.environ.items() if not k.startswith("STREAM_RETRIEVAL")}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
                             "tools/stream_retrieval"], cwd=tree, env=env, capture_output=True, text=True, timeout=3600)
    print(result.stdout[-4000:])
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-3000:]
    assert " passed" in result.stdout and " failed" not in result.stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="no git")
def test_make_streamret_arm_builds_reuses_and_rebuilds(tmp_path):
    """On a scratch repository holding this working tree's stack and overlay: the arm is HEAD plus one commit with
    the overlay, clean, reused when nothing changed and rebuilt when HEAD moves; the stack checkout is untouched."""
    repo = stack_tree(tmp_path / "repo")
    git = lambda *a, cwd=repo: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                                               "commit.gpgsign=false", *a], cwd=cwd, check=True, capture_output=True,
                                              text=True).stdout.strip()
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "stack")
    script = repo / "tools/stream_retrieval/make_streamret_arm.sh"
    build = lambda: subprocess.run(["bash", str(script), str(tmp_path / "work")], capture_output=True, text=True)
    first = build()
    assert first.returncode == 0, first.stderr
    arm = Path(first.stdout.strip().splitlines()[-1])
    assert arm == tmp_path / "work/streamret"
    assert git("rev-parse", "HEAD^", cwd=arm) == git("rev-parse", "HEAD")
    assert all((arm / f"track_1_short/{part}.c").exists() for part in PARTS)
    assert "on_spans" in (arm / "track_1_short/data.py").read_text()
    assert "stream_lm_out" in (arm / "track_1_short/model/gpt.py").read_text()
    assert git("status", "--porcelain", "--untracked-files=no", cwd=arm) == ""
    assert not [p for part in PARTS for p in (repo / "track_1_short").glob(f"{part}*")]
    assert git("status", "--porcelain") == ""
    arm_head = git("rev-parse", "HEAD", cwd=arm)
    (arm / "logs").mkdir()
    (arm / "logs/leg.txt").write_text("kept")
    second = build()
    assert second.returncode == 0 and "reusing" in second.stderr and git("rev-parse", "HEAD", cwd=arm) == arm_head
    (repo / "track_1_short/config.py").write_text((repo / "track_1_short/config.py").read_text() + "\n# moved\n")
    dirty = build()
    assert dirty.returncode != 0 and "uncommitted changes" in dirty.stderr
    git("commit", "-q", "-am", "stack moves")
    third = build()
    assert third.returncode == 0, third.stderr
    assert git("rev-parse", "HEAD^", cwd=arm) == git("rev-parse", "HEAD") and (arm / "logs/leg.txt").read_text() == "kept"
    assert (arm / "track_1_short/config.py").read_text().endswith("# moved\n")
