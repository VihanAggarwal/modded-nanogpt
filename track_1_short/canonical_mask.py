"""Canonical token masking for the final validation (record #350).

GPT-2's tokenizer never emits certain (prev, cur) token pairs. At the final validation those
tokens are masked out of the softmax, which can only lower the loss. The mask is built in a
separate process during training so its cost stays on the clock without stalling it. The build
itself lives in canonical_mask_build.py.
"""
import mmap
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from track_1_short.canonical_mask_build import build_canonical_mask

BUILD_SCRIPT = Path(__file__).resolve().with_name("canonical_mask_build.py")
# The builder starts at this step, on the clock: once the training loop is in steady state, so its CPU
# work does not compete with the host-bound first steps.
BUILD_START_STEP = 25


class BackgroundCanonicalMask:
    """Builds the canonical mask concurrently with training, in a separate process.

    The build is a few seconds of pure-Python work, so running it in a thread would hold the GIL
    and throttle the training loop's kernel launches. A separate process has its own interpreter and
    only computes, writing the result into shared memory.

    The process is a fresh interpreter, spawned (vfork + exec) at step BUILD_START_STEP: its startup,
    its imports, the tokenizer load and the build all run on the clock, off the training loop's path.
    Record #350 forked the trainer itself at the clock's start instead. fork() copies the page tables of
    the whole warmed-up 8-GPU process, and every page the trainer then writes takes a copy-on-write fault
    while the child lives, all on rank 0's launch path, which every rank waits for at its next
    collective. A spawn costs the trainer the same fraction of a millisecond at any size.
    """

    def __init__(self, vocab_size: int, owner: bool, print0):
        self.vocab_size = vocab_size
        self.print0 = print0
        self.fd = self.map = self.buf = self.proc = None
        self.pinned = False
        if owner:
            # Shared memory the builder maps too: its writes land in these pages.
            self.fd = os.memfd_create("canonical_mask")
            os.ftruncate(self.fd, vocab_size * (vocab_size // 8))
            self.map = mmap.mmap(self.fd, vocab_size * (vocab_size // 8))
            self.buf = torch.frombuffer(self.map, dtype=torch.uint8).view(vocab_size, vocab_size // 8)
            # Page-lock the buffer so that collect's H2D is a direct DMA rather than a staged
            # copy, roughly 10ms instead of 50. Registering is itself slow, which is why it
            # belongs here, before the clock. The mapping is MAP_SHARED, so the builder still
            # writes to these same pages.
            cudart = torch.cuda.cudart()
            err = cudart.cudaHostRegister(self.buf.data_ptr(), self.buf.nbytes, 0)
            self.pinned = err == cudart.cudaError.success
            if not self.pinned:
                print0(f"NOTE: could not page-lock the canonical mask buffer ({err}), "
                       "so its copy to device will be slower", console=True)
            # Before the clock: the builder's interpreter must import numpy and tiktoken and find the
            # tokenizer, or the build would fall back to rank 0, inline and on the clock.
            check = subprocess.run(self._command("--check"), capture_output=True, text=True, timeout=300)
            if check.returncode:
                raise RuntimeError(f"the canonical mask builder cannot run: {check.stderr.strip()}")
        self.started = False

    @staticmethod
    def _command(*args: str) -> list[str]:
        # -P: nothing beside the script shadows an import (the environment and site-packages still apply).
        return [sys.executable, "-P", str(BUILD_SCRIPT), *args]

    def start(self):
        """Spawn the builder (once). Called on the clock at step BUILD_START_STEP."""
        if self.buf is None or self.started:
            return
        self.started = True
        self.proc = subprocess.Popen(self._command(str(self.fd), str(self.vocab_size)), pass_fds=(self.fd,))

    def wait(self, timeout=60.0):
        """Block until the mask is ready. Called from the timed region."""
        self.start()  # a run shorter than BUILD_START_STEP steps
        if self.proc is None:
            return
        try:
            code = self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait()
        self.proc = None
        if code:
            self.print0(f"WARNING: background canonical mask build failed ({code}), building it inline", console=True)
            np.copyto(self.buf.numpy(), build_canonical_mask(self.vocab_size))

    def collect(self, out: Tensor):
        """Fill `out` on every rank, then release the shared buffer."""
        assert self.proc is None, "collect before wait"
        if self.buf is not None:
            out.copy_(self.buf)  # kept blocking: the source is released just below
            if self.pinned:
                torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
        dist.broadcast(out, 0)
        if self.buf is not None:
            # Dropped, not closed: the mapping goes when its last view does.
            self.buf = self.map = None
            os.close(self.fd)
