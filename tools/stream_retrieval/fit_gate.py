"""Fit the stream-retrieval gate's 4 constants on held-out TRAINING data (never val), from a STREAM_RETRIEVAL_FIT dump.

A dev run with STREAM_RETRIEVAL=1 STREAM_RETRIEVAL_FIT=<path> (untimed, after its final validation) writes
  <path>              u32 [world][n][4]: N, C, M, L* of every position of 64 training batches past the run's stream
  <path>.rank<r>.npy  float32 [n / chunk][chunk]: our model's eval NLL (-log(p + 1e-9)) at those positions
This fits W in lambda = sigmoid(W . [1, log2 N, M/N, log2 L*]) by L-BFGS on the mixture likelihood
q = (1 - lambda) p + lambda C/N, prints the held-out gain (2-fold by chunk) and the gain of the current W, and the line
to put in tools/stream_retrieval/arm/track_1_short/stream_memory.py (the overlay; the arm's copy is built from it).

usage: python tools/stream_retrieval/fit_gate.py PATH --world 8 --chunk 262144
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

OVERLAY_MODULE = Path(__file__).resolve().parent / "arm/track_1_short/stream_memory.py"


def current_w() -> tuple:
    """W from the overlay's stream_memory.py (runs in the stack checkout and in the arm alike)."""
    spec = importlib.util.spec_from_file_location("stream_memory_overlay", OVERLAY_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.W


def load(path: str, world: int, chunk: int):
    feat = np.fromfile(path, dtype=np.uint32).reshape(world, -1, 4)
    nll = np.stack([np.load(f"{path}.rank{r}.npy").reshape(-1) for r in range(world)])
    assert nll.shape == feat.shape[:2], f"features {feat.shape} and NLL {nll.shape} do not line up"
    chunks = feat.shape[1] // chunk
    fold = (np.arange(world)[:, None] * chunks + np.arange(feat.shape[1])[None, :] // chunk) % 2  # alternate chunks
    return feat.reshape(-1, 4).astype(np.float64), nll.reshape(-1).astype(np.float64), fold.reshape(-1)


def design(feat: np.ndarray) -> np.ndarray:
    n, m, lstar = feat[:, 0], feat[:, 2], feat[:, 3]
    return np.stack([np.ones_like(n), np.log2(np.maximum(n, 1)), m / np.maximum(n, 1), np.log2(np.maximum(lstar, 1))], 1)


def mixed_nll(w, x, p, r):
    lam = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
    return -np.log((1 - lam) * p + lam * r + 1e-9)


def fit(x, p, r, w0) -> np.ndarray:
    def objective(w):
        lam = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        q = (1 - lam) * p + lam * r + 1e-9
        g = (r - p) / q * lam * (1 - lam)
        return -np.log(q).sum(), -(x.T @ g)
    return minimize(objective, np.asarray(w0, dtype=np.float64), jac=True, method="L-BFGS-B").x


def gain_mnat(w, x, p, r, nll, total: int) -> float:
    """Mean over ALL positions (only matched ones change) of nll - mixed nll, in millinats."""
    return 1000 * (nll - mixed_nll(w, x, p, r)).sum() / total


def main(argv=None):
    W = current_w()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=262144)
    args = parser.parse_args(argv)
    feat, nll, fold = load(args.path, args.world, args.chunk)
    hit = feat[:, 0] > 0
    x, p = design(feat[hit]), np.clip(np.exp(-nll[hit]) - 1e-9, 0, None)
    r, nll_h, fold_h = feat[hit, 1] / feat[hit, 0], nll[hit], fold[hit]
    total = nll.size
    held_out = sum(gain_mnat(fit(x[fold_h != k], p[fold_h != k], r[fold_h != k], W), x[fold_h == k], p[fold_h == k],
                             r[fold_h == k], nll_h[fold_h == k], total) for k in (0, 1))
    w = fit(x, p, r, W)
    print(f"{total} positions, {hit.mean():.4f} matched; NLL {nll.mean():.4f}")
    print(f"current W {tuple(W)}: gain {gain_mnat(np.asarray(W), x, p, r, nll_h, total):.2f} mnat")
    print(f"fitted  W {tuple(np.round(w, 3))}: gain {gain_mnat(w, x, p, r, nll_h, total):.2f} mnat in sample, "
          f"{held_out:.2f} mnat held out (2-fold by chunk)")
    print(f"W = {tuple(float(v) for v in np.round(w, 3))}")
    return w


if __name__ == "__main__":
    main()
