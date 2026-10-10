"""Fit stream retrieval v2's gate (track_1_short/stream_memory.py: GATE_V2 / GATE_V2_LOW, mix_v2) on the run's OWN
consumed batches.

A dev run with STREAM_RETRIEVAL=1 STREAM_RETRIEVAL_FIT=<dir> (and the run's STREAM_RETRIEVAL_LOW) writes into <dir>,
after its final validation (stream_memory.fit_dump):
  fit                      the helper's FIT records (P1): the run's last STREAM_RETRIEVAL_FIT_K timed batches (default
                           16, one 262,144-token chunk per rank), queried against the memory as it stood before the
                           first of them was inserted (stream_memory.read_fit)
  fit.tokens               their inputs and targets, u16 [world][n][2]
  fit.low                  P2's exact low-order rows there (STREAM_RETRIEVAL_LOW=1), against the tables before them
  fit.ptr                  P3's pointer / vote / source-copy rows there (frozen memory)
  fit.lm.rank<r>.npz       the model's eval outputs there (stream_memory.save_fit_lm): nll, ent, mx and top_v /
                           top_in / top_rank at stream_memory.query_tokens() (one column per chain order, then P3's 3)
This fits, end to end (torch autograd, L-BFGS, as rg/combo/joint.py), the stick-breaking chain over the orders (P2's
1-5 when present, then the memory's levels): per order a gate w (on the standardized features of
stream_memory.chain_features) and 3 KN discount logits, plus P3's top-level softmax over [chain, pointer, vote, source
copy] on stream_memory.top_block's features; the features are built by the same code the eval runs. (An .npz may
instead carry low_orders / low_<field> and top_comps / top_avail / top_phi / top_names: the generic contract.)
It prints the in-sample gain and (--heldout) a 2-fold-by-document held-out gain on the FIT positions, and writes
the constants as JSON (--out; STREAM_RETRIEVAL_GATE=<it> uses them in a dev run) or into the overlay's
stream_memory.py (--module: into the GATE_V2 block, or GATE_V2_LOW for a chain with P2's orders, as the spec for the
dump's step count: it replaces a spec fitted at the same step count, and a spec fitted on the run's own model replaces
every placeholder).

The constants carry their provenance in "fit": the dev run's trained steps (total_steps, from the FIT dump's header;
a run uses the spec fitted at its own step count, and a record attempt requires one per cut), the dump, the positions,
the gains, --note, and --proxy TEXT when the model outputs are a proxy's (CPU placeholders: a record refuses them).

usage: python tools/stream_retrieval/fit_gate_v2.py DIR|PATH [--ranks 0,1,...] [--mode lm+su2+tp+hi] [--heldout]
                                                         [--out gate.json] [--module] [--note TEXT] [--proxy TEXT]
"""
import argparse
import ast
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

OVERLAY_MODULE = Path(__file__).resolve().parent / "arm/track_1_short/stream_memory.py"
L2_GATE = 1.0     # ridge on the chain gates' non-constant weights (joint.py: l2 1e-3 x 1e3)
L2_TOP = 10.0     # ridge on the top softmax's non-constant weights (joint.py's l2_top)
EPS = 1e-9        # CPLM's NLL is -log(p + EPS); the mixture's NLL is -log(q + EPS)
OVERLAY_LEVELS = (6, 7, 8, 10, 12, 16, 24, 32)  # the memory's levels (stream_memory.LEVELS)


def overlay():
    """stream_memory.py from the overlay (runs in the stack checkout and in the arm alike)."""
    spec = importlib.util.spec_from_file_location("stream_memory_overlay", OVERLAY_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOW_FIELDS = ("N", "C", "D", "M", "n1", "n2", "top")


def rank_data(sm, fit: dict, rank: int, lm: dict, mode: str) -> dict:
    """One rank's FIT positions as fitting data: p0, nll, the chain's counts and raw features, the top block."""
    x, y, rows, starts = sm.fit_inputs(fit, rank)
    n = x.numel()
    T = lambda a: torch.from_numpy(np.asarray(a))
    nll = T(lm["nll"]).to(torch.float64).reshape(-1)
    if nll.numel() != n:
        raise ValueError(f"rank {rank}: {nll.numel()} NLL values for {n} FIT positions")
    low, ptr = sm.fit_parts(fit, rank)
    counts = dict(rows.level_counts(y))
    if low is not None:  # P2's rows from the helper's dump
        lows = list(low)
        counts.update(low)
    else:  # the generic contract: low_orders / low_<field> in the .npz
        lows = [int(o) for o in lm.get("low_orders", [])]
        for i, o in enumerate(lows):
            counts[o] = {f: T(lm[f"low_{f}"][:, i]) for f in LOW_FIELDS}
            counts[o]["top"] = counts[o]["top"].to(torch.int64)
    orders = lows + list(sm.LEVELS)
    counts = {o: {k: (v.to(torch.int64) if k == "top" else v.to(torch.float64)).contiguous() for k, v in counts[o].items()}
              for o in orders}  # contiguous: columns of Rows' [P, 8] arrays would keep them alive
    blocks = mode.split("+")
    tp = None
    if "tp" in blocks:
        tp = {o: (T(lm["top_v"])[:, i], T(lm["top_in"])[:, i], T(lm["top_rank"])[:, i]) for i, o in enumerate(orders)}
    lp_lm = -T(lm["nll_lm"]).to(torch.float64) if "nll_lm" in lm else -nll
    phis = sm.chain_features(counts, orders, y, x, lp_lm, T(lm["ent"]), -nll, starts, mode,
                             mx=T(lm["mx"]) if "mx" in lm else None, tp=tp,
                             extra=T(lm["extra"]) if "extra" in lm else None)
    phis = [ph.to(torch.float32) for ph in phis]  # stored in float32 (chain_prob standardizes them in float64)
    top, top_names = None, None
    if ptr is not None:  # P3's rows from the helper's dump: the eval's own top block
        lmt = {k: T(lm[k]) for k in ("ent", "mx", "top_v", "top_in")}
        top = sm.top_block(ptr, n, x, y, starts, nll, lmt, len(orders), counts, orders, rows)
        top_names = list(sm.TOP_COMPS)
    elif "top_comps" in lm:
        top = dict(comps=T(lm["top_comps"]).to(torch.float64), avail=T(lm["top_avail"]).to(torch.bool),
                   phi=T(lm["top_phi"]).to(torch.float64))
        top_names = [str(v) for v in lm["top_names"]] if "top_names" in lm else None
    doc = torch.cumsum((x == sm.BOS).to(torch.int64), 0)
    return dict(n=n, nll=nll, p0=(torch.exp(-nll) - EPS).clamp_min(0), counts=counts, phis=phis, top=top, doc=doc,
                orders=orders, top_names=top_names, top_cols=sm.top_cols() if ptr is not None else None)


def _select(d: dict, mask: torch.Tensor, orders) -> dict:
    """The positions `mask` of a rank's data (features are computed on whole sequences first, so this keeps them)."""
    idx = mask.nonzero()[:, 0]
    out = dict(n=int(idx.numel()), nll=d["nll"][idx], p0=d["p0"][idx], doc=d["doc"][idx],
               counts={o: {k: v[idx] for k, v in d["counts"][o].items()} for o in orders},
               phis=[ph[idx] for ph in d["phis"]], top=None)
    if d["top"] is not None:
        out["top"] = {k: v[idx] for k, v in d["top"].items()}
    return out


def _cat(parts: list, orders) -> dict:
    out = dict(n=sum(p["n"] for p in parts), nll=torch.cat([p["nll"] for p in parts]),
               p0=torch.cat([p["p0"] for p in parts]),
               counts={o: {k: torch.cat([p["counts"][o][k] for p in parts]) for k in parts[0]["counts"][o]} for o in orders},
               phis=[torch.cat([p["phis"][i] for p in parts]) for i in range(len(orders))], top=None,
               doc=torch.cat([p["doc"] for p in parts]))
    if parts[0]["top"] is not None:
        out["top"] = {k: torch.cat([p["top"][k] for p in parts]) for k in parts[0]["top"]}
    return out


def moments(ph: torch.Tensor):
    """mean / sd of columns 1: (column 0 is the constant), float64; sd < 1e-9 -> 1 (joint.py's _moments)."""
    if ph.shape[0] == 0:
        return torch.zeros(ph.shape[1] - 1, dtype=torch.float64), torch.ones(ph.shape[1] - 1, dtype=torch.float64)
    p = ph[:, 1:].to(torch.float64)
    mu, sd = p.mean(0), p.std(0, unbiased=False)
    sd = torch.where(sd < 1e-9, torch.ones_like(sd), sd)
    return mu, sd


class Model:
    """The gate's parameters as one flat vector: per order [w (d_o), disc (3)], then the top block W [d_top, K]."""

    def __init__(self, sm, data: dict, orders, mode: str):
        self.sm, self.orders, self.mode = sm, list(orders), mode
        self.dims = [ph.shape[1] for ph in data["phis"]]
        self.stats = [moments(ph[data["counts"][o]["N"] > 0]) for o, ph in zip(self.orders, data["phis"])]
        self.top_dim = self.top_k = 0
        if data["top"] is not None:
            has = data["top"]["avail"].any(1)
            self.top_dim, self.top_k = data["top"]["phi"].shape[1], data["top"]["comps"].shape[1]
            self.top_stats = moments(data["top"]["phi"][has])
        self.size = sum(d + 3 for d in self.dims) + self.top_dim * self.top_k

    def unpack(self, x: torch.Tensor):
        w, disc, o = [], [], 0
        for d in self.dims:
            w.append(x[o:o + d])
            disc.append(x[o + d:o + d + 3])
            o += d + 3
        W = x[o:].reshape(self.top_dim, self.top_k) if self.top_k else None
        return w, disc, W

    def x0(self) -> np.ndarray:
        x = np.zeros(self.size)
        o = 0
        for d in self.dims:
            x[o] = -3.0
            x[o + d:o + d + 3] = (0.0, -1.0, -1.0)
            o += d + 3
        if self.top_k:
            x[o:o + self.top_k] = -3.0  # bias row (phi column 0 is the constant)
        return x

    def reg(self) -> np.ndarray:
        r = np.zeros(self.size)
        o = 0
        for d in self.dims:
            r[o + 1:o + d] = L2_GATE
            o += d + 3
        if self.top_k:
            r[o + self.top_k:] = L2_TOP
        return r

    def logq(self, x: torch.Tensor, data: dict):
        """(log q with EPS as the eval's mix_v2 returns it, the positions some component touches)."""
        sm = self.sm
        w, disc, W = self.unpack(x)
        P = sm.chain_prob(data["p0"], data["counts"], data["phis"], self.orders, w, disc,
                          [s[0] for s in self.stats], [s[1] for s in self.stats])
        lq = torch.log(P.clamp_min(1e-300))
        hit = torch.zeros(data["n"], dtype=torch.bool)
        for o in self.orders:
            hit |= data["counts"][o]["N"] > 0
        if W is not None:
            lq = sm.top_mixture(lq, data["top"], dict(W=W, mu=self.top_stats[0], sd=self.top_stats[1]))
            hit |= data["top"]["avail"].any(1)
        return torch.log(torch.exp(lq) + EPS), hit

    def fit(self, data: dict, maxiter=1500, verbose=True) -> np.ndarray:
        reg = torch.from_numpy(self.reg())
        n = float(data["n"])
        base = -data["nll"]
        it = [0]

        def f(xn):
            x = torch.tensor(xn, requires_grad=True)
            lq, hit = self.logq(x, data)
            loss = -(lq - base)[hit].sum() / n + 0.5 * (reg * x * x).sum() / n
            loss.backward()
            it[0] += 1
            return loss.item(), x.grad.numpy().copy()

        t0 = time.time()
        r = minimize(f, self.x0(), jac=True, method="L-BFGS-B",
                     options={"maxiter": maxiter, "maxfun": int(maxiter * 1.25), "gtol": 1e-8})
        if verbose:
            print(f"  fit: {self.size} constants, {it[0]} evaluations, {time.time() - t0:.0f} s, loss {r.fun:.6f} "
                  f"({r.message})", flush=True)
        self.x = r.x
        return r.x

    def gain_mnat(self, data: dict, x=None) -> float:
        """Mean over ALL positions of nll - mixed NLL, in millinats (positions no component touches contribute 0)."""
        with torch.no_grad():
            lq, hit = self.logq(torch.from_numpy(self.x if x is None else x), data)
        return 1000 * float(((lq + data["nll"]) * hit).sum()) / data["n"]

    def spec(self, note: dict) -> dict:
        note = dict(note)
        cols = note.pop("top_cols", None)
        w, disc, W = self.unpack(torch.from_numpy(self.x))
        r = lambda t: [float(f"{v:.7g}") for v in t.reshape(-1).tolist()]
        out = {"version": 2, "mode": self.mode, "orders": self.orders, "w": [r(v) for v in w], "disc": [r(v) for v in disc],
               "mu": [r(s[0]) for s in self.stats], "sd": [r(s[1]) for s in self.stats], "top": None, "fit": note}
        if W is not None:
            out["top"] = {"comps": note.get("top_comps", []), "cols": cols, "W": [r(row) for row in W],
                          "mu": r(self.top_stats[0]), "sd": r(self.top_stats[1])}
        return out


def module_block(spec: dict) -> str:
    """The constants block a spec belongs in: GATE_V2_LOW for a chain with P2's orders, else GATE_V2."""
    return "GATE_V2_LOW" if min(spec["orders"]) < min(OVERLAY_LEVELS) else "GATE_V2"


def read_module(path: Path = OVERLAY_MODULE, block: str = "GATE_V2") -> list:
    """The specs of a constants block of stream_memory.py, read from its text (no import)."""
    text = path.read_text()
    body = text[text.index(f"# {block}_BEGIN\n"):text.index(f"# {block}_END")]
    m = re.search(rf"{block}_JSON = (\(.*\))\s*$", body, re.S)
    if not m:
        return []
    spec = json.loads(ast.literal_eval(m.group(1)))
    return spec if isinstance(spec, list) else [spec]


def merge_specs(specs: list, spec: dict) -> list:
    """The block's specs with `spec` in: it replaces a spec fitted at the same step count, and, fitted on the run's own
    model (no proxy), every placeholder fitted on a proxy's outputs. Longest run first."""
    steps = spec["fit"].get("total_steps")
    own = not spec["fit"].get("proxy")
    keep = [s for s in specs if s["fit"].get("total_steps") != steps and not (own and s["fit"].get("proxy"))]
    return sorted(keep + [spec], key=lambda s: -(s["fit"].get("total_steps") or 0))


def write_module(spec: dict, path: Path = OVERLAY_MODULE, block: str | None = None, replace: bool = False):
    """Put the spec into its constants block of stream_memory.py (between the <block>_BEGIN / <block>_END markers):
    merged with the specs there (merge_specs), or alone (replace)."""
    block = block or module_block(spec)
    specs = [spec] if replace else merge_specs(read_module(path, block), spec)
    text = path.read_text()
    a, b = text.index(f"# {block}_BEGIN\n") + len(f"# {block}_BEGIN\n"), text.index(f"# {block}_END")
    js = json.dumps(specs, separators=(",", ":"))
    lines = [js[i:i + 112] for i in range(0, len(js), 112)]
    body = f"{block}_JSON = (\n" + "".join(f"    {line!r}\n" for line in lines) + ")\n"
    path.write_text(text[:a] + body + text[b:])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--ranks", default=None, help="comma-separated ranks (default: every rank with an .lm file)")
    parser.add_argument("--mode", default=None, help="feature blocks (default: stream_memory.GATE_MODE)")
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--heldout", action="store_true", help="also fit 2-fold by document and print the held-out gain")
    parser.add_argument("--out", default=None, help="write the constants as JSON (STREAM_RETRIEVAL_GATE=<it> uses them)")
    parser.add_argument("--module", action="store_true", help="write them into the overlay's stream_memory.py")
    parser.add_argument("--note", default="", help="provenance text stored with the constants")
    parser.add_argument("--proxy", default=None, metavar="TEXT",
                        help="the model outputs are a proxy's (TEXT says which): a placeholder a record refuses")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    sm = overlay()
    mode = args.mode or sm.GATE_MODE
    path = sm.fit_file(args.path)
    fit = sm.read_fit(path)
    ranks = [int(r) for r in args.ranks.split(",")] if args.ranks else \
        [r for r in range(fit["world"]) if Path(f"{path}.lm.rank{r}.npz").exists()]
    if not ranks:
        sys.exit(f"no {path}.lm.rank<r>.npz: the run's eval outputs at the FIT positions (stream_memory.save_fit_lm)")
    t0 = time.time()
    lms = [dict(np.load(f"{path}.lm.rank{r}.npz")) for r in ranks]
    parts, folds, doc_off = [], [], 0
    for r, lm in zip(ranks, lms):  # one rank at a time (its features), then the ranks' positions in one dataset
        p = rank_data(sm, fit, r, lm, mode)
        folds.append(((p["doc"] + doc_off) % 2).to(torch.bool))  # documents numbered across ranks: no fold splits one
        doc_off += int(p["doc"][-1]) + 1
        parts.append(p)
    orders = parts[0]["orders"]
    if any(p["orders"] != orders for p in parts):
        sys.exit("the ranks' dumps disagree on the chain's orders")
    top_names, top_cols = parts[0]["top_names"], parts[0]["top_cols"]
    data = _cat(parts, orders)
    del parts
    fold = torch.cat(folds)
    hits = int((torch.stack([data["counts"][o]["N"] > 0 for o in orders]).any(0)).sum())
    print(f"{data['n']} FIT positions from ranks {ranks} (the last {fit['k']} batches of a {fit['steps']}-step run; "
          f"memory frozen at entry {fit['freeze']}; chain orders {orders}"
          f"{'; top ' + str(top_names) if top_names else ''}), {hits / data['n']:.4f} with some order matched; mean NLL "
          f"{float(data['nll'].mean()):.4f}; features {time.time() - t0:.0f} s", flush=True)
    note = dict(source=str(path), ranks=ranks, positions=data["n"], matched=round(hits / data["n"], 5),
                batches=fit["k"], total_steps=fit["steps"], freeze=fit["freeze"], mode=mode, note=args.note,
                parts="P1" + (" + P2" if fit["low"] is not None else "") + (" + P3" if fit["ptr"] is not None else ""))
    if args.proxy:
        note["proxy"] = args.proxy
    if top_names is not None:
        note["top_comps"] = top_names
    if top_cols is not None:
        note["top_cols"] = top_cols
    if args.heldout:
        held = 0.0
        for k in (0, 1):
            tr, te = _select(data, fold != bool(k), orders), _select(data, fold == bool(k), orders)
            m = Model(sm, tr, orders, mode)
            m.fit(tr, args.maxiter)
            held += m.gain_mnat(te) * te["n"] / data["n"]
            del tr, te, m
        print(f"held out (2-fold by document): {held:.2f} mnat", flush=True)
        note["heldout_mnat"] = round(held, 3)
    model = Model(sm, data, orders, mode)
    model.fit(data, args.maxiter)
    g = model.gain_mnat(data)
    print(f"in sample: {g:.2f} mnat over the model's own NLL at the FIT positions", flush=True)
    note["in_sample_mnat"] = round(g, 3)
    spec = model.spec(note)
    if args.out:
        Path(args.out).write_text(json.dumps(spec))
        print(f"wrote {args.out}")
    if args.module:
        write_module(spec)
        print(f"wrote the spec for {fit['steps']} trained steps into the {module_block(spec)} block of {OVERLAY_MODULE}")
    return spec


if __name__ == "__main__":
    main()
