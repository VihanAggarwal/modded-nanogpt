"""CPU tests for the per-segment pointer beam, vote, source copy and doc-state counters (part P3 of stream retrieval v2:
track_1_short/stream_pointer.c and its binding stream_pointer.py). Standalone: they compile the overlay's copy with cc
into a temp dir and need no GPU, shards or the helper.

  1 layout: stream_pointer.py's ROW_DTYPE matches the C struct field by field (sp_layout)
  2 exact: every field of every row equals an independent Python reference (written from the research tools'
    semantics: rg/align/align4.c NREF 0, rg/docgate/srccopy6.c, docstate.c, rg/align/feats.py), on synthetic memories
    and val streams built by copying memory spans with substitutions, insertions and deletions (every recovery path of
    the beam), with BOS and chunk boundaries inside documents; and the per-position API (sp_step) equals the driver
  3 causality: rows <= t are bit-identical when every token after t (inputs, targets, so the candidates too) changes,
    and when only t's own target changes
  4 normalization: the vote shares sum to 1 and the source counts to N at li (when not truncated); by enumerating the
    vocabulary, each component (pointer, vote, source) sums to 1 over the vocabulary (component_probs)
  5 determinism: the same row bytes for 1..8 threads; a segment's rows do not depend on any other segment; the state
    resets at BOS and at chunk boundaries (a document cut by a chunk start equals its second part run alone)
  6 reads stay in the candidates' spans: memory past a freeze (FIT) can be anything without changing a row
  7 FIT-style sequences: explicit targets and runs (segments restart at batch starts)
  8 the memory summary equals stream_memory.c's record (query_one) on the same candidates, when that file compiles
  9 features() / component_probs() give the same values on numpy and torch
 10 the C file compiles warning-free, alone and with -DSP_NO_SIMD (the same rows)
 11 (STREAM_POINTER_REAL=1) the research parity on real data and the costs: tools/stream_retrieval/stream_pointer_bench.c
    over the 1050-step stream, compared with rg/align's c0.feat.f32 (align4) and the research libraries
"""
import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ARM = HERE / "arm/track_1_short"
sys.path.insert(0, str(ARM))
import stream_pointer as SP  # noqa: E402

BOS, SEP = SP.BOS, SP.SEP
LEVELS, LV = SP.LEVELS, SP.LV
_M = ctypes.CDLL(ctypes.util.find_library("m"))
for _fn in ("log2", "exp2"):
    getattr(_M, _fn).restype = ctypes.c_double
    getattr(_M, _fn).argtypes = [ctypes.c_double]
f32 = lambda v: float(np.float32(v))


@pytest.fixture(scope="module")
def lib():
    return SP.build_lib()


# ------------------------------------------------------------------------------------------------ data

def make_memory(rng, ndocs=60, vocab=30, minlen=20, maxlen=400):
    """The helper's tok: SEP, then spans (documents, BOS first) each followed by SEP. Documents are drawn from a small
    first-order Markov chain over a tiny vocabulary, and some are near-copies of others, so 6-token contexts recur
    with several continuations."""
    trans = rng.dirichlet(np.full(vocab, 0.3), vocab)
    docs = []
    for i in range(ndocs):
        if docs and rng.random() < 0.3:
            d = edit(rng, docs[int(rng.integers(len(docs)))][1:], vocab, 0.05)
            d = np.concatenate([[BOS], d[: int(rng.integers(minlen, maxlen))]]).astype(np.uint16)
        else:
            n = int(rng.integers(minlen, maxlen))
            d = np.empty(n, np.uint16)
            d[0] = BOS
            v = int(rng.integers(vocab))
            for k in range(1, n):
                v = int(rng.choice(vocab, p=trans[v]))
                d[k] = v
        docs.append(d)
    for _ in range(2):  # 'fan' documents: the bigram (0, 1) followed by 25 different tokens (source truncation)
        d = [BOS]
        for i in range(60):
            d += [0, 1, 2 + i % 25] + rng.integers(2, vocab, 3).tolist()
        docs.append(np.array(d, np.uint16))
    parts = [np.array([SEP], np.uint16)]
    for d in docs:
        parts += [d, np.array([SEP], np.uint16)]
    return np.concatenate(parts), docs


def edit(rng, a, vocab, rate):
    """a copy of a with substitutions, insertions and deletions at `rate` each."""
    out = []
    for v in a:
        r = rng.random()
        if r < rate:
            out.append(int(rng.integers(vocab)))        # substitution
        elif r < 2 * rate:
            out += [int(v), int(rng.integers(vocab))]    # insertion
        elif r < 3 * rate:
            continue                                     # deletion
        else:
            out.append(int(v))
    return np.array(out, np.uint16)


def make_val(rng, docs, n, vocab=30):
    """Val documents (BOS first): stretches copied from memory documents with edits, and random text."""
    parts = []
    tot = 0
    while tot < n + 1:
        d = docs[int(rng.integers(len(docs)))]
        a = int(rng.integers(1, max(2, d.size - 10)))
        piece = edit(rng, d[a:a + int(rng.integers(10, 200))], vocab, float(rng.choice([0.0, 0.02, 0.08])))
        noise = rng.integers(0, vocab, int(rng.integers(0, 30))).astype(np.uint16)
        doc = np.concatenate([[BOS], noise, piece, rng.integers(0, vocab, int(rng.integers(0, 10)))]).astype(np.uint16)
        if rng.random() < 0.3:  # a second copied stretch in the same document
            d2 = docs[int(rng.integers(len(docs)))]
            doc = np.concatenate([doc, edit(rng, d2[1:120], vocab, 0.03)]).astype(np.uint16)
        if rng.random() < 0.1:  # a stretch of a fan document (a source), then its bigram with new continuations
            f = docs[-1 - int(rng.integers(2))]
            a = int(rng.integers(1, f.size - 40))
            doc = np.concatenate([doc, f[a:a + 30], [0, 1, 7, 9, 0, 1, 4]]).astype(np.uint16)
        parts.append(doc)
        tot += doc.size
    return np.concatenate(parts)[:n + 1]


def index_memory(mem):
    """Exact index: 6-token context -> memory entries p (ascending) whose context lies in p's span and whose next
    token is in the span (stream_memory.c's insertion rule)."""
    idx = {}
    run = 0
    for p in range(mem.size - 1):
        run = 0 if mem[p] == SEP else run + 1
        if run >= 6 and mem[p + 1] != SEP:
            idx.setdefault(tuple(mem[p - 5:p + 1].tolist()), []).append(p)
    return idx


def walk(mem, idx, x, runs, limit=None):
    """Candidates of every position as stream_memory.c's walk finds them (exact buckets): the 32 most recent entries
    (below `limit`) with the 6-token context, lengths capped at min(32, run). Returns CSR arrays."""
    off, pos, ln = [0], [], []
    for t in range(x.size):
        r = int(runs[t])
        if r >= 6:
            ps = idx.get(tuple(x[t - 5:t + 1].tolist()), [])
            if limit is not None:
                ps = [p for p in ps if p < limit]
            lim = min(32, r)
            for p in ps[::-1][:32]:
                l = 6
                while l < lim and mem[p - l] == x[t - l]:
                    l += 1
                pos.append(p)
                ln.append(l)
        off.append(len(pos))
    return np.array(off, np.uint64), np.array(pos, np.uint32), np.array(ln, np.uint8)


def runs_of(x, chunk):
    return SP.segment_runs(x, chunk=chunk)


# ------------------------------------------------------------------------------------------------ the reference

def popcount(v):
    return bin(v).count("1")


def lowmask(n):
    return (1 << 64) - 1 if n >= 64 else (1 << n) - 1


class Hyp:
    __slots__ = ("j", "run", "hits", "miss", "hist", "nhist", "age", "since", "seedlen", "nrec", "how", "score")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, 0))

    def copy(self):
        return Hyp(**{k: getattr(self, k) for k in self.__slots__})


def score_of(h):
    n16, n64 = min(h.nhist, 16), h.nhist
    c16, c64 = popcount(h.hist & lowmask(n16)), popcount(h.hist & lowmask(n64))
    return f32(_M.log2(1.0 + h.run) + 0.5 * _M.log2(1.0 + h.hits) + 0.15 * c64 - 1.0 * (n16 - c16) - 0.25 * (n64 - c64))


def mem_summary(mem, x, t, run, cands, y):
    if not cands:
        return None
    maxl = max(l for _, l in cands)
    ls = max(k for k, L in enumerate(LEVELS) if L <= maxl)
    L = LEVELS[ls]
    cnt = Counter(int(mem[p + 1]) for p, l in cands if l >= L)
    m = max(cnt.values())
    top = min(v for v, c in cnt.items() if c == m)

    def full(p, l):
        if l < 32:
            return l
        lim = min(run, 4096)
        while l < lim and mem[p - l] == x[t - l]:
            l += 1
        return l
    irec = next(i for i, (_, l) in enumerate(cands) if l >= L)
    lrec = full(*cands[irec])
    best, ilong, bp = 0, 0, min(run, 4096)
    for i, (p, l0) in enumerate(cands):
        if best >= bp:
            break
        if l0 <= best and l0 < 32:
            continue
        l = lrec if i == irec else full(p, l0)
        if l > best:
            best, ilong = l, i
    return dict(top=top, n=sum(cnt.values()), m=m, c=cnt.get(int(y), 0), lstar=L, ncand=len(cands),
                pos_recent=cands[irec][0], len_recent=lrec, pos_longest=cands[ilong][0], len_longest=best)


class Track:
    """docstate.c over full per-segment histories (windows summed afresh)."""

    def __init__(self):
        self.h, self.c, self.r, self.hc, self.hr = [], [], [], [], []
        self.streak, self.since_wrong, self.since_hit = 0, 100000, 100000
        self.ema90 = self.ema98 = 0.0
        self.nlong32 = self.nlong16 = self.cont_run = self.cont_corr_run = 0
        self.prev_posr = None

    def snapshot(self):
        o = dict(npos=len(self.h), nhit=len(self.hc), ncorr=sum(self.hc), streak=self.streak,
                 since_wrong=self.since_wrong, since_hit=self.since_hit, nlong32=self.nlong32, nlong16=self.nlong16,
                 ema90=self.ema90, ema98=self.ema98)
        o["p_hit"] = [sum(self.h[-K:]) if self.h else 0 for K in (8, 32, 128)]
        o["p_corr"] = [sum(self.c[-K:]) if self.c else 0 for K in (8, 32, 128)]
        o["h_corr"] = [sum(self.hc[-K:]) if self.hc else 0 for K in (8, 32, 128)]
        r2 = 0.0
        for v in self.hr[::-1][:32]:
            r2 += v
        o["h32_r"] = r2
        o["prev_corr"] = self.c[-1] if self.c else 0
        o["prev_hit"] = self.h[-1] if self.h else 0
        return o

    def update(self, h, c, r, lenl):
        self.h.append(h)
        self.c.append(c)
        if h:
            self.hc.append(c)
            self.hr.append(r)
            self.since_hit = 0
        else:
            self.since_hit += 1
        self.since_wrong = 0 if (h and not c) else self.since_wrong + 1
        self.streak = self.streak + 1 if c else 0
        self.ema90 = 0.9 * self.ema90 + 0.1 * c
        self.ema98 = 0.98 * self.ema98 + 0.02 * c
        self.nlong32 += bool(h and lenl >= 32)
        self.nlong16 += bool(h and lenl >= 16)


def reference(mem, x, y, runs, cands_of, cfg=None):
    """Rows (dicts) of the active positions, from the research semantics, position by position."""
    K, SEED, DMAX, FWD, BACK, GRAM = 16, 6, 8, 64, 64, 2
    TH, MAXSRC, HALF, VOTES = 10, 8, 4096, 4
    ntok = mem.size
    rows = []
    for t in range(x.size):
        run = int(runs[t])
        if run == 1:
            H = []
            srcs, newest_t, newest_len, vb = [], -1, 0, {}
            vbins = []
            mt, st = Track(), Track()
            wh = []  # per position: (has, corr, hp, bptr, r)
            seg0 = t
        cands = cands_of(t)
        yv = int(y[t])
        m = mem_summary(mem, x, t, run, cands, yv)
        has = m is not None
        row = {}
        # ---- beam: seed, rank, row
        if run >= 6:
            lim = min(run, 512)
            for p, l in cands:
                if l >= 32:
                    while l < lim and mem[p - l] == x[t - l]:
                        l += 1
                f = next((h for h in H if h.j == p), None)
                if f is not None:
                    f.run = max(f.run, l)
                elif l >= SEED:
                    nh = min(l, 64)
                    H.append(Hyp(j=p, run=l, hits=l, since=l, seedlen=l, nhist=nh, hist=lowmask(nh)))
        for h in H:
            h.score = score_of(h)
        H = sorted(H, key=lambda h: -h.score)  # stable
        H = H[:K]
        best, wsum, toks, tw, tc, nvh = None, 0.0, [], [], [], 0
        for i, h in enumerate(H):
            v = int(mem[h.j + 1])
            if v in (SEP, BOS):
                continue
            nvh += 1
            if best is None:
                best = i
            w = _M.exp2(float(np.float32(h.score) - np.float32(H[best].score)))
            wsum += w
            if v not in toks:
                toks.append(v); tw.append(0.0); tc.append(0)
            q = toks.index(v)
            tw[q] += w
            tc[q] += 1
        hp = best is not None
        if hp:
            b = H[best]
            v = int(mem[b.j + 1])
            q = toks.index(v)
            s2 = None
            for i in range(len(toks)):
                if i != q and (s2 is None or tw[i] > tw[s2]):
                    s2 = i
            vt = 0
            for i in range(1, len(toks)):
                if tw[i] > tw[vt]:
                    vt = i
            n16 = min(b.nhist, 16)
            row.update(ptr_j=b.j, score=b.score, score2=H[best + 1].score if len(H) > best + 1 else -99.0,
                       share=f32(tw[q] / wsum), share2=f32(tw[s2] / wsum) if s2 is not None else 0.0,
                       vshare=f32(tw[vt] / wsum), pred=v, vtop=toks[vt], tok2=toks[s2] if s2 is not None else 0xFFFF,
                       run=b.run, hits=b.hits, miss=b.miss, age=b.age, since=b.since, seedlen=b.seedlen, nrec=b.nrec,
                       c16=popcount(b.hist & lowmask(n16)), n16=n16, c64=popcount(b.hist), nhist=b.nhist, how=b.how,
                       nvh=nvh, tcnt=tc[q], ntok=len(toks), vtok=toks, vw=[f32(w / wsum) for w in tw])
        # ---- source discovery
        if has:
            for w in (0, 1):
                ll, pp = (m["len_recent"], m["pos_recent"]) if w else (m["len_longest"], m["pos_longest"])
                strong = ll >= TH
                if not strong and VOTES > 0:
                    bn = pp >> 9
                    if bn not in vb and len(vb) < 256:
                        vb[bn] = 0
                    if bn in vb:
                        vb[bn] += 1
                        strong = vb[bn] >= VOTES
                if not strong:
                    continue
                dup = next((i for i, (a, bb) in enumerate(srcs) if a <= pp < bb), None)
                if dup is not None:
                    srcs.append(srcs.pop(dup))
                else:
                    a, bb = pp, pp + 1
                    while a > 0 and a > pp - HALF and mem[a] != BOS and mem[a - 1] != SEP:
                        a -= 1
                    while bb < ntok and bb < pp + HALF and mem[bb] != SEP and mem[bb] != BOS:
                        bb += 1
                    if len(srcs) == MAXSRC:
                        srcs.pop(0)
                    srcs.append((a, bb))
                newest_t, newest_len = t - seg0, ll
        row.update(src_nsrc=len(srcs), src_since=(t - seg0 - newest_t) if newest_t >= 0 else -1, src_newlen=newest_len)
        # ---- source query
        hs, sc_y, trunc = False, None, False
        if srcs and run >= 2:
            vmax = min(run, 32)
            sc = []
            for a, bb in srcs:
                for u in range(a + 1, bb - 1):
                    if mem[u] != x[t] or mem[u - 1] != x[t - 1]:
                        continue
                    nx = int(mem[u + 1])
                    if nx in (SEP, BOS):
                        continue
                    lim = min(vmax, u - a + 1)
                    l = 1
                    while l < lim and mem[u - l] == x[t - l]:
                        l += 1
                    sc.append((nx, l))
            if sc:
                hs = True
                N = [sum(1 for _, l in sc if l >= L) for L in LV]
                li = max(i for i in range(8) if N[i] > 0)
                cnt = Counter(nx for nx, l in sc if l >= LV[li])
                order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
                trunc = len(order) > 16
                row.update(src_n=N, src_li=li, src_m=max(cnt.values()), src_lbest=max(l for _, l in sc),
                           nsd=min(16, len(order)), stok=[k for k, _ in order[:16]], scnt=[c for _, c in order[:16]])
                sc_y = (cnt.get(yv, 0) if yv != BOS else 0, max(cnt.values()), N[li])
        # ---- memory fields and doc-state
        if has:
            row.update(mem_lenl=m["len_longest"], mem_lenr=m["len_recent"], mem_top=m["top"], mem_ncand=m["ncand"],
                       mem_lstar=m["lstar"], mem_n=m["n"], mem_m=m["m"])
        cont = bool(mt.h) and has and bool(mt.h[-1]) and mt.prev_posr is not None and m["pos_recent"] == mt.prev_posr + 1
        mt.cont_run = mt.cont_run + 1 if cont else 0
        mt.cont_corr_run = mt.cont_corr_run + 1 if (cont and mt.c and mt.c[-1]) else 0
        mt.prev_posr = m["pos_recent"] if has else None
        if hp or hs:
            d = mt.snapshot()
            row.update(d_npos=d["npos"], d_nhit=d["nhit"], d_ncorr=d["ncorr"], d_streak=d["streak"],
                       d_since_wrong=d["since_wrong"], d_since_hit=d["since_hit"], d_nlong32=d["nlong32"],
                       d_nlong16=d["nlong16"], d_cont_run=mt.cont_run, d_cont_corr_run=mt.cont_corr_run,
                       d_ema90=f32(d["ema90"]), d_ema98=f32(d["ema98"]), d_h32_r=f32(d["h32_r"]),
                       d_p_hit=d["p_hit"], d_p_corr=d["p_corr"], d_h_corr=d["h_corr"])
            s = st.snapshot()
            row.update(s_nhit=s["nhit"], s_ncorr=s["ncorr"], s_streak=s["streak"], s_h32_r=f32(s["h32_r"]),
                       s_h_corr=s["h_corr"][:2])
            T = len(wh)
            for w, W in ((0, 16), (1, 64)):
                win = wh[max(0, T - W):]
                for k, nm in ((0, "h_r_n"), (1, "h_r_c"), (2, "h_p_n"), (3, "h_p_c")):
                    row.setdefault(nm, [0, 0])[w] = sum(e[k] for e in win)
                row.setdefault("h_e", [0, 0])[w] = sum(e[1] | e[3] for e in win)
            win = wh[max(0, T - 32):]
            row.update(h32_r_n=sum(e[0] for e in win), h32_p_n=sum(e[2] for e in win), h32_p_c=sum(e[3] for e in win))
            rs = 0.0
            for e in win[::-1]:
                rs += e[4]
            rseg = 0.0
            for e in wh:
                rseg += e[4]
            row.update(h32_r_sum=f32(rs), h_r_sum_seg=f32(rseg), h_p_n_seg=sum(e[2] for e in wh),
                       h_p_c_seg=sum(e[3] for e in wh), h_e_seg=sum(e[1] | e[3] for e in wh))
            flags = (SP.F_HP if hp else 0) | (SP.F_HS if hs else 0) | (SP.F_HAS if has else 0)
            flags |= (SP.F_SRC_TRUNC if trunc else 0) | (SP.F_CONT if cont else 0)
            flags |= (SP.F_PREV_CORR if d["prev_corr"] else 0) | (SP.F_PREV_HIT if d["prev_hit"] else 0)
            flags |= SP.F_SPREV_CORR if s["prev_corr"] else 0
            row.update(pos=t, flags=flags)
            rows.append(row)
        # ---- outcome of t
        corr = int(has and m["top"] == yv)
        r = m["c"] / m["n"] if has else 0.0
        mt.update(int(has), corr, r, m["len_longest"] if has else 0)
        if hs:
            cy, mm, nli = sc_y
            st.update(1, int(cy == mm and cy > 0), cy / nli, 0)
        else:
            st.update(0, 0, 0.0, 0)
        bptr = int(hp and row["pred"] == yv)
        wh.append((int(has), corr, int(hp), bptr, r))
        # ---- beam update
        NH = []
        for h in H:
            p = h.copy()
            p.age += 1
            v = int(mem[p.j + 1])
            if v == yv and v != SEP:
                p.j += 1; p.run += 1; p.hits += 1; p.since += 1
                p.hist = ((p.hist << 1) | 1) & lowmask(64)
                p.nhist = min(p.nhist + 1, 64)
                p.how = 1
                NH.append(p)
                continue
            c = p.copy()
            c.run = 0; c.miss += 1; c.since = 0; c.hist = (c.hist << 1) & lowmask(64)
            c.nhist = min(c.nhist + 1, 64); c.nrec += 1
            if v != SEP:
                e = c.copy(); e.j = p.j + 1; e.how = 2; NH.append(e)
            e = c.copy(); e.how = 3; NH.append(e)
            for k in range(1, DMAX + 1):
                jj = p.j + 1 + k
                if jj >= ntok or mem[jj] == SEP:
                    break
                if mem[jj] == yv and mem[p.j + k] != SEP:
                    e = c.copy(); e.j = jj; e.how = 4; e.run = 1; NH.append(e)
                    break
            if GRAM > 0 and run + 1 >= GRAM:
                ctx = [yv] + [int(x[t - g]) for g in range(GRAM - 1)]   # tokens t + 1, t, t - 1, ...

                def gram_ok(i):
                    g = 1
                    while g < GRAM and i - g >= 0 and mem[i - g] == ctx[g]:
                        g += 1
                    return g == GRAM
                bestd, bi = 1 << 40, -1
                for i in range(p.j + 1, min(p.j + 1 + FWD, ntok - 1) + 1):
                    if mem[i] == SEP:
                        break
                    if mem[i] == yv and gram_ok(i):
                        bestd, bi = i - (p.j + 1), i
                        break
                for i in range(p.j, max(p.j - BACK, 0) - 1, -1):
                    if mem[i] == SEP:
                        break
                    if mem[i] == yv and gram_ok(i):
                        if p.j + 1 - i < bestd:
                            bestd, bi = p.j + 1 - i, i
                        break
                if bi >= 0 and bi != p.j + 1:
                    e = c.copy(); e.j = bi; e.how = 5; e.run = GRAM; NH.append(e)
        for e in NH:
            e.score = score_of(e)
        H2 = []
        for e in NH:
            dup = next((i for i, h in enumerate(H2) if h.j == e.j), None)
            if dup is None:
                H2.append(e)
            elif e.score > H2[dup].score:
                H2[dup] = e
        H = [h for h in H2 if min(h.nhist, 8) - popcount(h.hist & lowmask(min(h.nhist, 8))) < 6]
    return rows


ARRAY_FIELDS = {"vtok": "ntok", "vw": "ntok", "stok": "nsd", "scnt": "nsd"}


def compare(rows, ref):
    """Every field of every row (the reference leaves fields absent where the C row is 0)."""
    assert len(rows) == len(ref), (len(rows), len(ref))
    names = [n for n in SP.ROW_DTYPE.names if not n.startswith("pad")]
    for i, (r, e) in enumerate(zip(rows, ref)):
        for n in names:
            got = r[n]
            want = e.get(n, 0)
            if n in ARRAY_FIELDS:
                k = int(r[ARRAY_FIELDS[n]])
                want = np.zeros(got.shape, got.dtype)
                if k:
                    want[:k] = np.asarray(e[n][:k])
            elif np.ndim(got):
                w = np.zeros(got.shape, got.dtype)
                if n in e:
                    w[:len(e[n])] = e[n]
                want = w
            assert np.array_equal(np.asarray(got), np.asarray(want, dtype=np.asarray(got).dtype)), \
                f"row {i} (pos {r['pos']}) field {n}: {got} != {want}"


# ------------------------------------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(1)
    mem, docs = make_memory(rng)
    idx = index_memory(mem)
    x = make_val(rng, docs, 6000)
    chunk = 1500
    runs = runs_of(x[:-1], chunk)
    off, pos, ln = walk(mem, idx, x[:-1], runs)
    return dict(mem=mem, docs=docs, idx=idx, x=x, chunk=chunk, runs=runs, off=off, pos=pos, ln=ln)


def run_rows(w, threads=1, x=None, cands=None, y=None, runs=None, config=None, mem=None, chunk=None):
    x = w["x"] if x is None else x
    off, pos, ln = cands if cands is not None else (w["off"], w["pos"], w["ln"])
    return SP.run(w["mem"] if mem is None else mem, 0, x, off, pos, ln, y=y, runs=runs,
                  chunk=w["chunk"] if chunk is None else chunk, threads=threads, config=config)


def cands_fn(off, pos, ln):
    return lambda t: [(int(pos[k]), int(ln[k])) for k in range(int(off[t]), int(off[t + 1]))]


# ------------------------------------------------------------------------------------------------ tests

def test_layout(lib):
    lay = {}
    for e in lib.sp_layout().decode().split(","):
        name, o, sz = e.split()
        lay[name] = (int(o), int(sz))
    assert lay.pop("sizeof")[0] == SP.ROW_BYTES == lib.sp_row_bytes()
    for name, (o, sz) in lay.items():
        dt, off = SP.ROW_DTYPE.fields[name][:2]
        assert (off, dt.itemsize) == (o, sz), name
    named = {n for n in SP.ROW_DTYPE.names if not n.startswith("pad")}
    assert named == set(lay), named ^ set(lay)
    assert ctypes.sizeof(SP.Config) == 64


def test_exact_vs_reference(world):
    w = world
    rows = run_rows(w)
    ref = reference(w["mem"], w["x"][:-1], w["x"][1:], w["runs"], cands_fn(w["off"], w["pos"], w["ln"]))
    compare(rows, ref)
    f = rows["flags"]
    how = Counter(rows["how"][(f & SP.F_HP) != 0].tolist())
    # the data exercises every path: every hypothesis origin, sources, truncation, cont runs, chunk restarts
    assert set(how) == {0, 1, 2, 3, 4, 5}, how
    assert ((f & SP.F_HS) != 0).sum() > 200 and ((f & SP.F_HP) != 0).sum() > 500
    assert ((f & SP.F_SRC_TRUNC) != 0).any() and (rows["d_cont_run"] > 0).any() and (rows["src_nsrc"] > 1).any()
    assert (rows["ntok"] > 3).any() and (rows["src_since"] > 0).any()


def test_step_api_equals_driver(lib, world):
    w = world
    rows = run_rows(w, threads=3)
    st = lib.sp_state_new(ctypes.byref(SP.default_config()))
    mem, x, runs, off, pos, ln = w["mem"], w["x"], w["runs"], w["off"], w["pos"], w["ln"]
    row = np.zeros(1, SP.ROW_DTYPE)
    got = []
    for t in range(x.size - 1):
        a, b = int(off[t]), int(off[t + 1])
        cp = np.ascontiguousarray(pos[a:b]); cl = np.ascontiguousarray(ln[a:b])
        act = lib.sp_step(st, mem.ctypes.data, mem.size, x.ctypes.data + 2 * t, int(runs[t]),
                          cp.ctypes.data if b > a else None, cl.ctypes.data if b > a else None, b - a, int(x[t + 1]),
                          row.ctypes.data)
        if act:
            r = row.copy()
            r["pos"] = t
            got.append(r)
    lib.sp_state_free(st)
    assert np.concatenate(got).tobytes() == rows.tobytes()


def test_causality(world):
    w = world
    rng = np.random.default_rng(7)
    rows = run_rows(w)
    n = w["x"].size - 1
    for t0 in (0, 5, 777, 1499, 1500, 1501, 3000, n - 2):
        # every token after t0 changes: the targets from t0 on, the inputs after t0, and so the candidates after t0
        x2 = w["x"].copy()
        x2[t0 + 1:] = rng.integers(0, 30, x2.size - t0 - 1)
        x2[t0 + 1 + np.flatnonzero(rng.random(x2.size - t0 - 1) < 0.01)] = BOS
        runs2 = runs_of(x2[:-1], w["chunk"])
        assert np.array_equal(runs2[:t0 + 1], w["runs"][:t0 + 1])
        c2 = walk(w["mem"], w["idx"], x2[:-1], runs2)
        rows2 = run_rows(w, x=x2, cands=c2, runs=runs2)
        a, b = rows[rows["pos"] <= t0], rows2[rows2["pos"] <= t0]
        assert a.tobytes() == b.tobytes(), t0
        if t0 < n - 200:
            assert rows[rows["pos"] > t0].tobytes() != rows2[rows2["pos"] > t0].tobytes()
    # only t0's own target changes (explicit y): rows <= t0 identical, row t0 + 1 may differ
    y = w["x"][1:].copy()
    for t0 in rng.choice(np.flatnonzero((rows["flags"] & SP.F_HP) != 0), 20, replace=False):
        t0 = int(rows["pos"][t0])
        y2 = y.copy()
        y2[t0] = (y2[t0] + 1) % 30
        rows2 = run_rows(w, x=w["x"][:-1], y=y2, runs=w["runs"])
        assert rows[rows["pos"] <= t0].tobytes() == rows2[rows2["pos"] <= t0].tobytes()


def test_normalization(world):
    rows = run_rows(world)
    hp = (rows["flags"] & SP.F_HP) != 0
    hs = (rows["flags"] & SP.F_HS) != 0
    k = np.arange(16)[None]
    vs = (rows["vw"] * (k < rows["ntok"][:, None])).sum(1)
    assert np.abs(vs[hp] - 1).max() < 1e-6
    assert (rows["ntok"][~hp] == 0).all() and (rows["nsd"][~hs] == 0).all()
    tr = (rows["flags"] & SP.F_SRC_TRUNC) != 0
    li = rows["src_li"].astype(np.int64)
    nli = np.take_along_axis(rows["src_n"].astype(np.int64), li[:, None], 1)[:, 0]
    cs = (rows["scnt"].astype(np.int64) * (k < rows["nsd"][:, None])).sum(1)
    assert (cs[hs & ~tr] == nli[hs & ~tr]).all() and (cs[hs & tr] < nli[hs & tr]).all()
    assert (rows["nsd"][hs & tr] == 16).all()
    # vote tokens distinct, source tokens distinct and sorted by count
    for r in rows[hp]:
        assert len(set(r["vtok"][:r["ntok"]].tolist())) == r["ntok"]
    for r in rows[hs]:
        c = r["scnt"][:r["nsd"]].astype(np.int64)
        assert (np.diff(c) <= 0).all() and len(set(r["stok"][:r["nsd"]].tolist())) == r["nsd"]
    # by enumerating the vocabulary: every component sums to 1 (the source: when not truncated)
    sel = rows[(hp | hs)][:400]
    tot = np.zeros((3, sel.size))
    for v in list(range(30)) + [BOS]:
        p = SP.component_probs(sel, np.full(sel.size, v))
        tot += np.stack(p)
    shp = (sel["flags"] & SP.F_HP) != 0
    shs = ((sel["flags"] & SP.F_HS) != 0) & ((sel["flags"] & SP.F_SRC_TRUNC) == 0)
    assert np.allclose(tot[0][shp], 1) and np.allclose(tot[1][shp], 1, atol=1e-6) and np.allclose(tot[2][shs], 1)
    assert (tot[0][~shp] == 0).all() and (tot[1][~shp] == 0).all()


def test_determinism_threads(world):
    ref = run_rows(world).tobytes()
    for th in (2, 3, 4, 8):
        assert run_rows(world, threads=th).tobytes() == ref, th


def test_segments_independent(world):
    """A segment's rows depend on that segment only; a document cut by a chunk start restarts there."""
    w = world
    rows = run_rows(w)
    x = w["x"]
    starts = np.flatnonzero(w["runs"] == 1)
    rng = np.random.default_rng(3)
    for s in rng.choice(starts[1:-1], 6, replace=False):
        e = starts[np.searchsorted(starts, s) + 1]  # the segment [s, e)
        sub_x = x[s:e + 1]
        sub_runs = w["runs"][s:e]
        o = w["off"]
        a, b = int(o[s]), int(o[e])
        sub = SP.run(w["mem"], 0, sub_x, (o[s:e + 1] - o[s]).astype(np.uint64), w["pos"][a:b], w["ln"][a:b],
                     runs=sub_runs, chunk=0)
        mine = rows[(rows["pos"] >= s) & (rows["pos"] < e)].copy()
        mine["pos"] -= np.uint32(s)
        assert mine.tobytes() == sub.tobytes(), s
    # chunk starts that fall inside a document (not at a BOS) restart the segment
    inside = [s for s in starts if s % w["chunk"] == 0 and s > 0 and x[s] != BOS]
    assert inside, "the data should cut a document at a chunk start"


def test_freeze_reads(world):
    """Rows read memory only inside the candidates' spans: with candidates from below a freeze (FIT), whatever lies
    past the freeze (here: copies of val text, which would match if anything crossed) changes no row."""
    w = world
    rng = np.random.default_rng(5)
    freeze = w["mem"].size
    x = w["x"]
    rows = None
    for trial in range(3):
        tail = []
        for _ in range(30):
            a = int(rng.integers(0, x.size - 300))
            tail += [x[a:a + int(rng.integers(20, 300))], np.array([SEP], np.uint16)]
        mem2 = np.concatenate([w["mem"]] + tail).astype(np.uint16)
        if trial == 2:  # and an unterminated tail right at the end of the buffer
            mem2 = np.concatenate([mem2, x[100:400]]).astype(np.uint16)
        r = run_rows(w, mem=mem2)
        if rows is None:
            rows = r
        assert r.tobytes() == rows.tobytes()
    assert rows.tobytes() == run_rows(w).tobytes()
    assert freeze == w["mem"].size


def test_fit_style_sequences(world):
    """FIT: per rank, K batches of inputs and targets concatenated; segments restart at each batch and at BOS, and the
    target at a batch end is the batch's own last target (not the next batch's first input)."""
    w = world
    rng = np.random.default_rng(11)
    x = w["x"]
    nb, B = 4, 1200
    xs, ys, starts = [], [], []
    for b in range(nb):
        a = int(rng.integers(0, x.size - B - 2))
        buf = x[a:a + B + 1].copy()
        xs.append(buf[:-1]); ys.append(buf[1:]); starts.append(b * B)
    X = np.concatenate(xs); Y = np.concatenate(ys)
    st = np.zeros(X.size, bool); st[starts] = True
    runs = SP.segment_runs(X, chunk=0, starts=st)
    off, pos, ln = walk(w["mem"], w["idx"], X, runs)
    rows = SP.run(w["mem"], 0, X, off, pos, ln, y=Y, runs=runs, threads=2)
    ref = reference(w["mem"], X, Y, runs, cands_fn(off, pos, ln))
    compare(rows, ref)
    # each batch alone gives its own rows
    for b in range(nb):
        s, e = b * B, (b + 1) * B
        sub = SP.run(w["mem"], 0, X[s:e], (off[s:e + 1] - off[s]).astype(np.uint64), pos[int(off[s]):int(off[e])],
                     ln[int(off[s]):int(off[e])], y=Y[s:e], runs=runs[s:e])
        mine = rows[(rows["pos"] >= s) & (rows["pos"] < e)].copy()
        mine["pos"] -= np.uint32(s)
        assert mine.tobytes() == sub.tobytes()


def test_mem_summary_matches_helper(world, tmp_path):
    """sp_mem_summary (the record fields P3 recomputes from the candidates) equals stream_memory.c's query_one on the
    same memory: the helper's walk, levels and full lengths."""
    sm = ARM / "stream_memory.c"
    if not sm.exists():
        pytest.skip("no stream_memory.c")
    w = world
    mem, x = w["mem"], w["x"]
    (tmp_path / "mem.u16").write_bytes(mem.tobytes())
    (tmp_path / "val.u16").write_bytes(x.tobytes())
    w["runs"].astype(np.uint32).tofile(tmp_path / "runs.u32")
    prog = tmp_path / "t.c"
    prog.write_text(f'''
#define main sm_main
#include "{sm}"
#undef main
#define SP_API static
#include "{ARM / 'stream_pointer.c'}"
static void *rd(const char *p, size_t *n) {{ FILE *f = fopen(p, "rb"); fseek(f, 0, SEEK_END); *n = ftell(f);
    fseek(f, 0, SEEK_SET); void *b = malloc(*n); if (fread(b, 1, *n, f) != *n) exit(1); fclose(f); return b; }}
int main(int argc, char **argv) {{
    size_t nm, nv, nr; (void)argc;
    uint16_t *m = rd(argv[1], &nm), *x = rd(argv[2], &nv); uint32_t *runs = rd(argv[3], &nr);
    nm /= 2; nr /= 4;
    hash_bits = 20; tok = m; prev = calloc(nm, 4); head = calloc(1u << hash_bits, 4);
    uint32_t r = 0;
    for (size_t j = 0; j + 1 < nm; j++) {{
        r = m[j] == SEP ? 0 : r + 1;
        if (r >= KEY && m[j + 1] != SEP) {{ uint32_t h = bucket_of(m + j - (KEY - 1)); prev[j] = head[h]; head[h] = j; }}
    }}
    long bad = 0, hits = 0;
    for (size_t t = 0; t < nr; t++) {{
        rec_t rec; int got = query_one(x + t, runs[t], UINT32_MAX, &rec);
        // the candidates' positions: walk again (the record keeps next tokens and lengths, not positions)
        uint32_t cp[32]; uint8_t cl[32]; int nc = 0;
        if (runs[t] >= KEY) {{
            uint32_t lim = runs[t] < MAXLEN ? runs[t] : MAXLEN, vis = 0;
            for (uint32_t p = head[bucket_of(x + t - (KEY - 1))]; p && nc < CAP && vis < MAXVISIT; p = prev[p]) {{
                vis++;
                if (memcmp(m + p - (KEY - 1), x + t - (KEY - 1), 2 * KEY)) continue;
                uint32_t l = KEY; while (l < lim && m[p - l] == x[t - l]) l++;
                cp[nc] = p; cl[nc] = (uint8_t)l; nc++;
            }}
        }}
        sp_mem_t s; sp_mem_summary(m, x + t, runs[t], cp, cl, nc, x[t + 1], &s);
        if (got != (nc > 0)) {{ bad++; continue; }}
        if (!got) continue;
        hits++;
        bad += s.pos_recent != rec.pos_recent || s.pos_longest != rec.pos_longest || s.len_recent != rec.len_recent ||
               s.len_longest != rec.len_longest || s.top != rec.top[rec.lstar] || s.n != rec.n[rec.lstar] ||
               s.m != rec.m[rec.lstar] || s.ncand != rec.ncand || s.lstar != rec.lstar;
        for (int i = 0; i < nc; i++) bad += m[cp[i] + 1] != rec.nx[i] || cl[i] != rec.len[i];
    }}
    printf("%ld %ld\\n", hits, bad);
    return 0;
}}
''')
    exe = tmp_path / "t"
    cc = subprocess.run(["cc", "-O1", "-std=c11", "-pthread", "-w", str(prog), "-o", str(exe), "-lm"],
                        capture_output=True, text=True)
    if cc.returncode:
        pytest.skip(f"stream_memory.c does not compile with this test's assumptions (query_one, rec_t): {cc.stderr[:300]}")
    out = subprocess.run([str(exe), str(tmp_path / "mem.u16"), str(tmp_path / "val.u16"), str(tmp_path / "runs.u32")],
                         capture_output=True, text=True, check=True).stdout.split()
    hits, bad = int(out[0]), int(out[1])
    assert hits > 500 and bad == 0, (hits, bad)


def test_features_numpy_torch(world):
    torch = pytest.importorskip("torch")
    rows = run_rows(world)
    tr = {n: torch.from_numpy(np.ascontiguousarray(rows[n])) for n in SP.ROW_DTYPE.names}
    # torch has no uint16 arithmetic everywhere: widen as the GPU side would
    for n, a in list(tr.items()):
        if a.dtype in (torch.uint16, torch.uint32):
            tr[n] = a.to(torch.int64)
    fn = SP.features(rows)
    ft = SP.features(tr, xp=torch)
    assert fn.keys() == ft.keys()
    for k in fn:
        assert np.allclose(fn[k], ft[k].double().numpy(), rtol=1e-5, atol=1e-5), k
        assert np.isfinite(fn[k]).all(), k
    y = world["x"][1:][rows["pos"]]
    pn = SP.component_probs(rows, y)
    pt = SP.component_probs(tr, torch.from_numpy(y.astype(np.int64)), xp=torch)
    for a, b in zip(pn, pt):
        assert np.allclose(a, b.double().numpy(), atol=1e-6)
    groups = Counter(k.split("_")[0] for k in fn)
    # ptr 13 + 6 how + agree; ret 3; ind 2; hist 15 + hist2 4; mem 30 (pipe.HELPER); src 8 + 7; both 3
    assert groups == {"ptr": 20, "ret": 3, "ind": 2, "hist16": 5, "hist64": 5, "histseg": 5, "hist2": 4, "mem": 30,
                      "src": 15, "both": 3}, groups


def test_compiles_clean_and_scalar_path(world, tmp_path):
    src = ARM / "stream_pointer.c"
    for flags in ([], ["-DSP_NO_SIMD"]):
        r = subprocess.run(["cc", "-O2", "-std=c11", "-pthread", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                            *flags, str(src), "-o", str(tmp_path / "x.so"), "-lm"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    # one translation unit with the helper (and the low-order tables): no clashes, no warnings from this file
    parts = [f for f in ("stream_memory.c", "stream_lowtables.c") if (ARM / f).exists()]
    if parts:
        tu = tmp_path / "tu.c"
        inc = "".join(f'#define main sm_main\n#include "{ARM / f}"\n#undef main\n' if f == "stream_memory.c" else
                      f'#define LT_API static __attribute__((unused))\n#include "{ARM / f}"\n' for f in parts)
        tu.write_text(inc + f'#define SP_API static __attribute__((unused))\n#include "{src}"\nint main(void) {{ return 0; }}\n')
        r = subprocess.run(["cc", "-O2", "-std=c11", "-pthread", "-Wall", "-Wextra", "-c", str(tu), "-o",
                            str(tmp_path / "tu.o")], capture_output=True, text=True)
        assert r.returncode == 0 and "stream_pointer.c" not in r.stderr, r.stderr[-3000:]
    lib2 = SP.build_lib(extra_cflags=("-DSP_NO_SIMD",))
    w = world
    a = run_rows(w)
    b = SP.run(w["mem"], 0, w["x"], w["off"], w["pos"], w["ln"], chunk=w["chunk"], lib=lib2)
    assert a.tobytes() == b.tobytes()


def test_config_validation(lib):
    assert lib.sp_state_new(ctypes.byref(SP.default_config(beam_k=17))) is None
    assert lib.sp_state_new(ctypes.byref(SP.default_config(src_maxsrc=9))) is None
    st = lib.sp_state_new(ctypes.byref(SP.default_config()))
    assert st
    lib.sp_state_free(st)


# ------------------------------------------------------------------------------------------------ real data (optional)

def _ncand(out, qn):
    off = np.fromfile(out + ".cand.off", dtype=np.uint64)
    return np.diff(off.astype(np.int64)).astype(np.float64)


def research_parity(OUT, S, QN=1048576, CH=262144):
    """Parity of the harness rows OUT.rows with the research tools (align4 c0.feat.f32, libsrccopy6, libdocstate,
    feats.build, pipe.Set, stages.py formulas) on the same memory summary and segments. Prints a table; returns ok."""
    P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    sp = SP
    val = np.fromfile(S + "/sr/data/fineweb_val_000000.bin", dtype=np.uint16, offset=1024)[:QN + 1]
    x, y = val[:QN], val[1:QN + 1].astype(np.int64)
    rows = np.fromfile(OUT + ".rows", dtype=sp.ROW_DTYPE)
    ms = np.fromfile(OUT + ".mem", dtype=np.uint32).reshape(QN, 10)
    has = ms[:, 0] > 0
    top, Nl, Ml, Cl = ms[:, 1], ms[:, 2], ms[:, 3], ms[:, 4]
    lenl, posl, lenr, posr = ms[:, 6], ms[:, 7], ms[:, 8], ms[:, 9]
    t = np.arange(QN)
    start = (x == sp.BOS) | (t % CH == 0)
    seg = np.cumsum(start) - 1
    pos = rows["pos"].astype(np.int64)
    flags = rows["flags"]
    hp_d = np.zeros(QN, bool); hp_d[pos[(flags & sp.F_HP) != 0]] = True
    hs_d = np.zeros(QN, bool); hs_d[pos[(flags & sp.F_HS) != 0]] = True
    ok = True

    def check(name, a, b, tol=0.0):
        nonlocal ok
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        bad = np.abs(a - b) > tol * np.maximum(1, np.abs(b))
        print(f"  {name:28s} {'OK' if not bad.any() else 'MISMATCH'} ({bad.sum()} of {bad.size})", flush=True)
        if bad.any():
            i = np.flatnonzero(bad)[:5]
            print("     first:", i, a[i], b[i])
            ok = False

    # ---- 1. pointer vs align4
    F = np.fromfile(S + "/rg/align/c0.feat.f32", dtype=np.float32).reshape(-1, 32)[:QN].astype(np.float64)
    opos = np.fromfile(S + "/rg/align/c0.pos.u32", dtype=np.uint32)[:QN]
    print("pointer beam vs align4 (c0.feat.f32):")
    check("rows (hp)", hp_d, F[:, 0] > 0)
    R = rows[(flags & sp.F_HP) != 0]
    Fp = F[R["pos"].astype(np.int64)]
    cols = {"pred": 1, "run": 2, "hits": 3, "miss": 4, "c16": 5, "n16": 6, "c64": 7, "nhist": 8, "age": 9, "since": 10,
            "how": 11, "nvh": 12, "share": 13, "tcnt": 14, "ntok": 15, "share2": 17, "score": 18, "seedlen": 19,
            "nrec": 20, "vtop": 21, "vshare": 22, "score2": 25}
    for k, c in cols.items():
        check(k, R[k], Fp[:, c], 0.0)
    check("tok2", np.where(R["tok2"] == 0xFFFF, -1, R["tok2"].astype(np.int64)), Fp[:, 16])
    check("ptr_j (opos)", R["ptr_j"], opos[R["pos"]])
    yy = y[R["pos"]]
    pv = (R["vw"] * ((R["vtok"] == yy[:, None]) & (np.arange(16)[None] < R["ntok"][:, None]))).sum(1)
    check("vote p(y) [26]", pv, Fp[:, 26], 1e-6)
    print(f"  vote sums: max |sum - 1| = {np.abs((R['vw'] * (np.arange(16)[None] < R['ntok'][:, None])).sum(1) - 1).max():.2e}")

    # ---- 2. source copy vs srccopy6 (same memory summary and segments)
    lib6 = ctypes.CDLL(S + "/rg/docgate/libsrccopy6.so")
    st = np.fromfile(S + "/sr/mem/stream978.u16", dtype=np.uint16)
    NO = 4 + 3 * 8
    out = np.zeros((QN, NO), np.int32); outf = np.zeros((QN, 2), np.float32)
    doc = seg.astype(np.int64)
    posl_r = np.where(has, posl, 0xFFFFFFFF).astype(np.uint32); posr_r = np.where(has, posr, 0xFFFFFFFF).astype(np.uint32)
    args = [np.ascontiguousarray(val, np.uint16), doc, has.astype(np.uint8), lenl.astype(np.uint32), posl_r,
            lenr.astype(np.uint32), posr_r]
    lib6.srccopy6(ctypes.c_long(QN), *[P(a) for a in args], ctypes.c_int(1), P(st), ctypes.c_long(st.size),
                  ctypes.c_int(10), ctypes.c_int(8), ctypes.c_int(4096), P(out), P(outf), ctypes.c_int(4))
    print("source copy vs srccopy6 (votes 4, th 10, maxsrc 8, half 4096, recent 1; segments cut at chunks):")
    check("rows (hs = lbest > 0)", hs_d, out[:, 0] > 0)
    A = rows  # every active row has the src header (nsrc / since / newlen) even without SP_HS
    ap = A["pos"].astype(np.int64)
    check("nsrc (active rows)", A["src_nsrc"], out[ap, 1])
    check("since (active rows)", A["src_since"], out[ap, 2])
    check("newlen (active rows)", A["src_newlen"], out[ap, 3])
    Rs = rows[(flags & sp.F_HS) != 0]
    sp_ = Rs["pos"].astype(np.int64)
    o = out[sp_]
    check("lbest", Rs["src_lbest"], o[:, 0])
    N = o[:, 4::3]; C = o[:, 5::3]; M = o[:, 6::3]
    check("N per level", Rs["src_n"].astype(np.int64).ravel(), N.ravel())
    li = np.zeros(len(Rs), int)
    for v in range(8):
        li = np.where(N[:, v] > 0, v, li)
    check("li", Rs["src_li"], li)
    ar = np.arange(len(Rs))
    check("M at li", Rs["src_m"], M[ar, li])
    cnt = (Rs["scnt"] * ((Rs["stok"] == y[sp_][:, None]) & (np.arange(16)[None] < Rs["nsd"][:, None]))).sum(1)
    tr = (Rs["flags"] & sp.F_SRC_TRUNC) != 0
    check("C(y) at li (untruncated)", cnt[~tr], C[ar, li][~tr])
    sums = (Rs["scnt"] * (np.arange(16)[None] < Rs["nsd"][:, None])).sum(1)
    check("counts sum to N (untrunc)", sums[~tr], N[ar, li][~tr])
    r_exact = C[ar, li] / np.maximum(N[ar, li], 1)
    r_mine = cnt / np.maximum(N[ar, li], 1)
    print(f"  truncated rows {tr.sum()} of {len(Rs)}; target mass left out there: sum r_exact - r_kept = "
          f"{(r_exact - r_mine)[tr].sum():.2f} (vs total sum r over source rows {r_exact.sum():.0f}); "
          f"mean r_exact at truncated rows {r_exact[tr].mean():.4f}")

    # ---- 3. doc-state vs docstate.c
    libd = ctypes.CDLL(S + "/rg/docgate/libdocstate.so")
    FN = ["npos", "nhit", "ncorr", "w8p_hit", "w8p_corr", "w32p_hit", "w32p_corr", "w128p_hit", "w128p_corr",
          "w8h_n", "w8h_corr", "w32h_n", "w32h_corr", "w128h_n", "w128h_corr", "w8h_r", "w32h_r", "w128h_r",
          "streak", "since_wrong", "since_hit", "prev_corr", "prev_hit", "ema90", "ema98", "nlong32", "nlong16",
          "cont", "cont_run", "cont_corr_run", "sum_gain", "w32h_gain", "sum_llr"]

    def docstate(hit, corr, r, lenl_, posr_):
        o = np.zeros((QN, len(FN)))
        z = np.zeros(QN)
        a = [doc, np.ascontiguousarray(hit, np.uint8), np.ascontiguousarray(corr, np.uint8), np.ascontiguousarray(r, np.float64),
             np.ascontiguousarray(lenl_, np.uint32), np.ascontiguousarray(posr_, np.uint32), z, z]
        libd.docstate(ctypes.c_long(QN), *[P(v) for v in a], P(o))
        return {n: o[:, i] for i, n in enumerate(FN)}

    corr = has & (top == y)
    r = np.where(has, Cl / np.maximum(Nl, 1), 0.0)
    D = docstate(has, corr, r, lenl, posr_r)
    print("memory doc-state vs docstate.c (active rows):")
    for mine, ref in (("d_npos", "npos"), ("d_nhit", "nhit"), ("d_ncorr", "ncorr"), ("d_streak", "streak"),
                      ("d_since_wrong", "since_wrong"), ("d_since_hit", "since_hit"), ("d_nlong32", "nlong32"),
                      ("d_nlong16", "nlong16"), ("d_cont_run", "cont_run"), ("d_cont_corr_run", "cont_corr_run")):
        check(mine, A[mine], D[ref][ap])
    for i, K in enumerate((8, 32, 128)):
        check(f"d_p_hit[{K}]", A["d_p_hit"][:, i], D[f"w{K}p_hit"][ap])
        check(f"d_p_corr[{K}]", A["d_p_corr"][:, i], D[f"w{K}p_corr"][ap])
        check(f"d_h_corr[{K}]", A["d_h_corr"][:, i], D[f"w{K}h_corr"][ap])
    check("d_ema90", A["d_ema90"], D["ema90"][ap], 1e-6)
    check("d_ema98", A["d_ema98"], D["ema98"][ap], 1e-6)
    check("d_h32_r", A["d_h32_r"], D["w32h_r"][ap], 1e-6)
    check("cont", (A["flags"] & sp.F_CONT) != 0, D["cont"][ap])
    check("prev_corr", (A["flags"] & sp.F_PREV_CORR) != 0, D["prev_corr"][ap])
    check("prev_hit", (A["flags"] & sp.F_PREV_HIT) != 0, D["prev_hit"][ap])
    # the source track (research: stages.build hist=True on srccopy's r_src / corr_s)
    li_all = np.zeros(QN, int)
    for v in range(8):
        li_all = np.where(out[:, 4 + 3 * v] > 0, v, li_all)
    Nd = out[t, 4 + 3 * li_all]; Cd = out[t, 5 + 3 * li_all]; Md = out[t, 6 + 3 * li_all]
    hs_r = out[:, 0] > 0
    corr_s = hs_r & (Cd == Md) & (Cd > 0)
    r_src = np.where(hs_r, Cd / np.maximum(Nd, 1), 0.0)
    D2 = docstate(hs_r, corr_s, r_src, np.zeros(QN), np.zeros(QN))
    print("source doc-state vs docstate.c (active rows):")
    check("s_nhit", A["s_nhit"], D2["nhit"][ap])
    check("s_ncorr", A["s_ncorr"], D2["ncorr"][ap])
    check("s_streak", A["s_streak"], D2["streak"][ap])
    check("s_h_corr[8]", A["s_h_corr"][:, 0], D2["w8h_corr"][ap])
    check("s_h_corr[32]", A["s_h_corr"][:, 1], D2["w32h_corr"][ap])
    check("s_h32_r", A["s_h32_r"], D2["w32h_r"][ap], 1e-6)
    check("src prev_corr", (A["flags"] & sp.F_SPREV_CORR) != 0, D2["prev_corr"][ap])

    # ---- 4. windowed histories vs feats.py past_count
    first = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    segstart = np.zeros(QN, np.int64); segstart[first] = first; segstart = np.maximum.accumulate(segstart)

    def past(v, W=None):
        c = np.concatenate([[0.0], np.cumsum(np.asarray(v, np.float64))])
        lo = segstart if W is None else np.maximum(segstart, t - W)
        return c[t] - c[lo]
    bptr = np.zeros(QN, bool); bptr[R["pos"]] = R["pred"] == y[R["pos"]]
    either = corr | bptr
    print("windowed histories vs feats.py past_count (active rows):")
    for w, W in ((0, 16), (1, 64)):
        check(f"h_r_n[{W}]", A["h_r_n"][:, w], past(has, W)[ap])
        check(f"h_r_c[{W}]", A["h_r_c"][:, w], past(corr, W)[ap])
        check(f"h_p_n[{W}]", A["h_p_n"][:, w], past(hp_d, W)[ap])
        check(f"h_p_c[{W}]", A["h_p_c"][:, w], past(bptr, W)[ap])
        check(f"h_e[{W}]", A["h_e"][:, w], past(either, W)[ap])
    check("h32_r_n", A["h32_r_n"], past(has, 32)[ap])
    check("h32_p_n", A["h32_p_n"], past(hp_d, 32)[ap])
    check("h32_p_c", A["h32_p_c"], past(bptr, 32)[ap])
    check("h32_r_sum", A["h32_r_sum"], past(r, 32)[ap], 1e-5)
    check("h_r_sum_seg", A["h_r_sum_seg"], past(r)[ap], 1e-5)
    check("h_p_n_seg", A["h_p_n_seg"], past(hp_d)[ap])
    check("h_p_c_seg", A["h_p_c_seg"], past(bptr)[ap])
    check("h_e_seg", A["h_e_seg"], past(either)[ap])
    # ---- 5. features() vs the research feature code on the same inputs
    sys.path.insert(0, S + "/rg/base"); sys.path.insert(0, S + "/rg/align"); sys.path.insert(0, S + "/rg/docgate")
    from types import SimpleNamespace
    import feats as RF
    import pipe as RP
    feat = sp.features(A)
    Lv = np.where(has, ms[:, 5], 0)
    g = SimpleNamespace(has=has, N=Nl.astype(np.float64), C=Cl.astype(np.float64), M=Ml.astype(np.float64),
                        L=np.maximum(Lv, 1).astype(np.float64), r=r)
    kt = SimpleNamespace(Q=QN, x=x, y=y, lq_base=np.zeros(QN), mem={"top": top.astype(np.int64)})
    B = RF.build(kt, S + "/rg/align/c0", g=g, hist=True, hist2=True, lqb=np.zeros(QN), llr=False, use=("ptr",))
    print("features() vs feats.build (ptr_cols, ret_cols, ind, hist_cols; active rows):")
    names = ["ptr_run", "ptr_hits", "ptr_miss16", "ptr_acc64", "ptr_since", "ptr_age", "ptr_share", "ptr_tcnt",
             "ptr_ntok", "ptr_seedlen", "ptr_nrec", "ptr_margin", "ptr_score"] + [f"ptr_how{h}" for h in range(6)] + ["ptr_agree"]
    for nm, col in zip(names, B.ptr_cols):
        check(nm, feat[nm], col[ap], 1e-6)
    for nm, col in zip(["ret_logn", "ret_purity", "ret_logl"], B.ret_cols):
        check(nm, feat[nm], col[ap], 1e-6)
    check("ind_has", feat["ind_has"], B.ind[0][ap]); check("ind_hp", feat["ind_hp"], B.ind[1][ap])
    hn = []
    for W in (16, 64, "seg"):
        hn += [f"hist{W}_ret_c", f"hist{W}_ret_acc", f"hist{W}_ptr_c", f"hist{W}_ptr_acc", f"hist{W}_either"]
    hn += ["hist2_ret_r32", "hist2_ret_rseg", "hist2_ptr_r32", "hist2_ptr_rseg"]
    assert len(hn) == len(B.hist_cols), (len(hn), len(B.hist_cols))
    for nm, col in zip(hn, B.hist_cols):
        check(nm, feat[nm], col[ap], 1e-5)
    print("features() vs pipe.Set.helper_feats (mem_*):")
    Sset = RP.Set(doc, np.zeros(QN), Nl.astype(np.float64), Cl.astype(np.float64), Ml.astype(np.float64), Lv, top, y,
                  lenl.astype(np.float64), lenr.astype(np.float64), posr_r, ms[:, 0] * 0 + np.where(has, ms[:, 2] * 0, 0) + _ncand(OUT, QN))
    H = Sset.helper_feats()
    for c in RP.HELPER:
        check("mem_" + c, feat["mem_" + c], H[c][ap], 1e-5)
    print("features() vs stages.py src_cols + src_extra (from srccopy6's output):")
    L2b = lambda v: np.log2(np.maximum(v, 1))
    ref_src = {"src_n": L2b(Nd), "src_purity": Md / np.maximum(Nd, 1), "src_lbest": out[:, 0], "src_lbest_l2": L2b(out[:, 0]),
               "src_nsrc": out[:, 1], "src_since": L2b(out[:, 2] + 1), "src_newlen": L2b(out[:, 3]), "src_n2": L2b(out[:, 7])}
    smh = lambda c, n, a=0.3: (c + a) / (n + 1)
    ref_src.update({"src_acc_h8": smh(D2["w8h_corr"], D2["w8h_n"]), "src_acc_h32": smh(D2["w32h_corr"], D2["w32h_n"]),
                    "src_acc_doc": smh(D2["ncorr"], D2["nhit"]), "src_nhit": np.log1p(D2["nhit"]),
                    "src_streak": np.log1p(D2["streak"]), "src_prev_corr": D2["prev_corr"],
                    "src_r_h32": smh(D2["w32h_r"], D2["w32h_n"])})
    for k, v in ref_src.items():
        check(k, feat[k], v[ap], 1e-5)
    check("both_lstar", feat["both_lstar"], (np.log2(np.maximum(Lv, 1)) * has)[ap], 1e-6)
    print("PARITY", "PASS" if ok else "FAIL")
    return ok


SCRATCH = Path(os.environ.get("STREAM_POINTER_SCRATCH",
                              "/tmp/claude-0/-home-user/f1cd7961-00e0-5a67-b21c-7dd268b9bcc8/scratchpad"))


@pytest.mark.skipif(os.environ.get("STREAM_POINTER_REAL") != "1", reason="set STREAM_POINTER_REAL=1 (4 GB RAM, ~1 min)")
def test_bench_real(tmp_path):
    """The research parity and the costs on the real 1050-step stream and the first 1,048,576 val positions."""
    stream = SCRATCH / "sr/mem/stream978.u16"
    val = SCRATCH / "sr/data/fineweb_val_000000.bin"
    if not stream.exists() or not val.exists():
        pytest.skip("no real stream / val")
    exe = tmp_path / "spb"
    subprocess.run(["cc", "-O2", "-std=c11", "-pthread", f"-I{HERE}", str(HERE / "stream_pointer_bench.c"), "-o",
                    str(exe), "-lm"], check=True)
    out = subprocess.run([str(exe), str(stream), str(val), "1048576", str(tmp_path / "real"), "4"],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)
    print(json.dumps(st, indent=1))
    assert st["identical_across_threads"] == 1
    assert st["hp"] == 232122 and abs(st["ptr_top_correct"] - 0.1749) < 1e-4  # align4's pointer rows on this data
    need = ["rg/align/c0.feat.f32", "rg/align/c0.pos.u32", "rg/docgate/libsrccopy6.so", "rg/docgate/libdocstate.so",
            "rg/align/feats.py", "rg/docgate/pipe.py"]
    if all((SCRATCH / f).exists() for f in need):
        assert research_parity(str(tmp_path / "real"), str(SCRATCH))
    else:
        print("research outputs not found: parity skipped")
