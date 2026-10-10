"""Python side of stream_pointer.c (part P3 of stream retrieval v2): the per-segment pointer beam, vote, source copy and
doc-state counters. The C file's header has the design; this module builds it as a shared library, mirrors its row
layout (ROW_DTYPE), runs it over a whole sequence (`run`), and turns rows into what the eval's top-level mixture reads:
the components' probabilities of the realised token (`component_probs`, the GPU gathers) and the gate's feature
columns under the research names (`features`, rg/align/feats.py ptr_cols / hist_cols, rg/docgate pipe.HELPER /
stages src_cols). Both work on numpy arrays and on torch tensors (pass xp=torch).

Credits: as stream_memory.py's docstring (PR #367's StreamIndex ideas, PR #380's output mixture gated on training
positions); the beam, the source copy and the doc-state counters are this branch's research (rg/align, rg/docgate).
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().with_name("stream_pointer.c")
CFLAGS = ("-O2", "-std=c11", "-pthread", "-shared", "-fPIC")
SEP, BOS = 0xFFFF, 50256
CAP, KEY, VOTE, SRCD = 32, 6, 16, 16
LV = (1, 2, 3, 4, 6, 8, 12, 16)                 # source copy levels
LEVELS = (6, 7, 8, 10, 12, 16, 24, 32)          # stream_memory.c LEVELS (L*)
F_HP, F_HS, F_HAS, F_SRC_TRUNC, F_CONT, F_PREV_CORR, F_PREV_HIT, F_SPREV_CORR = 1, 2, 4, 8, 16, 32, 64, 128

# sp_row_t, field by field (C alignment: np.dtype(align=True) lays it out as the compiler does; tests compare it with
# the library's sp_layout()).
_FIELDS = [
    ("pos", "<u4"), ("flags", "<u2"), ("ntok", "u1"), ("nsd", "u1"),
    ("ptr_j", "<u4"), ("score", "<f4"), ("score2", "<f4"), ("share", "<f4"), ("share2", "<f4"), ("vshare", "<f4"),
    ("pred", "<u2"), ("vtop", "<u2"), ("tok2", "<u2"), ("run", "<u2"), ("hits", "<u2"), ("miss", "<u2"),
    ("age", "<u2"), ("since", "<u2"), ("seedlen", "<u2"), ("nrec", "<u2"),
    ("c16", "u1"), ("n16", "u1"), ("c64", "u1"), ("nhist", "u1"), ("how", "u1"), ("nvh", "u1"), ("tcnt", "u1"),
    ("pad0", "u1"),
    ("vtok", "<u2", (VOTE,)), ("vw", "<f4", (VOTE,)),
    ("src_n", "<u2", (8,)), ("src_m", "<u2"), ("src_newlen", "<u2"), ("src_since", "<i4"),
    ("src_lbest", "u1"), ("src_nsrc", "u1"), ("src_li", "u1"), ("pad1", "u1"),
    ("stok", "<u2", (SRCD,)), ("scnt", "<u2", (SRCD,)),
    ("mem_lenl", "<u2"), ("mem_lenr", "<u2"), ("mem_top", "<u2"), ("mem_ncand", "u1"), ("mem_lstar", "u1"),
    ("mem_n", "u1"), ("mem_m", "u1"), ("pad2", "<u2"),
    ("d_npos", "<u4"), ("d_nhit", "<u4"), ("d_ncorr", "<u4"), ("d_streak", "<u4"), ("d_since_wrong", "<u4"),
    ("d_since_hit", "<u4"), ("d_nlong32", "<u4"), ("d_nlong16", "<u4"), ("d_cont_run", "<u4"),
    ("d_cont_corr_run", "<u4"), ("d_ema90", "<f4"), ("d_ema98", "<f4"), ("d_h32_r", "<f4"),
    ("d_p_hit", "u1", (3,)), ("d_p_corr", "u1", (3,)), ("d_h_corr", "u1", (3,)), ("pad3", "u1", (3,)),
    ("s_nhit", "<u4"), ("s_ncorr", "<u4"), ("s_streak", "<u4"), ("s_h32_r", "<f4"), ("s_h_corr", "u1", (2,)),
    ("pad4", "u1", (2,)),
    ("h_r_n", "u1", (2,)), ("h_r_c", "u1", (2,)), ("h_p_n", "u1", (2,)), ("h_p_c", "u1", (2,)), ("h_e", "u1", (2,)),
    ("h32_r_n", "u1"), ("h32_p_n", "u1"), ("h32_p_c", "u1"), ("pad5", "u1"),
    ("h32_r_sum", "<f4"), ("h_r_sum_seg", "<f4"), ("h_p_n_seg", "<u4"), ("h_p_c_seg", "<u4"), ("h_e_seg", "<u4"),
]
ROW_DTYPE = np.dtype(_FIELDS, align=True)
ROW_BYTES = ROW_DTYPE.itemsize
assert ROW_BYTES == 380


class Config(ctypes.Structure):
    """sp_config_t (defaults: the research configuration, see default_config)."""
    _fields_ = [("beam_k", ctypes.c_int32), ("seed_min", ctypes.c_int32), ("dmax", ctypes.c_int32),
                ("fwd", ctypes.c_int32), ("back", ctypes.c_int32), ("gram", ctypes.c_int32),
                ("src_th", ctypes.c_int32), ("src_maxsrc", ctypes.c_int32), ("src_half", ctypes.c_int32),
                ("src_votes", ctypes.c_int32), ("src_use_recent", ctypes.c_int32), ("reserved", ctypes.c_int32 * 5)]


class Csr(ctypes.Structure):
    _fields_ = [("off", ctypes.c_void_p), ("pos", ctypes.c_void_p), ("len", ctypes.c_void_p)]


class MemSummary(ctypes.Structure):
    """sp_mem_t: stream_memory.c's record fields of one position, from its candidates."""
    _fields_ = [("pos_recent", ctypes.c_uint32), ("pos_longest", ctypes.c_uint32), ("len_recent", ctypes.c_uint32),
                ("len_longest", ctypes.c_uint32), ("n", ctypes.c_uint32), ("m", ctypes.c_uint32), ("c", ctypes.c_uint32),
                ("top", ctypes.c_uint16), ("lstar", ctypes.c_uint8), ("ncand", ctypes.c_uint8)]


_LIB = None


def build_lib(extra_cflags=()) -> ctypes.CDLL:
    """Compile stream_pointer.c into a shared library (cached by the source's hash) and load it."""
    global _LIB
    if _LIB is not None and not extra_cflags:
        return _LIB
    source = SOURCE.read_bytes()
    tag = hashlib.sha256(source + " ".join(extra_cflags).encode()).hexdigest()[:16]
    out = Path(tempfile.gettempdir()) / "stream_pointer_build" / f"libstream_pointer_{tag}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or "cc"
        cc = subprocess.run([compiler, *CFLAGS, *extra_cflags, str(SOURCE), "-o", str(tmp), "-lm"],
                            capture_output=True, text=True)
        if cc.returncode:
            raise RuntimeError(f"cannot compile {SOURCE}:\n{cc.stderr}")
        os.replace(tmp, out)
    lib = ctypes.CDLL(str(out))
    lib.sp_default_config.argtypes = [ctypes.POINTER(Config)]
    lib.sp_state_new.restype = ctypes.c_void_p
    lib.sp_state_new.argtypes = [ctypes.POINTER(Config)]
    lib.sp_state_free.argtypes = [ctypes.c_void_p]
    lib.sp_step.restype = ctypes.c_int
    lib.sp_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32,
                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint16, ctypes.c_void_p]
    lib.sp_run.restype = ctypes.c_int64
    lib.sp_run.argtypes = [ctypes.POINTER(Config), ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
    lib.sp_mem_summary.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_uint16, ctypes.POINTER(MemSummary)]
    lib.sp_free.argtypes = [ctypes.c_void_p]
    lib.sp_layout.restype = ctypes.c_char_p
    lib.sp_row_bytes.restype = ctypes.c_uint64
    if not extra_cflags:
        _LIB = lib
    return lib


def default_config(**over) -> Config:
    """The research configuration (align4: K 16, seed 6, DMAX 8, FWD/BACK 64, GRAM 2; srccopy6: th 10, maxsrc 8, half
    4096, votes 4, use_recent 1), with keyword overrides."""
    c = Config()
    build_lib().sp_default_config(ctypes.byref(c))
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def with_sentinel(stream: np.ndarray):
    """(buffer, offset) such that tok = buffer[offset:] is the stream and tok[-1] is a separator, as tok[0] in the
    helper: backward extensions from the first span stop there."""
    buf = np.empty(stream.size + 1, np.uint16)
    buf[0] = SEP
    buf[1:] = stream
    return buf, 1


def run(tok_buf: np.ndarray, tok_off: int, x: np.ndarray, cand_off: np.ndarray, cand_pos: np.ndarray,
        cand_len: np.ndarray, y=None, runs=None, chunk=262144, threads=1, config=None, lib=None) -> np.ndarray:
    """Rows (ROW_DTYPE, position order) of the active positions of the sequence x[0..n-1].
    tok_buf[tok_off:] is the memory (positions in cand_pos index it; it starts with a separator, as the helper's tok[0],
    or one precedes it: tok_buf[tok_off - 1]);
    position t's candidates are cand_pos/cand_len[cand_off[t]:cand_off[t + 1]] (walk order, most recent first; lengths
    capped at min(32, run)); y: targets (default x[t + 1], then x needs n + 1 tokens); runs: per position segment length
    (run == 1 starts a segment) or None (segments start at BOS and at every multiple of chunk)."""
    lib = lib or build_lib()
    x = np.ascontiguousarray(x, np.uint16)
    n = x.size - (1 if y is None else 0)
    if y is not None:
        y = np.ascontiguousarray(y, np.uint16)
        assert y.size == n
    off = np.ascontiguousarray(cand_off, np.uint64)
    cpos = np.ascontiguousarray(cand_pos, np.uint32)
    clen = np.ascontiguousarray(cand_len, np.uint8)
    assert off.size == n + 1 and cpos.size >= int(off[-1]) and clen.size >= int(off[-1])
    if runs is not None:
        runs = np.ascontiguousarray(runs, np.uint32)
        assert runs.size == n
    # no read goes below the memory's first entry: a separator must precede it (tok[-1]) or be it (tok[0], the helper's)
    assert tok_buf[tok_off - 1] == SEP if tok_off >= 1 else tok_buf[0] == SEP
    csr = Csr(_ptr(off), _ptr(cpos) if cpos.size else None, _ptr(clen) if clen.size else None)
    out = ctypes.c_void_p()
    err = ctypes.create_string_buffer(256)
    cfg = config if config is not None else default_config()
    tok_ptr = ctypes.c_void_p(tok_buf.ctypes.data + 2 * tok_off)
    m = lib.sp_run(ctypes.byref(cfg), tok_ptr, tok_buf.size - tok_off, _ptr(x), _ptr(y) if y is not None else None, n,
                   _ptr(runs) if runs is not None else None, chunk, ctypes.cast(lib.sp_cands_csr, ctypes.c_void_p),
                   ctypes.addressof(csr), threads, ctypes.byref(out), err, 256)
    if m < 0:
        raise RuntimeError(f"sp_run: {err.value.decode()}")
    rows = np.empty(m, ROW_DTYPE)
    if m:
        ctypes.memmove(rows.ctypes.data, out.value, m * ROW_BYTES)
    lib.sp_free(out)
    return rows


def segment_runs(x: np.ndarray, chunk=262144, starts=None) -> np.ndarray:
    """run[t] = t - sigma_t + 1 saturated at 4096 (sigma_t: the last BOS / chunk start / extra start at or before t)."""
    t = np.arange(x.size)
    st = (x == BOS) | (t % chunk == 0 if chunk else t == 0)
    st[0] = True
    if starts is not None:
        st |= starts
    sig = np.maximum.accumulate(np.where(st, t, 0))
    return np.minimum(t - sig + 1, 4096).astype(np.uint32)


# ------------------------------------------------------------------------------------------------ eval side

def _f(xp, a, dtype=None):
    if xp is np:
        return np.asarray(a, dtype=dtype or np.float64)
    return a.to(dtype or xp.float32)


def _clip_lo(xp, a, lo):
    return np.maximum(a, lo) if xp is np else xp.clamp(a, min=lo)


def _min(xp, a, hi):
    return np.minimum(a, hi) if xp is np else xp.clamp(a, max=hi)


def _at_li(xp, rows):
    """N at the source's deepest level li (src_n[li])."""
    li = rows["src_li"]
    if xp is np:
        return np.take_along_axis(rows["src_n"].astype(np.int64), li.astype(np.int64)[:, None], 1)[:, 0]
    return rows["src_n"].long().gather(1, li.long()[:, None])[:, 0]


def component_probs(rows, y, xp=np):
    """The three components' probabilities of the realised tokens y (one per row): pointer (pred == y), vote (the
    share of y), source copy (count of y at level li / N at li). Where a component is off (no SP_HP / SP_HS) it is 0.
    The GPU version is the same gathers on the row arrays."""
    flags = rows["flags"]
    hp = (flags & F_HP) != 0
    hs = (flags & F_HS) != 0
    yy = y.astype(np.int64) if xp is np else y.long()
    pred = rows["pred"].astype(np.int64) if xp is np else rows["pred"].long()
    p_ptr = _f(xp, hp & (pred == yy))
    vt = rows["vtok"].astype(np.int64) if xp is np else rows["vtok"].long()
    ntok = rows["ntok"]
    k = np.arange(VOTE) if xp is np else xp.arange(VOTE, device=vt.device)
    vmask = (vt == yy[:, None]) & (k[None, :] < ntok[:, None])
    p_vote = (_f(xp, rows["vw"]) * _f(xp, vmask)).sum(1) * _f(xp, hp)
    st = rows["stok"].astype(np.int64) if xp is np else rows["stok"].long()
    nsd = rows["nsd"]
    smask = (st == yy[:, None]) & (k[None, :] < nsd[:, None])
    cnt = (_f(xp, rows["scnt"]) * _f(xp, smask)).sum(1)
    p_src = cnt / _clip_lo(xp, _f(xp, _at_li(xp, rows)), 1.0) * _f(xp, hs)
    return p_ptr, p_vote, p_src


def features(rows, xp=np):
    """The gate's feature columns of the rows, named and computed as the research did (dict name -> column):
    ptr_* (rg/align/feats.py ptr_cols), ret_* / ind_* (feats.py ret_cols / ind), hist* (feats.py hist_cols and hist2
    for the memory 'ret' and the pointer; without the reference cache, and without the LLR sums, which need the model's
    probabilities of earlier targets and are computed on the GPU), mem_* (rg/docgate/pipe.py HELPER), src_*
    (rg/docgate/stages.py src_cols + src_extra), both_* (stages.py both). Masked as the research masks them."""
    F = lambda name: _f(xp, rows[name])
    flags = rows["flags"]
    hp = _f(xp, (flags & F_HP) != 0)
    hs = _f(xp, (flags & F_HS) != 0)
    has = _f(xp, (flags & F_HAS) != 0)
    L2a = lambda v: xp.log2(1 + _clip_lo(xp, v, 0.0))       # feats.py L2
    L2b = lambda v: xp.log2(_clip_lo(xp, v, 1.0))           # pipe.py / stages.py L2
    l1 = xp.log1p
    out = {}
    # pointer (feats.py ptr_cols; the 'how' one-hots)
    nh64 = _clip_lo(xp, F("nhist"), 1.0)
    pc = {"ptr_run": L2a(F("run")), "ptr_hits": L2a(F("hits")), "ptr_miss16": F("n16") - F("c16"),
          "ptr_acc64": F("c64") / nh64, "ptr_since": L2a(F("since")), "ptr_age": L2a(F("age")), "ptr_share": F("share"),
          "ptr_tcnt": L2a(F("tcnt")), "ptr_ntok": L2a(_f(xp, rows["ntok"])), "ptr_seedlen": L2a(F("seedlen")),
          "ptr_nrec": L2a(F("nrec")),
          "ptr_margin": _clip_lo(xp, _min(xp, F("score") - F("score2"), 20.0), -20.0) / 4,
          "ptr_score": F("score") / 4}
    how = rows["how"]
    for h in range(6):
        pc[f"ptr_how{h}"] = _f(xp, how == h)
    for k, v in pc.items():
        out[k] = v * hp
    # windowed histories (feats.py hist_cols / hist2 for 'ret' and 'ptr'); n = positions of the segment in the window
    npos = F("d_npos")
    for w, W in ((0, 16), (1, 64)):
        nr, cr = _f(xp, rows["h_r_n"][:, w]), _f(xp, rows["h_r_c"][:, w])
        npp, cp = _f(xp, rows["h_p_n"][:, w]), _f(xp, rows["h_p_c"][:, w])
        n = _clip_lo(xp, _min(xp, npos, float(W)), 1.0)
        out[f"hist{W}_ret_c"] = L2a(cr)
        out[f"hist{W}_ret_acc"] = cr / _clip_lo(xp, nr, 1.0)
        out[f"hist{W}_ptr_c"] = L2a(cp)
        out[f"hist{W}_ptr_acc"] = cp / _clip_lo(xp, npp, 1.0)
        out[f"hist{W}_either"] = _f(xp, rows["h_e"][:, w]) / n
    nr, cr, npp, cp = F("d_nhit"), F("d_ncorr"), F("h_p_n_seg"), F("h_p_c_seg")
    out["histseg_ret_c"] = L2a(cr)
    out["histseg_ret_acc"] = cr / _clip_lo(xp, nr, 1.0)
    out["histseg_ptr_c"] = L2a(cp)
    out["histseg_ptr_acc"] = cp / _clip_lo(xp, npp, 1.0)
    out["histseg_either"] = F("h_e_seg") / _clip_lo(xp, npos, 1.0)
    out["hist2_ret_r32"] = F("h32_r_sum") / _clip_lo(xp, F("h32_r_n"), 1.0)
    out["hist2_ret_rseg"] = F("h_r_sum_seg") / _clip_lo(xp, nr, 1.0)
    out["hist2_ptr_r32"] = F("h32_p_c") / _clip_lo(xp, F("h32_p_n"), 1.0)
    out["hist2_ptr_rseg"] = cp / _clip_lo(xp, npp, 1.0)
    # the memory's doc-state (pipe.py HELPER, in its order)
    sm = lambda c, n, a=0.4, b=1.0: (c + a * b) / (n + b)
    nhit = F("d_nhit")
    for i, K in enumerate((8, 32, 128)):
        hn = _min(xp, nhit, float(K))
        pn = _min(xp, npos, float(K))
        out[f"mem_acc_h{K}"] = sm(_f(xp, rows["d_h_corr"][:, i]), hn)
        out[f"mem_n_h{K}"] = l1(hn)
        out[f"mem_acc_p{K}"] = (_f(xp, rows["d_p_corr"][:, i]) + 0.03) / (pn + 1)
        out[f"mem_hit_p{K}"] = (_f(xp, rows["d_p_hit"][:, i]) + 0.07) / (pn + 1)
    out["mem_r_h32"] = sm(F("d_h32_r"), _min(xp, nhit, 32.0))
    out["mem_streak"] = l1(F("d_streak"))
    out["mem_since_wrong"] = l1(_min(xp, F("d_since_wrong"), 2000.0))
    out["mem_since_hit"] = l1(_min(xp, F("d_since_hit"), 2000.0))
    out["mem_prev_corr"] = _f(xp, (flags & F_PREV_CORR) != 0)
    out["mem_prev_hit"] = _f(xp, (flags & F_PREV_HIT) != 0)
    out["mem_ema90"] = F("d_ema90")
    out["mem_ema98"] = F("d_ema98")
    out["mem_nlong32"] = l1(F("d_nlong32"))
    out["mem_nlong16"] = l1(F("d_nlong16"))
    out["mem_acc_doc"] = sm(F("d_ncorr"), nhit)
    out["mem_lenl"] = L2b(F("mem_lenl")) * has
    out["mem_lenr"] = L2b(F("mem_lenr")) * has
    out["mem_cont"] = _f(xp, (flags & F_CONT) != 0)
    out["mem_cont_run"] = l1(F("d_cont_run"))
    out["mem_cont_corr_run"] = l1(F("d_cont_corr_run"))
    out["mem_npos"] = l1(npos)
    out["mem_ncand"] = L2b(F("mem_ncand")) * has
    # the source copy (stages.py src_cols + src_extra; L2 = log2(max(v, 1)))
    srcn = rows["src_n"]
    nd = _f(xp, _at_li(xp, rows))
    # (occurrence fields are 0 without SP_HS; nsrc / since / newlen are set on every row, as in the research)
    out.update({"src_n": L2b(nd), "src_purity": F("src_m") / _clip_lo(xp, nd, 1.0), "src_lbest": F("src_lbest"),
                "src_lbest_l2": L2b(F("src_lbest")), "src_nsrc": F("src_nsrc"), "src_since": L2b(F("src_since") + 1),
                "src_newlen": L2b(F("src_newlen")), "src_n2": L2b(_f(xp, srcn[:, 1]))})
    smh = lambda c, n, a=0.3: (c + a) / (n + 1)
    sn = F("s_nhit")
    out["src_acc_h8"] = smh(_f(xp, rows["s_h_corr"][:, 0]), _min(xp, sn, 8.0))
    out["src_acc_h32"] = smh(_f(xp, rows["s_h_corr"][:, 1]), _min(xp, sn, 32.0))
    out["src_acc_doc"] = smh(F("s_ncorr"), sn)
    out["src_nhit"] = l1(sn)
    out["src_streak"] = l1(F("s_streak"))
    out["src_prev_corr"] = _f(xp, (flags & F_SPREV_CORR) != 0)
    out["src_r_h32"] = smh(F("s_h32_r"), _min(xp, sn, 32.0))
    # the memory at L* (feats.py ret_cols / ind, stages.py both; the same values as stream_memory.c's record) and
    # agree (pointer == the memory's top)
    nm = _clip_lo(xp, F("mem_n"), 1.0)
    lstar = L2b(F("mem_lstar"))
    out["ret_logn"] = xp.log2(nm) * has
    out["ret_purity"] = F("mem_m") / nm * has
    out["ret_logl"] = lstar * has
    out["ind_has"] = has
    out["ind_hp"] = hp
    pred = rows["pred"].astype(np.int64) if xp is np else rows["pred"].long()
    mtop = rows["mem_top"].astype(np.int64) if xp is np else rows["mem_top"].long()
    out["ptr_agree"] = hp * has * _f(xp, pred == mtop)
    out["both_has"] = has
    out["both_lstar"] = lstar * has
    out["both_hs"] = hs
    return out
