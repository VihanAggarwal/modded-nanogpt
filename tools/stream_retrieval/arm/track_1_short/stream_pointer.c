// Per-segment pointer beam, vote, source copy and doc-state counters (part P3 of stream retrieval v2).
//
// The stream memory's chain walk (stream_memory.c, query_one) finds, for a val position t, up to 32 verified
// candidates: memory entries p (tok[p] = x[t], tok[p + 1] = the candidate's next token) with their match lengths. This
// file turns each SEGMENT's candidate lists into further components and gate features for the top level of the eval's
// mixture, sequentially within the segment (segment = the positions since the last BOS and since the start of t's
// 262,144-token chunk; for FIT, also since the start of each batch: `run` == 1 starts one):
//
//   pointer beam  up to K = 16 alignment hypotheses j (memory entry j ~ val position t) predicting tok[j + 1]; seeded
//                 from the candidates, ranked by their recent hit / miss history, advanced on a correct prediction,
//                 on a miss replaced by recovery successors (substitution j + 1, insertion j, deletion j + 1 + k,
//                 resync on the last 2 tokens within [j - 64, j + 65] of the same span). The row: the best
//                 hypothesis's next token (the POINTER, a one-hot component) and its history, and the VOTE (a
//                 component): the hypotheses' next tokens weighted 2^(score - best score), normalized.
//   source copy   up to 8 training documents of the memory become this segment's sources (a candidate whose full
//                 match is >= 10 tokens, or >= 4 candidates of the segment in one 512-entry bin of the memory); each
//                 is indexed by its token bigrams, and t reads every occurrence of (x[t-1], x[t]) in them, extends it
//                 backward (<= 32), and counts their next tokens per match level LV. The row: the counts N per level,
//                 and the next-token distribution at the deepest level reached (the SOURCE component, C(y)/N).
//   doc-state     counters of the segment's own track record over EARLIER positions: of the memory's top token at
//                 L* (hit, correct, C/N of the realised target; over the last 8 / 32 / 128 positions and hits, streaks,
//                 EMAs, continuation runs), of the source copy (hits, top correct, C/N) and of the pointer (hits,
//                 correct) over windows of 16 / 32 / 64 positions and the whole segment.
//
// Rows. One row (sp_row_t) per ACTIVE position (the pointer or the source copy has a prediction there; elsewhere the
// top level of the mixture is off). Every field of row t is a function of the memory and of the segment's tokens up to
// and including t, and of the outcomes (targets) of EARLIER positions of the segment: the target y of t itself is read
// only after row t is written, to update the state for t + 1. The eval reads each component at the realised token on
// the GPU (pointer: pred == y; vote: the share of y among vtok; source: scnt of y / src_n[li]), as it does for the
// memory's own candidates. Vote shares sum to 1 over vtok; source counts sum to src_n[li] (unless SP_SRC_TRUNC: more
// than SP_SRCD distinct next tokens, the rest are left out and the component sums to less than 1).
//
// Determinism. A segment's rows depend on that segment alone (the state resets at run == 1), so the driver
// (sp_run) processes whole segments on any number of threads and the rows are bit-identical across thread counts.
//
// Memory reads. Every read of tok[] stays inside the span of the candidate it starts from: separators bound the
// backward extensions, the beam's successors and resync window, and the source documents. So with the FIT path
// (candidates from the memory frozen before the run's last K batches), every read is below the freeze.
//
// Ported from this branch's research tools (no code from another PR): rg/align/align4.c (the beam, NREF = 0: no
// reference-doc cache), rg/docgate/srccopy6.c (the source copy, config votes = 4, th = 10, maxsrc = 8, half = 4096,
// use_recent = 1), rg/docgate/docstate.c (the doc-state counters) and rg/align/feats.py (the windowed histories). The
// memory summary (L*, top, pos/len of the most recent and the longest candidate) is stream_memory.c's query_one,
// recomputed here from the candidate list so this file depends on nothing else. Changes from the research tools: the
// doc-state and the source set also reset at chunk boundaries (research: BOS only); rows carry the components as
// sparse distributions (research: the probability of the target); the evidence table of the reference-doc cache is
// gone (it fed only the cache). The source copy's 512-entry vote bins are of the caller's memory positions: in the
// helper those are the stream index + 1 (its tok[0] is a separator), in the research and stream_pointer_bench.c the
// stream index, so the helper's rows differ from the research's in which candidates share a bin (~13k rows' source
// fields of the first 1,048,576 val positions; the gain is the same). Credits as stream_memory.py's docstring: the memory and candidates this reads follow
// PR #367's StreamIndex (Herman Brunborg); the components join PR #380's output mixture (Deven), gated on training
// positions (the run's own FIT batches).
//
// API (all exported names sp_*; internal names static):
//   void        sp_default_config(sp_config_t *cfg)
//   sp_state_t *sp_state_new(const sp_config_t *cfg)                  NULL: bad config or out of memory
//   int         sp_step(st, tok, ntok, x, run, cpos, clen, nc, y, row) one position; 1 = row written (active)
//   int64_t     sp_run(cfg, tok, ntok, x, y, n, run, chunk, cands, ctx, threads, &rows, err, errlen)
//                 whole sequence, segment-parallel; *rows = malloc'd active rows in position order (sp_free)
//   int         sp_cands_csr(ctx, t, x, run, pos, len)                a candidate source over CSR arrays (sp_csr_t)
//   void        sp_mem_summary(tok, x, run, cpos, clen, nc, y, out)   the memory record fields (as query_one)
//   const char *sp_layout(void)                                       "name offset size,..." of sp_row_t (tests)
//   void        sp_state_free(sp_state_t *), sp_free(void *)
//
// In the helper (stream_memory.c): make the query blocks start at segment starts (a block may then run past its
// nominal end to finish its last segment), give each query thread an sp_state_t, and after query_one(t) call
// sp_step(st, tok, entries, x + t, run, pos, lens, nc, y[t], &row) with query_one's candidate arrays (walk order, lens
// capped at min(32, run)); keep the active rows per block like the records.
//
// Cost (the 1050-step stream, the first 1,048,576 val positions, 1 thread of this 4-vCPU box; stream_pointer_bench.c):
// 0.74 us per val position (all positions; the beam and the doc-state 0.49 us, the source copy 0.26 us), on top of the
// shared walk (0.32 us); nothing during training. 33.3% of positions are active (pointer 22.1%, source 18.7%), so the
// rows take 127 bytes per val position (1.33 GB for 10,485,760 positions, 166 MB per rank). Per thread: ~1.4 MB of
// state (the source indices). For the 10,485,760 val positions: ~7.8 CPU-s, ~0.24 s on 32 threads.
//
// Build: cc -O2 -std=c11 -pthread -shared -fPIC stream_pointer.c -o libstream_pointer.so -lm (stream_pointer.py), or
// #include it into the helper after `#define SP_API static __attribute__((unused))` (one translation unit with
// stream_memory.c and stream_lowtables.c compiles without name clashes; test_stream_pointer.py checks it).
#ifndef STREAM_POINTER_C  // the helper (stream_memory.c) #includes this file; a second #include is a no-op
#define STREAM_POINTER_C
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__SSE2__) && !defined(SP_NO_SIMD)
#include <emmintrin.h>
#define SP_SIMD 1
#endif

#ifndef SP_API
#define SP_API __attribute__((visibility("default")))
#endif

#define SP_SEP 0xFFFFu
#define SP_BOS 50256u
#define SP_KEY 6
#define SP_MAXLEN 32       // the walk's match lengths are capped here (stream_memory.c MAXLEN)
#define SP_FULLCAP 4096    // full match lengths (and runs) saturate here (stream_memory.c FULLCAP)
#define SP_CAP 32          // candidates per position at most (stream_memory.c CAP)
#define SP_LCAP 512        // the beam's seed match length cap (align4 LCAP)
#define SP_NLEVELS 8
static const uint32_t SP_LEVELS[SP_NLEVELS] = {6, 7, 8, 10, 12, 16, 24, 32};  // stream_memory.c LEVELS
#define SP_NLV 8
static const uint32_t SP_LV[SP_NLV] = {1, 2, 3, 4, 6, 8, 12, 16};          // source copy levels (srccopy6 LV)
#define SP_CAPL 32         // source copy: match length cap
#define SP_VOTE 16         // vote entries (= the most distinct tokens K <= 16 hypotheses can have: exact)
#define SP_SRCD 16         // source distribution entries kept (most frequent first)
#define SP_KMAX 16
#define SP_MAXSRC 8
#define SP_VBINS 256       // source votes: distinct 512-entry bins tracked per segment
#define SP_RB 128          // history ring (positions and hits)
#define SP_NONE 0xFFFFFFFFu

// Flags of a row.
enum {
    SP_HP = 1,             // pointer row: the beam has a hypothesis with a next token (pred, vote valid)
    SP_HS = 2,             // source row: some source has an occurrence of (x[t-1], x[t]) (src_* valid)
    SP_HAS = 4,            // the memory has candidates at t (mem_* valid)
    SP_SRC_TRUNC = 8,      // the source distribution had more than SP_SRCD distinct tokens (the rest left out)
    SP_CONT = 16,          // docstate cont: hit at t and t - 1 with pos_recent[t] == pos_recent[t - 1] + 1
    SP_PREV_CORR = 32,     // memory top was correct at t - 1 (docstate prev_corr)
    SP_PREV_HIT = 64,      // memory hit at t - 1 (docstate prev_hit)
    SP_SPREV_CORR = 128,   // source top was correct at t - 1
};

// The configuration (research defaults: sp_default_config).
typedef struct {
    int32_t beam_k;        // hypotheses kept (K, <= 16)
    int32_t seed_min;      // a candidate seeds a hypothesis if its match is >= this
    int32_t dmax;          // deletion successors look this far ahead
    int32_t fwd, back;     // resync window after / before the hypothesis
    int32_t gram;          // resync on the last `gram` tokens (0: off)
    int32_t src_th;        // a memory candidate whose full match is >= th makes its document a source
    int32_t src_maxsrc;    // sources kept per segment (LRU, <= 8)
    int32_t src_half;      // a source document is cut to [j - half, j + half]
    int32_t src_votes;     // or >= votes candidates in one 512-entry bin (0: off)
    int32_t src_use_recent;// the most recent candidate at L* is a source candidate too (besides the longest)
    int32_t reserved[5];
} sp_config_t;

// One active position's row (416 bytes; stream_pointer.py ROW_DTYPE mirrors it, sp_layout() lists the offsets).
typedef struct {
    uint32_t pos;              // the position (index into the sequence given to sp_run; 0 from sp_step)
    uint16_t flags;            // SP_* flags
    uint8_t ntok;              // vote entries (distinct next tokens of the valid hypotheses), 0 without SP_HP
    uint8_t nsd;               // source distribution entries, 0 without SP_HS
    // ---- pointer beam (SP_HP; align4 feat column in brackets)
    uint32_t ptr_j;            // the best hypothesis's memory entry (its next token tok[ptr_j + 1] is pred)
    float score;               // [18] its score
    float score2;              // [25] the next hypothesis's score (-99 if none)
    float share;               // [13] the vote share of pred
    float share2;              // [17] the largest share of another token (0 if none)
    float vshare;              // [22] the vote's top share
    uint16_t pred;             // [1]  the pointer's token
    uint16_t vtop;             // [21] the vote's top token (ties: the first in rank order)
    uint16_t tok2;             // [16] that other token (0xFFFF if none)
    uint16_t run;              // [2]  current run of correct predictions (saturates at 65535)
    uint16_t hits, miss;       // [3], [4]
    uint16_t age, since;       // [9], [10]
    uint16_t seedlen, nrec;    // [19], [20]
    uint8_t c16, n16, c64, nhist;  // [5], [6], [7], [8]: correct of the last min(nhist,16) / all <= 64 outcomes
    uint8_t how;               // [11] how it was made: 0 seed, 1 advanced, 2 substitution, 3 insertion, 4 deletion, 5 resync
    uint8_t nvh;               // [12] valid hypotheses (next token not SEP / BOS)
    uint8_t tcnt;              // [14] hypotheses voting for pred
    uint8_t pad0;
    uint16_t vtok[SP_VOTE];    // the vote: tokens (ntok of them) ...
    float vw[SP_VOTE];         //           ... and their shares (sum 1)
    // ---- source copy (SP_HS; srccopy6 out columns)
    uint16_t src_n[SP_NLV];    // occurrences per level LV (match >= 1, 2, 3, 4, 6, 8, 12, 16); N1 == N2
    uint16_t src_m;            // the largest count of one next token at level li
    uint16_t src_newlen;       // the newest source's candidate match length (0 if none)
    int32_t src_since;         // positions since the newest source was found (-1 if none)
    uint8_t src_lbest;         // the longest occurrence's match length (2..32; 0 if none)
    uint8_t src_nsrc;          // sources of the segment
    uint8_t src_li;            // the deepest level with occurrences (index into LV)
    uint8_t pad1;
    uint16_t stok[SP_SRCD];    // the next tokens at level li, most frequent first (ties: lowest id) ...
    uint16_t scnt[SP_SRCD];    //   ... and their counts (sum src_n[li] unless SP_SRC_TRUNC)
    // ---- the memory at t (SP_HAS; stream_memory.c record fields, recomputed from the candidates)
    uint16_t mem_lenl, mem_lenr; // full match length of the longest / of the most recent candidate at L*
    uint16_t mem_top;          // the top next token at L*
    uint8_t mem_ncand;         // candidates
    uint8_t mem_lstar;         // L* (the level's length, 6..32)
    uint8_t mem_n, mem_m;      // N and M at L*
    uint16_t pad2;
    // ---- doc-state of the memory (docstate.c over hit = SP_HAS, corr = top at L* == y, r = C/N at L*)
    uint32_t d_npos;           // positions of the segment before t
    uint32_t d_nhit, d_ncorr;  // hits / correct hits before t
    uint32_t d_streak;         // correct in a row
    uint32_t d_since_wrong;    // positions since the last wrong hit (100000 + n at a segment start, as docstate)
    uint32_t d_since_hit;
    uint32_t d_nlong32, d_nlong16; // hits with a full match >= 32 / >= 16
    uint32_t d_cont_run, d_cont_corr_run;
    float d_ema90, d_ema98;    // EMA of correct
    float d_h32_r;             // C/N of the realised target summed over the last 32 hits
    uint8_t d_p_hit[3], d_p_corr[3];  // over the last 8 / 32 / 128 positions
    uint8_t d_h_corr[3];       // over the last 8 / 32 / 128 hits (their number: min(d_nhit, K))
    uint8_t pad3[3];
    // ---- doc-state of the source copy (hit = SP_HS, corr = y has the largest count at li, r = C(y)/N at li)
    uint32_t s_nhit, s_ncorr, s_streak;
    float s_h32_r;
    uint8_t s_h_corr[2];       // over the last 8 / 32 source hits
    uint8_t pad4[2];
    // ---- windowed histories (feats.py hist_cols): memory hit, memory correct (ret_ok), pointer row, pointer
    //      correct, either correct; over the last 16 / 64 positions of the segment (the whole segment: d_* and h_*_seg)
    uint8_t h_r_n[2], h_r_c[2], h_p_n[2], h_p_c[2], h_e[2];   // [0] = 16, [1] = 64
    uint8_t h32_r_n, h32_p_n, h32_p_c;                        // over the last 32 positions
    uint8_t pad5;
    float h32_r_sum;           // C/N of the realised target summed over memory hits in the last 32 positions
    float h_r_sum_seg;         // ... over the whole segment
    uint32_t h_p_n_seg, h_p_c_seg, h_e_seg;  // pointer rows / correct / either correct in the segment
} sp_row_t;
_Static_assert(sizeof(sp_row_t) == 380, "sp_row_t is 380 bytes");

// ------------------------------------------------------------------------------------------------ the state

typedef struct {
    uint32_t j, run, hits, miss;
    uint64_t hist;
    uint32_t nhist, age, since, seedlen, nrec;
    int32_t how;
    float score;
} sp_hyp;

typedef struct {
    int64_t a, b;              // the source document [a, b) in tok
    int32_t *head, *nxt;       // its bigram chain: head[h] = the first u - a (increasing u), nxt[u - a] = the next
    uint32_t hm;
} sp_src;

typedef struct { uint16_t nx; uint8_t l; } sp_scand;

// Track record of one predictor in the segment (docstate.c's counters).
typedef struct {
    uint32_t npos, nhit, ncorr, streak, since_wrong, since_hit, nlong32, nlong16, cont_run, cont_corr_run;
    double ema90, ema98;
    int prev_corr, prev_hit;
    uint32_t prev_posr;
    uint8_t ph[SP_RB], pc[SP_RB];      // per position: hit, correct
    uint8_t hc[SP_RB];                 // per hit: correct
    double hr[SP_RB];                  // per hit: r
    uint32_t p_hit[3], p_corr[3], h_corr[3];  // running sums over the last 8 / 32 / 128 positions / hits
} sp_track;

struct sp_state {
    sp_config_t cfg;
    // pointer beam
    sp_hyp *H, *NH;
    int nh, hcap;
    uint32_t dstamp[128], depoch;      // the successors' dedupe table (stamped)
    uint8_t dtab[128];
    // source copy
    sp_src src[SP_MAXSRC];
    int order[SP_MAXSRC], ns;          // source slots, least recently used first
    int64_t newest_t;
    uint32_t newest_len;
    uint32_t vbin[SP_VBINS];
    int vcnt[SP_VBINS], nv;
    sp_scand *cand;
    size_t capc;
    // token counting with stamps (source distribution)
    uint32_t *vstamp, *vcount, stamp;
    uint16_t *touched;
    size_t ntouched_cap;
    // position within the segment
    uint64_t t;
    // tracks
    sp_track mem, srct;
    // windowed histories over positions (ring of the last SP_RB positions)
    uint8_t wbits[SP_RB];              // bit0 has, bit1 ret_ok, bit2 hp, bit3 bptr, bit4 either correct
    double wr[SP_RB];                  // r at memory hits (0 elsewhere)
    uint64_t wsum[3];                  // running sums of the 5 bits over the last 16 / 64 / 32 positions (12-bit fields)
    uint32_t h_p_n_seg, h_p_c_seg, h_e_seg;
    double h_r_sum_seg;
};
typedef struct sp_state sp_state_t;

static inline int sp_pc(uint64_t x) {
#ifdef __POPCNT__
    return __builtin_popcountll(x);
#else  // without -mpopcnt the builtin is a libgcc call
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return (int)((x * 0x0101010101010101ull) >> 56);
#endif
}
static inline uint64_t sp_lowmask(uint32_t n) { return n >= 64 ? ~0ull : ((1ull << n) - 1); }
static inline uint16_t sp_sat16(uint64_t v) { return v > 65535 ? 65535 : (uint16_t)v; }

SP_API void sp_default_config(sp_config_t *c) {
    memset(c, 0, sizeof *c);
    c->beam_k = 16; c->seed_min = 6; c->dmax = 8; c->fwd = 64; c->back = 64; c->gram = 2;
    c->src_th = 10; c->src_maxsrc = 8; c->src_half = 4096; c->src_votes = 4; c->src_use_recent = 1;
}

static uint32_t sp_pow2_at_least(uint64_t v) { uint32_t h = 16; while (h < v) h <<= 1; return h; }

SP_API void sp_state_free(sp_state_t *s) {
    if (!s) return;
    free(s->H); free(s->NH); free(s->cand); free(s->vstamp); free(s->vcount); free(s->touched);
    for (int i = 0; i < SP_MAXSRC; i++) { free(s->src[i].head); free(s->src[i].nxt); }
    free(s);
}

static void sp_reset(sp_state_t *s);
static void sp_tables_init(void);

SP_API sp_state_t *sp_state_new(const sp_config_t *cfg) {
    sp_config_t c;
    if (cfg) c = *cfg; else sp_default_config(&c);
    if (c.beam_k < 1 || c.beam_k > SP_KMAX || c.seed_min < 1 || c.dmax < 0 || c.fwd < 0 || c.back < 0 || c.gram < 0 ||
        c.gram > 64 || c.src_maxsrc < 1 || c.src_maxsrc > SP_MAXSRC || c.src_half < 1 || c.src_half > (1 << 20) ||
        c.src_votes < 0 || c.src_th < 0)
        return NULL;
    sp_tables_init();
    sp_state_t *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->cfg = c;
    s->hcap = 4 * c.beam_k + SP_CAP + 8;
    s->H = malloc(sizeof(sp_hyp) * s->hcap);
    s->NH = malloc(sizeof(sp_hyp) * 4 * c.beam_k);
    s->capc = 1 << 12;
    s->cand = malloc(sizeof(sp_scand) * s->capc);
    s->vstamp = calloc(1 << 16, 4);
    s->vcount = calloc(1 << 16, 4);
    s->ntouched_cap = 1 << 16;
    s->touched = malloc(2 * s->ntouched_cap);
    const uint64_t nmax = 2 * (uint64_t)c.src_half;          // a source document is at most 2 half tokens
    const uint32_t hsmax = sp_pow2_at_least(2 * nmax);
    int ok = s->H && s->NH && s->cand && s->vstamp && s->vcount && s->touched;
    for (int i = 0; ok && i < c.src_maxsrc; i++) {
        s->src[i].head = malloc(4ull * hsmax);
        s->src[i].nxt = malloc(4ull * nmax);
        ok = s->src[i].head && s->src[i].nxt;
    }
    if (!ok) { sp_state_free(s); return NULL; }
    sp_reset(s);
    return s;
}

static void sp_track_reset(sp_track *k) {
    memset(k, 0, sizeof *k);
    k->since_wrong = 100000;
    k->since_hit = 100000;
    k->prev_posr = SP_NONE;
}

static void sp_reset(sp_state_t *s) {
    s->nh = 0;
    s->ns = 0;
    s->newest_t = -1;
    s->newest_len = 0;
    s->nv = 0;
    s->t = 0;
    sp_track_reset(&s->mem);
    sp_track_reset(&s->srct);
    s->h_p_n_seg = s->h_p_c_seg = s->h_e_seg = 0;
    s->h_r_sum_seg = 0;
    memset(s->wsum, 0, sizeof s->wsum);
}

// ------------------------------------------------------------------------------------------------ memory summary

// stream_memory.c's record fields of position t, from its candidates (as query_one computes them).
typedef struct {
    uint32_t pos_recent, pos_longest;
    uint32_t len_recent, len_longest;
    uint32_t n, m, c;          // at L*: candidates, the largest count of one next token, the count of y
    uint16_t top;
    uint8_t lstar, ncand;      // lstar: index into SP_LEVELS
} sp_mem_t;

static inline uint32_t sp_full_len(const uint16_t *x, const uint16_t *m, uint32_t l, uint32_t run) {
    if (l < SP_MAXLEN) return l;
    const uint32_t lim = run < SP_FULLCAP ? run : SP_FULLCAP;
    while (l < lim && m[-(ptrdiff_t)l] == x[-(ptrdiff_t)l]) l++;
    return l;
}

SP_API void sp_mem_summary(const uint16_t *tok, const uint16_t *x, uint32_t run, const uint32_t *cpos,
                           const uint8_t *clen, int nc, uint16_t y, sp_mem_t *r) {
    memset(r, 0, sizeof *r);
    r->pos_recent = r->pos_longest = SP_NONE;
    if (nc <= 0) return;
    uint32_t maxl = 0;
    for (int i = 0; i < nc; i++) maxl = clen[i] > maxl ? clen[i] : maxl;
    int ls = 0;
    for (int k = 0; k < SP_NLEVELS; k++) if (SP_LEVELS[k] <= maxl) ls = k;
    const uint32_t L = SP_LEVELS[ls];
    // top at L*: the most frequent next token among the candidates at least L deep, ties to the lowest id
    uint16_t nx[SP_CAP];
    int ns = 0;
    for (int i = 0; i < nc; i++) if (clen[i] >= L) {
        uint16_t v = tok[cpos[i] + 1];
        int q = ns++;
        while (q && nx[q - 1] > v) { nx[q] = nx[q - 1]; q--; }
        nx[q] = v;
    }
    uint32_t mm = 0, cy = 0;
    uint16_t top = 0;
    for (int i = 0; i < ns;) {
        int q = i;
        while (q < ns && nx[q] == nx[i]) q++;
        if ((uint32_t)(q - i) > mm) { mm = (uint32_t)(q - i); top = nx[i]; }
        if (nx[i] == y) cy = (uint32_t)(q - i);
        i = q;
    }
    int irec = 0;
    while (clen[irec] < L) irec++;
    const uint32_t best_possible = run < SP_FULLCAP ? run : SP_FULLCAP;
    uint32_t lrec = sp_full_len(x, tok + cpos[irec], clen[irec], run), best = 0;
    int ilong = 0;
    for (int i = 0; i < nc && best < best_possible; i++) {
        if (clen[i] <= best && clen[i] < SP_MAXLEN) continue;
        uint32_t l = i == irec ? lrec : sp_full_len(x, tok + cpos[i], clen[i], run);
        if (l > best) { best = l; ilong = i; }
    }
    r->pos_recent = cpos[irec];
    r->len_recent = lrec;
    r->pos_longest = cpos[ilong];
    r->len_longest = best;
    r->n = (uint32_t)ns;
    r->m = mm;
    r->c = cy;
    r->top = top;
    r->lstar = (uint8_t)ls;
    r->ncand = (uint8_t)nc;
}

// ------------------------------------------------------------------------------------------------ pointer beam

// log2(1 + v): a table of the same libm values below 4096
#define SP_L2N 4096
static double sp_l2p1_tab[SP_L2N];
static pthread_once_t sp_l2p1_once = PTHREAD_ONCE_INIT;
static void sp_l2p1_init(void) { for (int i = 0; i < SP_L2N; i++) sp_l2p1_tab[i] = log2(1.0 + i); }
static void sp_tables_init(void) { pthread_once(&sp_l2p1_once, sp_l2p1_init); }
static inline double sp_l2p1(uint32_t v) { return v < SP_L2N ? sp_l2p1_tab[v] : log2(1.0 + v); }

// align4's score_of: the run and hits so far, the outcomes of the last 16 / 64 predictions
static float sp_score(const sp_hyp *h) {
    const uint32_t n16 = h->nhist < 16 ? h->nhist : 16, n64 = h->nhist;
    const int c16 = sp_pc(h->hist & sp_lowmask(n16)), c64 = sp_pc(h->hist & sp_lowmask(n64));
    return (float)(sp_l2p1(h->run) + 0.5 * sp_l2p1(h->hits) + 0.15 * c64 - 1.0 * ((int)n16 - c16) -
                   0.25 * ((int)n64 - c64));
}

// Resync: does the memory before i match the val tokens before t + 1 (x[0], x[-1], ...) for gram - 1 tokens?
static inline int sp_gram_ok(const uint16_t *tok, int64_t i, const uint16_t *x, int gram) {
    int gg = 1;
    while (gg < gram && i - gg >= 0 && tok[i - gg] == x[1 - gg]) gg++;
    return gg == gram;
}

// The first i in [from, to] with tok[i] == y and the gram context, scanning up; -1 if none or a separator comes first.
static int64_t sp_scan_up(const uint16_t *tok, int64_t from, int64_t to, uint16_t y, const uint16_t *x, int gram) {
    int64_t i = from;
#ifdef SP_SIMD
    const __m128i vy = _mm_set1_epi16((short)y), vs = _mm_set1_epi16((short)SP_SEP);
    for (; i + 7 <= to; i += 8) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(tok + i));
        unsigned m = (unsigned)_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi16(v, vs), _mm_cmpeq_epi16(v, vy)));
        while (m) {
            const int64_t k = i + (__builtin_ctz(m) >> 1);
            if (tok[k] == SP_SEP) return -1;
            if (sp_gram_ok(tok, k, x, gram)) return k;
            m &= m - 1;
            m &= m - 1;  // both bytes of the lane
        }
    }
#endif
    for (; i <= to; i++) {
        if (tok[i] == SP_SEP) return -1;
        if (tok[i] == y && sp_gram_ok(tok, i, x, gram)) return i;
    }
    return -1;
}

// The same scanning down from `from` to `to` (to <= from).
static int64_t sp_scan_down(const uint16_t *tok, int64_t from, int64_t to, uint16_t y, const uint16_t *x, int gram) {
    int64_t i = from;
#ifdef SP_SIMD
    const __m128i vy = _mm_set1_epi16((short)y), vs = _mm_set1_epi16((short)SP_SEP);
    for (; i - 7 >= to; i -= 8) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(tok + i - 7));
        unsigned m = (unsigned)_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi16(v, vs), _mm_cmpeq_epi16(v, vy)));
        while (m) {
            const int hb = 31 - __builtin_clz(m);           // the highest set byte: the lane's upper byte
            const int64_t k = i - 7 + (hb >> 1);
            if (tok[k] == SP_SEP) return -1;
            if (sp_gram_ok(tok, k, x, gram)) return k;
            m &= ~(3u << (hb & ~1));
        }
    }
#endif
    for (; i >= to; i--) {
        if (tok[i] == SP_SEP) return -1;
        if (tok[i] == y && sp_gram_ok(tok, i, x, gram)) return i;
    }
    return -1;
}

// Seed (or refresh) hypotheses from t's candidates, rank, truncate to K, and write the pointer part of the row.
// Returns 1 if the beam has a valid hypothesis (SP_HP).
static int sp_beam_row(sp_state_t *s, const uint16_t *tok, const uint16_t *x, uint32_t run, const uint32_t *cpos,
                       const uint8_t *clen, int nc, sp_row_t *row) {
    const sp_config_t *c = &s->cfg;
    sp_hyp *H = s->H;
    int nh = s->nh;
    if (run >= SP_KEY) {
        const uint32_t lim = run < SP_LCAP ? run : SP_LCAP;
        for (int i = 0; i < nc; i++) {
            const uint32_t j = cpos[i];
            uint32_t l = clen[i];
            if (l >= SP_MAXLEN) while (l < lim && tok[j - l] == x[-(ptrdiff_t)l]) l++;  // past the walk's cap
            int found = -1;
            for (int k = 0; k < nh; k++) if (H[k].j == j) { found = k; break; }
            if (found >= 0) {
                if (H[found].run < l) { H[found].run = l; H[found].score = sp_score(&H[found]); }
            } else if ((int)l >= c->seed_min && nh < s->hcap) {
                sp_hyp *q = &H[nh++];
                memset(q, 0, sizeof *q);
                q->j = j; q->run = l; q->hits = l; q->since = l; q->seedlen = l;
                q->nhist = l < 64 ? l : 64;
                q->hist = sp_lowmask(q->nhist);
                q->score = sp_score(q);
            }
        }
    }
    // rank (stable insertion sort, best first) and truncate; every score is current (set where a field changed)
    for (int i = 1; i < nh; i++) {
        sp_hyp v = H[i];
        int q = i;
        while (q && H[q - 1].score < v.score) { H[q] = H[q - 1]; q--; }
        H[q] = v;
    }
    if (nh > c->beam_k) nh = c->beam_k;
    s->nh = nh;
    // the vote over the valid hypotheses
    int best = -1, nvh = 0, ntok = 0;
    double wsum = 0, tw[SP_KMAX];
    uint16_t toks[SP_KMAX];
    int tcnt[SP_KMAX];
    for (int i = 0; i < nh; i++) {
        const uint16_t v = tok[H[i].j + 1];
        if (v == SP_SEP || v == SP_BOS) continue;
        nvh++;
        if (best < 0) best = i;
        const double w = exp2((double)(H[i].score - H[best].score));
        wsum += w;
        int q = 0;
        while (q < ntok && toks[q] != v) q++;
        if (q == ntok) { toks[ntok] = v; tw[ntok] = 0; tcnt[ntok] = 0; ntok++; }
        tw[q] += w;
        tcnt[q]++;
    }
    if (best < 0) return 0;
    const sp_hyp *b = &H[best];
    const uint16_t v = tok[b->j + 1];
    const uint32_t n16 = b->nhist < 16 ? b->nhist : 16;
    int q = 0;
    while (toks[q] != v) q++;
    int s2 = -1;
    for (int i = 0; i < ntok; i++) if (i != q && (s2 < 0 || tw[i] > tw[s2])) s2 = i;
    int vt = 0;
    for (int i = 1; i < ntok; i++) if (tw[i] > tw[vt]) vt = i;
    row->ptr_j = b->j;
    row->score = b->score;
    row->score2 = nh > best + 1 ? H[best + 1].score : -99.0f;
    row->share = (float)(tw[q] / wsum);
    row->share2 = s2 >= 0 ? (float)(tw[s2] / wsum) : 0.0f;
    row->vshare = (float)(tw[vt] / wsum);
    row->pred = v;
    row->vtop = toks[vt];
    row->tok2 = s2 >= 0 ? toks[s2] : 0xFFFF;
    row->run = sp_sat16(b->run); row->hits = sp_sat16(b->hits); row->miss = sp_sat16(b->miss);
    row->age = sp_sat16(b->age); row->since = sp_sat16(b->since);
    row->seedlen = sp_sat16(b->seedlen); row->nrec = sp_sat16(b->nrec);
    row->c16 = (uint8_t)sp_pc(b->hist & sp_lowmask(n16));
    row->n16 = (uint8_t)n16;
    row->c64 = (uint8_t)sp_pc(b->hist);
    row->nhist = (uint8_t)b->nhist;
    row->how = (uint8_t)b->how;
    row->nvh = (uint8_t)nvh;
    row->tcnt = (uint8_t)tcnt[q];
    row->ntok = (uint8_t)ntok;
    for (int i = 0; i < ntok; i++) { row->vtok[i] = toks[i]; row->vw[i] = (float)(tw[i] / wsum); }
    return 1;
}

// Observe t's target y: advance the hypotheses that predicted it, replace the others by their recovery successors.
static void sp_beam_update(sp_state_t *s, const uint16_t *tok, uint64_t ntok, const uint16_t *x, uint32_t run,
                           uint16_t y) {
    const sp_config_t *c = &s->cfg;
    sp_hyp *H = s->H, *NH = s->NH;
    int nn = 0;
    for (int i = 0; i < s->nh; i++) {
        sp_hyp p = H[i];
        p.age++;
        const uint16_t v = tok[p.j + 1];
        if (v == y && v != SP_SEP) {
            p.j++; p.run++; p.hits++; p.since++; p.hist = (p.hist << 1) | 1;
            if (p.nhist < 64) p.nhist++;
            p.how = 1;
            NH[nn++] = p;
            continue;
        }
        sp_hyp h = p;
        h.run = 0; h.miss++; h.since = 0; h.hist <<= 1;
        if (h.nhist < 64) h.nhist++;
        h.nrec++;
        if (v != SP_SEP) { NH[nn] = h; NH[nn].j = p.j + 1; NH[nn].how = 2; nn++; }   // substitution
        NH[nn] = h; NH[nn].how = 3; nn++;                                          // insertion
        for (int k = 1; k <= c->dmax; k++) {                                       // deletion
            const uint64_t jj = (uint64_t)p.j + 1 + k;
            if (jj >= ntok || tok[jj] == SP_SEP) break;
            if (tok[jj] == y && tok[p.j + k] != SP_SEP) {
                NH[nn] = h; NH[nn].j = (uint32_t)jj; NH[nn].how = 4; NH[nn].run = 1; nn++;
                break;
            }
        }
        if (c->gram > 0 && run + 1 >= (uint32_t)c->gram) {                          // resync on the last gram tokens
            // tokens t + 1 - g for g = 0 .. gram - 1: y, then x[0], x[-1], ...
            // the nearest occurrence after (within fwd) or at / before (within back) j + 1, in j's span
            int64_t lo = (int64_t)p.j - c->back, hi = (int64_t)p.j + 1 + c->fwd;
            if (lo < 0) lo = 0;
            if (hi > (int64_t)ntok - 1) hi = (int64_t)ntok - 1;
            int64_t bestd = (int64_t)1 << 40, bi = sp_scan_up(tok, (int64_t)p.j + 1, hi, y, x, c->gram);
            if (bi >= 0) bestd = bi - ((int64_t)p.j + 1);
            const int64_t bd = sp_scan_down(tok, (int64_t)p.j, lo, y, x, c->gram);
            if (bd >= 0 && (int64_t)p.j + 1 - bd < bestd) { bestd = (int64_t)p.j + 1 - bd; bi = bd; }
            if (bi >= 0 && bi != (int64_t)p.j + 1) {
                NH[nn] = h; NH[nn].j = (uint32_t)bi; NH[nn].how = 5; NH[nn].run = (uint32_t)c->gram; nn++;
            }
        }
    }
    for (int i = 0; i < nn; i++) NH[i].score = sp_score(&NH[i]);
    int nh = 0;
    if (++s->depoch == 0) { memset(s->dstamp, 0, sizeof s->dstamp); s->depoch = 1; }
    for (int i = 0; i < nn; i++) {  // one hypothesis per memory entry: the first, unless a later one scores higher
        uint32_t hh = (NH[i].j * 0x9E3779B1u) >> 25;  // open addressing over 128 slots (nn <= 64)
        while (s->dstamp[hh] == s->depoch && H[s->dtab[hh]].j != NH[i].j) hh = (hh + 1) & 127;
        if (s->dstamp[hh] != s->depoch) {
            s->dstamp[hh] = s->depoch;
            s->dtab[hh] = (uint8_t)nh;
            H[nh++] = NH[i];
        } else if (NH[i].score > H[s->dtab[hh]].score) {
            H[s->dtab[hh]] = NH[i];
        }
    }
    int m = 0;
    for (int i = 0; i < nh; i++) {  // drop hypotheses with >= 6 misses in their last 8 outcomes
        const uint32_t n8 = H[i].nhist < 8 ? H[i].nhist : 8;
        if ((int)n8 - sp_pc(H[i].hist & sp_lowmask(n8)) >= 6) continue;
        H[m++] = H[i];
    }
    s->nh = m;
}

// ------------------------------------------------------------------------------------------------ source copy

static inline uint32_t sp_bh(uint32_t k) { k *= 0x9E3779B1u; return k ^ (k >> 15); }

static void sp_src_build(sp_src *S, const uint16_t *tok) {
    const int64_t n = S->b - S->a;
    const uint32_t hs = sp_pow2_at_least(2 * (uint64_t)(n > 0 ? n : 0));
    S->hm = hs - 1;
    memset(S->head, 0xFF, 4ull * hs);
    for (int64_t u = S->b - 2; u >= S->a + 1; u--) {  // decreasing u: chains list positions in increasing order
        const uint32_t k = ((uint32_t)tok[u - 1] << 16) | tok[u];
        const uint32_t h = sp_bh(k) & S->hm;
        S->nxt[u - S->a] = S->head[h];
        S->head[h] = (int32_t)(u - S->a);
    }
}

// Source discovery from t's memory record: its longest (and most recent at L*) candidate's document becomes a source
// when the match is long enough or its 512-entry bin has collected enough of the segment's candidates.
static void sp_src_discover(sp_state_t *s, const uint16_t *tok, uint64_t ntok, const sp_mem_t *m) {
    const sp_config_t *c = &s->cfg;
    if (!m->ncand) return;
    for (int w = 0; w < 1 + (c->src_use_recent != 0); w++) {
        const uint32_t ll = w ? m->len_recent : m->len_longest, pp = w ? m->pos_recent : m->pos_longest;
        if (pp == SP_NONE) continue;
        int strong = (int)ll >= c->src_th;
        if (!strong && c->src_votes > 0) {
            const uint32_t bn = pp >> 9;
            int vi = -1;
            for (int i = 0; i < s->nv; i++) if (s->vbin[i] == bn) { vi = i; break; }
            if (vi < 0 && s->nv < SP_VBINS) { s->vbin[s->nv] = bn; s->vcnt[s->nv] = 0; vi = s->nv++; }
            if (vi >= 0 && ++s->vcnt[vi] >= c->src_votes) strong = 1;
        }
        if (!strong) continue;
        const int64_t j = pp;
        int dup = -1;
        for (int i = 0; i < s->ns; i++) {
            const sp_src *S = &s->src[s->order[i]];
            if (j >= S->a && j < S->b) { dup = i; break; }
        }
        if (dup >= 0) {  // most recently used: to the end
            const int slot = s->order[dup];
            memmove(s->order + dup, s->order + dup + 1, sizeof(int) * (s->ns - dup - 1));
            s->order[s->ns - 1] = slot;
        } else {
            int64_t a = j, b = j + 1;
            while (a > 0 && a > j - c->src_half && tok[a] != SP_BOS && tok[a - 1] != SP_SEP) a--;
            while (b < (int64_t)ntok && b < j + c->src_half && tok[b] != SP_SEP && tok[b] != SP_BOS) b++;
            int slot = -1;
            if (s->ns == c->src_maxsrc) {  // evict the least recently used
                slot = s->order[0];
                memmove(s->order, s->order + 1, sizeof(int) * (s->ns - 1));
                s->ns--;
            } else {
                for (int f = 0; f < c->src_maxsrc && slot < 0; f++) {  // a free slot
                    int used = 0;
                    for (int i = 0; i < s->ns; i++) used |= s->order[i] == f;
                    if (!used) slot = f;
                }
            }
            s->src[slot].a = a;
            s->src[slot].b = b;
            sp_src_build(&s->src[slot], tok);
            s->order[s->ns++] = slot;
        }
        s->newest_t = (int64_t)s->t;
        s->newest_len = ll;
    }
}

// The source query: every occurrence of (x[-1], x[0]) in the sources, extended backward; per level the counts, and the
// next-token distribution at the deepest level. Returns 1 if there is an occurrence (SP_HS). Leaves the grouping of
// level li in the stamp arrays (for the target's count after the row).
static int sp_src_row(sp_state_t *s, const uint16_t *tok, const uint16_t *x, uint32_t run, sp_row_t *row,
                      uint32_t *li_out) {
    row->src_nsrc = (uint8_t)s->ns;
    row->src_since = s->newest_t >= 0 ? (int32_t)(s->t - (uint64_t)s->newest_t) : -1;
    row->src_newlen = sp_sat16(s->newest_len);
    if (!s->ns || run < 2) return 0;
    const uint32_t vmax = run < SP_CAPL ? run : SP_CAPL;
    const uint32_t key = ((uint32_t)x[-1] << 16) | x[0];
    size_t nc = 0;
    uint32_t best = 0;
    for (int i = 0; i < s->ns; i++) {
        const sp_src *S = &s->src[s->order[i]];
        for (int32_t q = S->head[sp_bh(key) & S->hm]; q >= 0; q = S->nxt[q]) {
            const int64_t u = S->a + q;
            if (tok[u] != x[0] || tok[u - 1] != x[-1]) continue;
            const uint16_t nx = tok[u + 1];
            if (nx == SP_SEP || nx == SP_BOS) continue;
            uint32_t lim = vmax;
            if ((uint64_t)(u - S->a + 1) < lim) lim = (uint32_t)(u - S->a + 1);
            uint32_t l = 1;
            while (l < lim && tok[u - l] == x[-(ptrdiff_t)l]) l++;
            if (nc == s->capc) {
                sp_scand *nb = realloc(s->cand, sizeof(sp_scand) * 2 * s->capc);
                if (!nb) break;  // out of memory: keep what was found (never happens at these sizes)
                s->cand = nb;
                s->capc *= 2;
            }
            s->cand[nc].nx = nx;
            s->cand[nc].l = (uint8_t)l;
            nc++;
            if (l > best) best = l;
        }
    }
    if (!nc) return 0;
    uint32_t N[SP_NLV] = {0};
    for (size_t i = 0; i < nc; i++)
        for (int v = 0; v < SP_NLV && s->cand[i].l >= SP_LV[v]; v++) N[v]++;
    uint32_t li = 0;
    for (int v = 0; v < SP_NLV; v++) if (N[v]) li = (uint32_t)v;
    // the distribution at level li: counts per next token (stamped arrays), then the most frequent SP_SRCD
    const uint32_t L = SP_LV[li];
    if (++s->stamp == 0) { memset(s->vstamp, 0, 4u << 16); s->stamp = 1; }
    size_t nt = 0;
    for (size_t i = 0; i < nc; i++) {
        if (s->cand[i].l < L) continue;
        const uint16_t v = s->cand[i].nx;
        if (s->vstamp[v] != s->stamp) {
            s->vstamp[v] = s->stamp;
            s->vcount[v] = 0;
            if (nt == s->ntouched_cap) break;  // cannot happen: at most 65536 distinct tokens
            s->touched[nt++] = v;
        }
        s->vcount[v]++;
    }
    uint32_t mm = 0;
    int nsd = 0;
    uint16_t st[SP_SRCD], sc[SP_SRCD];
    for (size_t i = 0; i < nt; i++) {
        const uint16_t v = s->touched[i];
        const uint32_t cv = s->vcount[v];
        if (cv > mm) mm = cv;
        // insert (cv, v) into the kept list ordered by count desc, then token asc
        int q = nsd < SP_SRCD ? nsd : SP_SRCD;
        if (q == SP_SRCD && (cv < sc[q - 1] || (cv == sc[q - 1] && v > st[q - 1]))) continue;
        if (nsd < SP_SRCD) nsd++;
        q = nsd - 1;
        while (q && (sc[q - 1] < cv || (sc[q - 1] == cv && st[q - 1] > v))) { st[q] = st[q - 1]; sc[q] = sc[q - 1]; q--; }
        st[q] = v;
        sc[q] = (uint16_t)cv;
    }
    for (int v = 0; v < SP_NLV; v++) row->src_n[v] = sp_sat16(N[v]);
    row->src_m = sp_sat16(mm);
    row->src_lbest = (uint8_t)best;
    row->src_li = (uint8_t)li;
    row->nsd = (uint8_t)nsd;
    for (int i = 0; i < nsd; i++) { row->stok[i] = st[i]; row->scnt[i] = sc[i]; }
    if (nt > SP_SRCD) row->flags |= SP_SRC_TRUNC;
    *li_out = li;
    return 1;
}

// ------------------------------------------------------------------------------------------------ tracks

// docstate.c's counters BEFORE t's outcome (the row): running sums over the last K positions / hits, and the C/N
// of the realised target summed over the last 32 hits (summed afresh, most recent first, as docstate.c) ...
static const uint32_t SP_KS[3] = {8, 32, 128};

static double sp_track_h32_r(const sp_track *k) {
    const uint32_t mh = k->nhit < 32 ? k->nhit : 32;
    double r2 = 0;
    for (uint32_t i = 1; i <= mh; i++) r2 += k->hr[(k->nhit - i) % SP_RB];
    return r2;
}

// ... and the update with t's outcome (hit, correct, r; lenl for the long-match counts).
static void sp_track_update(sp_track *k, int h, int c, double r, uint32_t lenl) {
    const uint32_t ip = k->npos % SP_RB;
    for (int ki = 0; ki < 3; ki++) {  // the position leaving each window (read before the ring slot is reused)
        if (k->npos >= SP_KS[ki]) {
            const uint32_t o = (k->npos - SP_KS[ki]) % SP_RB;
            k->p_hit[ki] -= k->ph[o];
            k->p_corr[ki] -= k->pc[o];
        }
    }
    k->ph[ip] = (uint8_t)h;
    k->pc[ip] = (uint8_t)c;
    for (int ki = 0; ki < 3; ki++) { k->p_hit[ki] += (uint32_t)h; k->p_corr[ki] += (uint32_t)c; }
    k->npos++;
    if (h) {
        for (int ki = 0; ki < 3; ki++)
            if (k->nhit >= SP_KS[ki]) k->h_corr[ki] -= k->hc[(k->nhit - SP_KS[ki]) % SP_RB];
        k->hc[k->nhit % SP_RB] = (uint8_t)c;
        k->hr[k->nhit % SP_RB] = r;
        for (int ki = 0; ki < 3; ki++) k->h_corr[ki] += (uint32_t)c;
        k->nhit++;
        k->ncorr += c;
        k->since_hit = 0;
    } else {
        k->since_hit++;
    }
    if (h && !c) k->since_wrong = 0; else k->since_wrong++;
    if (c) k->streak++; else k->streak = 0;
    k->ema90 = 0.9 * k->ema90 + 0.1 * c;
    k->ema98 = 0.98 * k->ema98 + 0.02 * c;
    if (h && lenl >= 32) k->nlong32++;
    if (h && lenl >= 16) k->nlong16++;
    k->prev_corr = c;
    k->prev_hit = h;
}

// The windowed histories (feats.py hist_cols): push t's bits and r.
static const uint32_t SP_WS[3] = {16, 64, 32};

static inline uint64_t sp_spread5(uint8_t b) {  // bit k of b to bit 12 k
    return (uint64_t)(b & 1) | (uint64_t)((b >> 1) & 1) << 12 | (uint64_t)((b >> 2) & 1) << 24 |
           (uint64_t)((b >> 3) & 1) << 36 | (uint64_t)((b >> 4) & 1) << 48;
}
static inline uint32_t sp_field(uint64_t w, int k) { return (uint32_t)(w >> (12 * k)) & 0xFFF; }

static void sp_window_update(sp_state_t *s, uint8_t bits, double r) {
    const uint64_t T = s->t, add = sp_spread5(bits);
    for (int w = 0; w < 3; w++) {
        s->wsum[w] += add;
        if (T >= SP_WS[w]) s->wsum[w] -= sp_spread5(s->wbits[(T - SP_WS[w]) % SP_RB]);
    }
    s->wbits[T % SP_RB] = bits;
    s->wr[T % SP_RB] = r;
}

// The counters of an active row (before t's outcome).
static void sp_snapshot(const sp_state_t *s, sp_row_t *row, int has, int hp, int hs, int cont) {
    const sp_track *k = &s->mem;
    row->d_npos = k->npos; row->d_nhit = k->nhit; row->d_ncorr = k->ncorr; row->d_streak = k->streak;
    row->d_since_wrong = k->since_wrong; row->d_since_hit = k->since_hit;
    row->d_nlong32 = k->nlong32; row->d_nlong16 = k->nlong16;
    row->d_cont_run = k->cont_run; row->d_cont_corr_run = k->cont_corr_run;
    row->d_ema90 = (float)k->ema90; row->d_ema98 = (float)k->ema98; row->d_h32_r = (float)sp_track_h32_r(k);
    for (int i = 0; i < 3; i++) {
        row->d_p_hit[i] = (uint8_t)k->p_hit[i]; row->d_p_corr[i] = (uint8_t)k->p_corr[i];
        row->d_h_corr[i] = (uint8_t)k->h_corr[i];
    }
    // doc-state of the source copy
    row->s_nhit = s->srct.nhit; row->s_ncorr = s->srct.ncorr; row->s_streak = s->srct.streak;
    row->s_h32_r = (float)sp_track_h32_r(&s->srct);
    row->s_h_corr[0] = (uint8_t)s->srct.h_corr[0];
    row->s_h_corr[1] = (uint8_t)s->srct.h_corr[1];
    // windowed histories over positions (bits: has, ret_ok, hp, bptr, either)
    for (int w = 0; w < 2; w++) {
        row->h_r_n[w] = (uint8_t)sp_field(s->wsum[w], 0); row->h_r_c[w] = (uint8_t)sp_field(s->wsum[w], 1);
        row->h_p_n[w] = (uint8_t)sp_field(s->wsum[w], 2); row->h_p_c[w] = (uint8_t)sp_field(s->wsum[w], 3);
        row->h_e[w] = (uint8_t)sp_field(s->wsum[w], 4);
    }
    row->h32_r_n = (uint8_t)sp_field(s->wsum[2], 0); row->h32_p_n = (uint8_t)sp_field(s->wsum[2], 2);
    row->h32_p_c = (uint8_t)sp_field(s->wsum[2], 3);
    {
        const uint64_t T = s->t, m32 = T < 32 ? T : 32;
        double rs = 0;
        for (uint64_t i = 1; i <= m32; i++) rs += s->wr[(T - i) % SP_RB];
        row->h32_r_sum = (float)rs;
    }
    row->h_r_sum_seg = (float)s->h_r_sum_seg;
    row->h_p_n_seg = s->h_p_n_seg; row->h_p_c_seg = s->h_p_c_seg; row->h_e_seg = s->h_e_seg;
    uint16_t flags = row->flags & SP_SRC_TRUNC;
    if (hp) flags |= SP_HP;
    if (hs) flags |= SP_HS;
    if (has) flags |= SP_HAS;
    if (cont) flags |= SP_CONT;
    if (k->npos > 0 && k->prev_corr) flags |= SP_PREV_CORR;
    if (k->npos > 0 && k->prev_hit) flags |= SP_PREV_HIT;
    if (s->srct.npos > 0 && s->srct.prev_corr) flags |= SP_SPREV_CORR;
    row->flags = flags;
}

// ------------------------------------------------------------------------------------------------ one position

SP_API int sp_step(sp_state_t *s, const uint16_t *tok, uint64_t ntok, const uint16_t *x, uint32_t run,
                   const uint32_t *cpos, const uint8_t *clen, int nc, uint16_t y, sp_row_t *row) {
    if (run <= 1) sp_reset(s);
    if (nc > SP_CAP) nc = SP_CAP;
    if (nc < 0) nc = 0;
    memset(row, 0, sizeof *row);
    // the memory record at t (target-independent fields; m.c is y's count, read only after the row)
    sp_mem_t m;
    sp_mem_summary(tok, x, run, cpos, clen, nc, y, &m);
    const int has = nc > 0;
    // pointer beam and vote
    const int hp = sp_beam_row(s, tok, x, run, cpos, clen, nc, row);
    // source copy
    sp_src_discover(s, tok, ntok, &m);
    uint32_t li = 0;
    const int hs = sp_src_row(s, tok, x, run, row, &li);
    // the memory fields
    if (has) {
        row->mem_lenl = sp_sat16(m.len_longest);
        row->mem_lenr = sp_sat16(m.len_recent);
        row->mem_top = m.top;
        row->mem_ncand = m.ncand;
        row->mem_lstar = (uint8_t)SP_LEVELS[m.lstar];
        row->mem_n = (uint8_t)m.n;
        row->mem_m = (uint8_t)m.m;
    }
    // doc-state of the memory: counters before t, and cont (t's own hit and pos_recent against t - 1's)
    sp_track *k = &s->mem;
    const int cont = k->npos > 0 && has && k->prev_hit && m.pos_recent == k->prev_posr + 1;
    if (cont) k->cont_run++; else k->cont_run = 0;
    if (cont && k->prev_corr) k->cont_corr_run++; else k->cont_corr_run = 0;
    k->prev_posr = has ? m.pos_recent : SP_NONE;
    if (hp || hs) sp_snapshot(s, row, has, hp, hs, cont);
    if (!hp) row->ntok = 0;
    if (!hs) row->nsd = 0;
    // ---- t's outcome (y): updates for t + 1 only
    const int corr = has && m.top == y;
    const double r = has ? (double)m.c / (double)m.n : 0.0;
    sp_track_update(k, has, corr, r, m.len_longest);
    {
        int sc = 0;
        double sr = 0;
        if (hs) {
            const uint32_t cy = (s->vstamp[y] == s->stamp && y != SP_BOS) ? s->vcount[y] : 0;
            sc = cy == row->src_m && cy > 0;
            sr = (double)cy / (double)row->src_n[li];
        }
        sp_track_update(&s->srct, hs, sc, sr, 0);
    }
    {
        const int bptr = hp && row->pred == y;
        sp_window_update(s, (uint8_t)(has | (corr << 1) | (hp << 2) | (bptr << 3) | ((corr | bptr) << 4)), r);
        s->h_p_n_seg += hp;
        s->h_p_c_seg += bptr;
        s->h_e_seg += corr | bptr;
        s->h_r_sum_seg += r;
    }
    sp_beam_update(s, tok, ntok, x, run, y);
    s->t++;
    return hp || hs;
}

// ------------------------------------------------------------------------------------------------ the driver

typedef int (*sp_cand_fn)(void *ctx, uint64_t t, const uint16_t *x, uint32_t run, uint32_t *pos, uint8_t *len);

// Candidates from CSR arrays: position t's are pos[off[t] .. off[t + 1]) (walk order), len likewise.
typedef struct { const uint64_t *off; const uint32_t *pos; const uint8_t *len; } sp_csr_t;

SP_API int sp_cands_csr(void *ctx, uint64_t t, const uint16_t *x, uint32_t run, uint32_t *pos, uint8_t *len) {
    (void)x; (void)run;
    const sp_csr_t *c = ctx;
    uint64_t a = c->off[t], n = c->off[t + 1] - a;
    if (n > SP_CAP) n = SP_CAP;
    memcpy(pos, c->pos + a, 4 * n);
    memcpy(len, c->len + a, n);
    return (int)n;
}

typedef struct { uint64_t lo, hi; sp_row_t *rows; uint64_t nrows; } sp_block_t;

typedef struct {
    const sp_config_t *cfg;
    const uint16_t *tok, *x, *y;
    uint64_t ntok;
    const uint32_t *run;
    sp_cand_fn cands;
    void *ctx;
    sp_block_t *blocks;
    uint64_t *order, nblocks;
    atomic_uint_fast64_t next;
    atomic_int failed;
} sp_job_t;

static void *sp_worker(void *arg) {
    sp_job_t *J = arg;
    sp_state_t *st = sp_state_new(J->cfg);
    sp_row_t *scratch = NULL;
    uint64_t scap = 0;
    if (!st) { atomic_store(&J->failed, 1); return NULL; }
    uint32_t cpos[SP_CAP];
    uint8_t clen[SP_CAP];
    for (;;) {
        const uint64_t k = atomic_fetch_add(&J->next, 1);
        if (k >= J->nblocks || atomic_load(&J->failed)) break;
        sp_block_t *B = &J->blocks[J->order[k]];
        const uint64_t n = B->hi - B->lo;
        if (n > scap) {
            free(scratch);
            scap = n;
            scratch = malloc(sizeof(sp_row_t) * scap);
            if (!scratch) { atomic_store(&J->failed, 1); break; }
        }
        uint64_t nr = 0;
        for (uint64_t t = B->lo; t < B->hi; t++) {
            const uint32_t run = J->run[t];
            const int nc = run >= SP_KEY ? J->cands(J->ctx, t, J->x + t, run, cpos, clen) : 0;
            const uint16_t y = J->y ? J->y[t] : J->x[t + 1];
            if (sp_step(st, J->tok, J->ntok, J->x + t, run, cpos, clen, nc, y, &scratch[nr])) {
                scratch[nr].pos = (uint32_t)t;
                nr++;
            }
        }
        B->nrows = nr;
        B->rows = nr ? malloc(sizeof(sp_row_t) * nr) : NULL;
        if (nr && !B->rows) { atomic_store(&J->failed, 1); break; }
        if (nr) memcpy(B->rows, scratch, sizeof(sp_row_t) * nr);
    }
    free(scratch);
    sp_state_free(st);
    return NULL;
}

static uint64_t *sp_sort_key;  // block lengths for the LPT order (qsort has no context argument)
static int sp_cmp_len_desc(const void *a, const void *b) {
    const uint64_t x = sp_sort_key[*(const uint64_t *)a], y = sp_sort_key[*(const uint64_t *)b];
    if (x != y) return x < y ? 1 : -1;
    const uint64_t i = *(const uint64_t *)a, j = *(const uint64_t *)b;
    return i < j ? -1 : i > j;
}
static pthread_mutex_t sp_sort_lock = PTHREAD_MUTEX_INITIALIZER;

static void sp_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

SP_API void sp_free(void *p) { free(p); }

// The whole sequence 0..n-1: x[t] its input (x[t + 1] must exist when y is NULL: y[t] = x[t + 1]); run[t] = t's
// segment length up to and including t (saturating at 4096; run == 1 starts a segment), or NULL: from x (a segment
// starts at t = 0, at every BOS and at every multiple of chunk; chunk 0 = no chunks). Segments are grouped into
// blocks of >= 4096 positions (whole segments), processed longest first on `threads` threads. Returns the number of
// active rows (*rows: malloc'd, in position order; free with sp_free) or -1 (err).
SP_API int64_t sp_run(const sp_config_t *cfg, const uint16_t *tok, uint64_t ntok, const uint16_t *x, const uint16_t *y,
                      uint64_t n, const uint32_t *run, uint64_t chunk, sp_cand_fn cands, void *ctx, int threads,
                      sp_row_t **rows, char *err, size_t errlen) {
    *rows = NULL;
    if (n >= (1ull << 32)) { sp_err(err, errlen, "n = %llu positions do not fit a row's pos", (unsigned long long)n); return -1; }
    sp_config_t c;
    if (cfg) c = *cfg; else sp_default_config(&c);
    sp_state_t *probe = sp_state_new(&c);
    if (!probe) { sp_err(err, errlen, "bad configuration or out of memory"); return -1; }
    sp_state_free(probe);
    uint32_t *own_run = NULL;
    if (!run) {
        own_run = malloc(4 * (n ? n : 1));
        if (!own_run) { sp_err(err, errlen, "out of memory"); return -1; }
        uint32_t r = 0;
        for (uint64_t t = 0; t < n; t++) {
            const int start = t == 0 || x[t] == SP_BOS || (chunk && t % chunk == 0);
            r = start ? 1 : (r < SP_FULLCAP ? r + 1 : SP_FULLCAP);
            own_run[t] = r;
        }
        run = own_run;
    } else if (n && run[0] != 1) {
        sp_err(err, errlen, "run[0] = %u: the sequence must start a segment", run[0]);
        return -1;
    }
    // blocks: whole segments, at least 4096 positions each (the last may be shorter)
    uint64_t nb = 0, cap = 16;
    sp_block_t *blocks = malloc(sizeof(sp_block_t) * cap);
    for (uint64_t t = 0; blocks && t < n;) {
        uint64_t e = t + 1;
        while (e < n && (e - t < 4096 || run[e] != 1)) e++;
        if (nb == cap) {
            sp_block_t *nbk = realloc(blocks, sizeof(sp_block_t) * 2 * cap);
            if (!nbk) { free(blocks); blocks = NULL; break; }
            blocks = nbk;
            cap *= 2;
        }
        blocks[nb].lo = t; blocks[nb].hi = e; blocks[nb].rows = NULL; blocks[nb].nrows = 0;
        nb++;
        t = e;
    }
    uint64_t *order = blocks ? malloc(8 * (nb ? nb : 1)) : NULL, *lens = order ? malloc(8 * (nb ? nb : 1)) : NULL;
    if (!blocks || !order || !lens) {
        free(blocks); free(order); free(lens); free(own_run);
        sp_err(err, errlen, "out of memory");
        return -1;
    }
    for (uint64_t i = 0; i < nb; i++) { order[i] = i; lens[i] = blocks[i].hi - blocks[i].lo; }
    pthread_mutex_lock(&sp_sort_lock);
    sp_sort_key = lens;
    qsort(order, nb, 8, sp_cmp_len_desc);
    pthread_mutex_unlock(&sp_sort_lock);
    free(lens);
    sp_job_t J = {.cfg = &c, .tok = tok, .x = x, .y = y, .ntok = ntok, .run = run, .cands = cands, .ctx = ctx,
                  .blocks = blocks, .order = order, .nblocks = nb};
    atomic_init(&J.next, 0);
    atomic_init(&J.failed, 0);
    int nt = threads < 1 ? 1 : (threads > 256 ? 256 : threads);
    pthread_t th[256];
    int started = 1;
    for (int i = 1; i < nt; i++) {
        if (pthread_create(&th[i], NULL, sp_worker, &J)) break;
        started++;
    }
    sp_worker(&J);
    for (int i = 1; i < started; i++) pthread_join(th[i], NULL);
    int64_t total = -1;
    if (!atomic_load(&J.failed)) {
        uint64_t tot = 0;
        for (uint64_t i = 0; i < nb; i++) tot += blocks[i].nrows;
        sp_row_t *out = malloc(sizeof(sp_row_t) * (tot ? tot : 1));
        if (out) {
            uint64_t o = 0;
            for (uint64_t i = 0; i < nb; i++) {
                if (blocks[i].nrows) memcpy(out + o, blocks[i].rows, sizeof(sp_row_t) * blocks[i].nrows);
                o += blocks[i].nrows;
            }
            *rows = out;
            total = (int64_t)tot;
        } else {
            sp_err(err, errlen, "out of memory");
        }
    } else {
        sp_err(err, errlen, "a worker ran out of memory");
    }
    for (uint64_t i = 0; i < nb; i++) free(blocks[i].rows);
    free(blocks);
    free(order);
    free(own_run);
    return total;
}

// ------------------------------------------------------------------------------------------------ layout

SP_API const char *sp_layout(void) {
    static char buf[4096];
    if (!buf[0]) {
        size_t o = 0;
#define SP_L(name) o += (size_t)snprintf(buf + o, sizeof buf - o, "%s %zu %zu,", #name, offsetof(sp_row_t, name), sizeof(((sp_row_t *)0)->name));
        SP_L(pos) SP_L(flags) SP_L(ntok) SP_L(nsd) SP_L(ptr_j) SP_L(score) SP_L(score2) SP_L(share) SP_L(share2)
        SP_L(vshare) SP_L(pred) SP_L(vtop) SP_L(tok2) SP_L(run) SP_L(hits) SP_L(miss) SP_L(age) SP_L(since)
        SP_L(seedlen) SP_L(nrec) SP_L(c16) SP_L(n16) SP_L(c64) SP_L(nhist) SP_L(how) SP_L(nvh) SP_L(tcnt) SP_L(vtok)
        SP_L(vw) SP_L(src_n) SP_L(src_m) SP_L(src_newlen) SP_L(src_since) SP_L(src_lbest) SP_L(src_nsrc) SP_L(src_li)
        SP_L(stok) SP_L(scnt) SP_L(mem_lenl) SP_L(mem_lenr) SP_L(mem_top) SP_L(mem_ncand) SP_L(mem_lstar) SP_L(mem_n)
        SP_L(mem_m) SP_L(d_npos) SP_L(d_nhit) SP_L(d_ncorr) SP_L(d_streak) SP_L(d_since_wrong) SP_L(d_since_hit)
        SP_L(d_nlong32) SP_L(d_nlong16) SP_L(d_cont_run) SP_L(d_cont_corr_run) SP_L(d_ema90) SP_L(d_ema98)
        SP_L(d_h32_r) SP_L(d_p_hit) SP_L(d_p_corr) SP_L(d_h_corr) SP_L(s_nhit) SP_L(s_ncorr) SP_L(s_streak)
        SP_L(s_h32_r) SP_L(s_h_corr) SP_L(h_r_n) SP_L(h_r_c) SP_L(h_p_n) SP_L(h_p_c) SP_L(h_e) SP_L(h32_r_n)
        SP_L(h32_p_n) SP_L(h32_p_c) SP_L(h32_r_sum) SP_L(h_r_sum_seg) SP_L(h_p_n_seg) SP_L(h_p_c_seg) SP_L(h_e_seg)
#undef SP_L
        o += (size_t)snprintf(buf + o, sizeof buf - o, "sizeof %zu %zu", sizeof(sp_row_t), sizeof(sp_row_t));
    }
    return buf;
}

SP_API uint64_t sp_row_bytes(void) { return sizeof(sp_row_t); }

#endif  // STREAM_POINTER_C
