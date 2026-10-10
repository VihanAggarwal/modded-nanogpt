// Exact low-order count tables of the run's trained stream (STREAM_RETRIEVAL_LOW=1; part P2 of stream retrieval v2).
//
// For every order k in ORDERS (default 1..5) and every context c of k tokens, the tables hold the exact statistics of
// the next tokens that followed c in the stream this run trained on:
//   N   positions with context c              M    the largest count of one next token
//   D   distinct next tokens                  n1, n2  next tokens seen exactly once / twice (modified-KN discounts)
//   top the most frequent next token (ties: the lowest id), and C(y) = the count of a given next token y
// and a query returns, for a val position t (input x[t], target y = x[t+1]) and each order, the row
// (N, C(y), M, D, n1, n2, top) of the context x[t-k+1..t], all zero where that context never occurred. The gated chain
// over orders that turns these rows into a distribution (KN components, stick-breaking gates) runs in the untimed eval
// (stream_memory.py); this file only counts.
//
// What is counted (the conventions of the research tools rg/loworders/ngcount.c and lotable.c, which this ports):
//   memory   the stream as the helper holds it: every rank's document spans, each followed by a separator (SEP).
//            Position j counts for order k when its k-token context x[j-k+1..j] lies in one document of one span
//            (since the span's start and since the last BOS; a context may start with that BOS) and its next token
//            x[j+1] is in the same document (not SEP, not BOS). So no distribution ever puts mass on BOS.
//   query    the context of t lies in t's segment: since the last BOS at or before t and since the start of t's
//            chunk (the val loader's 262,144-token windows), the context the model itself sees.
//
// Tables. Per order, open-addressing hash tables split into 64 partitions by the context's hash, so a context and
// everything about it live in one partition. Slots sit in 64-byte buckets (one cache line), probed linearly bucket by
// bucket:
//   stats  (fp u32, N u32, M u32, top u16, D u16, n1 u16, n2 u16)   20 bytes, 3 per bucket (+ an overflow flag)
//   pair   (fp u32, count u32)                                      8 bytes, 8 per bucket; keyed by (context, next)
// The count of a context's top token lives in its stats slot (M), not in the pair table: a pair entry exists only for
// a next token that is, or once was, not the top. Most contexts have one distinct next token, so most need no pair.
// Orders >= tier_from (default 4) add a first tier, since 90% or more of their contexts occur once:
//   index  (fp u32, top u16, flags u16)                             8 bytes, 8 per bucket
// Every context has an index slot; a context seen once is that slot alone (N = M = D = n1 = 1, the top); its second
// occurrence promotes it to a stats slot (flag PROMOTED). On the 1050-step stream this takes the tables from 21.6 GB
// to ~14 GB and makes most insertions and missed lookups touch one line.
// Update for one position (context c, next v), on c's stats slot:
//   new c:     N = M = D = n1 = 1, top = v                (tiered: the index slot; promoted on the second occurrence)
//   v == top:  N++, c_v = ++M
//   else:      N++, c_v = ++pair[c, v]; if c_v > M or (c_v == M and v < top): v becomes the top (pair[c, v] = 0, its
//              count moves to M; the old top's count M moves to pair[c, old top])
//   then       c_v == 1: D++, n1++;  c_v == 2: n1--, n2++;  c_v == 3: n2--
// so every field is exact at all times (up to ~60-bit hash collisions: a 32-bit fingerprint per slot on top of the
// partition and home-bucket bits). C(y) = M when y == top, else pair[c, y] (0 if absent). A bucket's overflow flag
// says some key went past it while it was full, so a lookup that misses in a full bucket without the flag stops.
//
// Insertion on the clock. lt_insert_block() takes one step's tokens (whole spans with SEP between them, as the helper
// appends them), copies them and returns; `threads` insertion threads (default 8) take every block in order, in two
// phases: each hashes its own slice of the block's positions (all orders) and hands each item to the thread that owns
// its partition (partition % threads); after a barrier, each applies the items it owns, slice by slice, so in stream
// order. Tables need no locks, and every partition sees its updates in stream order whatever the thread count or the
// block boundaries, so the tables (their very bytes, see lt_digest) and every row are identical across thread counts
// and batchings. Each thread hashes a group of its items, prefetches their buckets, peeks and prefetches the next
// level, then updates them in order (the DRAM latency overlaps). lt_finish() waits until every block is in and
// freezes the tables: the helper calls it at GO, before its val queries (the completion check).
//
// FIT positions (STREAM_RETRIEVAL_FIT, dev runs): the gate's constants are fitted on the run's OWN last K timed
// batches, queried against the memory as it stood BEFORE those batches were inserted. lt_hold(1) before the first
// of them: their blocks are queued but not inserted; lt_sync(), query them, then lt_hold(0) releases them in order
// (lt_finish() refuses to run while blocks are held). The final tables are the same as without the hold.
//
// Sizing. Tables are fixed-size and allocated (and, with prefault, touched by their owning threads, which also puts
// pages on the owner's NUMA node) at lt_create(), before the clock. Expected entries come from lt_default_entries()
// (measured on the real 1050-step stream and a 30M-token prefix of it) or the caller; slots = entries x a partition
// skew allowance / load (0.75). A partition that reaches ~98% load stops the insertion with an error (never silently
// wrong, never an endless probe); lt_stats() reports every table's fullest partition.
//
// API (all return 0 / non-NULL on success; lt_error() has the message):
//   lt_t *lt_create(const lt_config_t *cfg, char *err, size_t errlen)
//   int   lt_insert_block(lt_t *, const uint16_t *tok, size_t n)   copies; asynchronous; blocks are whole spans
//   int   lt_hold(lt_t *, int on)                                  hold new blocks / release the held ones
//   int   lt_sync(lt_t *)                                          wait until every released block is in
//   int   lt_finish(lt_t *)                                        sync, then freeze (no more blocks): before GO
//   int   lt_query_rows(const lt_t *, x, y, n, chunk, out, threads)  rows of positions 0..n-1, [n][norders]
//   void  lt_query_block(const lt_t *, x, y, run, n, out)          the same for a block, with the caller's run[]
//   void  lt_query_one(const lt_t *, const uint16_t *x, uint16_t y, uint32_t run, lt_row_t *out)
//   int   lt_stats(const lt_t *, uint64_t *words, int nwords)       see LT_S_* below
//   uint64_t lt_digest(lt_t *), int lt_census(lt_t *, k, out[4])   tests and diagnostics
//   const char *lt_error(const lt_t *);  void lt_destroy(lt_t *)
// Queries are read-only and thread-safe, valid once lt_sync() or lt_finish() returned 0 and while nothing is being
// inserted (held blocks are fine). Built into the helper (compile it with stream_memory.c, or #include it there: every
// internal name is lt_-prefixed and static) or as a shared library for stream_lowtables.py (cc -O2 -std=c11 -pthread
// -shared -fPIC).
//
// In the helper (stream_memory.c), with STREAM_RETRIEVAL_LOW=1:
//   start (before READY):  cfg = {norders 5, orders 1..5, threads 8, prefault 1, expected_positions = the schedule's
//                          stream tokens}; lt = lt_create(&cfg, err, sizeof err) (allocation and first touch, before
//                          the clock: ~14 GB for the 1050-step stream)
//   each STEP message:     after the step's spans are appended to tok[] at [start, entries) (each span then SEP),
//                          lt_insert_block(lt, tok + start, entries - start): a copy, queued; it returns at once
//   FIT (dev runs):        lt_hold(lt, 1) before the first of the last K steps; once they have arrived, lt_sync(lt),
//                          query their positions, lt_hold(lt, 0)
//   GO:                    lt_finish(lt) (every step in, or the error), then per query block lt_query_block(lt, x, y,
//                          run, n, out) with the block's run[] (or lt_query_rows on its own threads): 5 rows of 20
//                          bytes per val position, written next to the stream memory's rows
//
// Credits. The recipe is PR #380's (Deven): exact n-gram counts of the training data at low orders, a gated chain over
// the orders (stick-breaking interpolation with KN-style components), its gate fitted on training positions. Here the
// counts are of this run's consumed stream only, built on the clock by the helper. No code from #380: this file ports
// this branch's research tools rg/loworders/lotable.c and ngcount.c (incremental tables, partition-owning threads). The
// mixer is splitmix64's finalizer (public domain); linear probing, count-of-counts (Chen & Goodman, 1998) and a
// bucket overflow flag (as in Folly's F14) are textbook.
#ifndef STREAM_LOWTABLES_C  // the helper (stream_memory.c) #includes this file; a second #include is a no-op
#define STREAM_LOWTABLES_C
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef LT_API
#define LT_API __attribute__((visibility("default")))
#endif

#define LT_SEP 0xFFFFu
#define LT_BOS 50256u
#define LT_MAX_ORDERS 8         // orders per table set
#define LT_MAX_K 8              // the largest order supported
#define LT_TIER_FROM 4          // default: orders >= 4 have the index tier
#define LT_PBITS 6
#define LT_NPART (1 << LT_PBITS)
#define LT_MAX_THREADS LT_NPART
#ifndef LT_GROUP
#define LT_GROUP 32             // insertion: items hashed and prefetched together
#endif
#define LT_RING 32              // the query pipeline's ring (a power of two > LT_D1 + LT_D2 + LT_D3)
#define LT_D1 10                // queries: lookups between a first-level prefetch and its read
#define LT_D2 6                 // ... between the second level's prefetch (stats or pair) and its read
#define LT_D3 6                 // ... between the third level's prefetch (a tiered context's pair) and its read
#define LT_QBLOCK 4096          // lt_query_rows: positions per work item
#define LT_RUN_SAT 255

// lt_stats() words
enum { LT_S_NORDERS, LT_S_THREADS, LT_S_BLOCKS, LT_S_HELD, LT_S_TOKENS, LT_S_BUSY_NS, LT_S_BYTES, LT_S_STATE,
       LT_S_FINISH_NS, LT_S_CPU_NS, LT_S_PENDING, LT_S_ORDER0 };
enum { LT_SO_K, LT_SO_TIERED, LT_SO_POSITIONS, LT_SO_CONTEXTS, LT_SO_PROMOTED, LT_SO_PAIRS, LT_SO_IDX_SLOTS,
       LT_SO_CTX_SLOTS, LT_SO_PAIR_SLOTS, LT_SO_IDX_MAXLOAD_PPM, LT_SO_CTX_MAXLOAD_PPM, LT_SO_PAIR_MAXLOAD_PPM,
       LT_SO_BYTES, LT_SO_WORDS };
#define LT_STATS_WORDS (LT_S_ORDER0 + LT_MAX_ORDERS * LT_SO_WORDS)
enum { LT_ST_FINISHED = 1, LT_ST_FAILED = 2 };

typedef struct {
    int32_t norders;
    int32_t orders[LT_MAX_ORDERS];      // ascending, each in 1..LT_MAX_K
    int32_t threads;                    // insertion threads (partition owners), 1..64
    int32_t prefault;                   // touch every table page now (before the clock), each on its owning thread
    int32_t nice;                       // the insertion threads' nice value (0: inherit)
    int32_t tier_from;                  // orders >= this have the index tier (0: LT_TIER_FROM; > LT_MAX_K: none)
    int32_t reserved;
    uint64_t expected_positions;        // the stream's tokens, for the default sizes (lt_default_entries)
    uint64_t ctx_entries[LT_MAX_ORDERS];       // expected contexts per order (0: default)
    uint64_t pair_entries[LT_MAX_ORDERS];      // expected pair entries per order (0: default)
    uint64_t promoted_entries[LT_MAX_ORDERS];  // tiered orders: expected contexts seen twice or more (0: default)
    double load;                        // slots = entries x skew / load (<= 0: 0.75)
} lt_config_t;

typedef struct { uint32_t N, C, M; uint16_t D, n1, n2, top; } lt_row_t;  // 20 bytes; all zero: context unseen

typedef struct { uint32_t fp, cnt; } lt_pair_slot;
typedef struct __attribute__((packed)) { uint32_t fp, N, M; uint16_t top, D, n1, n2; } lt_ctx_slot;
typedef struct { uint32_t fp; uint16_t top, st; } lt_idx_slot;
#define LT_CB 3
#define LT_PB 8
#define LT_IB 8
#define LT_IDX_PROMOTED 1u      // lt_idx_slot.st: the context's statistics are in its stats slot
#define LT_IDX_OVER 0x8000u     // s[0].st of an index bucket: the bucket's overflow flag
// over: some context passed this bucket, full, to a later one; a lookup that misses in a full bucket without it is done
typedef struct __attribute__((aligned(64))) { lt_ctx_slot s[LT_CB]; uint32_t over; } lt_ctx_bucket;
typedef struct __attribute__((aligned(64))) { lt_pair_slot s[LT_PB]; } lt_pair_bucket;
typedef struct __attribute__((aligned(64))) { lt_idx_slot s[LT_IB]; } lt_idx_bucket;
_Static_assert(sizeof(lt_ctx_slot) == 20 && sizeof(lt_ctx_bucket) == 64 && sizeof(lt_pair_bucket) == 64 &&
               sizeof(lt_idx_bucket) == 64, "buckets");
_Static_assert(sizeof(lt_row_t) == 20, "row");
_Static_assert(LT_RING > LT_D1 + LT_D2 + LT_D3 + 1 && (LT_RING & (LT_RING - 1)) == 0, "pipeline ring");

typedef struct {
    lt_idx_bucket *ib;                  // tiered orders only
    lt_ctx_bucket *cb;
    lt_pair_bucket *pb;
    uint32_t inb, cnb, pnb;             // buckets
    uint32_t iused, cused, pused;       // entries
    uint32_t ilim, clim, plim;          // the most allowed (~98% of the slots)
} lt_part_t;

typedef struct {
    int k, tiered;
    void *imap, *cmap, *pmap;
    size_t ibytes, cbytes, pbytes;
    lt_part_t part[LT_NPART];
} lt_order_t;

typedef struct lt_block {
    size_t n;
    atomic_int refs;
    uint16_t tok[];
} lt_block_t;

typedef struct lt lt_t;

typedef struct { uint64_t acc; uint16_t v; uint8_t oi, k; } lt_item_t;  // 16 bytes: the scan's stores stay cheap
typedef struct { lt_item_t *a; size_t n, cap; } lt_items_t;

typedef struct __attribute__((aligned(128))) {  // a line pair of its own: no false sharing between threads
    lt_t *lt;
    int id;
    pthread_t thread;
    uint64_t done;                      // blocks processed (guarded by lt->mu, as the counters below)
    uint64_t busy_ns, cpu_ns;
    uint64_t positions[LT_MAX_ORDERS];
    lt_items_t out[2][LT_MAX_THREADS];  // [block parity][owner]: the items of this thread's slice, by owner
} lt_thread_t;

struct lt {
    int norders, nthreads, kmin, kmax, prefault, nice;
    int8_t oidx[LT_MAX_K + 1];          // order k -> index in orders[], or -1
    uint8_t owner[LT_NPART];
    lt_order_t ord[LT_MAX_ORDERS];
    lt_thread_t th[LT_MAX_THREADS];
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    pthread_barrier_t bar;              // between a block's two phases
    lt_block_t **blocks;                // by sequence number; NULL once every thread is done with it
    uint64_t nblocks, released, blocks_alloc, tokens, ready;
    uint64_t t_last_insert, finish_ns;
    int hold, stop, finished;
    atomic_int failed;                  // 1: an error (err); 2: lt_destroy draining
    char err[256];
};

// ------------------------------------------------------------------------------------------------ hashing

#define LT_G 0x9E3779B97F4A7C15ull
#define LT_SEED 0x2545F4914F6CDD1Dull

static inline uint64_t lt_mix64(uint64_t h) {  // splitmix64's finalizer
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 27; h *= 0x94d049bb133111ebull; return h ^ (h >> 31);
}
// The rolling context accumulator, newest token first: acc_1 = step(SEED, x[t]), acc_k = step(acc_{k-1}, x[t-k+1]).
// Each step is a bijection of acc for a given token, so contexts of one order differ unless 64-bit values collide.
// The partition is acc's top bits (the product's: they depend on every token); the slot and fingerprint come from a
// full mix of acc.
static inline uint64_t lt_step(uint64_t a, uint16_t w) { a = (a ^ w) * LT_G; return a ^ (a >> 29); }
static inline uint32_t lt_part_of(uint64_t acc) { return (uint32_t)(acc >> (64 - LT_PBITS)); }
static inline uint64_t lt_ctx_hash(uint64_t acc) { return lt_mix64(acc ^ 0x632BE59BD9B4E019ull); }
static inline uint32_t lt_fp(uint64_t h) { return (uint32_t)(h >> 32) | 1u; }  // never 0 (= empty)
static inline uint64_t lt_pair_hash(uint64_t h, uint16_t v) {
    return lt_mix64(h ^ (((uint64_t)v + 1) * 0xD6E8FEB86659FD93ull));
}
static inline uint32_t lt_home(uint64_t h, uint32_t nb) { return (uint32_t)(((uint64_t)(uint32_t)h * nb) >> 32); }

static uint64_t lt_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t lt_cpu_ns(void) {  // this thread's CPU time
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void lt_fail(lt_t *lt, const char *fmt, ...) {
    int zero = 0;
    if (!atomic_compare_exchange_strong(&lt->failed, &zero, 1)) return;  // the first error wins
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(lt->err, sizeof lt->err, fmt, ap);
    va_end(ap);
}

// ------------------------------------------------------------------------------------------------ table operations

static inline lt_ctx_slot *lt_ctx_find(const lt_part_t *P, uint64_t h) {
    uint32_t b = lt_home(h, P->cnb);
    const uint32_t fp = lt_fp(h);
    for (;;) {
        lt_ctx_bucket *B = &P->cb[b];
        for (int i = 0; i < LT_CB; i++) {
            if (B->s[i].fp == fp) return &B->s[i];
            if (!B->s[i].fp) return NULL;
        }
        if (!B->over) return NULL;
        if (++b == P->cnb) b = 0;
    }
}

// Find or create (fingerprint set, the rest zero) c's stats slot; NULL when the table is full.
static inline lt_ctx_slot *lt_ctx_get(lt_t *lt, lt_part_t *P, uint64_t h, int k, int *created) {
    uint32_t b = lt_home(h, P->cnb);
    const uint32_t fp = lt_fp(h);
    for (;;) {
        lt_ctx_bucket *B = &P->cb[b];
        for (int i = 0; i < LT_CB; i++) {
            lt_ctx_slot *s = &B->s[i];
            if (s->fp == fp) { *created = 0; return s; }
            if (!s->fp) {
                if (P->cused >= P->clim) {
                    lt_fail(lt, "the order-%d stats table is full (%u entries in a partition of %u slots): raise "
                            "%s", k, P->cused, P->cnb * LT_CB, P->ib ? "promoted_entries" : "ctx_entries");
                    return NULL;
                }
                P->cused++;
                s->fp = fp;
                *created = 1;
                return s;
            }
        }
        if (!B->over) B->over = 1;  // full: whatever is placed beyond it passed it (written once: lines stay clean)
        if (++b == P->cnb) b = 0;
    }
}

static inline const lt_idx_slot *lt_idx_find(const lt_part_t *P, uint64_t h) {
    uint32_t b = lt_home(h, P->inb);
    const uint32_t fp = lt_fp(h);
    for (;;) {
        const lt_idx_bucket *B = &P->ib[b];
        for (int i = 0; i < LT_IB; i++) {
            if (B->s[i].fp == fp) return &B->s[i];
            if (!B->s[i].fp) return NULL;
        }
        if (!(B->s[0].st & LT_IDX_OVER)) return NULL;
        if (++b == P->inb) b = 0;
    }
}

static inline uint32_t lt_pair_get(const lt_part_t *P, uint64_t hp) {
    uint32_t b = lt_home(hp, P->pnb);
    const uint32_t fp = lt_fp(hp);
    for (;;) {
        const lt_pair_bucket *B = &P->pb[b];
        for (int i = 0; i < LT_PB; i++) {
            if (B->s[i].fp == fp) return B->s[i].cnt;
            if (!B->s[i].fp) return 0;
        }
        if (++b == P->pnb) b = 0;
    }
}

static inline lt_pair_slot *lt_pair_upsert(lt_t *lt, lt_part_t *P, uint64_t hp, int k) {
    uint32_t b = lt_home(hp, P->pnb);
    const uint32_t fp = lt_fp(hp);
    for (;;) {
        lt_pair_bucket *B = &P->pb[b];
        for (int i = 0; i < LT_PB; i++) {
            lt_pair_slot *s = &B->s[i];
            if (s->fp == fp) return s;
            if (!s->fp) {
                if (P->pused >= P->plim) {
                    lt_fail(lt, "the order-%d pair table is full (%u entries in a partition of %u slots): raise "
                            "pair_entries", k, P->pused, P->pnb * LT_PB);
                    return NULL;
                }
                P->pused++;
                s->fp = fp;
                s->cnt = 0;
                return s;
            }
        }
        if (++b == P->pnb) b = 0;
    }
}

// One more next token v of c on its stats slot s (N >= 1 already). Returns 0, or -1 when a table is full.
static inline int lt_count(lt_t *lt, lt_part_t *P, lt_ctx_slot *s, uint64_t h, uint16_t v, int k) {
    if (s->N == UINT32_MAX) { lt_fail(lt, "an order-%d context count overflowed", k); return -1; }
    s->N++;
    uint32_t c;
    if (v == s->top) {
        c = ++s->M;
    } else {
        lt_pair_slot *ps = lt_pair_upsert(lt, P, lt_pair_hash(h, v), k);
        if (!ps) return -1;
        c = ++ps->cnt;
        if (c > s->M || (c == s->M && v < s->top)) {  // v becomes the top: swap where the two counts live
            const uint16_t old = s->top;
            const uint32_t oc = s->M;
            ps->cnt = 0;
            s->top = v;
            s->M = c;
            lt_pair_slot *po = lt_pair_upsert(lt, P, lt_pair_hash(h, old), k);
            if (!po) return -1;
            po->cnt = oc;
        }
    }
    if (c == 1) { s->D++; s->n1++; }
    else if (c == 2) { s->n1--; s->n2++; }
    else if (c == 3) s->n2--;
    return 0;
}

// One position (context hash h in partition P, next token v) of an order without the index tier.
static inline int lt_update(lt_t *lt, lt_part_t *P, uint64_t h, uint16_t v, int k) {
    int created;
    lt_ctx_slot *s = lt_ctx_get(lt, P, h, k, &created);
    if (!s) return -1;
    if (created) { s->N = 1; s->M = 1; s->top = v; s->D = 1; s->n1 = 1; s->n2 = 0; return 0; }
    return lt_count(lt, P, s, h, v, k);
}

// One position of a tiered order: a new context is an index slot; its second occurrence promotes it to a stats slot.
static inline int lt_update_tiered(lt_t *lt, lt_part_t *P, uint64_t h, uint16_t v, int k) {
    uint32_t b = lt_home(h, P->inb);
    const uint32_t fp = lt_fp(h);
    lt_idx_slot *x;
    for (;;) {
        lt_idx_bucket *B = &P->ib[b];
        for (int i = 0; i < LT_IB; i++) {
            x = &B->s[i];
            if (x->fp == fp) goto found;
            if (!x->fp) {
                if (P->iused >= P->ilim) {
                    lt_fail(lt, "the order-%d index table is full (%u entries in a partition of %u slots): raise "
                            "ctx_entries", k, P->iused, P->inb * LT_IB);
                    return -1;
                }
                P->iused++;
                x->fp = fp;
                x->top = v;  // st stays as it is: 0, or the bucket's flag in s[0] (never set on a bucket with room)
                return 0;
            }
        }
        if (!(B->s[0].st & LT_IDX_OVER)) B->s[0].st |= LT_IDX_OVER;
        if (++b == P->inb) b = 0;
    }
found:;
    int created;
    lt_ctx_slot *s = lt_ctx_get(lt, P, h, k, &created);
    if (!s) return -1;
    if (!(x->st & LT_IDX_PROMOTED)) {  // the second occurrence: the stats slot starts from the first one's token
        x->st |= LT_IDX_PROMOTED;
        if (created) { s->N = 1; s->M = 1; s->top = x->top; s->D = 1; s->n1 = 1; s->n2 = 0; }
    }
    return lt_count(lt, P, s, h, v, k);
}

// ------------------------------------------------------------------------------------------------ insertion

// Insertion runs over thread p's items (one per position and order) in stream order, LT_GROUP at a time: hash every
// item's context and prefetch its first bucket (index or stats); peek at it (in cache by now) and prefetch what the
// update will read next: the pair bucket (the context is there and its top is not v; a new context or v == top reads
// no pair), or a promoted context's stats bucket, whose peek then prefetches the pair; then update them one by one, in
// order, so the result is the plain sequential one.
static inline void lt_peek_stats(const lt_part_t *P, uint64_t h, uint16_t v) {
    const uint32_t b = lt_home(h, P->cnb);
    const lt_ctx_bucket *B = &P->cb[b];
    const uint32_t fp = lt_fp(h);
    for (int j = 0; j < LT_CB; j++) {
        if (B->s[j].fp == fp) {
            if (B->s[j].top != v) __builtin_prefetch(&P->pb[lt_home(lt_pair_hash(h, v), P->pnb)], 1);
            return;
        }
        if (!B->s[j].fp) return;
    }
    __builtin_prefetch(&P->cb[b + 1 < P->cnb ? b + 1 : 0], 1);  // full: the walk goes on
}

// 1: the context is promoted (its stats bucket was prefetched)
static inline int lt_peek_index(const lt_part_t *P, uint64_t h) {
    const uint32_t b = lt_home(h, P->inb);
    const lt_idx_bucket *B = &P->ib[b];
    const uint32_t fp = lt_fp(h);
    for (int j = 0; j < LT_IB; j++) {
        if (B->s[j].fp == fp) {
            if (!(B->s[j].st & LT_IDX_PROMOTED)) return 0;
            __builtin_prefetch(&P->cb[lt_home(h, P->cnb)], 1);
            return 1;
        }
        if (!B->s[j].fp) return 0;
    }
    __builtin_prefetch(&P->ib[b + 1 < P->inb ? b + 1 : 0], 1);
    return 0;
}

static void lt_apply(lt_t *lt, const lt_item_t *it, int m, uint64_t *positions) {
    lt_part_t *P[LT_GROUP];
    uint64_t h[LT_GROUP];
    uint8_t tiered[LT_GROUP], promoted[LT_GROUP];
    int ntp = 0;
    for (int i = 0; i < m; i++) {
        const lt_order_t *O = &lt->ord[it[i].oi];
        P[i] = (lt_part_t *)&O->part[lt_part_of(it[i].acc)];
        h[i] = lt_ctx_hash(it[i].acc);
        tiered[i] = (uint8_t)O->tiered;
        if (tiered[i]) __builtin_prefetch(&P[i]->ib[lt_home(h[i], P[i]->inb)], 1);
        else __builtin_prefetch(&P[i]->cb[lt_home(h[i], P[i]->cnb)], 1);
    }
    for (int i = 0; i < m; i++) {
        if (tiered[i]) ntp += promoted[i] = (uint8_t)lt_peek_index(P[i], h[i]);
        else lt_peek_stats(P[i], h[i], it[i].v);
    }
    if (ntp)
        for (int i = 0; i < m; i++)
            if (tiered[i] && promoted[i]) lt_peek_stats(P[i], h[i], it[i].v);
    for (int i = 0; i < m; i++) {
        const int k = it[i].k;
        if (tiered[i] ? lt_update_tiered(lt, P[i], h[i], it[i].v, k) : lt_update(lt, P[i], h[i], it[i].v, k)) return;
        positions[it[i].oi]++;
    }
}

// run(lo - 1), the tokens of lo - 1's document up to and including it (0 at a SEP, and before the block), or any
// value > kmax when it is larger: a block is whole spans, so its start counts as a SEP.
static uint32_t lt_run_before(const uint16_t *tok, size_t lo, int kmax) {
    for (size_t d = 0; d <= (size_t)kmax && d < lo; d++) {
        const uint16_t t = tok[lo - 1 - d];
        if (t == LT_SEP) return (uint32_t)d;
        if (t == LT_BOS) return (uint32_t)d + 1;
    }
    return lo <= (size_t)kmax ? (uint32_t)lo : (uint32_t)kmax + 1;
}

static int lt_push(lt_items_t *v, const lt_item_t *it) {
    if (v->n == v->cap) {
        const size_t cap = v->cap ? 2 * v->cap : 4096;
        lt_item_t *a = realloc(v->a, cap * sizeof *a);
        if (!a) return -1;
        v->a = a;
        v->cap = cap;
    }
    v->a[v->n++] = *it;
    return 0;
}

// Phase 1: thread p's slice of the block's positions [lo, hi) (j counts when j + 1 is in the block): every position
// whose context of order k lies in one document and whose next token is in that document, one item per order, handed
// to the owner of the context's partition, in stream order.
static void lt_scan(lt_t *lt, int p, const uint16_t *tok, size_t n, lt_items_t *out) {
    const size_t npos = n ? n - 1 : 0, lo = npos * (size_t)p / lt->nthreads, hi = npos * ((size_t)p + 1) / lt->nthreads;
    const int kmin = lt->kmin, kmax = lt->kmax;
    int8_t oidx[LT_MAX_K + 1];
    uint8_t owner[LT_NPART];
    memcpy(oidx, lt->oidx, sizeof oidx);
    memcpy(owner, lt->owner, sizeof owner);
    for (int q = 0; q < lt->nthreads; q++) out[q].n = 0;
    uint32_t run = lt_run_before(tok, lo, kmax);
    for (size_t j = lo; j < hi; j++) {
        const uint16_t t = tok[j], v = tok[j + 1];
        run = t == LT_SEP ? 0 : t == LT_BOS ? 1 : (run < LT_RUN_SAT ? run + 1 : LT_RUN_SAT);
        if (run < (uint32_t)kmin || v == LT_SEP || v == LT_BOS) continue;
        const int lim = run < (uint32_t)kmax ? (int)run : kmax;
        uint64_t acc = LT_SEED;
        for (int k = 1; k <= lim; k++) {
            acc = lt_step(acc, tok[j + 1 - k]);
            const int oi = oidx[k];
            if (oi < 0) continue;
            const lt_item_t it = {acc, v, (uint8_t)oi, (uint8_t)k};
            if (lt_push(&out[owner[lt_part_of(acc)]], &it)) { lt_fail(lt, "out of memory (insert items)"); return; }
        }
    }
}

// Phase 2: thread p's items from every slice, the slices in order.
static void lt_insert_own(lt_t *lt, int p, int par, uint64_t *positions) {
    for (int src = 0; src < lt->nthreads; src++) {
        const lt_items_t *v = &lt->th[src].out[par][p];
        for (size_t i = 0; i < v->n; i += LT_GROUP) {
            lt_apply(lt, v->a + i, v->n - i < LT_GROUP ? (int)(v->n - i) : LT_GROUP, positions);
            if (atomic_load_explicit(&lt->failed, memory_order_relaxed)) return;
        }
    }
}

static void lt_touch(const void *p, size_t bytes) {
    for (volatile char *c = (volatile char *)p, *e = c + bytes; c < e; c += 4096) *c = 0;
}

static void *lt_worker(void *arg) {
    lt_thread_t *th = arg;
    lt_t *lt = th->lt;
    const int p = th->id;
    if (lt->nice) setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), lt->nice);
    if (lt->prefault) {  // first touch by the owner: zero pages now, and on the owner's NUMA node
        for (int oi = 0; oi < lt->norders; oi++)
            for (int q = 0; q < LT_NPART; q++) {
                if (lt->owner[q] != p) continue;
                const lt_part_t *P = &lt->ord[oi].part[q];
                lt_touch(P->ib, (size_t)P->inb * sizeof *P->ib);
                lt_touch(P->cb, (size_t)P->cnb * sizeof *P->cb);
                lt_touch(P->pb, (size_t)P->pnb * sizeof *P->pb);
            }
    }
    pthread_mutex_lock(&lt->mu);
    lt->ready++;
    pthread_cond_broadcast(&lt->cv_done);
    for (;;) {
        while (th->done == lt->released && !lt->stop) pthread_cond_wait(&lt->cv_work, &lt->mu);
        if (th->done == lt->released) break;  // stop, and nothing left
        const uint64_t seq = th->done;
        lt_block_t *blk = lt->blocks[seq];
        pthread_mutex_unlock(&lt->mu);
        // Every thread takes every block through both phases and the barrier, failed or not, so none waits forever.
        // The items of block seq sit in the buffers of parity seq & 1: their next writer (block seq + 2) starts after
        // the barrier of block seq + 1, which every thread reaches only after reading them.
        const int par = (int)(seq & 1);
        const uint64_t t0 = lt_now_ns(), c0 = lt_cpu_ns();
        uint64_t positions[LT_MAX_ORDERS] = {0};  // counted locally: the thread structs stay out of the hot loop
        if (!atomic_load_explicit(&lt->failed, memory_order_relaxed)) lt_scan(lt, p, blk->tok, blk->n, th->out[par]);
        const uint64_t busy0 = lt_now_ns() - t0, cpu0 = lt_cpu_ns() - c0;
        pthread_barrier_wait(&lt->bar);
        const uint64_t t1 = lt_now_ns(), c1 = lt_cpu_ns();
        if (!atomic_load_explicit(&lt->failed, memory_order_relaxed)) lt_insert_own(lt, p, par, positions);
        const uint64_t busy = busy0 + lt_now_ns() - t1, cpu = cpu0 + lt_cpu_ns() - c1;  // the barrier's wait excluded
        const int last = atomic_fetch_sub(&blk->refs, 1) == 1;
        pthread_mutex_lock(&lt->mu);
        for (int oi = 0; oi < lt->norders; oi++) th->positions[oi] += positions[oi];
        th->busy_ns += busy;
        th->cpu_ns += cpu;
        if (last) {
            lt->blocks[seq] = NULL;
            free(blk);
        }
        th->done++;
        if (th->done == lt->released) pthread_cond_broadcast(&lt->cv_done);
    }
    pthread_mutex_unlock(&lt->mu);
    return NULL;
}

// ------------------------------------------------------------------------------------------------ sizing

// Expected entries for a stream of n tokens, from two anchors measured on the real 1050-step stream
// (tools/stream_retrieval/bench_lowtables.py): its first 30M tokens and all of it (289,983,448 tokens). Log-log
// interpolation between them and beyond; below the first anchor, entries that grow faster than linearly scale
// linearly, an overestimate. Then a 5% margin, at least min(n, 2^20), at most n; order-1 contexts: 65536.
// what: 0 contexts, 1 pair entries, 2 contexts seen twice or more (a tiered order's stats slots). Orders above 5 have
// no anchors: n (the bound).
#define LT_N30 29999704.0
#define LT_N290 289983448.0
static const double LT_ANCHOR[6][3][2] = {  // [k][what] = {at 30M, at 290M}
    {{0, 0}, {0, 0}, {0, 0}},
    {{49579, 49982}, {5212201, 22627026}, {49277, 49800}},
    {{5211077, 22612661}, {12366628, 89668226}, {1905335, 9.0e6}},
    {{16463948, 106722631}, {9221979, 100846811}, {2867130, 3.5e7}},
    {{24231838, 195038534}, {4086109, 60983879}, {1896095, 22870867}},
    {{27455504, 245415383}, {1521791, 27451588}, {970139, 14432433}},
};

LT_API uint64_t lt_default_entries(int k, uint64_t positions, int what) {
    if (k < 1 || k > LT_MAX_K || what < 0 || what > 2) return 0;
    const double n = (double)positions;
    double e = n;
    if (k <= 5) {
        const double v30 = LT_ANCHOR[k][what][0], v290 = LT_ANCHOR[k][what][1];
        double beta = log(v290 / v30) / log(LT_N290 / LT_N30);
        if (n < LT_N30 && beta > 1) beta = 1;
        e = 1.05 * v30 * pow(n / LT_N30, beta);
    }
    if (e < n && e < 1048576.0) e = n < 1048576.0 ? n : 1048576.0;  // small streams (tests): the bound, <= 2^20
    if (what != 1 && k == 1) e = 65536.0;            // order-1 contexts: at most one per token value
    if (e > n) e = n;                                // distinct <= positions
    return (uint64_t)e + 64;
}

// Partitions fill unevenly where a few contexts own many entries: all the pairs of a context live in its partition,
// and an order-1 context like "," or " the" has tens of thousands of distinct next tokens. Slots per partition are
// sized for the expected entries times this allowance (the fullest partition over the mean, measured on the real
// stream: order-1 pairs 1.39 on its first 30M tokens, 1.12 on all of it; order 2 1.11; others <= 1.08). And at
// orders 1-2 every pair partition has room for min(expected pairs, 65536) entries: on a small stream one context
// (" the") can own more pairs than the mean partition holds.
static const double LT_SKEW[LT_MAX_K + 1][2] = {  // {contexts, pairs}
    {1, 1}, {1.2, 1.6}, {1.03, 1.2}, {1.03, 1.1}, {1.03, 1.1}, {1.03, 1.1}, {1.03, 1.1}, {1.03, 1.1}, {1.03, 1.1},
};

// The layout the Python binding mirrors (stream_lowtables.py checks it): 0 sizeof(lt_config_t), 1 sizeof(lt_row_t),
// 2 LT_STATS_WORDS, 3 LT_MAX_ORDERS, 4 LT_MAX_K, 5 offsetof(lt_config_t, load), 6 LT_TIER_FROM.
LT_API uint64_t lt_abi(int what) {
    switch (what) {
    case 0: return sizeof(lt_config_t);
    case 1: return sizeof(lt_row_t);
    case 2: return LT_STATS_WORDS;
    case 3: return LT_MAX_ORDERS;
    case 4: return LT_MAX_K;
    case 5: return offsetof(lt_config_t, load);
    case 6: return LT_TIER_FROM;
    default: return 0;
    }
}

// ------------------------------------------------------------------------------------------------ create / destroy

static void lt_unmap(lt_t *lt) {
    for (int oi = 0; oi < LT_MAX_ORDERS; oi++) {
        lt_order_t *O = &lt->ord[oi];
        if (O->imap) munmap(O->imap, O->ibytes);
        if (O->cmap) munmap(O->cmap, O->cbytes);
        if (O->pmap) munmap(O->pmap, O->pbytes);
        O->imap = O->cmap = O->pmap = NULL;
    }
}

static void *lt_map(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return NULL;
    madvise(p, bytes, MADV_HUGEPAGE);  // transparent huge pages where the kernel gives them (fewer TLB misses)
    return p;
}

static uint32_t lt_buckets(double entries, double load, int per) {
    double b = ceil(entries / load / LT_NPART / per);
    if (b < 8) b = 8;
    if (b * per > 4.0e9) b = 4.0e9 / per;
    return (uint32_t)b;
}

static uint32_t lt_limit(uint32_t nb, int per) {  // ~98% of the slots
    const uint32_t s = nb * (uint32_t)per;
    return s - (s / 64 > 1 ? s / 64 : 1);
}

LT_API lt_t *lt_create(const lt_config_t *cfg, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
#define LT_CREATE_FAIL(...) do { if (err && errlen) snprintf(err, errlen, __VA_ARGS__); goto fail; } while (0)
    lt_t *lt = calloc(1, sizeof *lt);
    if (!lt) { if (err && errlen) snprintf(err, errlen, "out of memory"); return NULL; }
    int sync_made = 0, nstarted = 0;
    if (cfg->norders < 1 || cfg->norders > LT_MAX_ORDERS)
        LT_CREATE_FAIL("norders=%d is outside 1..%d", cfg->norders, LT_MAX_ORDERS);
    if (cfg->threads < 1 || cfg->threads > LT_MAX_THREADS)
        LT_CREATE_FAIL("threads=%d is outside 1..%d", cfg->threads, LT_MAX_THREADS);
    lt->norders = cfg->norders;
    lt->nthreads = cfg->threads;
    lt->prefault = cfg->prefault;
    lt->nice = cfg->nice;
    const int tier_from = cfg->tier_from > 0 ? cfg->tier_from : LT_TIER_FROM;
    memset(lt->oidx, -1, sizeof lt->oidx);
    for (int oi = 0; oi < lt->norders; oi++) {
        const int k = cfg->orders[oi];
        if (k < 1 || k > LT_MAX_K || (oi && k <= cfg->orders[oi - 1]))
            LT_CREATE_FAIL("orders must be ascending in 1..%d (order %d)", LT_MAX_K, k);
        lt->oidx[k] = (int8_t)oi;
        lt->ord[oi].k = k;
        lt->ord[oi].tiered = k >= tier_from;
    }
    lt->kmin = cfg->orders[0];
    lt->kmax = cfg->orders[lt->norders - 1];
    for (int q = 0; q < LT_NPART; q++) lt->owner[q] = (uint8_t)(q % lt->nthreads);
    const double load = cfg->load > 0 && cfg->load <= 0.95 ? cfg->load : 0.75;
    const uint64_t n = cfg->expected_positions;
    for (int oi = 0; oi < lt->norders; oi++) {
        lt_order_t *O = &lt->ord[oi];
        const int k = O->k;
        const double ce = (double)(cfg->ctx_entries[oi] ? cfg->ctx_entries[oi] : lt_default_entries(k, n, 0));
        const double pe = (double)(cfg->pair_entries[oi] ? cfg->pair_entries[oi] : lt_default_entries(k, n, 1));
        const double se = O->tiered ? (double)(cfg->promoted_entries[oi] ? cfg->promoted_entries[oi]
                                                                         : lt_default_entries(k, n, 2)) : ce;
        const uint32_t is = O->tiered ? lt_buckets(ce * LT_SKEW[k][0], load, LT_IB) : 0;
        const uint32_t cs = lt_buckets(se * LT_SKEW[k][0], load, LT_CB);
        uint32_t ps = lt_buckets(pe * LT_SKEW[k][1], load, LT_PB);
        if (k <= 2) {  // a small stream's heaviest order-1/2 context can own most of the pairs: room for them
            const uint32_t floor_b = (uint32_t)ceil((pe < 65536 ? pe : 65536) / load / LT_PB);
            if (ps < floor_b) ps = floor_b;
        }
        O->ibytes = (size_t)is * LT_NPART * sizeof(lt_idx_bucket);
        O->cbytes = (size_t)cs * LT_NPART * sizeof(lt_ctx_bucket);
        O->pbytes = (size_t)ps * LT_NPART * sizeof(lt_pair_bucket);
        if ((is && !(O->imap = lt_map(O->ibytes))) || !(O->cmap = lt_map(O->cbytes)) || !(O->pmap = lt_map(O->pbytes)))
            LT_CREATE_FAIL("mmap of the order-%d tables (%zu + %zu + %zu bytes) failed: %s", k, O->ibytes, O->cbytes,
                           O->pbytes, strerror(errno));
        for (int q = 0; q < LT_NPART; q++) {
            lt_part_t *P = &O->part[q];
            P->ib = is ? (lt_idx_bucket *)O->imap + (size_t)q * is : NULL;
            P->cb = (lt_ctx_bucket *)O->cmap + (size_t)q * cs;
            P->pb = (lt_pair_bucket *)O->pmap + (size_t)q * ps;
            P->inb = is;
            P->cnb = cs;
            P->pnb = ps;
            P->ilim = is ? lt_limit(is, LT_IB) : 0;
            P->clim = lt_limit(cs, LT_CB);
            P->plim = lt_limit(ps, LT_PB);
        }
    }
    lt->blocks_alloc = 1024;
    if (!(lt->blocks = calloc(lt->blocks_alloc, sizeof *lt->blocks))) LT_CREATE_FAIL("out of memory");
    pthread_mutex_init(&lt->mu, NULL);
    pthread_cond_init(&lt->cv_work, NULL);
    pthread_cond_init(&lt->cv_done, NULL);
    pthread_barrier_init(&lt->bar, NULL, (unsigned)lt->nthreads);
    sync_made = 1;
    for (int p = 0; p < lt->nthreads; p++) {
        lt->th[p].lt = lt;
        lt->th[p].id = p;
        if (pthread_create(&lt->th[p].thread, NULL, lt_worker, &lt->th[p])) LT_CREATE_FAIL("pthread_create failed");
        nstarted++;
    }
    pthread_mutex_lock(&lt->mu);
    while (lt->ready < (uint64_t)lt->nthreads) pthread_cond_wait(&lt->cv_done, &lt->mu);
    pthread_mutex_unlock(&lt->mu);
    return lt;
fail:
    if (nstarted) {
        pthread_mutex_lock(&lt->mu);
        lt->stop = 1;
        pthread_cond_broadcast(&lt->cv_work);
        pthread_mutex_unlock(&lt->mu);
        for (int p = 0; p < nstarted; p++) pthread_join(lt->th[p].thread, NULL);
    }
    if (sync_made) {
        pthread_mutex_destroy(&lt->mu);
        pthread_cond_destroy(&lt->cv_work);
        pthread_cond_destroy(&lt->cv_done);
        pthread_barrier_destroy(&lt->bar);
    }
    free(lt->blocks);
    lt_unmap(lt);
    free(lt);
    return NULL;
#undef LT_CREATE_FAIL
}

LT_API void lt_destroy(lt_t *lt) {
    if (!lt) return;
    pthread_mutex_lock(&lt->mu);
    lt->stop = 1;
    lt->released = lt->nblocks;  // held blocks too: the threads drain everything, then exit ...
    lt->hold = 0;
    atomic_fetch_or(&lt->failed, 2);  // ... without inserting it
    pthread_cond_broadcast(&lt->cv_work);
    pthread_mutex_unlock(&lt->mu);
    for (int p = 0; p < lt->nthreads; p++) pthread_join(lt->th[p].thread, NULL);
    for (uint64_t s = 0; s < lt->nblocks; s++) free(lt->blocks[s]);
    free(lt->blocks);
    for (int p = 0; p < lt->nthreads; p++)
        for (int par = 0; par < 2; par++)
            for (int q = 0; q < lt->nthreads; q++) free(lt->th[p].out[par][q].a);
    pthread_mutex_destroy(&lt->mu);
    pthread_cond_destroy(&lt->cv_work);
    pthread_cond_destroy(&lt->cv_done);
    pthread_barrier_destroy(&lt->bar);
    lt_unmap(lt);
    free(lt);
}

LT_API const char *lt_error(const lt_t *lt) { return atomic_load(&((lt_t *)lt)->failed) & 1 ? lt->err : ""; }

// ------------------------------------------------------------------------------------------------ blocks

LT_API int lt_insert_block(lt_t *lt, const uint16_t *tok, size_t n) {
    if (atomic_load(&lt->failed)) return -1;
    if (lt->finished) { lt_fail(lt, "a block after lt_finish"); return -1; }
    lt_block_t *b = malloc(sizeof *b + 2 * (n ? n : 1));
    if (!b) { lt_fail(lt, "out of memory (a block of %zu tokens)", n); return -1; }
    b->n = n;
    atomic_init(&b->refs, lt->nthreads);
    if (n) memcpy(b->tok, tok, 2 * n);
    pthread_mutex_lock(&lt->mu);
    if (lt->nblocks == lt->blocks_alloc) {
        lt_block_t **grown = realloc(lt->blocks, 2 * lt->blocks_alloc * sizeof *grown);
        if (!grown) {
            pthread_mutex_unlock(&lt->mu);
            free(b);
            lt_fail(lt, "out of memory (block list)");
            return -1;
        }
        memset(grown + lt->blocks_alloc, 0, lt->blocks_alloc * sizeof *grown);
        lt->blocks = grown;
        lt->blocks_alloc *= 2;
    }
    lt->blocks[lt->nblocks++] = b;
    lt->tokens += n;
    lt->t_last_insert = lt_now_ns();
    if (!lt->hold) {
        lt->released = lt->nblocks;
        pthread_cond_broadcast(&lt->cv_work);
    }
    pthread_mutex_unlock(&lt->mu);
    return 0;
}

LT_API int lt_hold(lt_t *lt, int on) {
    pthread_mutex_lock(&lt->mu);
    lt->hold = on != 0;
    if (!lt->hold && lt->released != lt->nblocks) {
        lt->released = lt->nblocks;
        pthread_cond_broadcast(&lt->cv_work);
    }
    pthread_mutex_unlock(&lt->mu);
    return atomic_load(&lt->failed) ? -1 : 0;
}

static int lt_idle_locked(const lt_t *lt) {
    for (int p = 0; p < lt->nthreads; p++)
        if (lt->th[p].done != lt->released) return 0;
    return 1;
}

LT_API int lt_sync(lt_t *lt) {
    pthread_mutex_lock(&lt->mu);
    while (!lt_idle_locked(lt)) pthread_cond_wait(&lt->cv_done, &lt->mu);
    pthread_mutex_unlock(&lt->mu);
    return atomic_load(&lt->failed) ? -1 : 0;
}

LT_API int lt_finish(lt_t *lt) {
    pthread_mutex_lock(&lt->mu);
    const uint64_t held = lt->nblocks - lt->released;
    pthread_mutex_unlock(&lt->mu);
    if (held) { lt_fail(lt, "lt_finish with %llu held blocks (lt_hold(0) first)", (unsigned long long)held); return -1; }
    if (lt_sync(lt)) return -1;
    const uint64_t t = lt_now_ns();
    if (!lt->finished) lt->finish_ns = lt->t_last_insert && t > lt->t_last_insert ? t - lt->t_last_insert : 0;
    lt->finished = 1;
    return 0;
}

// ------------------------------------------------------------------------------------------------ queries

static inline void lt_row_singleton(lt_row_t *r, uint16_t top, uint16_t y) {
    r->N = 1; r->M = 1; r->D = 1; r->n1 = 1; r->n2 = 0; r->top = top; r->C = y == top;
}

static inline void lt_row_stats(lt_row_t *r, const lt_ctx_slot *s) {
    r->N = s->N; r->M = s->M; r->D = s->D; r->n1 = s->n1; r->n2 = s->n2; r->top = s->top;
}

// The rows of one position: x points at its input token x[t]; run = tokens of t's segment up to and including t (any
// value >= the largest order is equivalent); out has norders rows.
LT_API void lt_query_one(const lt_t *lt, const uint16_t *x, uint16_t y, uint32_t run, lt_row_t *out) {
    memset(out, 0, sizeof *out * (size_t)lt->norders);
    const int lim = run < (uint32_t)lt->kmax ? (int)run : lt->kmax;
    uint64_t acc = LT_SEED;
    for (int k = 1; k <= lim; k++) {
        acc = lt_step(acc, x[1 - k]);
        const int oi = lt->oidx[k];
        if (oi < 0) continue;
        const lt_order_t *O = &lt->ord[oi];
        const lt_part_t *P = &O->part[lt_part_of(acc)];
        const uint64_t h = lt_ctx_hash(acc);
        lt_row_t *r = &out[oi];
        if (O->tiered) {
            const lt_idx_slot *xs = lt_idx_find(P, h);
            if (!xs) continue;
            if (!(xs->st & LT_IDX_PROMOTED)) { lt_row_singleton(r, xs->top, y); continue; }
        }
        const lt_ctx_slot *s = lt_ctx_find(P, h);
        if (!s) continue;
        lt_row_stats(r, s);
        r->C = y == s->top ? s->M : (y == LT_BOS ? 0 : lt_pair_get(P, lt_pair_hash(h, y)));
    }
}

// The rows of n consecutive positions: x[t], y[t], run[t] as lt_query_one's, out [n][norders]. A software pipeline
// over the (position, order) lookups, each read LT_D* lookups after its prefetch: the first level (the index, or the
// stats where an order has no index); then a promoted context's stats, or the pair C(y) needs (y is not the top);
// then a promoted context's pair.
typedef struct { const lt_part_t *P; uint64_t h, hp; lt_row_t *r; uint16_t y; uint8_t tiered, next; } lt_qitem_t;
enum { LT_Q_DONE, LT_Q_STATS, LT_Q_PAIR };

static inline void lt_q_stats(lt_qitem_t *it) {  // read the stats slot; prefetch the pair if C(y) needs it
    it->next = LT_Q_DONE;
    const lt_ctx_slot *s = lt_ctx_find(it->P, it->h);
    if (!s) return;
    lt_row_stats(it->r, s);
    if (it->y == s->top) it->r->C = s->M;
    else if (it->y != LT_BOS) {
        it->hp = lt_pair_hash(it->h, it->y);
        __builtin_prefetch(&it->P->pb[lt_home(it->hp, it->P->pnb)], 0);
        it->next = LT_Q_PAIR;
    }
}

static inline void lt_q_first(lt_qitem_t *it) {
    if (!it->tiered) { lt_q_stats(it); return; }
    it->next = LT_Q_DONE;
    const lt_idx_slot *xs = lt_idx_find(it->P, it->h);
    if (!xs) return;
    if (!(xs->st & LT_IDX_PROMOTED)) { lt_row_singleton(it->r, xs->top, it->y); return; }
    __builtin_prefetch(&it->P->cb[lt_home(it->h, it->P->cnb)], 0);
    it->next = LT_Q_STATS;
}

static inline void lt_q_later(lt_qitem_t *it) {
    if (it->next == LT_Q_STATS) lt_q_stats(it);
    else if (it->next == LT_Q_PAIR) { it->r->C = lt_pair_get(it->P, it->hp); it->next = LT_Q_DONE; }
}

LT_API void lt_query_block(const lt_t *lt, const uint16_t *x, const uint16_t *y, const uint32_t *run, size_t n,
                           lt_row_t *out) {
    const int no = lt->norders, kmax = lt->kmax;
    memset(out, 0, sizeof *out * n * (size_t)no);
    lt_qitem_t ring[LT_RING];
    uint64_t w = 0, c1 = 0, c2 = 0, c3 = 0;  // items pushed, read at the first, second, third level
#define LT_RI(i) (&ring[(i) & (LT_RING - 1)])
    for (size_t t = 0; t < n; t++) {
        const int lim = run[t] < (uint32_t)kmax ? (int)run[t] : kmax;
        uint64_t acc = LT_SEED;
        for (int k = 1; k <= lim; k++) {
            acc = lt_step(acc, x[t + 1 - k]);
            const int oi = lt->oidx[k];
            if (oi < 0) continue;
            const lt_order_t *O = &lt->ord[oi];
            lt_qitem_t *it = LT_RI(w++);
            it->P = &O->part[lt_part_of(acc)];
            it->h = lt_ctx_hash(acc);
            it->r = &out[t * no + oi];
            it->y = y[t];
            it->tiered = (uint8_t)O->tiered;
            if (O->tiered) __builtin_prefetch(&it->P->ib[lt_home(it->h, it->P->inb)], 0);
            else __builtin_prefetch(&it->P->cb[lt_home(it->h, it->P->cnb)], 0);
            if (w - c1 > LT_D1) lt_q_first(LT_RI(c1++));
            if (c1 - c2 > LT_D2) lt_q_later(LT_RI(c2++));
            if (c2 - c3 > LT_D3) lt_q_later(LT_RI(c3++));
        }
    }
    while (c1 < w) lt_q_first(LT_RI(c1++));
    while (c2 < w) lt_q_later(LT_RI(c2++));
    while (c3 < w) lt_q_later(LT_RI(c3++));
#undef LT_RI
}

typedef struct {
    const lt_t *lt;
    const uint16_t *x, *y;
    size_t n;
    uint64_t chunk;
    lt_row_t *out;
    atomic_size_t next;
} lt_qjob_t;

static void *lt_qworker(void *arg) {
    lt_qjob_t *job = arg;
    const uint16_t *x = job->x;
    uint32_t run[LT_QBLOCK];
    for (;;) {
        const size_t lo = atomic_fetch_add(&job->next, LT_QBLOCK);
        if (lo >= job->n) break;
        const size_t hi = lo + LT_QBLOCK < job->n ? lo + LT_QBLOCK : job->n;
        // t's segment restarts at every BOS and at every multiple of chunk: find lo's run by looking back, then count
        const size_t start = job->chunk ? lo - lo % job->chunk : 0;
        size_t next_chunk = job->chunk ? start + job->chunk : SIZE_MAX;
        uint32_t r = 1;
        for (size_t s = lo; s > start && x[s] != LT_BOS && r < LT_RUN_SAT; s--) r++;
        for (size_t t = lo; t < hi; t++) {
            if (t > lo) {
                if (t == next_chunk) { r = 1; next_chunk += job->chunk; }
                else r = x[t] == LT_BOS ? 1 : (r < LT_RUN_SAT ? r + 1 : LT_RUN_SAT);
            }
            run[t - lo] = r;
        }
        lt_query_block(job->lt, x + lo, job->y + lo, run, hi - lo, job->out + lo * (size_t)job->lt->norders);
    }
    return NULL;
}

// The rows of positions 0..n-1 (input x[t], target y[t]; segments restart at BOS and at multiples of chunk, 0 = never),
// out [n][norders], on `threads` threads (the caller's included). Refused unless every released block is in.
LT_API int lt_query_rows(const lt_t *lt_, const uint16_t *x, const uint16_t *y, size_t n, uint64_t chunk,
                         lt_row_t *out, int threads) {
    lt_t *lt = (lt_t *)lt_;
    pthread_mutex_lock(&lt->mu);
    const int idle = lt_idle_locked(lt);
    pthread_mutex_unlock(&lt->mu);
    if (!idle) { lt_fail(lt, "a query while blocks are being inserted (lt_sync or lt_finish first)"); return -1; }
    if (atomic_load(&lt->failed)) return -1;
    lt_qjob_t job = {.lt = lt, .x = x, .y = y, .n = n, .chunk = chunk, .out = out};
    atomic_init(&job.next, 0);
    const int nt = threads < 1 ? 1 : (threads > 256 ? 256 : threads);
    pthread_t th[256];
    int started = 1;
    for (; started < nt; started++)
        if (pthread_create(&th[started], NULL, lt_qworker, &job)) break;
    lt_qworker(&job);
    for (int i = 1; i < started; i++) pthread_join(th[i], NULL);
    return 0;
}

// ------------------------------------------------------------------------------------------------ statistics

static double lt_maxload(const lt_order_t *O, int which) {
    double m = 0;
    for (int q = 0; q < LT_NPART; q++) {
        const lt_part_t *P = &O->part[q];
        const double used = which == 0 ? P->iused : which == 1 ? P->cused : P->pused;
        const double slots = which == 0 ? (double)P->inb * LT_IB : which == 1 ? (double)P->cnb * LT_CB
                                                                              : (double)P->pnb * LT_PB;
        if (slots && used / slots > m) m = used / slots;
    }
    return m;
}

// Fill counts read while the insertion runs are a moment's (the return value is then 1).
LT_API int lt_stats(const lt_t *lt_, uint64_t *w, int nwords) {
    lt_t *lt = (lt_t *)lt_;
    if (nwords < LT_STATS_WORDS) return -1;
    memset(w, 0, sizeof *w * (size_t)nwords);
    pthread_mutex_lock(&lt->mu);
    const int idle = lt_idle_locked(lt);
    w[LT_S_NORDERS] = (uint64_t)lt->norders;
    w[LT_S_THREADS] = (uint64_t)lt->nthreads;
    w[LT_S_BLOCKS] = lt->nblocks;
    w[LT_S_HELD] = lt->nblocks - lt->released;
    w[LT_S_TOKENS] = lt->tokens;
    uint64_t least = lt->released;
    for (int p = 0; p < lt->nthreads; p++) {
        w[LT_S_BUSY_NS] += lt->th[p].busy_ns;
        w[LT_S_CPU_NS] += lt->th[p].cpu_ns;
        if (lt->th[p].done < least) least = lt->th[p].done;
    }
    w[LT_S_PENDING] = lt->released - least;  // released blocks the slowest thread has not finished
    w[LT_S_STATE] = (lt->finished ? LT_ST_FINISHED : 0) | (atomic_load(&lt->failed) & 1 ? LT_ST_FAILED : 0);
    w[LT_S_FINISH_NS] = lt->finish_ns;
    for (int oi = 0; oi < lt->norders; oi++) {
        const lt_order_t *O = &lt->ord[oi];
        uint64_t *o = w + LT_S_ORDER0 + oi * LT_SO_WORDS;
        o[LT_SO_K] = (uint64_t)O->k;
        o[LT_SO_TIERED] = (uint64_t)O->tiered;
        for (int p = 0; p < lt->nthreads; p++) o[LT_SO_POSITIONS] += lt->th[p].positions[oi];
        for (int q = 0; q < LT_NPART; q++) {
            const lt_part_t *P = &O->part[q];
            o[LT_SO_CONTEXTS] += O->tiered ? P->iused : P->cused;
            o[LT_SO_PROMOTED] += O->tiered ? P->cused : 0;
            o[LT_SO_PAIRS] += P->pused;
            o[LT_SO_IDX_SLOTS] += (uint64_t)P->inb * LT_IB;
            o[LT_SO_CTX_SLOTS] += (uint64_t)P->cnb * LT_CB;
            o[LT_SO_PAIR_SLOTS] += (uint64_t)P->pnb * LT_PB;
        }
        o[LT_SO_IDX_MAXLOAD_PPM] = (uint64_t)(lt_maxload(O, 0) * 1e6 + 0.5);
        o[LT_SO_CTX_MAXLOAD_PPM] = (uint64_t)(lt_maxload(O, 1) * 1e6 + 0.5);
        o[LT_SO_PAIR_MAXLOAD_PPM] = (uint64_t)(lt_maxload(O, 2) * 1e6 + 0.5);
        o[LT_SO_BYTES] = O->ibytes + O->cbytes + O->pbytes;
        w[LT_S_BYTES] += o[LT_SO_BYTES];
    }
    pthread_mutex_unlock(&lt->mu);
    return idle ? 0 : 1;
}

// Diagnostics (a scan of every slot): out[0..3] = contexts of order k with N == 1, N == 2, D == 1, and N >= 2 with
// D == 1. Waits for the released blocks first. Returns -1 for an order not in the set.
LT_API int lt_census(lt_t *lt, int k, uint64_t *out) {
    if (k < 1 || k > LT_MAX_K || lt->oidx[k] < 0) return -1;
    lt_sync(lt);
    memset(out, 0, 4 * sizeof *out);
    const lt_order_t *O = &lt->ord[lt->oidx[k]];
    for (int q = 0; q < LT_NPART; q++) {
        const lt_part_t *P = &O->part[q];
        for (uint32_t b = 0; b < P->inb; b++)
            for (int i = 0; i < LT_IB; i++)
                if (P->ib[b].s[i].fp && !(P->ib[b].s[i].st & LT_IDX_PROMOTED)) { out[0]++; out[2]++; }
        for (uint32_t b = 0; b < P->cnb; b++)
            for (int i = 0; i < LT_CB; i++) {
                const lt_ctx_slot *s = &P->cb[b].s[i];
                if (!s->fp) continue;
                out[0] += s->N == 1;
                out[1] += s->N == 2;
                out[2] += s->D == 1;
                out[3] += s->N >= 2 && s->D == 1;
            }
    }
    return 0;
}

// A hash of every table byte and fill count, in a fixed order (tests: identical tables across thread counts and
// block boundaries). Waits for the released blocks first.
LT_API uint64_t lt_digest(lt_t *lt) {
    lt_sync(lt);
    uint64_t h = 0xCBF29CE484222325ull;
    for (int oi = 0; oi < lt->norders; oi++)
        for (int q = 0; q < LT_NPART; q++) {
            const lt_part_t *P = &lt->ord[oi].part[q];
            h = lt_mix64(h ^ ((uint64_t)P->iused << 42 ^ (uint64_t)P->cused << 21 ^ P->pused));
            const struct { const void *p; size_t bytes; } t[3] = {{P->ib, (size_t)P->inb * sizeof *P->ib},
                                                                  {P->cb, (size_t)P->cnb * sizeof *P->cb},
                                                                  {P->pb, (size_t)P->pnb * sizeof *P->pb}};
            for (int j = 0; j < 3; j++)
                for (size_t i = 0; i + 8 <= t[j].bytes; i += 8) {
                    uint64_t v;
                    memcpy(&v, (const char *)t[j].p + i, 8);
                    h = (h ^ v) * 0x100000001B3ull;
                }
        }
    return lt_mix64(h);
}

#endif  // STREAM_LOWTABLES_C
