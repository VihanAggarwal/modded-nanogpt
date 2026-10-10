// The stream-only retrieval memory (STREAM_RETRIEVAL=1): a helper process that rank 0 spawns
// (track_1_short/stream_memory.py drives it; that file's docstring has the design and the rules argument).
//
// The memory holds exactly the tokens this run trains on. For every timed step, rank 0's loader sends the
// document spans Shard.next_batch gave every rank: the shard file's index and the [a, e) token ranges. The
// helper reads those ranges from the shard file itself (rank 0 just read them, so they come from the page
// cache), appends each span and a separator to `tok`, and indexes every position j that has at least KEY
// tokens of its span up to and including j and whose next token is also in the span. The index is a hash
// chain over the KEY-token context ending at j, as in LZ77/zlib: head[h] = the latest position in bucket h,
// prev[j] = the position before j in j's bucket. Positions start at 1 (tok[0] is a separator), so 0 = none.
//
// Three parts read that memory (stream retrieval v2):
//   P1 records   (always)     at GO, one RECORD per val position t that matched (input val[t]; its target is never read):
//     segment  sigma_t = max(the last BOS at or before t, the start of t's chunk): the context the model sees
//     walk     t's bucket, most recent first; verify the KEY context tokens; extend each match backward,
//              within both segments, up to MAXLEN; stop after CAP verified candidates or MAXVISIT entries
//     record   the candidates (next token, match length), and per level L of LEVELS over the candidates whose
//              match length is >= L: N, D (distinct next tokens), M (largest count of one), n1 / n2 (tokens seen
//              once / twice), top (the most frequent next token, ties to the lowest id); L* (the deepest level
//              reached); the most recent candidate at L* and the candidate with the longest full match (lengths
//              extended past MAXLEN, up to FULLCAP), with their memory positions
//     Every field is a function of the memory and val[<= t] only. The eval reads the target's count C_L(y) =
//     |{candidates with match >= L and next == y}| from the candidate list on the GPU, as a cross-entropy gather
//     reads p(y); the gate (stream_memory.mix_v2) reads only target-independent fields.
//   P2 low orders (low=1, STREAM_RETRIEVAL_LOW=1; stream_lowtables.c)  exact count tables of the stream at orders
//     1-5, filled on the clock: each step's tokens (its spans, SEP between them) go to the tables' insertion
//     threads right after the step is appended to tok[]. At GO lt_finish() (every step in, or the error) on a
//     thread of its own while the P1 / P3 queries run (any insertion backlog drains alongside them), then per val
//     position one 20-byte row per order: (N, C(y), M, D, n1, n2, top) of the context in t's segment.
//     C(y) is the count of the realised target y = val[t + 1], read at the target as a cross-entropy gather reads
//     p(y) (the tables are too large to ship); every other field and the gate are target-independent.
//   P3 pointer   (pointer=1; stream_pointer.c)  per segment, sequentially: the pointer beam, the vote, the source
//     copy and the doc-state counters, from P1's candidate list at every position (sp_step). A row per ACTIVE
//     position (the pointer or the source copy predicts); its fields read val[<= t] and the outcomes of EARLIER
//     positions of the segment only (t's target is read after row t is written, to update the state for t + 1).
//     So the query blocks start at segment starts and run whole segments (one sp_state per query thread); they are
//     handed out longest first, so that a long segment never starts last.
//
// Rows go to a file that every rank maps: a header (state word, statistics, a checksum of every chunk's val
// tokens, the P1 record count and the P3 row count of every chunk) and the regions, rank-major ([world][val_steps]):
// the P1 records of each chunk's matched positions in position order (rec_t below), P2's rows of every position
// ([chunk][norders] lt_row_t), P3's active rows in position order (sp_row_t). Each query thread appends its blocks'
// rows to its own arena (touched before the clock), and the gather copies them into the regions in position order.
// One insert thread in message order (P2: partition-owning threads, deterministic too), and queries that are pure
// functions of the final memory and of each segment: the rows are bit-identical across runs, query thread counts
// and message batching.
//
// FIT (fit=PATH fit_k=K; dev runs): the run's OWN last K timed batches, queried against the memory as it stood
// before the first of them was inserted (the walk skips every entry from the freeze on, without counting it as a
// visit, so it is the walk over the frozen memory; P3's reads stay in the candidates' spans, below the freeze; P2
// holds the K steps' blocks until their rows are queried, lt_hold). Per rank, the K batches' inputs buf[:-1] and
// targets buf[1:] are concatenated (segments restart at every batch and at every BOS). P2's FIT rows are queried
// when the last step has arrived (then the held blocks are inserted, so a LOW FIT run's GO waits for those K
// steps' insertion: time the arm with a run without FIT); P1's and P3's after the val rows are done, when rank 0
// sends FIT after the clock has stopped. Written to PATH (P1), PATH.tokens, PATH.low (P2), PATH.ptr (P3).
//
// Credits (stream_memory.py's docstring has the full list). The memory's layout and the walk follow PR #367's
// StreamIndex (Herman Brunborg, exact_match/src/stream.rs): the stream of every rank's documents (inputs plus the
// last target) step by step with a STOP after each, positions keyed on their last 6 tokens, resolved against the
// most recent occurrences, the match levels, the next tokens of the occurrences matching at least that deep. P2's
// exact low-order counts and the gated chain over orders they feed are PR #380's recipe (Deven). No code from #367
// or #380: this file (the hash chain, the C helper, the protocol, the records) is written for this branch.
// The index is the classic LZ77/zlib hash chain; the bucket hash uses MurmurHash3's 64-bit finalizer.
//
// Build: stream_memory.py's build_helper() (cc -O2 -std=c11 -pthread stream_memory.c -o stream_memory -lm; the
// parts are #included from this directory)
// Usage: stream_memory key=value... -- TRAIN_SHARD...   (stdin: the messages below; keys in main())
// Messages (u32 words): STEP     1 step file_idx world n_0..n_{world-1} a e a e ...   (spans in rank order)
//                       GO       2 total_steps
//                       FIT      3 fit_k                                            (after GO; with fit= only)
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
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
#include <time.h>
#include <unistd.h>

// The parts (their names are all lt_ / LT_ and sp_ / SP_ prefixed, internal ones static).
#ifndef LT_API
#define LT_API static __attribute__((unused))
#endif
#include "stream_lowtables.c"
#ifndef SP_API
#define SP_API static __attribute__((unused))
#endif
#include "stream_pointer.c"

#define SEP 0xFFFFu
#define BOS 50256u
#define KEY 6
#define MAXLEN 32
#define CAP 32
#define MAXVISIT 128
#define FULLCAP 4096  // full match lengths (and segment runs) saturate here
#define NLEVELS 8
static const uint32_t LEVELS[NLEVELS] = {6, 7, 8, 10, 12, 16, 24, 32};
#define SHARD_HEADER_BYTES 1024
#define SHARD_MAGIC 20240520
#define MAX_CHUNKS 384
#define ERRMSG_OFFSET 512
#define CHECKSUM_OFFSET 1024
#define COUNTS_OFFSET 4096
#define PCOUNTS_OFFSET 8192
#define ROWS_OFFSET 16384
#define REGION_ALIGN 4096
#define BLOCK 4096
#define LOW_NORDERS 5
static const int32_t LOW_ORDERS[LOW_NORDERS] = {1, 2, 3, 4, 5};

enum { MSG_STEP = 1, MSG_GO = 2, MSG_FIT = 3 };
enum { ST_STARTING = 0, ST_READY = 1, ST_DONE = 2, ST_FIT_DONE = 3, ST_ERROR = 14 };
// Header words (u64), mirrored by stream_memory.py's H_* constants.
enum { H_MAGIC, H_STATE, H_ERROR, H_WORLD, H_VAL_STEPS, H_CHUNK, H_STEPS, H_ENTRIES, H_INSERTED, H_HITS, H_CANDS,
       H_QUERIED, H_T_READY, H_T_GO, H_T_DONE, H_INSERT_NS, H_QUERY_NS, H_BYTES_READ, H_REC_BYTES, H_LEVEL0,
       H_FIT_POSITIONS = H_LEVEL0 + NLEVELS, H_FIT_HITS, H_FIT_FREEZE, H_T_FIT_DONE,
       H_LOW_ORDERS, H_PTR, H_LOW_OFFSET, H_PTR_OFFSET, H_PTR_ROWS, H_LOW_WAIT_NS, H_LOW_MAX_PENDING, H_LOW_BYTES,
       H_LOW_CPU_NS, H_FIT_PTR_ROWS, H_LOW_FIT_NS, H_LOW_ROW_BYTES, H_PTR_ROW_BYTES, H_LOW_DRAIN_NS,
       H_LOW_LATE_PENDING, H_NWORDS };
#define HEADER_MAGIC 0x324d454d52545353ull  // "SSTRMEM2"
#define FIT_MAGIC 0x3354494652545353ull     // "SSTRFIT3": a 16-word header (word 8: the run's trained steps)
#define FIT_HEAD_WORDS 16
#define LATE_STEPS 100                      // H_LOW_LATE_PENDING: P2's largest backlog over the run's last 100 steps
_Static_assert(H_NWORDS * 8 <= ERRMSG_OFFSET, "the header words fit before the error message");

// One matched position's record (176 bytes; stream_memory.py's REC_DTYPE mirrors it). Candidates in walk order
// (most recent first); unused candidate slots have len 0. Levels index LEVELS; a level with n == 0 has top 0.
typedef struct {
    uint32_t pos;                 // the position within its chunk (val) or within its rank's FIT sequence
    uint32_t pos_recent;          // memory entry p (tok[p] = the candidate's last context token, tok[p + 1] its next)
    uint32_t pos_longest;         //   of the most recent candidate at L*, and of the longest full match
    uint16_t len_recent;          // their full match lengths (past MAXLEN, within both segments, <= FULLCAP)
    uint16_t len_longest;
    uint16_t top[NLEVELS];
    uint16_t nx[CAP];             // candidates: next token
    uint8_t len[CAP];             //             match length capped at MAXLEN (KEY..MAXLEN; 0 = no candidate)
    uint8_t n[NLEVELS], d[NLEVELS], m[NLEVELS], n1[NLEVELS], n2[NLEVELS];
    uint8_t ncand;                // verified candidates
    uint8_t lstar;                // index of L* in LEVELS
    uint8_t pad[6];
} rec_t;
_Static_assert(sizeof(rec_t) == 176, "rec_t is 176 bytes");

static uint16_t *tok;            // the stream: spans, each followed by SEP; tok[0] = SEP
static uint32_t *prev, *head;
static uint64_t cap, entries = 1, inserted;
static int hash_bits = 29;
static volatile uint64_t *hdr;   // the rows file's header (mapped)
static char *rows;               // the rows file (mapped from its start)
static FILE *readlog;
static lt_t *lt;                 // P2's tables (low=1), else NULL
static int pointer_on;           // P3 (pointer=1)
static sp_config_t sp_cfg;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void die(int code, const char *fmt, ...) {
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fprintf(stderr, "stream_memory: %s\n", msg);
    if (hdr) {
        memcpy((char *)hdr + ERRMSG_OFFSET, msg, sizeof msg);
        hdr[H_ERROR] = (uint64_t)code;
        __atomic_store_n(&hdr[H_STATE], ST_ERROR, __ATOMIC_RELEASE);
    }
    exit(2);
}

static void *xalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die(1, "out of memory (%zu bytes)", n);
    return p;
}

static void *xgrow(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die(1, "out of memory (%zu bytes)", n);
    return p;
}

// Anonymous memory for the big arrays: huge pages where the kernel gives them, and `prefault` bytes touched now,
// before the clock, so that insertion never takes a page fault. The rest of `bytes` is reserved but untouched.
// Without huge pages, insertion (one thread, prefetched) measured ~40 ns per entry, still ~3x under the ~119 ns
// per token at which training consumes the stream.
static void *big_alloc(size_t bytes, size_t prefault) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) die(1, "mmap of %zu bytes failed: %s", bytes, strerror(errno));
    madvise(p, bytes, MADV_HUGEPAGE);
    for (size_t i = 0; i < prefault && i < bytes; i += 4096) ((volatile char *)p)[i] = 0;
    return p;
}

static void read_all(int fd, void *buf, size_t n, off_t off, const char *what) {
    char *p = buf;
    while (n) {
        ssize_t r = off >= 0 ? pread(fd, p, n, off) : read(fd, p, n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) die(3, "short read of %s (%s)", what, r < 0 ? strerror(errno) : "end of file");
        p += r;
        n -= (size_t)r;
        if (off >= 0) off += r;
    }
}

// Reads one message's next n words from stdin; returns 0 on a clean end of file before the first byte.
static int read_words(uint32_t *buf, size_t n, int eof_ok) {
    size_t got = 0, want = n * 4;
    while (got < want) {
        ssize_t r = read(0, (char *)buf + got, want - got);
        if (r < 0 && errno == EINTR) continue;
        if (r == 0 && got == 0 && eof_ok) return 0;
        if (r <= 0) die(4, "the message pipe closed mid-message");
        got += (size_t)r;
    }
    return 1;
}

// MurmurHash3's 64-bit finalizer (public domain), used as a mixer.
static inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ull; k ^= k >> 33;
    return k;
}

// The bucket of the KEY tokens w[0..5].
static inline uint32_t bucket_of(const uint16_t *w6) {
    uint64_t lo = (uint64_t)w6[0] | (uint64_t)w6[1] << 16 | (uint64_t)w6[2] << 32 | (uint64_t)w6[3] << 48;
    uint64_t hi = (uint64_t)w6[4] | (uint64_t)w6[5] << 16;
    return (uint32_t)(fmix64(lo ^ fmix64(hi + 0x9e3779b97f4a7c15ull)) >> (64 - hash_bits));
}

static uint64_t align_up(uint64_t v) { return (v + REGION_ALIGN - 1) / REGION_ALIGN * REGION_ALIGN; }

// ------------------------------------------------------------------------------------------------ insertion

static int nfiles;
static int *fds;
static uint32_t *ins_pos, *ins_bucket;
static size_t ins_n, ins_alloc;

// FIT: the spans (memory entry of the first token, length) of every rank's batch in the last fit_k steps.
static uint64_t fit_k, fit_freeze;
typedef struct { uint32_t *start, *len, *batch_first; size_t n, alloc, nbatch; } fit_spans_t;
static fit_spans_t *fspans;

// Reads the tokens [a, e) of training shard file_idx into dst.
static void read_span(uint32_t file_idx, uint32_t a, uint32_t e, uint16_t *dst) {
    if (file_idx >= (uint32_t)nfiles) die(5, "span from file %u of %d", file_idx, nfiles);
    if (e <= a) die(5, "empty span [%u, %u)", a, e);
    size_t nbytes = 2 * (size_t)(e - a);
    off_t off = SHARD_HEADER_BYTES + 2 * (off_t)a;
    read_all(fds[file_idx], dst, nbytes, off, "a training shard");
    hdr[H_BYTES_READ] += nbytes;
    if (readlog) fprintf(readlog, "train %u %lld %zu\n", file_idx, (long long)off, nbytes);
}

static void fit_note(uint64_t r, uint32_t start, uint32_t len, int first_of_batch) {
    fit_spans_t *f = &fspans[r];
    if (f->n == f->alloc) {
        f->alloc = f->alloc ? 2 * f->alloc : 1024;
        f->start = xgrow(f->start, 4 * f->alloc);
        f->len = xgrow(f->len, 4 * f->alloc);
    }
    if (first_of_batch) {
        f->batch_first = xgrow(f->batch_first, 4 * (f->nbatch + 1));
        f->batch_first[f->nbatch++] = (uint32_t)f->n;
    }
    f->start[f->n] = start;
    f->len[f->n++] = len;
}

// One STEP message: append every span (and a separator), then index the step's positions in stream order, and hand
// the step's tokens to P2's insertion threads (a copy; they insert it while the next steps train).
// fit: this step is one of the last fit_k (its spans are kept per rank for the FIT queries); late: one of the last
// LATE_STEPS (P2's backlog there is what GO may have to wait for).
static void insert_step(uint32_t file_idx, uint64_t world, const uint32_t *counts, const uint32_t *spans, int fit,
                        int late) {
    uint64_t t0 = now_ns();
    const uint64_t step_start = entries;
    ins_n = 0;
    for (uint64_t r = 0, s = 0; r < world; r++) {
        for (uint32_t i = 0; i < counts[r]; i++, s++) {
            uint32_t a = spans[2 * s], e = spans[2 * s + 1];
            uint64_t len = e - a, start = entries;
            if (e <= a || entries + len + 1 > cap) die(6, "the stream outgrew its %llu entries", (unsigned long long)cap);
            read_span(file_idx, a, e, tok + start);
            if (fit) fit_note(r, (uint32_t)start, (uint32_t)len, i == 0);
            entries += len;
            tok[entries++] = SEP;
            if (ins_n + len > ins_alloc) {
                ins_alloc = 2 * (ins_n + len);
                ins_pos = xgrow(ins_pos, ins_alloc * 4);
                ins_bucket = xgrow(ins_bucket, ins_alloc * 4);
            }
            // j has >= KEY tokens of the span up to and including it, and j + 1 is in the span
            for (uint64_t j = start + KEY - 1; j + 1 < start + len; j++) {
                ins_pos[ins_n] = (uint32_t)j;
                ins_bucket[ins_n++] = bucket_of(tok + j - (KEY - 1));
            }
        }
        if (fit && !counts[r]) die(8, "rank %llu has no span in a FIT step", (unsigned long long)r);
    }
    if (lt && entries > step_start && lt_insert_block(lt, tok + step_start, entries - step_start))
        die(13, "low-order tables: %s", lt_error(lt));
    for (size_t i = 0; i < ins_n; i++) {  // the head[] lines miss the cache: prefetch them 16 entries ahead
        if (i + 16 < ins_n) __builtin_prefetch(&head[ins_bucket[i + 16]], 1);
        uint32_t h = ins_bucket[i], j = ins_pos[i];
        prev[j] = head[h];
        head[h] = j;
    }
    inserted += ins_n;
    hdr[H_INSERTED] = inserted;
    hdr[H_ENTRIES] = entries;
    hdr[H_INSERT_NS] += now_ns() - t0;
    if (lt) {  // P2's backlog: released blocks its slowest insertion thread has not finished
        uint64_t w[LT_STATS_WORDS];
        lt_stats(lt, w, LT_STATS_WORDS);
        if (w[LT_S_PENDING] > hdr[H_LOW_MAX_PENDING]) hdr[H_LOW_MAX_PENDING] = w[LT_S_PENDING];
        if (late && w[LT_S_PENDING] > hdr[H_LOW_LATE_PENDING]) hdr[H_LOW_LATE_PENDING] = w[LT_S_PENDING];
    }
}

// ------------------------------------------------------------------------------------------------ queries

// The full match length of a candidate whose capped length is MAXLEN: extended backward past MAXLEN while both
// sides agree, within t's segment (`run` tokens) and below FULLCAP; the memory side stops at its span's separator.
static inline uint32_t full_len(const uint16_t *x, const uint16_t *m, uint32_t l, uint32_t run) {
    if (l < MAXLEN) return l;
    const uint32_t lim = run < FULLCAP ? run : FULLCAP;
    while (l < lim && m[-(ptrdiff_t)l] == x[-(ptrdiff_t)l]) l++;
    return l;
}

// The candidates of position t (x points at its input token, `run` = tokens of t's segment up to and including t,
// saturated at FULLCAP; `limit` = the first memory entry the walk ignores): memory entries, match lengths capped at
// min(run, MAXLEN), next tokens, in walk order (most recent first). Returns their number.
static int walk(const uint16_t *x, uint32_t run, uint32_t limit, uint32_t *pos, uint8_t *lens, uint16_t *next) {
    if (run < KEY) return 0;
    const uint32_t lim = run < MAXLEN ? run : MAXLEN;
    int nc = 0, visited = 0;
    uint32_t p = head[bucket_of(x - (KEY - 1))];
    while (p >= limit) p = prev[p];  // FIT: entries from the freeze on are newer than all others; not visits
    for (; p && nc < CAP && visited < MAXVISIT; p = prev[p]) {
        visited++;
        const uint16_t *m = tok + p;
        if (m[0] != x[0] || m[-1] != x[-1] || m[-2] != x[-2] || m[-3] != x[-3] || m[-4] != x[-4] || m[-5] != x[-5])
            continue;
        // Extend backward. The memory side stops at its span's start by itself (a separator precedes every span
        // and never equals a token); the val side stops at its segment's start through `lim`.
        uint32_t l = KEY;
        while (l < lim && m[-(ptrdiff_t)l] == x[-(ptrdiff_t)l]) l++;
        pos[nc] = p;
        lens[nc] = (uint8_t)l;
        next[nc++] = m[1];
    }
    return nc;
}

// The record of position t from its nc > 0 candidates.
static void make_record(const uint16_t *x, uint32_t run, const uint32_t *pos, const uint8_t *lens, const uint16_t *next,
                        int nc, rec_t *r) {
    memset(r, 0, sizeof *r);
    uint32_t maxl = 0;
    for (int i = 0; i < nc; i++) {
        maxl = lens[i] > maxl ? lens[i] : maxl;
        r->nx[i] = next[i];
        r->len[i] = lens[i];
    }
    int ls = 0;
    for (int k = 0; k < NLEVELS; k++) if (LEVELS[k] <= maxl) ls = k;
    // per level: group the candidates by next token (indices sorted by token; insertion sort, nc <= 32)
    uint8_t ord[CAP];
    for (int i = 0; i < nc; i++) {
        int q = i;
        while (q && next[ord[q - 1]] > next[i]) { ord[q] = ord[q - 1]; q--; }
        ord[q] = (uint8_t)i;
    }
    for (int k = 0; k <= ls; k++) {
        const uint32_t L = LEVELS[k];
        uint32_t n = 0, d = 0, mm = 0, n1 = 0, n2 = 0;
        uint16_t top = 0;
        for (int i = 0; i < nc;) {
            int q = i;
            uint32_t c = 0;
            const uint16_t v = next[ord[i]];
            for (; q < nc && next[ord[q]] == v; q++) c += lens[ord[q]] >= L;
            if (c) {
                n += c;
                d++;
                n1 += c == 1;
                n2 += c == 2;
                if (c > mm) { mm = c; top = v; }  // tokens ascending: ties keep the lowest id
            }
            i = q;
        }
        r->n[k] = (uint8_t)n; r->d[k] = (uint8_t)d; r->m[k] = (uint8_t)mm; r->n1[k] = (uint8_t)n1; r->n2[k] = (uint8_t)n2;
        r->top[k] = top;
    }
    // the most recent candidate at L*, and the longest full match (the most recent on ties)
    int irec = 0;
    while (lens[irec] < LEVELS[ls]) irec++;
    const uint32_t best_possible = run < FULLCAP ? run : FULLCAP;
    uint32_t lrec = full_len(x, tok + pos[irec], lens[irec], run), best = 0;
    int ilong = 0;
    for (int i = 0; i < nc && best < best_possible; i++) {
        if (lens[i] <= best && lens[i] < MAXLEN) continue;  // its full length is its capped length: no better
        uint32_t l = i == irec ? lrec : full_len(x, tok + pos[i], lens[i], run);
        if (l > best) { best = l; ilong = i; }
    }
    r->pos_recent = pos[irec];
    r->len_recent = (uint16_t)lrec;
    r->pos_longest = pos[ilong];
    r->len_longest = (uint16_t)best;
    r->ncand = (uint8_t)nc;
    r->lstar = (uint8_t)ls;
}

// The record of position t (see walk). Returns 0 if nothing matched. (Not called by the helper: test_stream_pointer.py
// compiles this file and checks P3's memory summary against it.)
__attribute__((unused)) static int query_one(const uint16_t *x, uint32_t run, uint32_t limit, rec_t *r) {
    uint32_t pos[CAP];
    uint8_t lens[CAP];
    uint16_t next[CAP];
    const int nc = walk(x, run, limit, pos, lens, next);
    if (!nc) return 0;
    make_record(x, run, pos, lens, next, nc, r);
    return 1;
}

// t - sigma_t + 1, saturated at FULLCAP: t's segment restarts at a BOS and at the start of its chunk. A query block
// finds it for its first position by looking back, then counts forward.
static uint32_t run_at(const uint16_t *x, size_t t, size_t chunk_start) {
    uint32_t r = 1;
    for (size_t s = t; s > chunk_start && x[s] != BOS && r < FULLCAP; s--) r++;
    return r;
}

// Each query thread appends its blocks' records and P3 rows to its own arena, which the gather then copies into the
// rows file in position order: one copy per row on the clock. The arenas are allocated and touched before the clock
// (arenas_create, at the expected share of val's rows; they grow by mremap past it) and reused by the FIT queries.
typedef struct { rec_t *recs; size_t nrec, rec_bytes; sp_row_t *prow; size_t nptr, ptr_bytes; } arena_t;
#define MAX_QUERY_THREADS 256
static arena_t arenas[MAX_QUERY_THREADS];
#define ARENA_REC_SHARE 0.10   // P1 records: 7.4-7.6% of val positions on the real stream
#define ARENA_PTR_SHARE 0.40   // P3 rows: 33.4-34.1%

static void *arena_resize(void *p, size_t *bytes, size_t need) {
    if (need <= *bytes) return p;
    size_t nb = *bytes ? 2 * *bytes : (2u << 20);
    if (nb < need) nb = need;
    nb = (nb + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    void *q = *bytes ? mremap(p, *bytes, nb, MREMAP_MAYMOVE)
                     : mmap(NULL, nb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (q == MAP_FAILED) die(1, "out of memory (a query arena of %zu bytes)", nb);
    if (!*bytes) madvise(q, nb, MADV_HUGEPAGE);
    *bytes = nb;
    return q;
}

// Before the clock: every query thread's arena, touched at the expected share of `positions` (val) per thread.
static void arenas_create(int nt, uint64_t positions, int with_ptr) {
    for (int k = 0; k < nt; k++) {
        arena_t *A = &arenas[k];
        A->recs = arena_resize(A->recs, &A->rec_bytes, (size_t)(ARENA_REC_SHARE * positions / nt + 1) * sizeof(rec_t));
        for (size_t i = 0; i < A->rec_bytes; i += 4096) ((volatile char *)A->recs)[i] = 0;
        if (with_ptr) {
            A->prow = arena_resize(A->prow, &A->ptr_bytes, (size_t)(ARENA_PTR_SHARE * positions / nt + 1) * sizeof(sp_row_t));
            for (size_t i = 0; i < A->ptr_bytes; i += 4096) ((volatile char *)A->prow)[i] = 0;
        }
    }
}

// A query block: positions [lo, hi) of one chunk; its rows are n records at arenas[tid].recs + roff and np P3 rows at
// arenas[tid].prow + poff.
typedef struct { uint64_t lo, hi, roff, poff; uint32_t n, np, tid; } block_out_t;

typedef struct {
    const uint16_t *x;       // inputs of nchunks chunks of `chunk` positions each (x[t] is t's input)
    const uint16_t *y;       // their targets (P2, P3), or NULL
    const uint32_t *runs;    // per position (FIT), or NULL: from BOS and the chunk starts
    uint64_t chunk, nchunks;
    uint32_t limit;          // memory entries >= limit are ignored (FIT); else above every entry
    int do_rec, do_ptr;      // P1 records; P3 rows (then blocks are whole segments)
    lt_row_t **low;          // P2: per chunk, the destination of its [chunk][norders] rows; NULL: no P2
    uint64_t blocks_per_chunk, nblocks;
    block_out_t *out;        // per block (index order = position order): its bounds and rows
    uint32_t *order;         // the blocks in the order they are handed out: longest first
    atomic_size_t next, bnext;
    atomic_int tids;
    uint64_t hits, cands, levels[NLEVELS], prows;
    pthread_mutex_t lock;
} query_job_t;

static inline int seg_start_at(const query_job_t *job, size_t t, size_t cs) {
    return t == cs || (job->runs ? job->runs[t] == 1 : job->x[t] == BOS);
}

// With P3 a block is whole segments: the segments that start in its nominal range [lo, hi) of chunk [cs, ce).
static size_t seg_bound(const query_job_t *job, size_t t, size_t cs, size_t ce) {
    while (t < ce && !seg_start_at(job, t, cs)) t++;
    return t;
}

// The blocks' bounds (on the query threads: with P3 each bound scans to the next segment start).
static void *bounds_worker(void *arg) {
    query_job_t *job = arg;
    for (;;) {
        const size_t b = atomic_fetch_add(&job->bnext, 1);
        if (b >= job->nblocks) break;
        const size_t c = b / job->blocks_per_chunk, cs = c * job->chunk, ce = cs + job->chunk;
        size_t lo = cs + (b % job->blocks_per_chunk) * BLOCK;
        size_t hi = lo + BLOCK < ce ? lo + BLOCK : ce;
        if (job->do_ptr) {
            lo = seg_bound(job, lo, cs, ce);
            hi = seg_bound(job, hi, cs, ce);
        }
        job->out[b].lo = lo;
        job->out[b].hi = hi > lo ? hi : lo;
    }
    return NULL;
}

static int cmp_u64(const void *a, const void *b) {
    const uint64_t u = *(const uint64_t *)a, v = *(const uint64_t *)b;
    return u < v ? -1 : u > v;
}

static void *query_worker(void *arg) {
    query_job_t *job = arg;
    const int tid = atomic_fetch_add(&job->tids, 1);
    if (tid >= MAX_QUERY_THREADS) die(1, "more than %d query threads", MAX_QUERY_THREADS);
    arena_t *A = &arenas[tid];
    uint64_t hits = 0, cands = 0, levels[NLEVELS] = {0}, prows = 0;
    size_t rcap = 0;
    uint32_t *runbuf = NULL;
    sp_state_t *st = job->do_ptr ? sp_state_new(&sp_cfg) : NULL;
    if (job->do_ptr && !st) die(1, "out of memory (pointer state)");
    const int no = lt ? lt->norders : 0;
    for (;;) {
        const size_t i = atomic_fetch_add(&job->next, 1);
        if (i >= job->nblocks) break;
        const size_t b = job->order[i];
        block_out_t *o = &job->out[b];
        const size_t c = b / job->blocks_per_chunk, cs = c * job->chunk, lo = o->lo, hi = o->hi;
        o->n = o->np = 0;
        o->tid = (uint32_t)tid;
        o->roff = A->nrec;
        o->poff = A->nptr;
        if (lo >= hi) continue;
        const size_t len = hi - lo;
        // room for a row per position, written in place (the arena grows by mremap: its offsets stay valid)
        if (job->do_rec) A->recs = arena_resize(A->recs, &A->rec_bytes, (A->nrec + len) * sizeof(rec_t));
        if (job->do_ptr) A->prow = arena_resize(A->prow, &A->ptr_bytes, (A->nptr + len) * sizeof(sp_row_t));
        rec_t *recs = job->do_rec ? A->recs + A->nrec : NULL;
        sp_row_t *prs = job->do_ptr ? A->prow + A->nptr : NULL;
        if (job->low && len > rcap) runbuf = xgrow(runbuf, 4 * (rcap = len));
        uint32_t run = job->runs ? 0 : run_at(job->x, lo, cs), nrec = 0, np = 0;
        for (size_t t = lo; t < hi; t++) {
            if (job->runs) run = job->runs[t];
            else if (t > lo) run = job->x[t] == BOS ? 1 : (run < FULLCAP ? run + 1 : FULLCAP);
            if (job->low) runbuf[t - lo] = run;
            if (!job->do_rec && !job->do_ptr) continue;
            uint32_t pos[CAP];
            uint8_t lens[CAP];
            uint16_t next[CAP];
            const int nc = walk(job->x + t, run, job->limit, pos, lens, next);
            if (job->do_rec && nc) {
                rec_t *r = recs + nrec++;
                make_record(job->x + t, run, pos, lens, next, nc, r);
                r->pos = (uint32_t)(t - cs);
                hits++;
                cands += r->ncand;
                levels[r->lstar]++;
            }
            if (job->do_ptr) {
                sp_row_t *pr = prs + np;
                if (sp_step(st, tok, entries, job->x + t, run, pos, lens, nc, job->y[t], pr)) {
                    pr->pos = (uint32_t)(t - cs);
                    np++;
                }
            }
        }
        if (job->low) lt_query_block(lt, job->x + lo, job->y + lo, runbuf, len, job->low[c] + (lo - cs) * (size_t)no);
        o->n = nrec;
        o->np = np;
        A->nrec += nrec;
        A->nptr += np;
        prows += np;
    }
    free(runbuf);
    if (st) sp_state_free(st);
    pthread_mutex_lock(&job->lock);
    job->hits += hits;
    job->cands += cands;
    job->prows += prows;
    for (int k = 0; k < NLEVELS; k++) job->levels[k] += levels[k];
    pthread_mutex_unlock(&job->lock);
    return NULL;
}

static int query_threads = 32;

static int nthreads(void) { return query_threads < 1 ? 1 : (query_threads > MAX_QUERY_THREADS ? MAX_QUERY_THREADS : query_threads); }

static void run_threads(void *(*fn)(void *), void *arg) {
    const int nt = nthreads();
    pthread_t th[MAX_QUERY_THREADS];
    for (int k = 1; k < nt; k++)
        if (pthread_create(&th[k], NULL, fn, arg)) die(1, "pthread_create failed");
    fn(arg);
    for (int k = 1; k < nt; k++) pthread_join(th[k], NULL);
}

// The job's blocks: their bounds, then handed out longest first (with P3 a block runs whole segments, and a long one
// started last would set the end of the queries alone), each thread's rows into its arena.
static void run_queries(query_job_t *job) {
    job->blocks_per_chunk = (job->chunk + BLOCK - 1) / BLOCK;
    job->nblocks = job->nchunks * job->blocks_per_chunk;
    job->out = calloc(job->nblocks ? job->nblocks : 1, sizeof(block_out_t));
    job->order = xalloc(sizeof(uint32_t) * (job->nblocks ? job->nblocks : 1));
    if (!job->out) die(1, "out of memory");
    for (int k = 0; k < MAX_QUERY_THREADS; k++) arenas[k].nrec = arenas[k].nptr = 0;
    atomic_init(&job->bnext, 0);
    run_threads(bounds_worker, job);
    uint64_t *keys = xalloc(8 * (job->nblocks ? job->nblocks : 1));
    for (uint64_t b = 0; b < job->nblocks; b++)  // length descending, then index: a deterministic order
        keys[b] = (uint64_t)(0xFFFFFFFFu - (uint32_t)(job->out[b].hi - job->out[b].lo)) << 32 | b;
    qsort(keys, job->nblocks, 8, cmp_u64);
    for (uint64_t i = 0; i < job->nblocks; i++) job->order[i] = (uint32_t)keys[i];
    free(keys);
    atomic_init(&job->next, 0);
    atomic_init(&job->tids, 0);
    job->hits = job->cands = job->prows = 0;
    memset(job->levels, 0, sizeof job->levels);
    pthread_mutex_init(&job->lock, NULL);
    run_threads(query_worker, job);
    pthread_mutex_destroy(&job->lock);
}

static void free_job(query_job_t *job) {
    free(job->out);
    free(job->order);
    job->out = NULL;
    job->order = NULL;
}

// Chunk c's records and P3 rows, in position order, from the arenas to dst / pdst (room for a whole chunk each, or
// NULL); returns their counts.
static void gather_chunk(const query_job_t *job, uint64_t c, char *dst, char *pdst, uint64_t *n, uint64_t *np) {
    *n = *np = 0;
    for (uint64_t b = c * job->blocks_per_chunk; b < (c + 1) * job->blocks_per_chunk; b++) {
        const block_out_t *o = &job->out[b];
        const arena_t *A = &arenas[o->tid];
        if (o->n) memcpy(dst + *n * sizeof(rec_t), A->recs + o->roff, sizeof(rec_t) * o->n);
        if (o->np) memcpy(pdst + *np * sizeof(sp_row_t), A->prow + o->poff, sizeof(sp_row_t) * o->np);
        *n += o->n;
        *np += o->np;
    }
}

// The rows file's regions: chunk c = step * world + rank goes to region rank * val_steps + step.
static struct { uint64_t world, val_steps, chunk, low_offset, ptr_offset; } lay;

// Before the clock: the rows file's pages GO will write (tmpfs allocates a page at its first write, ~1 us each):
// P2's region whole, and of every P1 / P3 region the share its rows fill on FineWeb (7.4% / 33.4% of positions), with
// room (15% / 50%).
#define PREFAULT_REC_SHARE 0.15
#define PREFAULT_PTR_SHARE 0.50
static void touch(char *p, uint64_t n) {
    for (uint64_t i = 0; i < n; i += 4096) ((volatile char *)p)[i] = 0;
}

static char *rec_region(uint64_t region) { return rows + ROWS_OFFSET + region * lay.chunk * sizeof(rec_t); }
static char *ptr_region(uint64_t region) { return rows + lay.ptr_offset + region * lay.chunk * sizeof(sp_row_t); }
static lt_row_t *low_region(uint64_t region) {
    return (lt_row_t *)(rows + lay.low_offset + region * lay.chunk * LOW_NORDERS * sizeof(lt_row_t));
}

// The val gather, one chunk at a time on the query threads: its records and P3 rows to the chunk's regions, their
// counts to the header.
typedef struct { query_job_t *job; atomic_size_t next; } gather_t;

static void *gather_worker(void *arg) {
    gather_t *g = arg;
    for (;;) {
        size_t c = atomic_fetch_add(&g->next, 1);
        if (c >= g->job->nchunks) break;
        const uint64_t s = c / lay.world, r = c % lay.world, region = r * lay.val_steps + s;
        uint64_t n, np;
        gather_chunk(g->job, c, rec_region(region), pointer_on ? ptr_region(region) : NULL, &n, &np);
        ((volatile uint64_t *)((char *)hdr + COUNTS_OFFSET))[region] = n;
        ((volatile uint64_t *)((char *)hdr + PCOUNTS_OFFSET))[region] = np;
    }
    return NULL;
}

// The checksum of one chunk's L + 1 val tokens (its inputs and its last target), as stream_memory.py computes it.
static uint64_t chunk_checksum(const uint16_t *x, uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; i++) s += ((uint64_t)x[i] + 1) * (i * 0x9e3779b97f4a7c15ull + 1);
    return s;
}

// The val checks, on a thread of their own while the queries run (neither feeds the other): no val token equals
// the separator (the backward match extension relies on it), and every chunk's checksum, for stream_memory.check().
typedef struct { const uint16_t *val; uint64_t n, chunks, chunk; int sep_found; } val_check_t;

static void *val_check_worker(void *arg) {
    val_check_t *c = arg;
    for (uint64_t i = 0; i <= c->n; i++) c->sep_found |= c->val[i] == SEP;
    for (uint64_t k = 0; k < c->chunks; k++)
        ((volatile uint64_t *)((char *)hdr + CHECKSUM_OFFSET))[k] = chunk_checksum(c->val + k * c->chunk, c->chunk + 1);
    return NULL;
}

// ------------------------------------------------------------------------------------------------ messages

static struct {
    uint64_t world, val_tokens, chunk, val_steps, total_steps;
    const char *fit_path, *dump_path;
    int vfd;
} cfg;
static uint64_t steps;
static int went, fitted;

static void write_file(const char *path, const void *data, size_t size, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f || (n && fwrite(data, size, n, f) != n) || fclose(f)) die(11, "cannot write %s", path);
}

// The FIT positions: every rank's last fit_k batches as one sequence (inputs buf[:-1], targets buf[1:] per batch;
// segments restart at each batch and each BOS). Every rank's batch k has the same length (the schedule's batch
// size). Built once, from tok[] (the K steps are appended there, past the freeze).
static struct { uint64_t n; uint64_t *blen; uint16_t *x, *y; uint32_t *runs; lt_row_t *low; } fs;

static void build_fit_seq(void) {
    if (fs.x) return;
    const uint64_t world = cfg.world;
    uint64_t n = 0, *blen = xalloc(8 * fit_k);
    for (uint64_t r = 0; r < world; r++) {
        uint64_t nr = 0;
        fit_spans_t *f = &fspans[r];
        if (f->nbatch != fit_k) die(12, "rank %llu has %zu FIT batches of %llu", (unsigned long long)r, f->nbatch,
                                    (unsigned long long)fit_k);
        for (size_t b = 0; b < f->nbatch; b++) {
            const size_t e = b + 1 < f->nbatch ? f->batch_first[b + 1] : f->n;
            uint64_t len = 0;
            for (size_t i = f->batch_first[b]; i < e; i++) len += f->len[i];
            if (len < 2) die(12, "a FIT batch of %llu tokens", (unsigned long long)len);
            if (r && blen[b] != len - 1)
                die(12, "FIT batch %zu holds %llu positions on rank 0 and %llu on rank %llu", b, (unsigned long long)blen[b],
                    (unsigned long long)(len - 1), (unsigned long long)r);
            blen[b] = len - 1;
            nr += len - 1;
        }
        n = nr;
    }
    uint16_t *x = xalloc(2 * world * n), *y = xalloc(2 * world * n);
    uint32_t *runs = xalloc(4 * world * n);
    for (uint64_t r = 0; r < world; r++) {
        fit_spans_t *f = &fspans[r];
        uint64_t o = r * n;
        for (size_t b = 0; b < f->nbatch; b++) {
            const size_t e = b + 1 < f->nbatch ? f->batch_first[b + 1] : f->n;
            uint64_t len = 0, k = 0;
            for (size_t i = f->batch_first[b]; i < e; i++) len += f->len[i];
            uint16_t *buf = xalloc(2 * len);
            for (size_t i = f->batch_first[b]; i < e; i++) {
                memcpy(buf + k, tok + f->start[i], 2 * (size_t)f->len[i]);
                k += f->len[i];
            }
            uint32_t run = 0;
            for (uint64_t j = 0; j + 1 < len; j++, o++) {
                x[o] = buf[j];
                y[o] = buf[j + 1];
                run = (j == 0 || buf[j] == BOS) ? 1 : (run < FULLCAP ? run + 1 : FULLCAP);
                runs[o] = run;
            }
            free(buf);
        }
    }
    fs.n = n;
    fs.blen = blen;
    fs.x = x;
    fs.y = y;
    fs.runs = runs;
}

// P2's FIT rows (low=1, fit=): once the last step has arrived, its K steps' blocks still held, the tables are the
// memory as it stood before the first of them. Then the held blocks are released (GO's lt_finish waits for them).
static void low_fit_queries(void) {
    const uint64_t t0 = now_ns();
    build_fit_seq();
    if (lt_sync(lt)) die(13, "low-order tables: %s", lt_error(lt));
    const uint64_t world = cfg.world;
    fs.low = xalloc(sizeof(lt_row_t) * LOW_NORDERS * world * fs.n);
    lt_row_t **dst = xalloc(sizeof *dst * world);
    for (uint64_t r = 0; r < world; r++) dst[r] = fs.low + r * fs.n * LOW_NORDERS;
    query_job_t job = {.x = fs.x, .y = fs.y, .runs = fs.runs, .chunk = fs.n, .nchunks = world, .limit = UINT32_MAX,
                       .low = dst};
    run_queries(&job);
    free_job(&job);
    free(dst);
    if (lt_hold(lt, 0)) die(13, "low-order tables: %s", lt_error(lt));
    hdr[H_LOW_FIT_NS] = now_ns() - t0;
}

// P2 at GO: lt_finish() (every step's block in, or the error) on a thread of its own, so that the insertion threads
// drain their backlog while the P1 / P3 queries run.
typedef struct { int rc; uint64_t done_ns; } low_finish_t;

static void *low_finish_worker(void *arg) {
    low_finish_t *f = arg;
    f->rc = lt_finish(lt);
    f->done_ns = now_ns();
    return NULL;
}

// GO: every training step is in the memory. Read val and compute every position's rows (and the checksums): P1's
// records and P3's rows first, while P2's insertion threads finish (P1's walk and P3 read only tok / head / prev,
// complete since the last step's message), then P2's rows once its tables are complete.
static void on_go(uint32_t t) {
    hdr[H_T_GO] = now_ns();
    if (went) die(9, "a second GO");
    if (t != cfg.total_steps || steps != cfg.total_steps)
        die(9, "GO for %u steps after %llu step messages; the memory expects %llu", t, (unsigned long long)steps,
            (unsigned long long)cfg.total_steps);
    went = 1;
    const uint64_t n = cfg.val_tokens;
    uint16_t *val = xalloc(2 * (n + 1));
    int32_t vh[3];
    read_all(cfg.vfd, vh, sizeof vh, 0, "the val shard's header");
    if (vh[0] != SHARD_MAGIC || vh[1] != 1 || (uint64_t)vh[2] <= n + 1)  // as the val loader: no second val shard
        die(10, "the val shard has a bad header or <= %llu tokens", (unsigned long long)(n + 1));
    read_all(cfg.vfd, val, 2 * (n + 1), SHARD_HEADER_BYTES, "the val shard");
    if (readlog) fprintf(readlog, "val 0 %d %llu\n", SHARD_HEADER_BYTES, (unsigned long long)(2 * (n + 1)));
    hdr[H_BYTES_READ] += 2 * (n + 1);
    val_check_t check = {.val = val, .n = n, .chunks = cfg.world * cfg.val_steps, .chunk = cfg.chunk};
    pthread_t check_thread;
    if (pthread_create(&check_thread, NULL, val_check_worker, &check)) die(1, "pthread_create failed");
    const uint64_t nchunks = cfg.world * cfg.val_steps;
    pthread_t finisher;
    low_finish_t fin = {0};
    if (lt && pthread_create(&finisher, NULL, low_finish_worker, &fin)) die(1, "pthread_create failed");
    uint64_t t0 = now_ns();
    query_job_t job = {.x = val, .y = val + 1, .chunk = cfg.chunk, .nchunks = nchunks, .limit = UINT32_MAX,
                       .do_rec = 1, .do_ptr = pointer_on};
    run_queries(&job);
    gather_t g = {.job = &job};
    atomic_init(&g.next, 0);
    run_threads(gather_worker, &g);
    free_job(&job);
    if (lt) {  // every step's block is in (the completion check), the tables frozen: P2's rows
        const uint64_t tw = now_ns();
        pthread_join(finisher, NULL);
        hdr[H_LOW_WAIT_NS] = now_ns() - tw;  // P2's drain past the P1 / P3 queries: the part GO waits for
        hdr[H_LOW_DRAIN_NS] = fin.done_ns > hdr[H_T_GO] ? fin.done_ns - hdr[H_T_GO] : 0;
        if (fin.rc) die(13, "low-order tables: %s", lt_error(lt));
        lt_row_t **low = xalloc(sizeof *low * nchunks);
        for (uint64_t c = 0; c < nchunks; c++) low[c] = low_region((c % lay.world) * lay.val_steps + c / lay.world);
        query_job_t lj = {.x = val, .y = val + 1, .chunk = cfg.chunk, .nchunks = nchunks, .limit = UINT32_MAX,
                          .low = low};
        run_queries(&lj);
        free_job(&lj);
        free(low);
    }
    hdr[H_QUERY_NS] = now_ns() - t0;
    pthread_join(check_thread, NULL);
    if (check.sep_found) die(10, "a val token equals the separator");
    hdr[H_QUERIED] = n;
    hdr[H_HITS] = job.hits;
    hdr[H_CANDS] = job.cands;
    hdr[H_PTR_ROWS] = job.prows;
    for (int k = 0; k < NLEVELS; k++) hdr[H_LEVEL0 + k] = job.levels[k];
    if (lt) {
        uint64_t w[LT_STATS_WORDS];
        lt_stats(lt, w, LT_STATS_WORDS);
        hdr[H_LOW_BYTES] = w[LT_S_BYTES];
        hdr[H_LOW_CPU_NS] = w[LT_S_CPU_NS];
    }
    free(val);
    if (cfg.dump_path) write_file(cfg.dump_path, tok + 1, 2, entries - 1);
    if (readlog) fflush(readlog);
    hdr[H_T_DONE] = now_ns();
    __atomic_store_n(&hdr[H_STATE], ST_DONE, __ATOMIC_RELEASE);
}

// FIT (after the val rows are out): the FIT positions' P1 records and P3 rows against the memory before the freeze.
// PATH: u64 [16] (magic, world, positions per rank, record bytes, fit_k, freeze, parts (1 low | 2 pointer), low
// orders, the run's trained steps, 0...), u64 [world] record counts, u64 [fit_k] positions per batch, then every rank's records; PATH.tokens:
// u16 [world][positions][2] (input, target); PATH.low: lt_row_t [world][positions][low orders]; PATH.ptr: u64 [world]
// row counts, then every rank's P3 rows.
static void on_fit(void) {
    const uint64_t world = cfg.world;
    build_fit_seq();
    const uint64_t n = fs.n;
    query_job_t job = {.x = fs.x, .y = fs.y, .runs = fs.runs, .chunk = n, .nchunks = world,
                       .limit = (uint32_t)fit_freeze, .do_rec = 1, .do_ptr = pointer_on};
    run_queries(&job);
    char *recs = xalloc(sizeof(rec_t) * n * world), *prow = pointer_on ? xalloc(sizeof(sp_row_t) * n * world) : NULL;
    uint64_t *counts = xalloc(8 * world), *pcounts = xalloc(8 * world), total = 0, ptotal = 0;
    for (uint64_t r = 0; r < world; r++) {
        gather_chunk(&job, r, recs + total * sizeof(rec_t), prow ? prow + ptotal * sizeof(sp_row_t) : NULL, &counts[r],
                     &pcounts[r]);
        total += counts[r];
        ptotal += pcounts[r];
    }
    free_job(&job);
    const uint64_t parts = (lt ? 1u : 0u) | (pointer_on ? 2u : 0u);
    uint64_t head_words[FIT_HEAD_WORDS] = {FIT_MAGIC, world, n, sizeof(rec_t), fit_k, fit_freeze, parts,
                                           lt ? LOW_NORDERS : 0, cfg.total_steps};
    FILE *fo = fopen(cfg.fit_path, "wb");
    if (!fo || fwrite(head_words, 8, FIT_HEAD_WORDS, fo) != FIT_HEAD_WORDS || fwrite(counts, 8, world, fo) != world ||
        fwrite(fs.blen, 8, fit_k, fo) != fit_k || (total && fwrite(recs, sizeof(rec_t), total, fo) != total) || fclose(fo))
        die(11, "cannot write %s", cfg.fit_path);
    char path[4096];
    uint16_t *xy = xalloc(4 * world * n);
    for (uint64_t i = 0; i < world * n; i++) { xy[2 * i] = fs.x[i]; xy[2 * i + 1] = fs.y[i]; }
    snprintf(path, sizeof path, "%s.tokens", cfg.fit_path);
    write_file(path, xy, 4, world * n);
    free(xy);
    if (lt) {
        snprintf(path, sizeof path, "%s.low", cfg.fit_path);
        write_file(path, fs.low, sizeof(lt_row_t), world * n * LOW_NORDERS);
    }
    if (pointer_on) {
        snprintf(path, sizeof path, "%s.ptr", cfg.fit_path);
        FILE *fp = fopen(path, "wb");
        if (!fp || fwrite(pcounts, 8, world, fp) != world || (ptotal && fwrite(prow, sizeof(sp_row_t), ptotal, fp) != ptotal)
            || fclose(fp))
            die(11, "cannot write %s", path);
    }
    free(recs); free(prow); free(counts); free(pcounts);
    hdr[H_FIT_POSITIONS] = n * world;
    hdr[H_FIT_HITS] = total;
    hdr[H_FIT_PTR_ROWS] = ptotal;
    hdr[H_FIT_FREEZE] = fit_freeze;
    hdr[H_T_FIT_DONE] = now_ns();
    __atomic_store_n(&hdr[H_STATE], ST_FIT_DONE, __ATOMIC_RELEASE);
}

// ------------------------------------------------------------------------------------------------ main

static const char *arg_of(int argc, char **argv, const char *key, const char *dflt) {
    size_t k = strlen(key);
    for (int i = 1; i < argc && strcmp(argv[i], "--"); i++)
        if (!strncmp(argv[i], key, k) && argv[i][k] == '=') return argv[i] + k + 1;
    return dflt;
}

int main(int argc, char **argv) {
    // rows=PATH (made by rank 0, header + regions) val=PATH world=W val_tokens=N chunk=L steps=T
    // cap=ENTRIES prefault=ENTRIES [hash_bits=29] [threads=32] [fit=PATH fit_k=K]
    // [low=0|1 low_threads=8 low_tokens=N (the stream's expected tokens, for the tables' sizes) low_entries=N (tests:
    // entries per order and table, in place of the sizes measured on FineWeb)] [pointer=0|1],
    // and for tests and diagnostics [dump=PATH] (the stream) [readlog=PATH] (every read: file, byte offset, bytes)
    // -- TRAIN_SHARD...
    const char *rows_path = arg_of(argc, argv, "rows", NULL), *val_path = arg_of(argc, argv, "val", NULL);
    if (!rows_path || !val_path) { fprintf(stderr, "stream_memory: rows= and val= are required\n"); return 2; }
    cfg.world = strtoull(arg_of(argc, argv, "world", "8"), NULL, 10);
    cfg.val_tokens = strtoull(arg_of(argc, argv, "val_tokens", "10485760"), NULL, 10);
    cfg.chunk = strtoull(arg_of(argc, argv, "chunk", "262144"), NULL, 10);
    cfg.total_steps = strtoull(arg_of(argc, argv, "steps", "0"), NULL, 10);
    cfg.fit_path = arg_of(argc, argv, "fit", NULL);
    cfg.dump_path = arg_of(argc, argv, "dump", NULL);
    fit_k = strtoull(arg_of(argc, argv, "fit_k", "0"), NULL, 10);
    cap = strtoull(arg_of(argc, argv, "cap", "0"), NULL, 10);
    const uint64_t prefault = strtoull(arg_of(argc, argv, "prefault", "0"), NULL, 10);
    hash_bits = atoi(arg_of(argc, argv, "hash_bits", "29"));
    query_threads = atoi(arg_of(argc, argv, "threads", "32"));
    const int low_on = atoi(arg_of(argc, argv, "low", "0"));
    const int low_threads = atoi(arg_of(argc, argv, "low_threads", "8"));
    const uint64_t low_tokens = strtoull(arg_of(argc, argv, "low_tokens", "0"), NULL, 10);
    const uint64_t low_entries = strtoull(arg_of(argc, argv, "low_entries", "0"), NULL, 10);
    pointer_on = atoi(arg_of(argc, argv, "pointer", "0"));
    const char *readlog_path = arg_of(argc, argv, "readlog", NULL);

    int rfd = open(rows_path, O_RDWR);
    if (rfd < 0) { fprintf(stderr, "stream_memory: cannot open %s: %s\n", rows_path, strerror(errno)); return 2; }
    const uint64_t world = cfg.world, chunk = cfg.chunk;
    cfg.val_steps = world && chunk ? cfg.val_tokens / (world * chunk) : 0;
    const uint64_t nreg = world * cfg.val_steps;
    lay.world = world;
    lay.val_steps = cfg.val_steps;
    lay.chunk = chunk;
    lay.low_offset = align_up(ROWS_OFFSET + nreg * chunk * sizeof(rec_t));
    lay.ptr_offset = align_up(lay.low_offset + (low_on ? nreg * chunk * LOW_NORDERS * sizeof(lt_row_t) : 0));
    const uint64_t rows_bytes = lay.ptr_offset + (pointer_on ? nreg * chunk * sizeof(sp_row_t) : 0);
    void *rmap = mmap(NULL, (size_t)rows_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, rfd, 0);
    if (rmap == MAP_FAILED) { fprintf(stderr, "stream_memory: cannot map %s\n", rows_path); return 2; }
    close(rfd);
    hdr = rmap;
    rows = (char *)rmap;
    hdr[H_MAGIC] = HEADER_MAGIC;
    hdr[H_WORLD] = world;
    hdr[H_VAL_STEPS] = cfg.val_steps;
    hdr[H_CHUNK] = chunk;
    hdr[H_REC_BYTES] = sizeof(rec_t);
    hdr[H_LOW_ORDERS] = low_on ? LOW_NORDERS : 0;
    hdr[H_PTR] = pointer_on ? 1 : 0;
    hdr[H_LOW_OFFSET] = lay.low_offset;
    hdr[H_PTR_OFFSET] = lay.ptr_offset;
    hdr[H_LOW_ROW_BYTES] = sizeof(lt_row_t);
    hdr[H_PTR_ROW_BYTES] = sizeof(sp_row_t);
    if (!cfg.val_steps || cfg.val_steps * world * chunk != cfg.val_tokens || nreg > MAX_CHUNKS)
        die(2, "val_tokens=%llu is not a whole number of world=%llu x chunk=%llu steps (<= %d chunks)",
            (unsigned long long)cfg.val_tokens, (unsigned long long)world, (unsigned long long)chunk, MAX_CHUNKS);
    if (chunk >= (1ull << 32)) die(2, "chunk=%llu does not fit a record's position", (unsigned long long)chunk);
    if (hash_bits < 4 || hash_bits > 32) die(2, "bad hash_bits=%d", hash_bits);
    if (cap < 2 || cap > 0xFFFFFFF0ull) die(2, "cap=%llu entries is outside [2, 2^32)", (unsigned long long)cap);
    if (!cfg.fit_path != !fit_k || fit_k >= cfg.total_steps)
        die(2, "fit=PATH and fit_k=K (0 < K < steps=%llu) go together", (unsigned long long)cfg.total_steps);
    if (fit_k && !(fspans = calloc(world, sizeof *fspans))) die(1, "out of memory");
    sp_default_config(&sp_cfg);

    int sep = 1;
    while (sep < argc && strcmp(argv[sep], "--")) sep++;
    nfiles = argc - sep - 1;
    if (nfiles < 1) die(2, "no training shards after --");
    fds = xalloc(sizeof(int) * (size_t)nfiles);
    for (int i = 0; i < nfiles; i++)
        if ((fds[i] = open(argv[sep + 1 + i], O_RDONLY)) < 0) die(2, "cannot open %s", argv[sep + 1 + i]);
    if ((cfg.vfd = open(val_path, O_RDONLY)) < 0) die(2, "cannot open %s", val_path);
    if (readlog_path && !(readlog = fopen(readlog_path, "w"))) die(2, "cannot open %s", readlog_path);

    // Allocation before the clock: the stream and its chain (cap entries reserved, prefault touched), the heads, and
    // P2's tables (allocated and first touched by their owning threads).
    tok = big_alloc(2 * cap, 2 * prefault);
    prev = big_alloc(4 * cap, 4 * prefault);
    head = big_alloc(4ull << hash_bits, 4ull << hash_bits);
    tok[0] = SEP;
    if (low_on) {
        lt_config_t lc;
        memset(&lc, 0, sizeof lc);
        lc.norders = LOW_NORDERS;
        memcpy(lc.orders, LOW_ORDERS, sizeof LOW_ORDERS);
        lc.threads = low_threads;
        lc.prefault = 1;
        lc.expected_positions = low_tokens ? low_tokens : prefault;
        for (int i = 0; low_entries && i < LOW_NORDERS; i++)  // tests: an upper bound in place of the FineWeb defaults
            lc.ctx_entries[i] = lc.pair_entries[i] = lc.promoted_entries[i] = low_entries;
        char err[256];
        if (!(lt = lt_create(&lc, err, sizeof err))) die(13, "low-order tables: %s", err);
    }
    arenas_create(nthreads(), cfg.val_tokens, pointer_on);
    for (uint64_t region = 0; region < nreg; region++) {
        touch(rec_region(region), (uint64_t)(PREFAULT_REC_SHARE * chunk) * sizeof(rec_t));
        if (low_on) touch((char *)low_region(region), chunk * LOW_NORDERS * sizeof(lt_row_t));
        if (pointer_on) touch(ptr_region(region), (uint64_t)(PREFAULT_PTR_SHARE * chunk) * sizeof(sp_row_t));
    }
    hdr[H_ENTRIES] = entries;
    hdr[H_T_READY] = now_ns();
    __atomic_store_n(&hdr[H_STATE], ST_READY, __ATOMIC_RELEASE);

    // The message loop: until rank 0 closes the pipe.
    uint32_t *spans = NULL, counts[MAX_CHUNKS], kind, words[3];
    size_t spans_alloc = 0;
    while (read_words(&kind, 1, 1)) {
        if (kind == MSG_GO) {
            read_words(words, 1, 0);
            on_go(words[0]);
            continue;
        }
        if (kind == MSG_FIT) {
            read_words(words, 1, 0);
            if (!went || !cfg.fit_path || fitted || words[0] != fit_k)
                die(12, "FIT %u: %s", words[0], !went ? "before GO" : !cfg.fit_path ? "without fit=" : fitted ? "twice" : "not fit_k");
            fitted = 1;
            on_fit();
            continue;
        }
        if (kind != MSG_STEP) die(7, "unknown message kind %u", kind);
        read_words(words, 3, 0);  // step, file_idx, world
        if (words[2] != world) die(7, "a message for world %u, expected %llu", words[2], (unsigned long long)world);
        read_words(counts, world, 0);
        size_t nspans = 0;
        for (uint64_t r = 0; r < world; r++) nspans += counts[r];
        if (2 * nspans > spans_alloc) {
            spans_alloc = 2 * nspans;
            free(spans);
            spans = xalloc(4 * spans_alloc);
        }
        read_words(spans, 2 * nspans, 0);
        if (went) die(8, "a training step after GO");
        if (words[0] != steps || steps >= cfg.total_steps)
            die(8, "step message %u, expected %llu of %llu", words[0], (unsigned long long)steps,
                (unsigned long long)cfg.total_steps);
        const int fit = fit_k && steps >= cfg.total_steps - fit_k;
        if (fit && steps == cfg.total_steps - fit_k) {
            fit_freeze = entries;  // the memory before the last K steps
            if (lt && lt_hold(lt, 1)) die(13, "low-order tables: %s", lt_error(lt));  // P2: hold their blocks
        }
        insert_step(words[1], world, counts, spans, fit, steps + LATE_STEPS >= cfg.total_steps);
        hdr[H_STEPS] = ++steps;
        if (lt && fit_k && steps == cfg.total_steps) low_fit_queries();
    }
    if (readlog) fclose(readlog);
    return 0;
}
