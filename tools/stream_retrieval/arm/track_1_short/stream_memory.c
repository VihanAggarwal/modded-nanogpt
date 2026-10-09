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
// At GO (the last step) the helper reads the val shard and computes one row (a, b) per val position t
// (input val[t], target y = val[t+1]):
//   segment  sigma_t = max(the last BOS at or before t, the start of t's chunk): the context the model sees
//   walk     t's bucket, most recent first; verify the KEY context tokens; extend each match backward,
//            within both segments, up to MAXLEN; stop after CAP verified candidates or MAXVISIT entries
//   level    L* = the largest of LEVELS that some candidate's match length reaches; S = the candidates
//            whose match length is at least L*; N = |S|; M = the largest count of one next token in S;
//            C = the count of next tokens equal to y
//   weight   lambda = sigmoid(w0 + w1 log2 N + w2 M/N + w3 log2 L*)
//   row      (a, b) = (1 - lambda, lambda C / N); (1, 0) where nothing matched
// so that the validation mixes q = a p + b, i.e. (1 - lambda) p + lambda r(y) with r the next-token
// distribution of S, a function of the memory and val[<= t] only.
//
// Rows go to a file that every rank maps: a 4096-byte header (state word, statistics, a checksum of every
// chunk's val tokens) and f32 [world][val_steps][chunk][2]. One insert thread in message order and queries that
// are pure functions of the final memory: the rows are bit-identical across runs and query thread counts.
//
// Credits (stream_memory.py's docstring has the full list). The memory's layout and its row rule are those of
// PR #367's StreamIndex (Herman Brunborg, exact_match/src/stream.rs): the stream of every rank's documents (inputs
// plus the last target) step by step with a STOP after each, positions keyed on their last 6 tokens, resolved
// against the most recent occurrences, the deepest of the match levels reached, the row from the next tokens of
// the occurrences matching at least that deep, and its (length, count, top share) as the gate's inputs. No code
// from #367: this file (the hash chain, the C helper, the protocol, the query over the validation stream) is
// written for this branch. The index is the classic LZ77/zlib hash chain; the bucket hash uses MurmurHash3's
// 64-bit finalizer.
//
// Build: stream_memory.py's build_helper() (cc -O2 -std=c11 -pthread stream_memory.c -o stream_memory -lm)
// Usage: stream_memory key=value... -- TRAIN_SHARD...   (stdin: the messages below; keys in main())
// Messages (u32 words): STEP     1 step file_idx world n_0..n_{world-1} a e a e ...   (spans in rank order)
//                       GO       2 total_steps
//                       FIT_STEP 3 k file_idx world n_0..n_{world-1} a e ...          (query only, after GO)
//                       FIT_GO   4 k_total
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

#define SEP 0xFFFFu
#define BOS 50256u
#define KEY 6
#define MAXLEN 32
#define CAP 32
#define MAXVISIT 128
#define NLEVELS 6
static const uint32_t LEVELS[NLEVELS] = {6, 8, 12, 16, 24, 32};
#define SHARD_HEADER_BYTES 1024
#define SHARD_MAGIC 20240520
#define ROWS_OFFSET 4096
#define MAX_CHUNKS 384
#define BLOCK 4096

enum { MSG_STEP = 1, MSG_GO = 2, MSG_FIT_STEP = 3, MSG_FIT_GO = 4 };
enum { ST_STARTING = 0, ST_READY = 1, ST_DONE = 2, ST_FIT_DONE = 3, ST_ERROR = 14 };
// Header words (u64), mirrored by stream_memory.py's H_* constants.
enum { H_MAGIC, H_STATE, H_ERROR, H_WORLD, H_VAL_STEPS, H_CHUNK, H_STEPS, H_ENTRIES, H_INSERTED, H_HITS, H_CHITS,
       H_QUERIED, H_T_READY, H_T_GO, H_T_DONE, H_INSERT_NS, H_QUERY_NS, H_BYTES_READ, H_LEVEL0,
       H_FIT_POSITIONS = H_LEVEL0 + NLEVELS, H_NWORDS };
#define HEADER_MAGIC 0x314d454d52545353ull  // "SSTRMEM1"
#define ERRMSG_OFFSET 512
#define CHECKSUM_OFFSET 1024

typedef struct { uint32_t n, c, m, lstar; } feat_t;

static uint16_t *tok;            // the stream: spans, each followed by SEP; tok[0] = SEP
static uint32_t *prev, *head;
static uint64_t cap, entries = 1, inserted;
static int hash_bits = 29;
static double w[4];              // the gate's constants: w= (stream_memory.py's W), required
static volatile uint64_t *hdr;   // the rows file's header (mapped)
static float *rows;
static FILE *readlog;

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

// ------------------------------------------------------------------------------------------------ insertion

static int nfiles;
static int *fds;
static uint32_t *ins_pos, *ins_bucket;
static size_t ins_n, ins_alloc;

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

// One STEP message: append every span (and a separator), then index the step's positions in stream order.
static void insert_step(uint32_t file_idx, uint32_t nspans, const uint32_t *spans) {
    uint64_t t0 = now_ns();
    ins_n = 0;
    for (uint32_t s = 0; s < nspans; s++) {
        uint32_t a = spans[2 * s], e = spans[2 * s + 1];
        uint64_t len = e - a, start = entries;
        if (e <= a || entries + len + 1 > cap) die(6, "the stream outgrew its %llu entries", (unsigned long long)cap);
        read_span(file_idx, a, e, tok + start);
        entries += len;
        tok[entries++] = SEP;
        if (ins_n + len > ins_alloc) {
            ins_alloc = 2 * (ins_n + len);
            ins_pos = realloc(ins_pos, ins_alloc * 4);
            ins_bucket = realloc(ins_bucket, ins_alloc * 4);
            if (!ins_pos || !ins_bucket) die(1, "out of memory");
        }
        // j has >= KEY tokens of the span up to and including it, and j + 1 is in the span
        for (uint64_t j = start + KEY - 1; j + 1 < start + len; j++) {
            ins_pos[ins_n] = (uint32_t)j;
            ins_bucket[ins_n++] = bucket_of(tok + j - (KEY - 1));
        }
    }
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
}

// ------------------------------------------------------------------------------------------------ queries

// The features of position t: x points at the input token, `run` = tokens of t's segment up to and including t.
static feat_t query_one(const uint16_t *x, uint16_t y, uint32_t run) {
    feat_t f = {0, 0, 0, 0};
    if (run < KEY) return f;
    const uint32_t lim = run < MAXLEN ? run : MAXLEN;
    uint32_t lens[CAP];
    uint16_t next[CAP];
    int nc = 0, visited = 0;
    for (uint32_t p = head[bucket_of(x - (KEY - 1))]; p && nc < CAP && visited < MAXVISIT; p = prev[p]) {
        visited++;
        const uint16_t *m = tok + p;
        if (m[0] != x[0] || m[-1] != x[-1] || m[-2] != x[-2] || m[-3] != x[-3] || m[-4] != x[-4] || m[-5] != x[-5])
            continue;
        // Extend backward. The memory side stops at its span's start by itself (a separator precedes every span
        // and never equals a token); the val side stops at its segment's start through `lim`.
        uint32_t l = KEY;
        while (l < lim && m[-(ptrdiff_t)l] == x[-(ptrdiff_t)l]) l++;
        lens[nc] = l;
        next[nc++] = m[1];
    }
    if (!nc) return f;
    uint32_t maxl = 0;
    for (int i = 0; i < nc; i++) maxl = lens[i] > maxl ? lens[i] : maxl;
    uint32_t lstar = LEVELS[0];
    for (int k = 0; k < NLEVELS; k++) if (LEVELS[k] <= maxl) lstar = LEVELS[k];
    uint16_t s[CAP];
    uint32_t n = 0;
    for (int i = 0; i < nc; i++) if (lens[i] >= lstar) s[n++] = next[i];
    for (uint32_t i = 1; i < n; i++) {  // insertion sort: n <= 32
        uint16_t v = s[i];
        uint32_t k = i;
        while (k && s[k - 1] > v) { s[k] = s[k - 1]; k--; }
        s[k] = v;
    }
    uint32_t m = 0, c = 0;
    for (uint32_t i = 0; i < n;) {
        uint32_t k = i;
        while (k < n && s[k] == s[i]) k++;
        if (k - i > m) m = k - i;
        if (s[i] == y) c = k - i;
        i = k;
    }
    f.n = n; f.c = c; f.m = m; f.lstar = lstar;
    return f;
}

static inline void row_of(feat_t f, float *out) {
    if (!f.n) { out[0] = 1.0f; out[1] = 0.0f; return; }
    double z = w[0] + w[1] * log2((double)f.n) + w[2] * (double)f.m / f.n + w[3] * log2((double)f.lstar);
    double lam = 1.0 / (1.0 + exp(-z));
    out[0] = (float)(1.0 - lam);
    out[1] = (float)(lam * f.c / f.n);
}

// t - sigma_t + 1, saturated at 255 (more than MAXLEN is never needed): t's segment restarts at a BOS and at every
// multiple of `chunk`. Each query block finds it for its first position by looking back, then counts forward.
#define RUN_SAT 255
static uint32_t run_at(const uint16_t *x, size_t t, uint64_t chunk) {
    const size_t start = t - t % chunk;
    uint32_t r = 1;
    for (size_t s = t; s > start && x[s] != BOS && r < RUN_SAT; s--) r++;
    return r;
}

typedef struct {
    const uint16_t *x, *y;  // inputs, targets (y[t] is t's target)
    uint64_t chunk;         // segments restart at every multiple of chunk (and at every BOS)
    size_t n;
    float *rows;            // n x 2, or NULL
    uint32_t *feat;         // n x 4, or NULL
    uint64_t (*index)(size_t t);  // where t's row goes (rows only)
    atomic_size_t next;
    uint64_t hits, chits, levels[NLEVELS];
    pthread_mutex_t lock;
} query_job_t;

static uint64_t g_world, g_val_steps, g_chunk;

static uint64_t val_row_index(size_t t) {  // f32 [world][val_steps][chunk][2], chunk c = step * world + rank
    uint64_t c = t / g_chunk, i = t % g_chunk, s = c / g_world, r = c % g_world;
    return (r * g_val_steps + s) * g_chunk + i;
}

static void *query_worker(void *arg) {
    query_job_t *job = arg;
    uint64_t hits = 0, chits = 0, levels[NLEVELS] = {0};
    for (;;) {
        size_t lo = atomic_fetch_add(&job->next, BLOCK);
        if (lo >= job->n) break;
        size_t hi = lo + BLOCK < job->n ? lo + BLOCK : job->n;
        uint32_t run = run_at(job->x, lo, job->chunk);
        size_t next_chunk = lo - lo % job->chunk + job->chunk;
        for (size_t t = lo; t < hi; t++) {
            if (t > lo) {
                if (t == next_chunk) { run = 1; next_chunk += job->chunk; }
                else run = job->x[t] == BOS ? 1 : (run < RUN_SAT ? run + 1 : RUN_SAT);
            }
            feat_t f = query_one(job->x + t, job->y[t], run);
            if (f.n) {
                hits++;
                chits += f.c > 0;
                for (int k = 0; k < NLEVELS; k++) levels[k] += f.lstar == LEVELS[k];
            }
            if (job->rows) row_of(f, job->rows + 2 * job->index(t));
            if (job->feat) {
                uint32_t *o = job->feat + 4 * t;
                o[0] = f.n; o[1] = f.c; o[2] = f.m; o[3] = f.lstar;
            }
        }
    }
    pthread_mutex_lock(&job->lock);
    job->hits += hits;
    job->chits += chits;
    for (int k = 0; k < NLEVELS; k++) job->levels[k] += levels[k];
    pthread_mutex_unlock(&job->lock);
    return NULL;
}

static int query_threads = 32;

static void run_queries(query_job_t *job) {
    atomic_init(&job->next, 0);
    job->hits = job->chits = 0;
    memset(job->levels, 0, sizeof job->levels);
    pthread_mutex_init(&job->lock, NULL);
    int nt = query_threads < 1 ? 1 : (query_threads > 256 ? 256 : query_threads);
    pthread_t th[256];
    for (int k = 1; k < nt; k++)
        if (pthread_create(&th[k], NULL, query_worker, job)) die(1, "pthread_create failed");
    query_worker(job);
    for (int k = 1; k < nt; k++) pthread_join(th[k], NULL);
    pthread_mutex_destroy(&job->lock);
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
    const char *fit_path, *dump_path, *features_path;
    int vfd;
} cfg;
static uint64_t steps, fit_steps;
static int went;
static uint16_t **fx, **fy;  // FIT: per rank, the val-shaped inputs and targets of the query-only batches
static size_t *fn, *falloc;

static void write_file(const char *path, const void *data, size_t size, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, size, n, f) != n || fclose(f)) die(11, "cannot write %s", path);
}

// GO: every training step is in the memory. Read val and compute every position's row (and its checksums).
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
    uint64_t t0 = now_ns();
    uint32_t *feat = cfg.features_path ? xalloc(16 * n) : NULL;
    query_job_t job = {.x = val, .y = val + 1, .chunk = cfg.chunk, .n = n, .rows = rows, .feat = feat,
                       .index = val_row_index};
    run_queries(&job);
    hdr[H_QUERY_NS] = now_ns() - t0;
    pthread_join(check_thread, NULL);
    if (check.sep_found) die(10, "a val token equals the separator");
    hdr[H_QUERIED] = n;
    hdr[H_HITS] = job.hits;
    hdr[H_CHITS] = job.chits;
    for (int k = 0; k < NLEVELS; k++) hdr[H_LEVEL0 + k] = job.levels[k];
    free(val);
    if (feat) {
        write_file(cfg.features_path, feat, 16, n);
        free(feat);
    }
    if (cfg.dump_path) write_file(cfg.dump_path, tok + 1, 2, entries - 1);
    if (readlog) fflush(readlog);
    hdr[H_T_DONE] = now_ns();
    __atomic_store_n(&hdr[H_STATE], ST_DONE, __ATOMIC_RELEASE);
}

// FIT_STEP (after GO, never inserted): rank r's batch adds inputs buf[:-1] and targets buf[1:] to its val-shaped
// chunks, the layout train_gpt.py's eval forward sees (stream_memory.pseudo_val_batches).
static void on_fit_step(uint32_t step, uint32_t file_idx, const uint32_t *counts, const uint32_t *spans) {
    if (!went || !cfg.fit_path) die(8, "a FIT step %s", went ? "without fit=" : "before GO");
    if (step != fit_steps++) die(8, "FIT step %u out of order", step);
    if (!fx) {
        fx = calloc(cfg.world, sizeof *fx); fy = calloc(cfg.world, sizeof *fy);
        fn = calloc(cfg.world, sizeof *fn); falloc = calloc(cfg.world, sizeof *falloc);
        if (!fx || !fy || !fn || !falloc) die(1, "out of memory");
    }
    for (uint64_t r = 0; r < cfg.world; r++) {
        size_t len = 0;
        for (uint32_t s = 0; s < counts[r]; s++) len += spans[2 * s + 1] - spans[2 * s];
        if (len < 2) die(8, "a FIT batch of %zu tokens", len);
        uint16_t *buf = xalloc(2 * len);
        for (size_t o = 0, s = 0; s < counts[r]; s++, spans += 2) {
            read_span(file_idx, spans[0], spans[1], buf + o);
            o += spans[1] - spans[0];
        }
        if (fn[r] + len > falloc[r]) {
            falloc[r] = 2 * (fn[r] + len);
            fx[r] = realloc(fx[r], 2 * falloc[r]);
            fy[r] = realloc(fy[r], 2 * falloc[r]);
            if (!fx[r] || !fy[r]) die(1, "out of memory");
        }
        memcpy(fx[r] + fn[r], buf, 2 * (len - 1));
        memcpy(fy[r] + fn[r], buf + 1, 2 * (len - 1));
        fn[r] += len - 1;
        free(buf);
    }
}

// FIT_GO: the features (N, C, M, L*) of every FIT position, u32 [world][positions][4], to fit=PATH.
static void on_fit_go(uint32_t k) {
    if (!cfg.fit_path || !fx || k != fit_steps) die(12, "FIT_GO for %u steps after %llu", k, (unsigned long long)fit_steps);
    for (uint64_t r = 1; r < cfg.world; r++)
        if (fn[r] != fn[0]) die(12, "FIT ranks hold %zu and %zu positions", fn[0], fn[r]);
    uint32_t *feat = xalloc(16 * fn[0] * cfg.world);
    for (uint64_t r = 0; r < cfg.world; r++) {
        query_job_t job = {.x = fx[r], .y = fy[r], .chunk = cfg.chunk, .n = fn[r], .feat = feat + 4 * r * fn[0]};
        run_queries(&job);
    }
    write_file(cfg.fit_path, feat, 16, fn[0] * cfg.world);
    free(feat);
    hdr[H_FIT_POSITIONS] = fn[0] * cfg.world;
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
    // rows=PATH (made by rank 0, header + rows) val=PATH world=W val_tokens=N chunk=L steps=T cap=ENTRIES
    // prefault=ENTRIES w=w0,w1,w2,w3 [hash_bits=29] [threads=32] [fit=PATH], and for tests
    // and diagnostics [dump=PATH] (the stream) [features=PATH] (val N, C, M, L* as u32 [val_tokens][4])
    // [readlog=PATH] (every read: file, byte offset, bytes) -- TRAIN_SHARD...
    const char *rows_path = arg_of(argc, argv, "rows", NULL), *val_path = arg_of(argc, argv, "val", NULL);
    if (!rows_path || !val_path) { fprintf(stderr, "stream_memory: rows= and val= are required\n"); return 2; }
    cfg.world = strtoull(arg_of(argc, argv, "world", "8"), NULL, 10);
    cfg.val_tokens = strtoull(arg_of(argc, argv, "val_tokens", "10485760"), NULL, 10);
    cfg.chunk = strtoull(arg_of(argc, argv, "chunk", "262144"), NULL, 10);
    cfg.total_steps = strtoull(arg_of(argc, argv, "steps", "0"), NULL, 10);
    cfg.fit_path = arg_of(argc, argv, "fit", NULL);
    cfg.dump_path = arg_of(argc, argv, "dump", NULL);
    cfg.features_path = arg_of(argc, argv, "features", NULL);
    cap = strtoull(arg_of(argc, argv, "cap", "0"), NULL, 10);
    const uint64_t prefault = strtoull(arg_of(argc, argv, "prefault", "0"), NULL, 10);
    hash_bits = atoi(arg_of(argc, argv, "hash_bits", "29"));
    query_threads = atoi(arg_of(argc, argv, "threads", "32"));
    const char *ws = arg_of(argc, argv, "w", NULL), *readlog_path = arg_of(argc, argv, "readlog", NULL);
    if (!ws || sscanf(ws, "%lf,%lf,%lf,%lf", &w[0], &w[1], &w[2], &w[3]) != 4) {
        fprintf(stderr, "stream_memory: w=w0,w1,w2,w3 is required (got %s)\n", ws ? ws : "none");
        return 2;
    }

    int rfd = open(rows_path, O_RDWR);
    if (rfd < 0) { fprintf(stderr, "stream_memory: cannot open %s: %s\n", rows_path, strerror(errno)); return 2; }
    const uint64_t world = cfg.world, chunk = cfg.chunk;
    cfg.val_steps = world && chunk ? cfg.val_tokens / (world * chunk) : 0;
    void *rmap = mmap(NULL, ROWS_OFFSET + (size_t)(world * cfg.val_steps * chunk) * 8, PROT_READ | PROT_WRITE,
                      MAP_SHARED, rfd, 0);
    if (rmap == MAP_FAILED) { fprintf(stderr, "stream_memory: cannot map %s\n", rows_path); return 2; }
    close(rfd);
    hdr = rmap;
    rows = (float *)((char *)rmap + ROWS_OFFSET);
    hdr[H_MAGIC] = HEADER_MAGIC;
    hdr[H_WORLD] = world;
    hdr[H_VAL_STEPS] = cfg.val_steps;
    hdr[H_CHUNK] = chunk;
    if (!cfg.val_steps || cfg.val_steps * world * chunk != cfg.val_tokens || world * cfg.val_steps > MAX_CHUNKS)
        die(2, "val_tokens=%llu is not a whole number of world=%llu x chunk=%llu steps (<= %d chunks)",
            (unsigned long long)cfg.val_tokens, (unsigned long long)world, (unsigned long long)chunk, MAX_CHUNKS);
    if (hash_bits < 4 || hash_bits > 32) die(2, "bad hash_bits=%d", hash_bits);
    if (cap < 2 || cap > 0xFFFFFFF0ull) die(2, "cap=%llu entries is outside [2, 2^32)", (unsigned long long)cap);
    g_world = world; g_val_steps = cfg.val_steps; g_chunk = chunk;

    int sep = 1;
    while (sep < argc && strcmp(argv[sep], "--")) sep++;
    nfiles = argc - sep - 1;
    if (nfiles < 1) die(2, "no training shards after --");
    fds = xalloc(sizeof(int) * (size_t)nfiles);
    for (int i = 0; i < nfiles; i++)
        if ((fds[i] = open(argv[sep + 1 + i], O_RDONLY)) < 0) die(2, "cannot open %s", argv[sep + 1 + i]);
    if ((cfg.vfd = open(val_path, O_RDONLY)) < 0) die(2, "cannot open %s", val_path);
    if (readlog_path && !(readlog = fopen(readlog_path, "w"))) die(2, "cannot open %s", readlog_path);

    // Allocation before the clock: the stream and its chain (cap entries reserved, prefault touched) and the heads.
    tok = big_alloc(2 * cap, 2 * prefault);
    prev = big_alloc(4 * cap, 4 * prefault);
    head = big_alloc(4ull << hash_bits, 4ull << hash_bits);
    tok[0] = SEP;
    hdr[H_ENTRIES] = entries;
    hdr[H_T_READY] = now_ns();
    __atomic_store_n(&hdr[H_STATE], ST_READY, __ATOMIC_RELEASE);

    // The message loop: until rank 0 closes the pipe.
    uint32_t *spans = NULL, counts[MAX_CHUNKS], kind, words[3];
    size_t spans_alloc = 0;
    while (read_words(&kind, 1, 1)) {
        if (kind == MSG_GO || kind == MSG_FIT_GO) {
            read_words(words, 1, 0);
            if (kind == MSG_GO) on_go(words[0]); else on_fit_go(words[0]);
            continue;
        }
        if (kind != MSG_STEP && kind != MSG_FIT_STEP) die(7, "unknown message kind %u", kind);
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
        if (kind == MSG_FIT_STEP) {
            on_fit_step(words[0], words[1], counts, spans);
            continue;
        }
        if (went) die(8, "a training step after GO");
        if (words[0] != steps || steps >= cfg.total_steps)
            die(8, "step message %u, expected %llu of %llu", words[0], (unsigned long long)steps,
                (unsigned long long)cfg.total_steps);
        insert_step(words[1], (uint32_t)nspans, spans);
        hdr[H_STEPS] = ++steps;
    }
    if (readlog) fclose(readlog);
    return 0;
}
