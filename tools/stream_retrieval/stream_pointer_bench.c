// Real-data harness for stream_pointer.c (part P3): not part of the helper. It rebuilds the research chain index of
// rg/align/align4.c over a stream file (key-6 hash48 chain, CAP 32 verified candidates within 128 visits, so the
// candidate lists are those align4 saw), walks every val position once (the walk P1's query_one shares with P3), then
// runs sp_run on those candidates and reports the costs. test_stream_pointer.py compares its outputs with the research
// tools' (align4's pointer rows, srccopy6, docstate) when their outputs are on disk.
//
// usage: stream_pointer_bench STREAM.u16 VAL.bin QN OUT_PREFIX [THREADS]
// writes OUT.rows (sp_row_t [n]), OUT.mem (per position: u32 has, top, n, m, c, lstar, lenl, posl, lenr, posr),
//        OUT.cand.off (u64 [QN + 1]), OUT.cand.pos (u32), OUT.cand.len (u8); prints one JSON line of statistics.
// Positions are those of the stream file (a separator sits before its first token, as tok[0] in the helper).
#include "arm/track_1_short/stream_pointer.c"

#include <time.h>

#define CHUNK 262144u

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline uint64_t mix64(uint64_t h) {
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 27; h *= 0x94d049bb133111ebull; return h ^ (h >> 31);
}
static inline uint64_t hash48(const uint16_t *w, int k) {
    uint64_t h = (uint64_t)k * 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < k; i++) { h = (h ^ (uint64_t)w[i]) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    return mix64(h) >> 16;
}

static uint16_t *read_u16(const char *path, size_t skip, size_t count, size_t pad_front, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t bytes = (size_t)ftell(f) - skip;
    fseek(f, (long)skip, SEEK_SET);
    size_t n = bytes / 2;
    if (count && count < n) n = count;
    uint16_t *a = malloc((n + pad_front) * 2);
    for (size_t i = 0; i < pad_front; i++) a[i] = SP_SEP;
    if (fread(a + pad_front, 2, n, f) != n) { perror("fread"); exit(1); }
    fclose(f);
    *n_out = n;
    return a;
}

static void write_file(const char *pref, const char *suf, const void *p, size_t bytes) {
    char path[4096];
    snprintf(path, sizeof path, "%s%s", pref, suf);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(p, 1, bytes, f) != bytes || fclose(f)) { perror(path); exit(1); }
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: stream_pointer_bench STREAM VAL QN OUT [THREADS]\n"); return 2; }
    const size_t qn = strtoull(argv[3], 0, 10);
    const int threads = argc > 5 ? atoi(argv[5]) : 4;
    double t0 = now();
    size_t n, nv;
    uint16_t *buf = read_u16(argv[1], 0, 0, 1, &n);
    const uint16_t *mem = buf + 1;  // mem[-1] = SEP
    uint16_t *val = read_u16(argv[2], 1024, qn + 1, 0, &nv);
    if (nv != qn + 1) { fprintf(stderr, "val too short\n"); return 1; }
    // align4's index: positions j with >= KEY tokens of their span up to j and a next token in the span (not SEP/BOS)
    int hbits = 64 - __builtin_clzll((unsigned long long)n);
    uint64_t mask = (1ull << hbits) - 1;
    uint32_t *head = calloc(1ull << hbits, 4), *prev = calloc(n, 4);
    if (!head || !prev) { fprintf(stderr, "oom\n"); return 1; }
    {
        uint32_t mrun = 0;
        for (size_t j = 0; j + 1 < n; j++) {
            mrun = mem[j] == SP_SEP ? 0 : mrun + 1;
            uint16_t nx = mem[j + 1];
            if (mrun < SP_KEY || nx == SP_SEP || nx == SP_BOS) continue;
            size_t h = hash48(mem + j + 1 - SP_KEY, SP_KEY) & mask;
            prev[j] = head[h];
            head[h] = (uint32_t)(j + 1);
        }
    }
    double t_index = now() - t0;
    uint32_t *run = malloc(4 * qn);
    {
        uint32_t cur = 0;
        for (size_t t = 0; t < qn; t++) {
            cur = (val[t] == SP_BOS || t % CHUNK == 0) ? 1 : (cur < SP_FULLCAP ? cur + 1 : SP_FULLCAP);
            run[t] = cur;
        }
    }
    // the walk (shared with P1): candidates of every position, as align4 finds them; lens as query_one caps them
    uint64_t *off = malloc(8 * (qn + 1));
    size_t ccap = 1 << 22, nc_tot = 0;
    uint32_t *cpos = malloc(4 * ccap);
    uint8_t *clen = malloc(ccap);
    double t1 = now();
    for (size_t t = 0; t < qn; t++) {
        off[t] = nc_tot;
        if (run[t] < SP_KEY) continue;
        const uint16_t *ctx = val + t + 1 - SP_KEY;
        size_t h = hash48(ctx, SP_KEY) & mask;
        int nc = 0, walked = 0;
        const uint32_t lim = run[t] < SP_MAXLEN ? run[t] : SP_MAXLEN;
        for (uint32_t p = head[h]; p && nc < SP_CAP && walked < 4 * SP_CAP;) {
            size_t j = p - 1;
            walked++;
            if (!memcmp(mem + j + 1 - SP_KEY, ctx, 2 * SP_KEY)) {
                uint32_t l = SP_KEY;
                while (l < lim && mem[j - l] == val[t - l]) l++;
                if (nc_tot + 1 > ccap) { ccap *= 2; cpos = realloc(cpos, 4 * ccap); clen = realloc(clen, ccap); }
                cpos[nc_tot] = (uint32_t)j;
                clen[nc_tot] = (uint8_t)l;
                nc_tot++;
                nc++;
            }
            p = prev[j];
        }
    }
    off[qn] = nc_tot;
    double t_walk = now() - t1;
    free(head);
    free(prev);
    // the memory summary of every position (for the research tools' inputs)
    uint32_t *ms = malloc(4 * 10 * qn);
    double t2 = now();
    for (size_t t = 0; t < qn; t++) {
        sp_mem_t m;
        sp_mem_summary(mem, val + t, run[t], cpos + off[t], clen + off[t], (int)(off[t + 1] - off[t]), val[t + 1], &m);
        uint32_t *o = ms + 10 * t;
        o[0] = m.ncand > 0; o[1] = m.top; o[2] = m.n; o[3] = m.m; o[4] = m.c; o[5] = SP_LEVELS[m.lstar] * (m.ncand > 0);
        o[6] = m.len_longest; o[7] = m.pos_longest; o[8] = m.len_recent; o[9] = m.pos_recent;
    }
    double t_summary = now() - t2;
    sp_csr_t csr = {.off = off, .pos = cpos, .len = clen};
    char err[256] = {0};
    sp_row_t *rows1 = NULL, *rowsT = NULL, *rows_nosrc = NULL;
    double t3 = now();
    int64_t n1 = sp_run(NULL, mem, n, val, NULL, qn, run, 0, sp_cands_csr, &csr, 1, &rows1, err, sizeof err);
    double t_p3_1 = now() - t3;
    if (n1 < 0) { fprintf(stderr, "sp_run: %s\n", err); return 1; }
    double t4 = now();
    int64_t nT = sp_run(NULL, mem, n, val, NULL, qn, run, 0, sp_cands_csr, &csr, threads, &rowsT, err, sizeof err);
    double t_p3_T = now() - t4;
    int same = nT == n1 && !memcmp(rows1, rowsT, sizeof(sp_row_t) * (size_t)n1);
    sp_config_t c;
    sp_default_config(&c);
    c.src_th = 1 << 30;
    c.src_votes = 0;  // no sources: the beam and the doc-state alone
    double t5 = now();
    int64_t n0 = sp_run(&c, mem, n, val, NULL, qn, run, 0, sp_cands_csr, &csr, 1, &rows_nosrc, err, sizeof err);
    double t_p3_nosrc = now() - t5;
    // distribution statistics
    uint64_t hp = 0, hs = 0, trunc = 0, ntok_hist[SP_VOTE + 1] = {0}, nsd_gt8 = 0, ntok_gt8 = 0, ptr_ok = 0, src_y_lost = 0;
    for (int64_t i = 0; i < n1; i++) {
        const sp_row_t *r = &rows1[i];
        hp += !!(r->flags & SP_HP);
        hs += !!(r->flags & SP_HS);
        trunc += !!(r->flags & SP_SRC_TRUNC);
        ntok_hist[r->ntok]++;
        ntok_gt8 += r->ntok > 8;
        nsd_gt8 += r->nsd > 8;
        ptr_ok += (r->flags & SP_HP) && r->pred == val[r->pos + 1];
        if (r->flags & SP_SRC_TRUNC) {  // is the target among the left-out tokens? (count it from the occurrences)
            int in = 0;
            for (int k = 0; k < r->nsd; k++) in |= r->stok[k] == val[r->pos + 1];
            src_y_lost += !in;
        }
    }
    write_file(argv[4], ".rows", rows1, sizeof(sp_row_t) * (size_t)n1);
    write_file(argv[4], ".mem", ms, 4 * 10 * qn);
    write_file(argv[4], ".cand.off", off, 8 * (qn + 1));
    write_file(argv[4], ".cand.pos", cpos, 4 * nc_tot);
    write_file(argv[4], ".cand.len", clen, nc_tot);
    printf("{\"qn\": %zu, \"stream\": %zu, \"index_s\": %.2f, \"walk_s\": %.3f, \"walk_ns_per_pos\": %.1f, "
           "\"cands\": %zu, \"summary_ns_per_pos\": %.1f, \"rows\": %lld, \"hp\": %llu, \"hs\": %llu, "
           "\"ptr_top_correct\": %.4f, \"p3_1thread_s\": %.3f, \"p3_ns_per_pos_1thread\": %.1f, "
           "\"p3_%dthreads_s\": %.3f, \"identical_across_threads\": %d, \"p3_nosrc_ns_per_pos\": %.1f, \"rows_nosrc\": %lld, "
           "\"src_trunc\": %llu, \"src_trunc_target_left_out\": %llu, \"ntok_gt8\": %llu, \"nsd_gt8\": %llu, \"ntok_hist\": [",
           qn, n, t_index, t_walk, 1e9 * t_walk / qn, nc_tot, 1e9 * t_summary / qn, (long long)n1,
           (unsigned long long)hp, (unsigned long long)hs, hp ? (double)ptr_ok / hp : 0.0, t_p3_1, 1e9 * t_p3_1 / qn,
           threads, t_p3_T, same, 1e9 * t_p3_nosrc / qn, (long long)n0, (unsigned long long)trunc,
           (unsigned long long)src_y_lost, (unsigned long long)ntok_gt8, (unsigned long long)nsd_gt8);
    for (int i = 0; i <= SP_VOTE; i++) printf("%s%llu", i ? ", " : "", (unsigned long long)ntok_hist[i]);
    printf("], \"row_bytes\": %zu}\n", sizeof(sp_row_t));
    sp_free(rows1); sp_free(rowsT); sp_free(rows_nosrc);
    return same ? 0 : 3;
}
