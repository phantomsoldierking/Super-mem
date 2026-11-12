/* memtrace runtime: private-cache filter + SimpleO3 trace writer.
 *
 * Linked into binaries compiled with MemTracePass. Behaviour is controlled by
 * environment variables (all instruction counts are LLVM IR instructions):
 *
 *   MT_OUT    output trace path (required to trace; otherwise only counts)
 *   MT_SKIP   instructions to fast-forward without cache modelling (default 0)
 *   MT_WARM   instructions to warm the private caches without output (default 50M)
 *   MT_INSTS  instructions to trace (default 100M); the process exits afterwards
 *
 * The private hierarchy (L1D 48 KiB/12-way, L2 2 MiB/16-way, 64 B lines,
 * write-back/write-allocate, LRU) filters accesses. Every L2 miss becomes one
 * SimpleO3 trace line "<bubble> <paddr> [<dirty-victim paddr>]" where bubble is
 * the number of non-miss instructions since the previous line. Virtual pages
 * are mapped to physical frames in first-touch order, so physical addresses are
 * compact and the frame count is the touched footprint.
 *
 * A JSON statistics file is written next to the trace (MT_OUT + ".json").
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define LINE 64
#define PAGE_SHIFT 12

enum { MODE_SKIP = 0, MODE_WARM = 1, MODE_TRACE = 2, MODE_DONE = 3 };

static int mt_mode = MODE_SKIP;
static uint64_t mt_insts, mt_next_switch = UINT64_MAX;
static uint64_t warm_insts = 50000000ULL, trace_insts = 100000000ULL, skip_insts;
static uint64_t trace_start_inst, last_emit_inst;
static FILE *out;
static char out_path[4096];
static int initialized;

/* see push_pending_wb */
#define PENDING_MAX 1024
static uint64_t pending_wb[PENDING_MAX];
static int pending_n;
static uint64_t s_wb_dropped;
static uint64_t s_loads, s_stores, s_l1_miss, s_l2_miss, s_l2_wb, s_lines, s_l1_access;

/* ---------------- caches ---------------- */
typedef struct {
  uint64_t *tag; /* line address + 1, 0 = invalid */
  uint64_t *lru;
  uint8_t *dirty;
  int sets, ways;
  uint64_t clock;
} cache_t;

static cache_t L1, L2;

static void cache_init(cache_t *c, size_t bytes, int ways) {
  c->ways = ways;
  c->sets = (int)(bytes / LINE / ways);
  size_t n = (size_t)c->sets * ways;
  c->tag = calloc(n, sizeof(uint64_t));
  c->lru = calloc(n, sizeof(uint64_t));
  c->dirty = calloc(n, 1);
  c->clock = 1;
}

/* Returns 1 on hit. On miss, inserts the line and reports a dirty victim. */
static int cache_access(cache_t *c, uint64_t line, int write, uint64_t *victim, int *victim_dirty) {
  size_t set = (size_t)(line % (uint64_t)c->sets) * c->ways;
  uint64_t key = line + 1;
  int lru_way = 0;
  uint64_t lru_val = UINT64_MAX;
  for (int w = 0; w < c->ways; w++) {
    if (c->tag[set + w] == key) {
      c->lru[set + w] = ++c->clock;
      c->dirty[set + w] |= (uint8_t)write;
      return 1;
    }
    if (c->lru[set + w] < lru_val) {
      lru_val = c->lru[set + w];
      lru_way = w;
    }
  }
  size_t i = set + lru_way;
  *victim_dirty = c->tag[i] && c->dirty[i];
  *victim = c->tag[i] ? c->tag[i] - 1 : 0;
  c->tag[i] = key;
  c->lru[i] = ++c->clock;
  c->dirty[i] = (uint8_t)write;
  return 0;
}

static void cache_mark_dirty(cache_t *c, uint64_t line) {
  size_t set = (size_t)(line % (uint64_t)c->sets) * c->ways;
  for (int w = 0; w < c->ways; w++)
    if (c->tag[set + w] == line + 1) { c->dirty[set + w] = 1; return; }
}

/* ---------------- first-touch page map ---------------- */
#define PMAP_BITS 23
static uint64_t *pmap_key; /* vpage + 1 */
static uint64_t *pmap_val;
static uint64_t frames;

static uint64_t translate(uint64_t vaddr) {
  uint64_t vp = vaddr >> PAGE_SHIFT;
  uint64_t mask = (1ULL << PMAP_BITS) - 1;
  uint64_t h = (vp * 0x9E3779B97F4A7C15ULL) >> (64 - PMAP_BITS);
  for (;;) {
    if (pmap_key[h] == vp + 1) break;
    if (pmap_key[h] == 0) {
      pmap_key[h] = vp + 1;
      pmap_val[h] = frames++;
      break;
    }
    h = (h + 1) & mask;
  }
  return (pmap_val[h] << PAGE_SHIFT) | (vaddr & ((1ULL << PAGE_SHIFT) - 1));
}

/* ---------------- phases ---------------- */
static void write_stats(const char *reason) {
  if (!out_path[0]) return;
  char path[4200];
  snprintf(path, sizeof path, "%s.json", out_path);
  FILE *f = fopen(path, "w");
  if (!f) return;
  long hwm_kb = 0;
  FILE *st = fopen("/proc/self/status", "r");
  if (st) {
    char line[256];
    while (fgets(line, sizeof line, st))
      if (!strncmp(line, "VmHWM:", 6)) hwm_kb = atol(line + 6);
    fclose(st);
  }
  uint64_t traced = mt_mode >= MODE_TRACE ? mt_insts - trace_start_inst : 0;
  fprintf(f,
          "{\"end_reason\":\"%s\",\"skip_insts\":%llu,\"warm_insts\":%llu,\"trace_insts\":%llu,"
          "\"total_insts_executed\":%llu,\"l1d_accesses\":%llu,\"loads\":%llu,\"stores\":%llu,"
          "\"l1d_misses\":%llu,\"l2_misses\":%llu,\"l2_dirty_writebacks\":%llu,\"trace_lines\":%llu,"
          "\"touched_pages_in_window\":%llu,\"peak_rss_bytes\":%llu,"
          "\"pending_writebacks_at_end\":%d,\"writebacks_dropped\":%llu,"
          "\"l1d\":{\"bytes\":%d,\"ways\":%d},\"l2\":{\"bytes\":%d,\"ways\":%d},\"line_bytes\":%d}\n",
          reason, (unsigned long long)skip_insts, (unsigned long long)warm_insts,
          (unsigned long long)traced, (unsigned long long)mt_insts, (unsigned long long)s_l1_access,
          (unsigned long long)s_loads, (unsigned long long)s_stores, (unsigned long long)s_l1_miss,
          (unsigned long long)s_l2_miss, (unsigned long long)s_l2_wb, (unsigned long long)s_lines,
          (unsigned long long)frames, (unsigned long long)hwm_kb * 1024ULL, pending_n,
          (unsigned long long)s_wb_dropped, L1.sets * L1.ways * LINE,
          L1.ways, L2.sets * L2.ways * LINE, L2.ways, LINE);
  fclose(f);
}

static void finish(const char *reason) {
  if (mt_mode == MODE_DONE) return;
  if (out) fflush(out);
  write_stats(reason);
  if (out) fclose(out);
  out = NULL;
  mt_mode = MODE_DONE;
}

static void at_exit_handler(void) { finish("program_exit"); }

static uint64_t env_u64(const char *name, uint64_t def) {
  const char *v = getenv(name);
  return v && *v ? strtoull(v, NULL, 10) : def;
}

static void mt_init(void) {
  initialized = 1;
  const char *o = getenv("MT_OUT");
  skip_insts = env_u64("MT_SKIP", 0);
  warm_insts = env_u64("MT_WARM", warm_insts);
  trace_insts = env_u64("MT_INSTS", trace_insts);
  if (!o || !*o) { mt_mode = MODE_DONE; return; }
  snprintf(out_path, sizeof out_path, "%s", o);
  cache_init(&L1, 48 * 1024, 12);
  cache_init(&L2, 2 * 1024 * 1024, 16);
  pmap_key = calloc(1ULL << PMAP_BITS, sizeof(uint64_t));
  pmap_val = calloc(1ULL << PMAP_BITS, sizeof(uint64_t));
  atexit(at_exit_handler);
  mt_mode = MODE_SKIP;
  mt_next_switch = skip_insts;
}

static void phase_switch(void) {
  if (mt_mode == MODE_SKIP) {
    mt_mode = MODE_WARM;
    mt_next_switch = mt_insts + warm_insts;
    if (warm_insts) return;
  }
  if (mt_mode == MODE_WARM) {
    out = fopen(out_path, "w");
    if (!out) { perror("memtrace: MT_OUT"); _exit(97); }
    setvbuf(out, NULL, _IOFBF, 1 << 22);
    s_loads = s_stores = s_l1_miss = s_l2_miss = s_l2_wb = s_lines = s_l1_access = 0;
    mt_mode = MODE_TRACE;
    trace_start_inst = last_emit_inst = mt_insts;
    mt_next_switch = mt_insts + trace_insts;
    return;
  }
  if (mt_mode == MODE_TRACE) {
    finish("trace_window_complete");
    _exit(0);
  }
}

void __mt_bb(uint32_t n) {
  if (__builtin_expect(!initialized, 0)) mt_init();
  mt_insts += n;
  if (__builtin_expect(mt_insts >= mt_next_switch, 0) && mt_mode != MODE_DONE) phase_switch();
}

/* Dirty L2 victims displaced by L1 write-backs have no load of their own; SimpleO3
 * lines carry at most one write-back, so they ride on the next miss line that has
 * none. Overflow (never expected with a 1024-entry buffer) is counted. */

static void push_pending_wb(uint64_t line) {
  if (pending_n < PENDING_MAX) pending_wb[pending_n++] = line;
  else s_wb_dropped++;
}

static void access_line(uint64_t vline_addr, int write) {
  uint64_t paddr = translate(vline_addr);
  uint64_t line = paddr / LINE;
  uint64_t victim;
  int vd = 0;
  if (mt_mode == MODE_TRACE) s_l1_access++;
  if (cache_access(&L1, line, write, &victim, &vd)) return;
  if (vd) { /* L1 dirty eviction is written into the (non-inclusive) L2 */
    uint64_t v2; int d2 = 0;
    if (!cache_access(&L2, victim, 1, &v2, &d2) && d2 && mt_mode == MODE_TRACE) push_pending_wb(v2);
  }
  if (mt_mode == MODE_TRACE) s_l1_miss++;
  vd = 0;
  if (cache_access(&L2, line, 0, &victim, &vd)) return;
  if (mt_mode != MODE_TRACE) return;
  s_l2_miss++;
  /* Each trace line is <bubble> non-missing instructions plus the missing one.
   * last_emit_inst counts instructions already represented by emitted lines, so
   * several misses inside one basic block borrow from the instructions that
   * follow and the line total tracks the executed instruction count. */
  uint64_t bubble = mt_insts > last_emit_inst ? mt_insts - last_emit_inst - 1 : 0;
  last_emit_inst += bubble + 1;
  if (bubble > 2000000000ULL) bubble = 2000000000ULL;
  if (!vd && pending_n) { /* attach a write-back displaced by an L1 eviction */
    victim = pending_wb[--pending_n];
    vd = 1;
  }
  if (vd) {
    s_l2_wb++;
    fprintf(out, "%llu %llu %llu\n", (unsigned long long)bubble, (unsigned long long)(line * LINE),
            (unsigned long long)(victim * LINE));
  } else {
    fprintf(out, "%llu %llu\n", (unsigned long long)bubble, (unsigned long long)(line * LINE));
  }
  s_lines++;
}

static inline void record(const void *p, uint64_t size, int write) {
  if (__builtin_expect(mt_mode == MODE_SKIP || mt_mode == MODE_DONE, 1)) return;
  if (mt_mode == MODE_TRACE) { if (write) s_stores++; else s_loads++; }
  uint64_t a = (uint64_t)(uintptr_t)p;
  if (size == 0) size = 1;
  uint64_t first = a & ~(uint64_t)(LINE - 1), last = (a + size - 1) & ~(uint64_t)(LINE - 1);
  for (uint64_t l = first; l <= last; l += LINE) access_line(l, write);
}

void __mt_ld(const void *p, uint32_t size) { record(p, size, 0); }
void __mt_st(const void *p, uint32_t size) { record(p, size, 1); }
void __mt_range(const void *p, uint64_t size, uint32_t write) {
  if (size > (64ULL << 20)) size = 64ULL << 20; /* bound pathological intrinsics */
  record(p, size, (int)write);
}
