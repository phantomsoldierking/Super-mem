"""Ramulator 2.1 backend: build DRAM memory systems from architecture configs and
run them with Ramulator's C++ trace frontends.

Two uses:
  * ``calibrate``   - replay access-pattern microtraces (weight streaming, KV-cache
                      page gathers, embedding-row gathers, writes) with the
                      LoadStoreTrace frontend and report achieved bandwidth.
  * ``run_cpu_trace`` - replay SPEC CPU L2-miss traces through Ramulator's SimpleO3
                      out-of-order core + shared LLC model in front of the DRAM.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import random
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from .paths import CACHE_DIR, RAMULATOR_DIR

_loaded = None


def ramulator():
    """Import the Ramulator Python bindings without requiring LD_LIBRARY_PATH."""
    global _loaded
    if _loaded is None:
        lib = RAMULATOR_DIR / "libramulator.so"
        if not lib.exists():
            raise RuntimeError(f"{lib} not found; build Ramulator (see ramulator2/README.md)")
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        sys.path.insert(0, str(RAMULATOR_DIR / "python"))
        import ramulator as r  # noqa: E402

        _loaded = r
    return _loaded


_CONTROLLERS = {"DDR5": "GenericDDR", "GDDR6": "GenericDDR", "LPDDR5": "LPDDR5",
                "HBM3": "HBM34", "HBM4": "HBM34"}


def dram_clock_ps(dram_cfg) -> int:
    """Period of one controller tick in ps.

    Ramulator counts controller cycles in ticks. For most standards a tick is one
    CK, but HBM3/HBM4 use ``tick_multiplier = 2`` (half-CK ticks): the spec class
    divides tCK_ps by the multiplier before the controller sees it. Cycle counts
    from the controller (latencies, ``cycles``) and the frontend/memory clock
    ratio must therefore use the tick period, not the preset CK."""
    r = ramulator()
    std = getattr(r.dram, dram_cfg["standard"])
    return std.timing_presets[dram_cfg["timing_preset"]]["tCK_ps"] // getattr(std, "tick_multiplier", 1)


def _dram_class(std):
    """DRAM spec class. HBM pseudo-channels natively move 32 B per access; every
    frontend here uses 64 B lines, so HBM is modelled with a 64 B payload whose
    burst occupies the bus twice as long (two back-to-back BL8 bursts to the same
    open row), which keeps per-channel bandwidth unchanged."""
    r = ramulator()
    base = getattr(r.dram, std)
    if std.startswith("HBM"):
        return type(f"{std}_64B", (base,), {"data_payload_bytes": 64})
    return base


def _overrides(dram_cfg):
    o = dict(dram_cfg.get("overrides", {}))
    if dram_cfg["standard"].startswith("HBM"):
        r = ramulator()
        nbl = getattr(r.dram, dram_cfg["standard"]).timing_presets[dram_cfg["timing_preset"]]["nBL"]
        o.setdefault("nBL", 2 * nbl)
    return o


def build_memory_system(dram_cfg, clock_ratio=1, queue_depth=None):
    """Return a GenericDRAM memory system with one controller per channel."""
    r = ramulator()
    std = dram_cfg["standard"]
    channels = dram_cfg["channels"]
    if channels & (channels - 1):
        raise ValueError("Ramulator CacheLineInterleave needs a power-of-two channel count")
    dram = _dram_class(std)(org_preset=dram_cfg["org_preset"], timing_preset=dram_cfg["timing_preset"],
                            **_overrides(dram_cfg))
    if std.startswith("HBM"):
        refresh = r.refresh_manager.HBM34PerBankRefresh()
    elif std == "LPDDR5":
        refresh = r.refresh_manager.PerBank()
    else:
        refresh = r.refresh_manager.AllBank()
    q = queue_depth or dram_cfg.get("queue_depth", 64)
    ctrl = getattr(r.controller, _CONTROLLERS[std])(
        dram=dram, scheduler=r.scheduler.FRFCFS(), refresh_manager=refresh,
        row_policy=r.row_policy.Open(), addr_mapper=r.addr_mapper.MOP4CLXOR(),
        read_buffer_size=q, write_buffer_size=q)
    return r.memory_system.GenericDRAM(clock_ratio=clock_ratio, controllers=[ctrl] * channels,
                                       channel_mapper=r.channel_mapper.CacheLineInterleave())


def peak_bandwidth_GBps(dram_cfg) -> float:
    """Pin bandwidth: channels x width x data rate."""
    r = ramulator()
    std = getattr(r.dram, dram_cfg["standard"])
    org = dict(std.org_presets[dram_cfg["org_preset"]])
    org.update({k: v for k, v in dram_cfg.get("overrides", {}).items() if k in org})
    rate = std.timing_presets[dram_cfg["timing_preset"]]["rate"]
    return dram_cfg["channels"] * org["channel_width"] * rate / 8 / 1000


def _controllers(stats):
    c = stats["memory_system"]["controller"]
    return c if isinstance(c, list) else [c]


# --------------------------------------------------------------------------
# Access-pattern calibration
# --------------------------------------------------------------------------
PATTERNS = {
    # name: description
    "stream_read": "sequential 100% read (weight streaming)",
    "kv_page_gather": "random 4 KiB pages read sequentially inside each page (paged KV cache)",
    "row_gather_256B": "random 256 B rows (embedding-table lookups)",
    "random_64B": "uniform random 64 B lines (pointer chasing / hash tables)",
    "stream_rw_2to1": "sequential, 2 reads : 1 write (activations / optimizer-free mixed)",
}


def pattern_trace(pattern, n, line, span, seed=1):
    rnd = random.Random(seed)
    out = []
    if pattern == "stream_read":
        out = [("LD", i * line) for i in range(n)]
    elif pattern == "stream_rw_2to1":
        for i in range(n):
            out.append(("ST" if i % 3 == 2 else "LD", i * line))
    elif pattern in ("kv_page_gather", "row_gather_256B"):
        chunk = 4096 if pattern == "kv_page_gather" else 256
        per = max(1, chunk // line)
        while len(out) < n:
            base = rnd.randrange(span // chunk) * chunk
            out.extend(("LD", base + j * line) for j in range(per))
        out = out[:n]
    elif pattern == "random_64B":
        out = [("LD", rnd.randrange(span // line) * line) for _ in range(n)]
    else:
        raise KeyError(pattern)
    return out


def calibrate(dram_cfg, patterns=None, requests=None, cache=True, refresh=False):
    """Achieved bandwidth (GB/s) and efficiency per access pattern.

    ``cache=True`` reads and writes results/cache/; ``refresh=True`` recomputes and
    overwrites the cached entry; ``cache=False`` neither reads nor writes it."""
    patterns = patterns or list(PATTERNS)
    key = json.dumps(dram_cfg, sort_keys=True) + f"|n={requests}"
    cache_file = CACHE_DIR / f"calib_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
    if cache and not refresh and cache_file.exists():
        data = json.loads(cache_file.read_text())
        if all(p in data["patterns"] for p in patterns):
            return data
    r = ramulator()
    tck = dram_clock_ps(dram_cfg)
    peak = peak_bandwidth_GBps(dram_cfg)
    # Enough requests to keep every channel busy for a long time.
    n = requests or max(120_000, dram_cfg["channels"] * 2000)
    span = min(dram_cfg.get("capacity_bytes", 1 << 34), 1 << 34) // 2
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for p in patterns:
            mem = build_memory_system(dram_cfg)
            line = 64
            trace = Path(tmp) / f"{p}.trace"
            trace.write_text("".join(f"{op} {addr}\n" for op, addr in pattern_trace(p, n, line, span)))
            # One request per frontend tick; frontend runs many ticks per memory tick.
            fe = r.frontend.LoadStoreTrace(clock_ratio=max(16, dram_cfg["channels"]), path=str(trace))
            sim = r.Simulation(fe, mem)
            sim.run()
            stats = sim.stats
            ctrls = _controllers(stats)
            reads = sum(c["num_read_reqs"] for c in ctrls)
            writes = sum(c["num_write_reqs"] for c in ctrls)
            # Controllers report throughput using the device transaction size.
            bw = sum(c["total_throughput_MBps"] for c in ctrls) / 1000
            hits = sum(c.get("row_hits", 0) for c in ctrls)
            misses = sum(c.get("row_misses", 0) for c in ctrls)
            conflicts = sum(c.get("row_conflicts", 0) for c in ctrls)
            lat = [c["avg_read_latency"] for c in ctrls if c.get("num_read_reqs")]
            results[p] = dict(achieved_GBps=bw, efficiency=bw / peak, requests=reads + writes,
                              row_hit_rate=hits / max(1, hits + misses + conflicts),
                              avg_loaded_read_latency_ns=(sum(lat) / len(lat)) * tck / 1000 if lat else None)
    data = dict(dram=dram_cfg, peak_GBps=peak, tCK_ps=tck, patterns=results)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, indent=2))
    return data


# --------------------------------------------------------------------------
# CPU trace replay (SPEC CPU)
# --------------------------------------------------------------------------
def _ratio(core_ghz, tck_ps):
    """Frontend:memory clock ratio as small integers."""
    mem_ghz = 1000 / tck_ps
    f = Fraction(core_ghz / mem_ghz).limit_denominator(16)
    return f.numerator, f.denominator


def llc_geometry(target_bytes, cores, min_ways=16, line=64):
    """SimpleO3's LLC indexes with ``sets - 1`` as a bit mask, so the set count
    must be a power of two. Keep the capacity close to the target by choosing the
    largest power-of-two set count with at least ``min_ways`` ways and rounding
    the associativity. Returns (total_bytes, ways, sets)."""
    lines = target_bytes // line
    sets = 1 << max(0, (lines // min_ways).bit_length() - 1)
    while sets * line % (cores * 1024):  # per-core capacity is passed in whole KB
        sets <<= 1
    ways = max(1, round(lines / sets))
    return sets * ways * line, ways, sets


def run_cpu_trace(arch, traces, insts, core_cfg):
    """Replay one SimpleO3 trace per core. Returns Ramulator stats dict + derived values."""
    r = ramulator()
    dram_cfg = arch.cpu_dram()
    tck = dram_clock_ps(dram_cfg)
    fe_ratio, mem_ratio = _ratio(core_cfg["clock_ghz"], tck)
    llc_cycles = max(1, round(arch.sram_latency_ns(len(traces)) * core_cfg["clock_ghz"]))
    llc_bytes, llc_ways, llc_sets = llc_geometry(arch.cpu_llc_bytes(len(traces)), len(traces),
                                                 core_cfg["llc_ways"])
    per_core = llc_bytes // len(traces)
    mem = build_memory_system(dram_cfg, clock_ratio=mem_ratio)
    fe = r.frontend.SimpleO3(
        clock_ratio=fe_ratio, num_expected_insts=int(insts), traces=[str(t) for t in traces],
        ipc=core_cfg["ipc"], inst_window_depth=core_cfg["window"],
        llc_latency=llc_cycles, llc_linesize=64, llc_associativity=llc_ways,
        llc_capacity_per_core=f"{per_core // 1024}KB",
        llc_num_mshr_per_core=core_cfg["mshrs_per_core"],
        translation=r.translation.NoTranslation(max_addr=1 << 46))
    sim = r.Simulation(fe, mem)
    sim.run()
    stats = sim.stats
    ctrls = _controllers(stats)
    f = stats["frontend"]
    cycles = [f[f"cycles_recorded_core_{i}"] for i in range(len(traces))]
    lat = [(c["avg_read_latency"], c["num_read_reqs"]) for c in ctrls if c.get("num_read_reqs")]
    nreq = sum(n for _, n in lat)
    return dict(
        core_cycles=cycles,
        llc_read_access=f["llc_read_access"], llc_read_misses=f["llc_read_misses"],
        llc_write_access=f["llc_write_access"], llc_write_misses=f["llc_write_misses"],
        dram_reads=sum(c["num_read_reqs"] for c in ctrls),
        dram_writes=sum(c["num_write_reqs"] for c in ctrls),
        dram_avg_read_latency_ns=(sum(l * n for l, n in lat) / nreq * tck / 1000) if nreq else 0.0,
        row_hits=sum(c.get("row_hits", 0) for c in ctrls),
        row_misses=sum(c.get("row_misses", 0) for c in ctrls),
        row_conflicts=sum(c.get("row_conflicts", 0) for c in ctrls),
        llc_latency_cycles=llc_cycles, llc_bytes=llc_bytes, llc_ways=llc_ways, llc_sets=llc_sets, clock_ratio=[fe_ratio, mem_ratio],
        dram=dram_cfg)
