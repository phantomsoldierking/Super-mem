"""SPEC CPU 2026 trace replay on every memory architecture.

The same out-of-order core (Ramulator SimpleO3: 4-wide, 128-entry window,
4 GHz) runs the same L2-miss trace on each architecture; only the memory
system behind the private L2 changes:

  * last-level SRAM: capacity and latency of the architecture's on-chip SRAM
    (LLC for CPU/GPU, scratchpad/resident SRAM used as a cache otherwise),
  * DRAM: the architecture's DRAM simulated in Ramulator (SRAM-only chips use
    a small host DDR5 spill memory, which the SPEC footprints rarely reach).

Warm-cache measurement: Ramulator replays a trace cyclically, so each
configuration is simulated for N and for 2N instructions and the second pass
(2N - N) is reported. Its LLC is warmed by the first pass.

Multi-core ("rate") runs give each core its own copy of the trace in a
disjoint physical address range, sharing the LLC and DRAM.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .architectures import load_all
from .paths import RESULTS_DIR, TRACE_DIR

CORE = dict(clock_ghz=4.0, ipc=4, window=128, llc_ways=16, mshrs_per_core=32)
OFFSET_ALIGN = 1 << 30
# Instructions per measured pass (per core). Each job simulates N + 2N.
DEFAULT_INSTS = {1: 100_000_000, 4: 10_000_000}


def _trace_info(trace: Path):
    stats = json.loads(Path(str(trace) + ".json").read_text())
    return stats


def _prepare_core_traces(trace: Path, cores: int, max_lines: int, workdir: Path):
    """Write per-core copies (first max_lines lines) in disjoint address ranges.

    Returns (paths, instructions represented by one copy)."""
    workdir.mkdir(parents=True, exist_ok=True)
    lines, insts, max_addr = [], 0, 0
    with open(trace) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            parts = line.split()
            insts += int(parts[0]) + 1
            max_addr = max(max_addr, *(int(x) for x in parts[1:]))
            lines.append(parts)
    span = (max_addr // OFFSET_ALIGN + 1) * OFFSET_ALIGN
    paths = []
    for c in range(cores):
        p = workdir / f"{trace.stem}.core{c}.trace"
        # Copies are keyed by max_lines (directory name) and rebuilt if the source trace is newer.
        if not p.exists() or p.stat().st_mtime < trace.stat().st_mtime:
            off = c * span
            with open(p, "w") as f:
                for parts in lines:
                    f.write(" ".join([parts[0], *(str(int(x) + off) for x in parts[1:])]) + "\n")
        paths.append(p)
    return paths, insts


def _simulate(job):
    """Run one (arch, benchmark, cores) job: N and 2N instruction passes."""
    import contextlib
    import io

    from .architectures import load_all as _load
    from .ramulator_backend import run_cpu_trace

    arch = _load([job["arch"]])[0]
    traces = [Path(p) for p in job["traces"]]
    n = job["insts"]
    with contextlib.redirect_stdout(io.StringIO()):
        first = run_cpu_trace(arch, traces, n, CORE)
        both = run_cpu_trace(arch, traces, 2 * n, CORE)
    cores = len(traces)
    warm_cycles = [b - a for a, b in zip(first["core_cycles"], both["core_cycles"])]
    d = {k: both[k] - first[k] for k in ("dram_reads", "dram_writes", "llc_read_access", "llc_read_misses",
                                          "row_hits", "row_misses", "row_conflicts")}
    lat_sum = both["dram_avg_read_latency_ns"] * both["dram_reads"] - first["dram_avg_read_latency_ns"] * first["dram_reads"]
    seconds = max(warm_cycles) / (CORE["clock_ghz"] * 1e9)
    ipc = [n / c for c in warm_cycles]
    rows = d["row_hits"] + d["row_misses"] + d["row_conflicts"]
    return dict(
        arch=job["arch"], benchmark=job["bench"], cores=cores, insts_per_core=n,
        ipc_per_core=sum(ipc) / cores, ipc_min=min(ipc),
        # LLC misses are reported as DRAM accesses per kilo-instruction (dram_apki, see postprocess): every
        # primary LLC miss issues one DRAM read. Ramulator's llc_read_misses/llc_read_access counters are bumped
        # before the MSHR check and so also count MSHR merges and retries; they are not used.
        dram_GBps=(d["dram_reads"] + d["dram_writes"]) * 64 / seconds / 1e9,
        dram_avg_read_latency_ns=lat_sum / d["dram_reads"] if d["dram_reads"] else 0.0,
        row_hit_rate=d["row_hits"] / rows if rows else 0.0,
        llc_latency_cycles=both["llc_latency_cycles"], llc_bytes=both["llc_bytes"], llc_ways=both["llc_ways"],
        dram=f"{both['dram']['standard']} x{both['dram']['channels']}")


def run(benchmarks=None, archs=None, core_counts=(1, 4), insts=None, mc_lines=1_500_000, jobs=None,
        trace_dir=None, out_dir=None, merge=False):
    """Replay every (benchmark, architecture, core count). With ``merge=True`` the new
    results replace matching entries of an existing spec_results.json instead of
    overwriting the whole file (used to re-run a subset of architectures)."""
    trace_dir = Path(trace_dir or TRACE_DIR / "spec")
    out_dir = Path(out_dir or RESULTS_DIR / "spec")
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = sorted(trace_dir.glob("*.trace"))
    if benchmarks:
        traces = [t for t in traces if t.stem in benchmarks]
    if not traces:
        raise FileNotFoundError(f"No traces in {trace_dir}; run spec/trace_spec.py first")
    arch_names = [a.name for a in load_all(archs)]
    work = []
    meta = {}
    for t in traces:
        meta[t.stem] = _trace_info(t)
        for cores in core_counts:
            lines = 10**12 if cores == 1 else mc_lines
            paths, n = _prepare_core_traces(t, cores, lines, trace_dir / "multicore" / f"{cores}c_{lines}l") \
                if cores > 1 \
                else ([t], None)
            if n is None:
                n = meta[t.stem]["trace_insts"]
            limit = (insts or DEFAULT_INSTS).get(cores, min((insts or DEFAULT_INSTS).values()))
            n = min(n, limit)
            for a in arch_names:
                work.append(dict(arch=a, bench=t.stem, traces=[str(p) for p in paths], insts=n))
    # Larger multi-core jobs first so the pool stays busy.
    work.sort(key=lambda j: -len(j["traces"]))
    results = []
    with ProcessPoolExecutor(jobs or max(1, (os.cpu_count() or 2) - 2)) as pool:
        for r in pool.map(_simulate, work):
            results.append(r)
            print(f"{r['benchmark']:>16} {r['arch']:>11} x{r['cores']}: IPC {r['ipc_per_core']:.3f} "
                  f"DRAM {r['dram_GBps']:.1f} GB/s "
                  f"lat {r['dram_avg_read_latency_ns']:.0f} ns", flush=True)
    if merge and (out_dir / "spec_results.json").exists():
        new = {(r["benchmark"], r["arch"], r["cores"]) for r in results}
        old = json.loads((out_dir / "spec_results.json").read_text())
        results += [r for r in old["results"] if (r["benchmark"], r["arch"], r["cores"]) not in new]
        meta = {**old["traces"], **meta}
        arch_names = [a.name for a in load_all()]
    postprocess(results)
    results.sort(key=lambda r: (r["benchmark"], r["cores"], arch_names.index(r["arch"])))
    (out_dir / "spec_results.json").write_text(json.dumps(dict(core=CORE, traces=meta, results=results), indent=2))
    _write_csv(out_dir / "spec_results.csv", results)
    return results


def postprocess(results):
    """Derived columns: DRAM accesses per kilo-instruction and speedup over the CPU + DDR5 baseline."""
    for r in results:
        seconds = r["insts_per_core"] / (r["ipc_min"] * CORE["clock_ghz"] * 1e9)
        r["dram_apki"] = 1000 * r["dram_GBps"] * 1e9 * seconds / 64 / (r["insts_per_core"] * r["cores"])
        for k in ("llc_miss_rate", "l2_mpki", "llc_mpki"):  # superseded by dram_apki (see _simulate)
            r.pop(k, None)
    base = {(r["benchmark"], r["cores"]): r["ipc_per_core"] for r in results if r["arch"] == "cpu_ddr5"}
    for r in results:
        b = base.get((r["benchmark"], r["cores"]))
        r["speedup_vs_cpu_ddr5"] = r["ipc_per_core"] / b if b else None
    return results


def _write_csv(path, rows):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
