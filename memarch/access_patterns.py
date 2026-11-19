"""Memory access-pattern characterization of the SPEC CPU 2026 traces.

Two levels, following the inter-arrival / power-spectral-density methodology of
Agarwal et al., "Towards Improving Memory Access in Large-Scale NoC-based Systems"
(ISCAS 2025):

1. **Trace level** (architecture-independent): the L2-miss stream written by the
   LLVM tracer. Spatial pattern (address vs time), locality of consecutive misses,
   re-reference within the window, and the spectrum of the inter-miss interval
   series (in instructions).

2. **Memory-controller level** (per architecture): a warm 1-core SimpleO3 replay
   with Ramulator's CmdTraceRecorder plugin on every channel controller. The
   column commands (RD/WR/RDA/WRA) of all channels are merged into one time
   series; the interval between consecutive commands is the inter-request time at
   the memory controllers (the plugin observes command *issue*, so this is the
   service-side counterpart of the paper's inter-packet arrival time). Its power
   spectral density S(f) is estimated with Welch's method and summarised by the
   log-log slope beta (S ~ 1/f^beta):

     * stationary   - flat spectrum: |beta| < 0.3 and both halves within +-0.4,
     * monofractal  - one straight 1/f^beta line,
     * multifractal - the low- and high-frequency halves have slopes that differ
                      by >= 0.5 (more than one scaling regime),
     * anti-persistent - beta <= -0.3: power rises with frequency (short and long
                      gaps alternate).

Results go to results/spec/access_patterns.json; figures.py plots them and the
report builder turns them into tables.
"""
from __future__ import annotations

import contextlib
import glob
import io
import itertools
import json
import os
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .paths import RESULTS_DIR, TRACE_DIR

# DRAM-backed architectures; the SRAM-only designs only reach a small host DDR5 spill memory.
MC_ARCHS = ["cpu_ddr5", "gpu_hbm3", "gpu_hbm4", "mesh_gddr6", "npu_lpddr5"]
MC_INSTS = 30_000_000  # replay 2 x MC_INSTS; the second half (by time) is analysed with a warm LLC
SCATTER_POINTS = 1000  # paper Fig. 1: first 1000 requests
ADDR_POINTS = 6000
LOCALITY = ["next line", "same 4 KiB page", "within 1 MiB", "far"]
STREAM_WINDOW = 16
COLUMN_CMDS = {"RD", "WR", "RDA", "WRA"}
ACT_CMDS = {"ACT", "ACT1"}  # LPDDR5 opens a row with ACT1 + ACT2; count ACT1 only


# --------------------------------------------------------------------------- spectra
def welch_psd(x, nperseg=None):
    """One-sided Welch PSD of a series sampled once per request (frequency in cycles/request)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if nperseg is None:
        nperseg = int(2 ** np.clip(np.floor(np.log2(max(n, 1) / 8)), 8, 14))
    nperseg = min(nperseg, n)
    step = nperseg // 2
    win = np.hanning(nperseg)
    scale = 1.0 / (win ** 2).sum()
    acc, k = np.zeros(nperseg // 2 + 1), 0
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start + nperseg]
        seg = (seg - seg.mean()) * win
        acc += np.abs(np.fft.rfft(seg)) ** 2
        k += 1
    psd = 2 * scale * acc / max(k, 1)
    f = np.fft.rfftfreq(nperseg)
    return f[1:], psd[1:]


def summarise_psd(f, s, bins=32):
    """Log-binned spectrum plus slope fits (whole range, low half, high half) and a class label."""
    edges = np.logspace(np.log10(f[0]), np.log10(f[-1]), bins + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)  # include the Nyquist bin
    fb, sb = [], []
    for lo, hi in itertools.pairwise(edges):
        m = (f >= lo) & (f < hi) & (s > 0)
        if m.any():
            fb.append(float(np.exp(np.log(f[m]).mean())))
            sb.append(float(np.exp(np.log(s[m]).mean())))
    if len(fb) < 3:
        return {"f": fb, "psd": sb, "beta": float("nan"), "beta_low": float("nan"), "beta_high": float("nan"),
                    "psd_class": "too short"}
    lf, ls = np.log10(fb), np.log10(sb)

    def slope(mask):
        return float(-np.polyfit(lf[mask], ls[mask], 1)[0]) if mask.sum() >= 3 else float("nan")

    mid = (lf[0] + lf[-1]) / 2
    beta, lo, hi = slope(np.ones_like(lf, bool)), slope(lf <= mid), slope(lf > mid)
    return {"f": fb, "psd": sb, "beta": beta, "beta_low": lo, "beta_high": hi, "psd_class": classify(beta, lo, hi)}


def classify(beta, lo, hi):
    """PSD class from the whole-range slope and the slopes of the low/high-frequency halves."""
    if abs(lo - hi) >= 0.5:
        return "multifractal"
    if beta <= -0.3:
        return "anti-persistent"  # power rises with frequency: short and long gaps alternate
    if abs(beta) < 0.3 and max(abs(lo), abs(hi)) < 0.4:
        return "stationary"
    return "monofractal"


# --------------------------------------------------------------------------- trace level
def read_trace(path):
    bubble, addr, wb = [], [], 0
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            bubble.append(int(p[0]))
            addr.append(int(p[1]))
            wb += len(p) > 2
    return np.array(bubble, dtype=np.int64), np.array(addr, dtype=np.int64), wb


def analyse_trace(path):
    path = Path(path)
    meta = json.loads(Path(str(path) + ".json").read_text())
    bubble, addr, wb = read_trace(path)
    ia = bubble + 1  # instructions from one miss to the next
    pos = np.cumsum(ia)
    delta = np.diff(addr)
    adist = np.abs(delta)
    same_page = (addr[1:] >> 12) == (addr[:-1] >> 12)
    cls = np.select([delta == 64, same_page, adist < (1 << 20)], [0, 1, 2], default=3)
    locality = {name: float((cls == i).mean()) for i, name in enumerate(LOCALITY)}
    lines = addr >> 6
    unique = np.unique(lines).size
    # Stream share: the miss is the next line after one of the previous STREAM_WINDOW misses (what a stream
    # prefetcher with that many trackers would catch). Interleaved streams count here but not as "next line".
    stream = np.zeros(len(lines), bool)
    for k in range(1, STREAM_WINDOW + 1):
        stream[k:] |= lines[k:] - lines[:-k] == 1
    stride, counts = np.unique(delta, return_counts=True)
    top = int(np.argmax(counts))
    sel = np.linspace(0, len(addr) - 1, min(ADDR_POINTS, len(addr))).astype(int)
    psd = summarise_psd(*welch_psd(ia))
    return dict(
        benchmark=path.stem, misses=len(addr), insts=int(pos[-1]),
        mpki=1000 * len(addr) / pos[-1], writebacks_per_miss=wb / len(addr),
        interval_mean=float(ia.mean()), interval_median=float(np.median(ia)), interval_cv=float(ia.std() / ia.mean()),
        interval_p99=float(np.percentile(ia, 99)),
        locality=locality, stream_share=float(stream[1:].mean()), rereference=1 - unique / len(addr), unique_MiB=unique * 64 / 2**20,
        footprint_MiB=meta["touched_pages_in_window"] * 4096 / 2**20,
        top_stride_bytes=int(stride[top]), top_stride_share=float(counts[top] / len(delta)),
        scatter_pos=(pos[sel] / 1e6).round(3).tolist(), scatter_addr=(addr[sel] / 2**20).round(3).tolist(),
        intervals=ia[:SCATTER_POINTS].tolist(), **{f"trace_{k}": v for k, v in psd.items()})


# --------------------------------------------------------------------------- controller level
def _read_cmd_trace(path):
    raw = Path(path).read_bytes()
    levels, ncmd = struct.unpack_from("<II", raw)
    off, names = 8, []
    for _ in range(levels + ncmd):
        end = raw.index(b"\0", off)
        names.append(raw[off:end].decode())
        off = end + 1
    cmds = names[levels:]
    rec = np.dtype([("clk", "<u8"), ("cmd", "<i4"), ("type", "<i4"), ("src", "<i4"), ("addr", "<i4", (levels,))])
    data = np.frombuffer(raw, dtype=rec, offset=off, count=(len(raw) - off) // rec.itemsize)
    col = np.isin(data["cmd"], [i for i, c in enumerate(cmds) if c in COLUMN_CMDS])
    act = np.isin(data["cmd"], [i for i, c in enumerate(cmds) if c in ACT_CMDS])
    return data["clk"][col], data["clk"][act]


def _mc_job(job):
    from .architectures import load_all
    from .cpu_study import CORE
    from .ramulator_backend import dram_clock_ps, run_cpu_trace

    arch = load_all([job["arch"]])[0]
    Path(job["tmp"]).mkdir(parents=True, exist_ok=True)
    prefix = Path(job["tmp"]) / f"{job['bench']}.{job['arch']}"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            stats = run_cpu_trace(arch, [Path(job["trace"])], 2 * job["insts"], CORE, cmd_trace=prefix)
        cols, acts = [], []
        for f in glob.glob(f"{prefix}.ch*"):
            c, a = _read_cmd_trace(f)
            cols.append(c)
            acts.append(a)
    finally:
        for f in glob.glob(f"{prefix}.ch*"):
            os.remove(f)
    tck_ns = dram_clock_ps(arch.cpu_dram()) / 1000
    col = np.sort(np.concatenate(cols)).astype(float) * tck_ns
    # Every read/write request is served by exactly one column command (reads forwarded from the write queue
    # excepted), so the recording must account for (almost) all of them; a short file means a failed write.
    served = stats["dram_reads"] + stats["dram_writes"]
    recorded = len(col) / max(1, served)
    if not 0.97 <= recorded <= 1.01:
        raise RuntimeError(f"{job['bench']} {job['arch']}: command trace holds {len(col):,} column commands "
                           f"for {served:,} requests ({recorded:.1%}); recording incomplete")
    act = np.concatenate(acts).astype(float) * tck_ns
    t0, t1 = col[-1] / 2, col[-1]  # warm half: the LLC was filled by the first half
    warm = col[col >= t0]
    nact = int((act >= t0).sum())
    ipat = np.diff(warm)
    psd = summarise_psd(*welch_psd(ipat))
    return dict(
        benchmark=job["bench"], arch=job["arch"], requests=len(warm), window_us=(t1 - t0) / 1000,
        recorded_fraction=recorded,
        throughput_GBps=len(warm) * 64 / (t1 - t0), act_per_column=nact / max(1, len(warm)),
        ipat_mean_ns=float(ipat.mean()), ipat_median_ns=float(np.median(ipat)),
        ipat_cv=float(ipat.std() / ipat.mean()), ipat_zero_share=float((ipat == 0).mean()),
        intervals_ns=ipat[:SCATTER_POINTS].round(3).tolist(), **psd)


def run(benchmarks=None, archs=None, insts=MC_INSTS, jobs=None, out_dir=None, tmp_dir=None, merge=False):
    """``merge=True`` replaces only the re-run (benchmark, arch) entries of an existing access_patterns.json.
    Command traces (up to ~0.5 GB per HBM job) go to ``tmp_dir`` on disk, not to a RAM-backed tmpfs."""
    out_dir = Path(out_dir or RESULTS_DIR / "spec")
    tmp_dir = Path(tmp_dir or TRACE_DIR / "spec" / "cmd_tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    traces = sorted((TRACE_DIR / "spec").glob("*.trace"))
    if benchmarks:
        traces = [t for t in traces if t.stem in benchmarks]
    archs = archs or MC_ARCHS
    work = [{"bench": t.stem, "trace": str(t), "arch": a, "insts": insts, "tmp": tmp_dir} for t in traces for a in archs]
    # HBM (64 channels) first: the slowest jobs.
    work.sort(key=lambda j: not j["arch"].startswith("gpu"))
    with ProcessPoolExecutor(jobs or 6) as pool:
        tr = {r["benchmark"]: r for r in pool.map(analyse_trace, traces)}
        for b, r in sorted(tr.items()):
            print(f"{b:>16} trace: MPKI {r['mpki']:.2f} next-line {r['locality']['next line']:.0%} "
                  f"stream {r['stream_share']:.0%} "
                  f"re-ref {r['rereference']:.0%} beta {r['trace_beta']:.2f} ({r['trace_psd_class']})", flush=True)
        mc = []
        for r in pool.map(_mc_job, work):
            mc.append(r)
            print(f"{r['benchmark']:>16} {r['arch']:>10}: {r['requests']:>8,} req {r['throughput_GBps']:6.2f} GB/s "
                  f"ACT/col {r['act_per_column']:.2f} beta {r['beta']:.2f} ({r['psd_class']})", flush=True)
    old = out_dir / "access_patterns.json"
    if merge and old.exists():
        prev = json.loads(old.read_text())
        done = {(r["benchmark"], r["arch"]) for r in mc}
        mc += [r for r in prev["mc"] if (r["benchmark"], r["arch"]) not in done]
        tr = {**{t["benchmark"]: t for t in prev["traces"]}, **tr}
        archs = [a for a in MC_ARCHS if a in set(prev["archs"]) | set(archs)]
    order = {a: i for i, a in enumerate(archs)}
    mc.sort(key=lambda r: (r["benchmark"], order[r["arch"]]))
    out = {"method": {"mc_insts_per_pass": insts, "warm": "second half of a 2N-instruction 1-core replay (by time)",
                           "psd": "Welch, Hann window, 50% overlap; slope fitted on 32 log-spaced bins",
                           "classes": "multifractal: |beta_low-beta_high|>=0.5; anti-persistent: beta<=-0.3; stationary: "
                                   "|beta|<0.3 and both halves within +-0.4; otherwise monofractal"},
               "archs": archs, "traces": [tr[b] for b in sorted(tr)], "mc": mc}
    (out_dir / "access_patterns.json").write_text(json.dumps(out, indent=1))
    return out
