"""Figures for the report (matplotlib, PNG + PDF vector)."""
from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .architectures import load_all
from .paths import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "figures"
# Validated categorical order (dataviz reference palette, light mode), one slot per architecture.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#d95926", "#f3d3c4", "#eeeeec", "#c7dbf3", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc0ec", "#2a78d6", "#123f75"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
    "axes.titleweight": "bold", "axes.titlesize": 10, "axes.axisbelow": True,
})


def _archs():
    archs = load_all()
    return archs, {a.name: SERIES[i % len(SERIES)] for i, a in enumerate(archs)}


def _plain_log(axis):
    from matplotlib.ticker import FuncFormatter, LogLocator

    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax, matrix, rows, cols, fmt, norm, cmap, label_fn=None):
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = matrix[i][j]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            ax.text(j, i, label_fn(v) if label_fn else fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color=INK)
    return im


# ---------------------------------------------------------------------------
def fig_landscape():
    archs, color = _archs()
    calib = json.loads((RESULTS_DIR / "ai/calibration.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for a in archs:
        pts = [(a.sram_bytes / 2**30, a.sram["bandwidth_GBps"] / 1000, "o", "SRAM")]
        if a.has_dram:
            pts.append((a.dram_bytes / 2**30, calib[a.name]["patterns"]["stream_read"]["achieved_GBps"] / 1000,
                        "s", "DRAM"))
        xs, ys = zip(*[(p[0], p[1]) for p in pts])
        ax.plot(xs, ys, color=color[a.name], lw=1.2, alpha=0.6, zorder=1)
        for x, y, m, _ in pts:
            ax.scatter(x, y, s=46, marker=m, color=color[a.name], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        dx, dy = {"gpu_hbm4": (-8, 6), "gpu_hbm3": (-6, -12), "npu_lpddr5": (14, -10), "mesh_gddr6": (8, 6),
                  "cpu_ddr5": (8, -10)}.get(a.name, (6, 4))
        ax.annotate(a.label, (pts[0][0], pts[0][1]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                    color=INK, ha="right" if dx < 0 else "left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.02, 1500)
    ax.set_xlabel("Capacity per chip (GiB)")
    ax.set_ylabel("Bandwidth per chip (TB/s)")
    ax.set_title("Capacity vs bandwidth per chip (circle = on-chip SRAM, square = DRAM, achieved stream BW)")
    _save(fig, "01_landscape")


def fig_calibration():
    archs, _ = _archs()
    calib = json.loads((RESULTS_DIR / "ai/calibration.json").read_text())
    pats = ["stream_read", "stream_rw_2to1", "kv_page_gather", "row_gather_256B", "random_64B"]
    rows, mat = [], []
    for a in archs:
        if not calib.get(a.name):
            continue
        c = calib[a.name]
        rows.append(f"{a.label}\n{c['dram']['standard']} {c['peak_GBps']:.0f} GB/s peak")
        mat.append([c["patterns"][p]["efficiency"] * 100 for p in pats])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = _heatmap(ax, mat, rows, [p.replace("_", " ") for p in pats], "{:.0f}%",
                  matplotlib.colors.Normalize(0, 100), SEQUENTIAL)
    for t in ax.texts:
        if float(t.get_text().rstrip("%")) > 60:
            t.set_color("white")
    fig.colorbar(im, ax=ax, label="% of pin bandwidth (Ramulator)", shrink=0.8)
    ax.set_title("Achieved DRAM bandwidth by access pattern")
    _save(fig, "02_dram_calibration")


def fig_ai_heatmap():
    archs, _ = _archs()
    rows = json.loads((RESULTS_DIR / "ai/ai_results.json").read_text())
    wls = list(dict.fromkeys(r["workload"] for r in rows))
    get = {(r["workload"], r["arch"]): r for r in rows}
    for metric, name, title in [
        ("speedup_vs_gpu_hbm3", "03_ai_latency_speedup", "Step latency speedup vs GPU + HBM3 (system with minimum chips)"),
        ("per_chip_vs_gpu_hbm3", "04_ai_per_chip", "Throughput per chip vs GPU + HBM3"),
    ]:
        mat = [[math.log2(get[(w, a.name)][metric]) for a in archs] for w in wls]
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        lim = max(abs(v) for row in mat for v in row)
        im = _heatmap(ax, mat, wls, [a.label for a in archs], None, TwoSlopeNorm(0, -lim, lim), DIVERGING,
                      label_fn=lambda v: f"{2 ** v:.2g}x")
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("log2 ratio (blue = better than GPU + HBM3)")
        ax.set_title(title)
        _save(fig, name)
    # chips required
    mat = [[get[(w, a.name)]["chips"] for a in archs] for w in wls]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = _heatmap(ax, [[math.log10(v) for v in r] for r in mat], wls, [a.label for a in archs], "{:.1f}",
                  matplotlib.colors.Normalize(0, 3), SEQUENTIAL)
    for t, v in zip(ax.texts, [v for r in mat for v in r]):
        t.set_text(str(v))
        if v >= 30:
            t.set_color("white")
    ax.set_title("Chips needed to hold the model state (capacity-driven)")
    _save(fig, "05_ai_chips")


def fig_breakdown():
    archs, _ = _archs()
    rows = json.loads((RESULTS_DIR / "ai/ai_results.json").read_text())
    get = {(r["workload"], r["arch"]): r for r in rows}
    parts = [("compute_bound", "Compute-bound time", "#2a78d6"), ("memory_bound", "Memory-bound time", "#eb6834"),
             ("comm_s", "Communication", "#1baf7a"), ("overhead_s", "Per-op overhead", "#eda100")]
    wls = ["llama8b_decode_b1", "llama8b_decode_b32", "llama70b_decode_b1", "llama8b_prefill"]
    fig, axes = plt.subplots(1, len(wls), figsize=(10.5, 3.6), sharey=False)
    for ax, w in zip(axes, wls):
        labels = [a.label for a in archs]
        bottoms = [0.0] * len(archs)
        for key, lab, col in parts:
            vals = []
            for a in archs:
                r = get[(w, a.name)]
                total = r["step_s"]
                cb, mb = r["compute_bound_s"], r["memory_bound_s"]
                v = {"compute_bound": cb, "memory_bound": mb}.get(key, r.get(key, 0))
                vals.append(v / total * 100 if total else 0)
            ax.barh(labels, vals, left=bottoms, color=col, label=lab, height=0.7, edgecolor=SURFACE, linewidth=1)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_title(w, fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of step time")
        ax.invert_yaxis()
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.4, 1.28), ncol=4)
    _save(fig, "06_time_breakdown")


def fig_energy():
    archs, color = _archs()
    rows = json.loads((RESULTS_DIR / "ai/ai_results.json").read_text())
    get = {(r["workload"], r["arch"]): r for r in rows}
    wls = [("llama8b_decode_b1", "Llama-8B decode, batch 1"), ("llama8b_decode_b32", "Llama-8B decode, batch 32"),
           ("llama70b_decode_b32", "Llama-70B decode, batch 32")]
    fig, axes = plt.subplots(2, len(wls), figsize=(10.5, 5.6))
    for j, (w, title) in enumerate(wls):
        for i, (key, ylabel) in enumerate([("memory_energy_j_per_unit", "Memory + link energy (mJ/token)"),
                                           ("tdp_energy_j_per_unit", "TDP x time (J/token)")]):
            ax = axes[i][j]
            scale = 1e3 if i == 0 else 1
            vals = [get[(w, a.name)][key] * scale for a in archs]
            ax.bar(range(len(archs)), vals, color=[color[a.name] for a in archs], width=0.7, edgecolor=SURFACE)
            ax.set_yscale("log")
            _plain_log(ax.yaxis)
            ax.set_xticks(range(len(archs)), [a.label for a in archs], rotation=40, ha="right", fontsize=7)
            if i == 0:
                ax.set_title(title, fontsize=9)
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ylabel)
            for k, v in enumerate(vals):
                ax.text(k, v, f"{v:,.0f}" if v >= 100 else f"{v:.3g}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    _save(fig, "07_energy")


def fig_sweeps():
    archs, color = _archs()
    for fname, xkey, xlabel, title, out in [
        ("batch_sweep_llama3_1_8b.json", "batch", "Batch size (sequences)",
         "Llama-8B decode: aggregate tokens/s per chip vs batch (2k context)", "08_batch_sweep"),
        ("context_sweep_llama3_1_8b.json", "context", "Context length (tokens)",
         "Llama-8B decode, batch 8: step latency vs context (KV-cache growth)", "09_context_sweep"),
    ]:
        rows = json.loads((RESULTS_DIR / "ai" / fname).read_text())
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for a in archs:
            pts = sorted((r[xkey], r) for r in rows if r["arch"] == a.name)
            xs = [p[0] for p in pts]
            ys = [p[1]["throughput_per_chip"] if xkey == "batch" else p[1]["latency_ms"] for p in pts]
            ax.plot(xs, ys, color=color[a.name], lw=2, marker="o", ms=4, label=a.label)
            ax.annotate(a.label, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                        va="center", color=INK)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Tokens/s per chip" if xkey == "batch" else "Decode step latency (ms)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.18, 1))
        _save(fig, out)


def fig_spec():
    path = RESULTS_DIR / "spec/spec_results.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    res = data["results"]
    archs, color = _archs()
    names = [a.name for a in archs if any(r["arch"] == a.name for r in res)]
    labels = {a.name: a.label for a in archs}
    benches = sorted({r["benchmark"] for r in res})
    get = {(r["benchmark"], r["arch"], r["cores"]): r for r in res}
    cores = sorted({r["cores"] for r in res})
    fig, axes = plt.subplots(1, len(cores), figsize=(5.2 * len(cores), 4.0))
    axes = axes if len(cores) > 1 else [axes]
    for ax, c in zip(axes, cores):
        mat = [[math.log2(get[(b, a, c)]["speedup_vs_cpu_ddr5"]) if (b, a, c) in get else float("nan")
                for a in names] for b in benches]
        lim = max(0.5, max(abs(v) for row in mat for v in row if not math.isnan(v)))
        im = _heatmap(ax, mat, benches, [labels[a] for a in names], None, TwoSlopeNorm(0, -lim, lim), DIVERGING,
                      label_fn=lambda v: f"{2 ** v:.2f}")
        ax.set_title(f"SPEC CPU 2026 IPC vs CPU + DDR5 ({c} core{'s' if c > 1 else ''})")
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="log2 IPC ratio (blue = faster)")
    _save(fig, "10_spec_ipc")

    # Memory-level parallelism / bandwidth demand on the CPU baseline vs GPU HBM3
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    c = max(cores)
    for ax, key, ylabel in [(axes[0], "dram_avg_read_latency_ns", "DRAM read latency at controller (ns)"),
                            (axes[1], "dram_GBps", f"DRAM bandwidth used, {c} cores (GB/s)")]:
        width = 0.8 / len(names)
        for i, a in enumerate(names):
            vals = [get[(b, a, c)][key] if (b, a, c) in get else 0 for b in benches]
            ax.bar([k + i * width for k in range(len(benches))], vals, width=width, color=color[a], label=labels[a],
                   edgecolor=SURFACE, linewidth=0.6)
        ax.set_xticks([k + 0.4 - width / 2 for k in range(len(benches))], benches, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        if key == "dram_GBps":
            ax.set_yscale("symlog", linthresh=1)
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1))
    axes[0].set_title("Loaded DRAM latency")
    axes[1].set_title("DRAM bandwidth demand")
    _save(fig, "11_spec_memory")


# ---------------------------------------------------------------------------
# Access-pattern characterization (memarch/access_patterns.py)
LOCALITY_COLORS = ["#123f75", "#2a78d6", "#9cc0ec", "#d9d8d3"]  # near -> far


def _short(bench):
    return bench.split(".", 1)[1].removesuffix("_r")


def fig_patterns():
    path = RESULTS_DIR / "spec/access_patterns.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    archs, color = _archs()
    labels = {a.name: a.label for a in archs}
    traces = data["traces"]
    mc = {(r["benchmark"], r["arch"]): r for r in data["mc"]}

    # 12: spatial pattern of the L2-miss stream
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.0))
    for ax, t in zip(axes.flat, traces):
        ax.scatter(t["scatter_pos"], t["scatter_addr"], s=1.2, color=SERIES[0], alpha=0.5, linewidths=0)
        ax.set_title(_short(t["benchmark"]), loc="left", pad=15)
        ax.text(0, 1.02, f"stream {t['stream_share']:.0%} · re-referenced {t['rereference']:.0%}",
                transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
    for ax in axes[1]:
        ax.set_xlabel("Instructions in trace window (M)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Address (MiB, first-touch)")
    fig.tight_layout()
    _save(fig, "12_access_spatial")

    # 13: inter-request interval at the memory controllers vs request index (paper Fig. 1)
    show = [a for a in ("cpu_ddr5", "gpu_hbm3") if a in data["archs"]]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.0), sharex=True)
    for ax, t in zip(axes.flat, traces):
        b = t["benchmark"]
        for a, m in zip(show, ("x", "+")):
            r = mc.get((b, a))
            if r:
                y = r["intervals_ns"]
                ax.scatter(range(len(y)), y, s=9, marker=m, color=color[a], linewidths=0.7, alpha=0.75,
                           label=labels[a])
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(_short(b), loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Data request index")
    for ax in axes[:, 0]:
        ax.set_ylabel("Time since last request (ns)")
    axes[0, -1].legend(fontsize=7, loc="upper right", markerscale=1.5)
    fig.tight_layout()
    _save(fig, "13_mc_interarrival")

    # 14: log-log PSD of the inter-request time series (paper Figs. 3-5)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
    for ax, t in zip(axes.flat, traces):
        b = t["benchmark"]
        for a in data["archs"]:
            r = mc.get((b, a))
            if r:
                ax.loglog(r["f"], r["psd"], color=color[a], linewidth=1.4, marker=".", markersize=3,
                          label=f"{labels[a]}")
        ax.set_title(_short(b), loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Frequency (cycles / request)")
    for ax in axes[:, 0]:
        ax.set_ylabel("PSD of inter-request time")
    handles, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=len(lab), fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "14_mc_psd")

    # 15: locality of consecutive misses (trace) and DRAM row-buffer hit rate (Ramulator, 1 core)
    spec = RESULTS_DIR / "spec/spec_results.json"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    names = [_short(t["benchmark"]) for t in traces]
    left = [0.0] * len(traces)
    for k, (cat, c) in enumerate(zip(["next line", "same 4 KiB page", "within 1 MiB", "far"], LOCALITY_COLORS)):
        vals = [100 * t["locality"][cat] for t in traces]
        ax1.barh(names, vals, left=left, color=c, label=cat, edgecolor=SURFACE, linewidth=1.5, height=0.7)
        left = [x + v for x, v in zip(left, vals)]
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of consecutive L2-miss pairs (%)")
    ax1.set_title("Distance to the previous miss")
    ax1.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.07))
    ax1.grid(axis="y", visible=False)
    if spec.exists():
        res = json.loads(spec.read_text())["results"]
        get = {(r["benchmark"], r["arch"]): r for r in res if r["cores"] == 1}
        cols = [a.name for a in archs]
        mat = [[100 * get[(t["benchmark"], a)]["row_hit_rate"] if (t["benchmark"], a) in get else float("nan")
                for a in cols] for t in traces]
        im = _heatmap(ax2, mat, names, [labels[a] for a in cols], "{:.0f}", plt.Normalize(0, 100), SEQUENTIAL)
        fig.colorbar(im, ax=ax2, shrink=0.85, label="DRAM row-buffer hit rate (%)")
        ax2.set_title("Row-buffer hits at the DRAM (1 core)")
    fig.tight_layout()
    _save(fig, "15_locality")


def make_all():
    fig_landscape()
    fig_calibration()
    fig_ai_heatmap()
    fig_breakdown()
    fig_energy()
    fig_sweeps()
    fig_spec()
    fig_patterns()
    print(f"figures -> {FIG_DIR}")
