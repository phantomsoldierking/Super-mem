#!/usr/bin/env python3
"""Render the report templates into PDFs.

    .venv/bin/python report/build_report.py

Templates (hand-written prose) live in report/*.md and contain tokens that are
replaced with tables and figures generated from results/:

    {{arch_table}}  {{calib_table}}  {{ai_overview}}  {{ai_table:<scenario>}}
    {{spec_traces}} {{spec_table:<cores>}}  {{public_check}}  {{gpu_bound_pct}}  {{fig:<name>|<caption>}}
    {{replay_config}}  {{pattern_table}}  {{mc_table:throughput|psd}}

Outputs:
    output/memory_architecture_study_detailed.pdf  (from report/detailed.md)
    output/memory_architecture_study_summary.pdf   (from report/summary.md)
Intermediate HTML is written to a temporary directory and discarded.
"""
import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memarch.architectures import load_all

RES = ROOT / "results"
FIG = RES / "figures"
OUT = ROOT / "output"
DOCS = {"detailed.md": "memory_architecture_study_detailed.pdf",
        "summary.md": "memory_architecture_study_summary.pdf"}


def gb(x):
    return f"{x / 2**30:,.0f} GiB" if x >= 2**30 else f"{x / 2**20:,.0f} MiB"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


class Data:
    def __init__(self):
        self.archs = load_all()
        self.calib = json.loads((RES / "ai/calibration.json").read_text())
        self.ai = json.loads((RES / "ai/ai_results.json").read_text())
        p = RES / "spec/spec_results.json"
        self.spec = json.loads(p.read_text()) if p.exists() else None
        p = RES / "spec/access_patterns.json"
        self.patterns = json.loads(p.read_text()) if p.exists() else None

    def ai_row(self, workload, arch):
        return next(r for r in self.ai if r["workload"] == workload and r["arch"] == arch)


def arch_table(d, _):
    rows = []
    for a in d.archs:
        c = d.calib.get(a.name)
        dram = (f"{a.dram['standard']} x{a.dram['channels']}, {gb(a.dram_bytes)}, {c['peak_GBps']:,.0f} GB/s"
                if a.has_dram else "none")
        rows.append([f"**{a.label}**<br>{a.analogue}",
                     f"{gb(a.sram_bytes)} {a.sram_role}<br>{a.sram['bandwidth_GBps'] / 1000:,.0f} TB/s, {a.sram['latency_ns']} ns",
                     dram, f"{a.compute['matrix_tflops']:,}", f"{a.interconnect['link_GBps']}", f"{a.tdp_w:,}"])
    return table(["Architecture", "On-chip SRAM (role)", "DRAM (pin BW)", "TFLOPS bf16", "Link GB/s", "TDP W"], rows)


def calib_table(d, _):
    pats = ["stream_read", "stream_rw_2to1", "kv_page_gather", "row_gather_256B", "random_64B"]
    rows = []
    for a in d.archs:
        c = d.calib.get(a.name)
        if c:
            rows.append([a.label, f"{c['peak_GBps']:,.0f}"] +
                        [f"{c['patterns'][p]['achieved_GBps']:,.0f} ({c['patterns'][p]['efficiency']:.0%})" for p in pats])
    return table(["DRAM system", "Pin GB/s"] + [p.replace("_", " ") for p in pats], rows)


def ai_table(d, workload):
    out = []
    for a in d.archs:
        r = d.ai_row(workload, a.name)
        out.append([a.label, r["chips"], f"{r['latency_ms']:.3g}", f"{r['throughput_per_s']:,.4g}",
                    f"{r['throughput_per_chip']:,.4g}", f"{r['speedup_vs_gpu_hbm3']:.2f}x",
                    f"{r['memory_energy_j_per_unit'] * 1e3:.3g}", f"{r['tdp_energy_j_per_unit']:.3g}", r["bound"]])
    unit = d.ai_row(workload, d.archs[0].name)["unit"]
    sc = d.ai_row(workload, d.archs[0].name)
    head = (f"*{sc['model']}, {sc['phase']}, batch {sc['batch']}"
            + (f", context {sc['context']}" if sc.get("context") else "")
            + f"; persistent state {sc['resident_GB']:.1f} GB.*\n\n")
    return head + table(["Architecture", "Chips", "Step ms", f"{unit}s/s", f"{unit}s/s/chip", "Latency vs GPU",
                         f"Mem+link mJ/{unit}", f"TDP J/{unit}", "Bound by"], out)


def ai_overview(d, _):
    wls = list(dict.fromkeys(r["workload"] for r in d.ai))
    rows = []
    for w in wls:
        rs = [d.ai_row(w, a.name) for a in d.archs]
        fastest = max(rs, key=lambda r: 1 / r["step_s"])
        per_chip = max(rs, key=lambda r: r["throughput_per_chip"])
        per_kw = max(rs, key=lambda r: r["throughput_per_kw"])
        lab = {a.name: a.label for a in d.archs}
        rows.append([w, f"{lab[fastest['arch']]} ({fastest['chips']} chips)", lab[per_chip["arch"]], lab[per_kw["arch"]]])
    return table(["Scenario", "Lowest latency", "Most throughput per chip", "Most throughput per kW (TDP)"], rows)


PUBLIC = [  # (scenario, arch, public tokens/s per user, source)
    ("llama8b_decode_b1", "lpu_sram", "~750 (2024) / 635-850 (Artificial Analysis)", "Groq via press and artificialanalysis.ai"),
    ("llama70b_decode_b1", "lpu_sram", "~250 (2024)", "press coverage of Groq launch figures"),
    ("llama8b_decode_b1", "wafer_sram", "~1,800 (2024)", "Cerebras inference launch blog"),
    ("llama70b_decode_b1", "wafer_sram", "~450 (2024), ~2,100 after later software", "Cerebras blogs"),
]


def public_check(d, _):
    rows = []
    lab = {a.name: a.label for a in d.archs}
    for w, a, pub, src in PUBLIC:
        r = d.ai_row(w, a)
        rows.append([w, lab[a], r["chips"], f"{r['throughput_per_s']:,.0f}", pub, src])
    gpu = d.ai_row("llama8b_decode_b1", "gpu_hbm3")
    bound = d.calib["gpu_hbm3"]["peak_GBps"] * 1e9 / (gpu["dram_bytes"])
    rows.append(["llama8b_decode_b1", lab["gpu_hbm3"], gpu["chips"], f"{gpu['throughput_per_s']:,.0f}",
                 f"<= {bound:,.0f} (pin-bandwidth bound)",
                 f"{gpu['dram_bytes'] / 1e9:.1f} GB DRAM traffic per token / {d.calib['gpu_hbm3']['peak_GBps'] / 1000:.2f} TB/s"])
    return table(["Scenario", "Model of", "Chips", "Model tokens/s", "Public figure tokens/s", "Source"], rows)


def gpu_bound_pct(d, _):
    """GPU + HBM3 Llama-8B batch-1 throughput as a share of its pin-bandwidth bound."""
    gpu = d.ai_row("llama8b_decode_b1", "gpu_hbm3")
    bound = d.calib["gpu_hbm3"]["peak_GBps"] * 1e9 / gpu["dram_bytes"]
    return f"{gpu['throughput_per_s'] / bound:.0%}"


def spec_traces(d, _):
    if not d.spec:
        return "*SPEC results not available.*"
    rows = []
    for b, s in sorted(d.spec["traces"].items()):
        rows.append([b, f"{s['trace_insts'] / 1e6:.0f}M", f"{1000 * s['l2_misses'] / s['trace_insts']:.2f}",
                     f"{s['l2_dirty_writebacks'] / max(1, s['l2_misses']):.0%}",
                     f"{s['touched_pages_in_window'] * 4096 / 2**20:,.0f}", f"{s['peak_rss_bytes'] / 2**20:,.0f}",
                     f"`{s['command'].split()[0] if s['command'] else ''}`"])
    return table(["Benchmark", "Traced insts", "L2 MPKI", "Dirty WB / miss", "Window MiB", "Peak RSS MiB",
                  "First arg"], rows)


def spec_table(d, cores):
    if not d.spec:
        return ""
    cores = int(cores)
    res = [r for r in d.spec["results"] if r["cores"] == cores]
    get = {(r["benchmark"], r["arch"]): r for r in res}
    rows = []
    for b in sorted({r["benchmark"] for r in res}):
        base = get[(b, "cpu_ddr5")]
        row = [b, f"{base['ipc_per_core']:.2f}", f"{base['dram_apki']:.2f}"]
        row += [f"{get[(b, a.name)]['speedup_vs_cpu_ddr5']:.2f}" for a in d.archs]
        rows.append(row)
    import math

    geo = ["**geomean**", "", ""]
    for a in d.archs:
        v = [get[(b, a.name)]["speedup_vs_cpu_ddr5"] for b in sorted({r["benchmark"] for r in res})]
        geo.append(f"**{math.exp(sum(math.log(x) for x in v) / len(v)):.2f}**")
    rows.append(geo)
    return table(["Benchmark", "CPU IPC", "CPU DRAM acc./kinst"] + [a.label for a in d.archs], rows)


def replay_config(d, _):
    """System configuration of the SPEC replay (cf. Table I of Agarwal et al., ISCAS 2025)."""
    if not d.spec:
        return ""
    core = d.spec["core"]
    general = table(["Parameter", "Value"], [
        ["Traces", ("8 SPEC CPU 2026 rate benchmarks (refrate, first workload); 30B instructions skipped, "
                   "100M warm-up, 200M traced (LLVM IR instructions)")],
        ["Private L1D / L2 (tracer)", "48 KiB 12-way / 2 MiB 16-way, 64 B lines, LRU, write-back"],
        ["Core (Ramulator SimpleO3)", f"{core['ipc']}-wide, {core['window']}-entry window, {core['clock_ghz']:g} GHz"],
        ["Shared LLC", (f"architecture's SRAM, power-of-two sets, at least {core['llc_ways']} ways (per-architecture "
                        f"ways below), {core['mshrs_per_core']} MSHRs per core")],
        ["Memory controller", "one per channel; FR-FCFS, open row, MOP4CLXOR mapping, refresh, 64-entry queues"],
        ["IPC study", ("1 and 4 cores; run(2N) - run(N), i.e. instructions N..2N with the LLC warmed by the first N; "
                       "N = 100M (1 core) / 10M per core (4 cores)")],
        ["Access-pattern study", ("1 core; 2 x 30M instructions, instructions 30M-60M analysed; every DRAM "
                                 "command recorded by Ramulator's CmdTraceRecorder")],
    ])
    get = {r["arch"]: r for r in d.spec["results"] if r["cores"] == 1}
    rows = []
    for a in d.archs:
        r = get.get(a.name)
        if r:
            rows.append([a.label, gb(r["llc_bytes"]), f"{r['llc_ways']}", f"{r['llc_latency_cycles']}",
                         f"{r['llc_latency_cycles'] / core['clock_ghz']:.0f}", r["dram"]])
    per_arch = table(["Architecture", "LLC (1 core)", "Ways", "LLC hit (cycles)", "LLC hit (ns)", "DRAM"], rows)
    return general + "\n\n" + per_arch


def pattern_table(d, _):
    if not d.patterns:
        return "*Access-pattern results not available (run `python -m memarch patterns`).*"
    rows = []
    for t in d.patterns["traces"]:
        loc = t["locality"]
        rows.append([t["benchmark"], f"{t['mpki']:.2f}", f"{t['writebacks_per_miss']:.2f}",
                     f"{t['interval_mean']:,.0f} / {t['interval_median']:,.0f}", f"{t['interval_cv']:.1f}",
                     f"{loc['next line']:.0%}", f"{loc['same 4 KiB page']:.0%}", f"{loc['within 1 MiB']:.0%}",
                     f"{loc['far']:.0%}", f"{t['stream_share']:.0%}", f"{t['rereference']:.0%}", f"{t['unique_MiB']:,.0f}",
                     f"{t['trace_beta']:.2f} ({t['trace_psd_class']})"])
    return table(["Benchmark", "L2 MPKI", "WB / miss", "Inter-miss mean / median (insts)", "CV", "Next line",
                  "Same page", "≤1 MiB", "Far", "Stream", "Re-ref.", "Unique MiB", "PSD β (class)"], rows)


def mc_table(d, kind):
    if not d.patterns:
        return ""
    archs = [a for a in d.archs if a.name in d.patterns["archs"]]
    get = {(r["benchmark"], r["arch"]): r for r in d.patterns["mc"]}
    benches = [t["benchmark"] for t in d.patterns["traces"]]
    short = {"stationary": "S", "monofractal": "M", "multifractal": "MF", "anti-persistent": "AP", "too short": "–"}
    rows = []
    for b in benches:
        row = [b]
        for a in archs:
            r = get.get((b, a.name))
            if r is None:
                row.append("–")
            elif kind == "throughput":
                row.append(f"{r['throughput_GBps']:.2f}")
            else:
                row.append(f"{r['beta']:.2f} {short[r['psd_class']]}")
        rows.append(row)
    if kind == "throughput":
        return table(["Benchmark"] + [f"{a.label} (GB/s)" for a in archs], rows)
    return table(["Benchmark"] + [f"{a.label} β" for a in archs], rows)


def fig(_, arg):
    name, caption = arg.split("|", 1)
    return f'<figure><img src="{(FIG / (name + ".png")).as_uri()}"><figcaption>{caption}</figcaption></figure>'


TOKENS = {"arch_table": arch_table, "calib_table": calib_table, "ai_table": ai_table, "ai_overview": ai_overview,
          "public_check": public_check, "gpu_bound_pct": gpu_bound_pct, "spec_traces": spec_traces, "spec_table": spec_table, "fig": fig,
          "replay_config": replay_config, "pattern_table": pattern_table, "mc_table": mc_table}


def render(text, d):
    def sub(m):
        name, _, arg = m.group(1).partition(":")
        return TOKENS[name](d, arg)

    text = text.replace("{{date}}", datetime.date.today().isoformat())
    return re.sub(r"\{\{([^}]+)\}\}", sub, text)


CSS = """
@page{size:A4;margin:16mm 15mm}
body{font-family:'Inter','Helvetica Neue',Arial,sans-serif;font-size:10pt;line-height:1.45;color:#111;background:#fff}
h1{font-size:19pt;margin:0 0 4pt}h2{font-size:13.5pt;border-bottom:1px solid #ccc;padding-bottom:2pt;margin-top:18pt;
break-after:avoid}h3{font-size:11pt;break-after:avoid;margin-top:12pt}
table{border-collapse:collapse;font-size:7.8pt;margin:6pt 0 10pt;width:100%;break-inside:avoid}
th,td{border-bottom:1px solid #e4e3df;padding:2.5pt 4pt;vertical-align:top}th{text-align:left;background:#f4f3f0}
td:not(:first-child){text-align:right}
figure{margin:8pt 0 12pt;break-inside:avoid;text-align:center}figure img{max-width:100%;max-height:118mm}
figcaption{font-size:8.5pt;color:#555;margin-top:2pt}
code{font-size:8.5pt;background:#f4f3f0;padding:0 2pt;border-radius:2px}
pre{background:#f4f3f0;padding:6pt 8pt;font-size:7.8pt;line-height:1.35;overflow:hidden;white-space:pre-wrap;break-inside:avoid}
pre code{background:none;padding:0}blockquote{border-left:3px solid #2a78d6;margin:8pt 0;padding:2pt 10pt;color:#333}
.subtitle{color:#555;font-size:10pt;margin-bottom:12pt}
"""


def main(only=None):
    import markdown

    d = Data()
    chrome = shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("chrome")
    if not chrome:
        raise SystemExit("Chromium is required to print the PDF")
    OUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for src, pdf_name in DOCS.items():
            if only and src not in only:
                continue
            md = render((ROOT / "report" / src).read_text(), d)
            body = markdown.markdown(md, extensions=["tables", "fenced_code", "md_in_html"])
            page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{pdf_name}</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
            html_path = Path(tmp) / (src + ".html")
            html_path.write_text(page)
            pdf = OUT / pdf_name
            subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                            "--allow-file-access-from-files", f"--print-to-pdf={pdf}", html_path.as_uri()],
                           check=True, capture_output=True)
            print("wrote", pdf)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
