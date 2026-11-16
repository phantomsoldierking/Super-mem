"""AI inference study: every scenario in configs/scenarios.json on every architecture."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .architectures import load_all
from .paths import RESULTS_DIR
from .perf import evaluate
from .ramulator_backend import calibrate
from .workloads import all_workloads


def calibrations(archs):
    return {a.name: (calibrate(a.dram) if a.has_dram else None) for a in archs}


def run(archs=None, scenarios=None, out_dir=None):
    out_dir = Path(out_dir or RESULTS_DIR / "ai")
    out_dir.mkdir(parents=True, exist_ok=True)
    archs = load_all(archs)
    calib = calibrations(archs)
    wls = all_workloads(scenarios)
    rows = []
    for wl in wls:
        for a in archs:
            r = evaluate(a, wl, calib[a.name]).row()
            r.update(model=wl.model["name"], phase=wl.scenario["phase"], batch=wl.scenario["batch"],
                     context=wl.scenario.get("context"), resident_GB=wl.resident_bytes / 1e9,
                     arch_label=a.label)
            rows.append(r)
    base = {r["workload"]: r for r in rows if r["arch"] == "gpu_hbm3"}
    for r in rows:
        b = base[r["workload"]]
        r["speedup_vs_gpu_hbm3"] = b["step_s"] / r["step_s"]
        r["per_chip_vs_gpu_hbm3"] = r["throughput_per_chip"] / b["throughput_per_chip"]
        r["energy_vs_gpu_hbm3"] = r["energy_j_per_unit"] / b["energy_j_per_unit"]
    with open(out_dir / "ai_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "ai_results.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2))
    return rows


def sweep_batch(arch_names=None, model="llama3_1_8b", context=2048, batches=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                out_dir=None):
    """Decode throughput vs latency as batch grows (memory-bound -> compute-bound)."""
    from .workloads import build

    out_dir = Path(out_dir or RESULTS_DIR / "ai")
    archs = load_all(arch_names)
    calib = calibrations(archs)
    rows = []
    for a in archs:
        for b in batches:
            wl = build({"id": f"{model}_b{b}", "model": model, "phase": "decode", "batch": b, "context": context})
            r = evaluate(a, wl, calib[a.name]).row()
            r.update(batch=b, context=context, model=model)
            rows.append(r)
    (out_dir / f"batch_sweep_{model}.json").write_text(json.dumps(rows, indent=2))
    return rows


def sweep_context(arch_names=None, model="llama3_1_8b", batch=8,
                  contexts=(512, 2048, 8192, 32768, 131072), out_dir=None):
    """KV-cache growth: decode step time as context grows."""
    from .workloads import build

    out_dir = Path(out_dir or RESULTS_DIR / "ai")
    archs = load_all(arch_names)
    calib = calibrations(archs)
    rows = []
    for a in archs:
        for c in contexts:
            wl = build({"id": f"{model}_c{c}", "model": model, "phase": "decode", "batch": batch, "context": c})
            r = evaluate(a, wl, calib[a.name]).row()
            r.update(batch=batch, context=c, model=model)
            rows.append(r)
    (out_dir / f"context_sweep_{model}.json").write_text(json.dumps(rows, indent=2))
    return rows
