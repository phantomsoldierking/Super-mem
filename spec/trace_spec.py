#!/usr/bin/env python3
"""Collect SimpleO3 memory traces from instrumented SPEC CPU 2026 binaries.

Requires spec/build_spec.sh to have built the benchmarks and set up their
run directories. For each benchmark, the first workload of the chosen input
size is executed directly in its run directory with the memtrace runtime
enabled: fast-forward MT_SKIP instructions, warm private caches for MT_WARM,
then record MT_INSTS instructions of L2 misses.

    python3 spec/trace_spec.py --skip 30000000000 --insts 200000000
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command_for(run_dir: Path, index: int):
    lines = [l for l in (run_dir / "speccmds.cmd").read_text().splitlines()
             if l.startswith("-o ")]
    words = shlex.split(lines[index])
    # "-o out -e err EXE args... > out 2>> err"
    words = words[4:]
    if ">" in words:
        words = words[:words.index(">")]
    exe = (run_dir / words[0]).resolve()
    return [str(exe), *words[1:]]


def trace_one(bench, args):
    run_dir = Path(args.spec_dir) / "benchspec/CPU" / bench / "run" / f"run_base_{args.size}_memtrace.0000"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / f"{bench}.trace"
    cmd = command_for(run_dir, args.workload)
    env = dict(os.environ, MT_OUT=str(trace), MT_SKIP=str(args.skip), MT_WARM=str(args.warm),
               MT_INSTS=str(args.insts))
    stats_path = Path(str(trace) + ".json")
    stats_path.unlink(missing_ok=True)  # never pick up the stats of an earlier run
    t0 = time.time()
    proc = subprocess.run(cmd, check=False, cwd=run_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          timeout=args.timeout)
    elapsed = time.time() - t0
    if not stats_path.exists():
        return bench, {"error": f"no stats (exit {proc.returncode}): {proc.stderr.decode()[-400:]}"}
    stats = json.loads(stats_path.read_text())
    stats.update(benchmark=bench, command=" ".join(cmd[1:]), input_size=args.size, workload_index=args.workload,
                 wall_seconds=round(elapsed, 1), exit_code=proc.returncode)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    if stats.get("end_reason") != "trace_window_complete":
        stats["error"] = f"trace window incomplete: {stats.get('end_reason')} (exit {proc.returncode})"
    return bench, stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmarks", nargs="*")
    p.add_argument("--spec-dir", default=os.environ.get("SPEC_DIR", str(Path.home() / "spec/cpu2026")))
    p.add_argument("--size", default="refrate")
    p.add_argument("--workload", type=int, default=0, help="index of the workload in speccmds.cmd")
    p.add_argument("--skip", type=int, default=30_000_000_000)
    p.add_argument("--warm", type=int, default=100_000_000)
    p.add_argument("--insts", type=int, default=200_000_000)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--output", default=str(ROOT / "traces/spec"))
    args = p.parse_args()
    benches = args.benchmarks or [l.strip() for l in (ROOT / "spec/benchmarks.txt").read_text().splitlines()
                                  if l.strip() and not l.startswith("#")]
    with ThreadPoolExecutor(args.jobs) as pool:
        for bench, stats in pool.map(lambda b: trace_one(b, args), benches):
            if "error" in stats:
                print(f"{bench}: ERROR {stats['error']}", file=sys.stderr, flush=True)
            else:
                mpki = 1000 * stats["l2_misses"] / max(1, stats["trace_insts"])
                print(f"{bench}: {stats['end_reason']} traced={stats['trace_insts']:,} insts "
                      f"L2 MPKI={mpki:.2f} lines={stats['trace_lines']:,} "
                      f"RSS={stats['peak_rss_bytes'] / 2**20:.0f} MiB in {stats['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
