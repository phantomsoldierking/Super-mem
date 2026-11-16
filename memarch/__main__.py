"""Command line: python -m memarch <command>

  calibrate          Ramulator access-pattern calibration of every DRAM config
  ai                 AI inference study (scenarios x architectures) + sweeps
  spec               SPEC CPU 2026 trace replay (needs traces/spec/*.trace)
  patterns           Access-pattern characterization of the SPEC traces (locality, inter-request PSD)
  figures            Regenerate figures from results/
  all                ai + spec + figures
"""
import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="memarch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["calibrate", "ai", "spec", "patterns", "figures", "all"])
    p.add_argument("--archs", nargs="+")
    p.add_argument("--benchmarks", nargs="+")
    p.add_argument("--cores", nargs="+", type=int, default=[1, 4])
    p.add_argument("--insts", type=int, help="instructions per measured pass (all core counts)")
    p.add_argument("--jobs", type=int)
    p.add_argument("--fresh", action="store_true", help="calibrate: ignore and overwrite the calibration cache")
    p.add_argument("--merge", action="store_true", help="spec/patterns: merge into existing results (re-run a subset)")
    p.add_argument("--out", help="spec/patterns: results directory (default results/spec; e.g. /dev/shm/... for a smoke test)")
    a = p.parse_args(argv)
    if a.command == "calibrate":
        from .ai_study import calibrations
        from .architectures import load_all
        if a.fresh:
            from .ramulator_backend import calibrate
            for arch in load_all(a.archs):
                if arch.has_dram:
                    calibrate(arch.dram, refresh=True)
        for name, c in calibrations(load_all(a.archs)).items():
            if c:
                print(name, {k: f"{v['achieved_GBps']:.0f} GB/s ({v['efficiency']:.0%})" for k, v in c["patterns"].items()})
    if a.command in ("ai", "all"):
        from . import ai_study
        rows = ai_study.run(a.archs)
        ai_study.sweep_batch()
        ai_study.sweep_context()
        print(f"AI study: {len(rows)} results -> results/ai/")
    if a.command in ("spec", "all"):
        from . import cpu_study
        insts = {c: a.insts for c in a.cores} if a.insts else None
        rows = cpu_study.run(a.benchmarks, a.archs, tuple(a.cores), insts=insts, jobs=a.jobs, merge=a.merge,
                             out_dir=a.out)
        print(f"SPEC study: {len(rows)} results -> {a.out or 'results/spec'}/")
    if a.command in ("patterns", "all"):
        from . import access_patterns
        kw = {"insts": a.insts} if a.insts else {}
        out = access_patterns.run(a.benchmarks, a.archs, jobs=a.jobs, out_dir=a.out, merge=a.merge, **kw)
        print(f"Access patterns: {len(out['traces'])} traces, {len(out['mc'])} replays -> "
              f"{a.out or 'results/spec'}/access_patterns.json")
    if a.command in ("figures", "all"):
        from . import figures
        figures.make_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
