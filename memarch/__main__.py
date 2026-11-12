"""Command line: python -m memarch <command>

  calibrate          Ramulator access-pattern calibration of every DRAM config
  ai                 AI inference study (scenarios x architectures) + sweeps
  spec               SPEC CPU 2026 trace replay (needs traces/spec/*.trace)
  figures            Regenerate figures from results/
  all                ai + spec + figures
"""
import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="memarch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["calibrate", "ai", "spec", "figures", "all"])
    p.add_argument("--archs", nargs="+")
    p.add_argument("--benchmarks", nargs="+")
    p.add_argument("--cores", nargs="+", type=int, default=[1, 4])
    p.add_argument("--insts", type=int, help="instructions per measured pass (all core counts)")
    p.add_argument("--jobs", type=int)
    a = p.parse_args(argv)
    if a.command == "calibrate":
        from .ai_study import calibrations
        from .architectures import load_all
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
        rows = cpu_study.run(a.benchmarks, a.archs, tuple(a.cores), insts=insts, jobs=a.jobs)
        print(f"SPEC study: {len(rows)} results -> results/spec/")
    if a.command in ("figures", "all"):
        from . import figures
        figures.make_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
