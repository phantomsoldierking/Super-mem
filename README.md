# Memory architecture of AI inference chips

How do the memory systems of AI inference chips — HBM GPUs, SRAM+GDDR6 meshes,
SRAM+LPDDR5 NPUs, SRAM-only LPUs and wafer-scale SRAM — behave on the AI
workloads they were built for, and on general-purpose code (SPEC CPU 2026)?

**Reports:**
[detailed report](output/memory_architecture_study_detailed.pdf) (method, tools, all results, validation) and
[summary](output/memory_architecture_study_summary.pdf) (concise version for submission).
Both are generated from `results/` by `make report` from the templates in [report/](report).

## What is in here

| Path | Contents |
|---|---|
| [configs/architectures/](configs/architectures) | 7 memory-system archetypes (SRAM size/role/latency/BW, DRAM standard and channels, compute, links, TDP) |
| [configs/models/](configs/models), [configs/scenarios.json](configs/scenarios.json) | Llama-3.1 8B/70B, Mixtral-8x7B, ResNet-50, DLRM-DCNv2; 12 inference scenarios |
| [memarch/ramulator_backend.py](memarch/ramulator_backend.py) | Builds Ramulator 2.1 DRAM systems; access-pattern calibration; SimpleO3 CPU-trace replay |
| [memarch/workloads.py](memarch/workloads.py) | Operator graphs with FLOPs and bytes per traffic class |
| [memarch/perf.py](memarch/perf.py) | Placement (cache / scratchpad / resident SRAM), roofline per op, collectives, energy |
| [memarch/ai_study.py](memarch/ai_study.py), [memarch/cpu_study.py](memarch/cpu_study.py) | The two studies |
| [tracer/](tracer) | LLVM 23 pass plugin + runtime that turns any C/C++ program into a Ramulator trace |
| [spec/](spec) | SPEC CPU 2026 config, build script and trace driver |
| [results/](results) | CSV/JSON results and figures |
| [report/](report) | Report templates (`detailed.md`, `summary.md`) and the PDF builder |
| [tests/](tests) | `make test` |
| [papers/](papers) | Ramulator and NVMExplorer papers |

## Quick start

```bash
make venv          # numpy, matplotlib, pytest, markdown, ruff
make ai figures    # Ramulator calibration + AI study (~5 min first time, cached after)
make report        # PDF via headless Chromium
make test
```

SPEC CPU 2026 (from `cpu2026-1.0.1.iso`, licensed separately and not redistributed):

```bash
make spec-install  # extract ISO to ~/spec/iso, install to ~/spec/cpu2026
make spec-build    # clang + MemTracePass build of the benchmarks in spec/benchmarks.txt
make spec-trace    # 200M-instruction L2-miss traces into traces/spec/
make spec          # replay on all architectures, 1 and 4 cores (about 3 hours on 12 threads)
```

The vendor toolset links `libcrypt.so.1`; on distributions that ship only
`libcrypt.so.2` the Makefile/`spec/build_spec.sh` add a symlink shim in
`spec/.compat-lib` (same `XCRYPT_2.0` symbols). Fortran benchmarks are skipped
(no Fortran compiler).

Requirements: Python 3.14 (the bundled Ramulator extension is built for it; rebuild
Ramulator per `ramulator2/README.md` for other versions), clang/LLVM 23, Chromium for
the PDF.

## Scope

Architecture parameters are public-spec class values, not vendor-calibrated
models. DRAM behaviour is cycle-accurate (Ramulator 2.1); the AI model is an
operator-level roofline on top of it; SPEC is replayed on a common
out-of-order core so only the memory system differs. See the Limitations
section of the report.
