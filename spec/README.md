# SPEC CPU 2026 tracing

* `memtrace-llvm.cfg` — runcpu config: clang `-O2 -march=x86-64-v3` with the
  MemTracePass plugin; links `tracer/build/memtrace_rt.o`.
* `build_spec.sh [size] [benchmarks]` — builds and sets up run directories
  (`REBUILD=1` forces a rebuild). Adds a `libcrypt.so.1` shim for the vendor tools.
* `trace_spec.py` — runs the first workload of each benchmark in its run
  directory with the tracer enabled; traces go to `traces/spec/`.
* `benchmarks.txt` — the 8 C/C++ rate benchmarks used.

SPEC CPU 2026 is licensed software; the kit, binaries and traces are not part of
this repository.
