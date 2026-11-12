# memtrace: LLVM pass + runtime for Ramulator SimpleO3 traces

```bash
make                     # build/libMemTracePass.so and build/memtrace_rt.o (clang/LLVM 23)
clang -O2 -fpass-plugin=build/libMemTracePass.so prog.c build/memtrace_rt.o -o prog
MT_OUT=prog.trace MT_SKIP=1000000000 MT_WARM=100000000 MT_INSTS=200000000 ./prog
make test                # STREAM-triad sanity kernel
```

* `MemTracePass.cpp` runs at the end of the optimisation pipeline and inserts
  `__mt_bb(n)` at each basic block and `__mt_ld/__mt_st/__mt_range` before every
  load, store, atomic and memory intrinsic.
* `memtrace_rt.c` skips `MT_SKIP` instructions, warms a 48 KiB L1D and 2 MiB L2
  for `MT_WARM`, then writes each L2 miss of the next `MT_INSTS` instructions as
  `<instructions since previous line> <address> [<dirty victim>]` and exits.
  Statistics go to `$MT_OUT.json`. Instruction counts are LLVM IR instructions.
