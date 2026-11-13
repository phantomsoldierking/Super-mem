The archtypes simulated in ramulator_2.1 are [here](https://github.com/phantomsoldierking/Super-mem/#1-purpose) .
To generate the memory usage trace(L2-cache miss) in [/tracer](https://github.com/phantomsoldierking/Super-mem/tree/main/tracer) we have a **compiler plugin** that adds hooks into the program, and a **runtime library** that those hooks call while the program runs. Together they turn a SPEC benchmark into a memory trace that Ramulator can replay.

pass plugin loaded with `clang -fpass-plugin=libMemTracePass.so`. It registers at the **OptimizerLastEP** point, so it runs after `-O2` has finished. The code is instrumented as optimized, which means loads that the optimizer removed are not traced.

**What it does to every function:**

| It finds                        | It inserts before it                                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Start of each basic block       | `__mt_bb(n)`, where n is the number of IR instructions in the block (PHIs and debug info are skipped)                                      |
| `load`                          | `__mt_ld(ptr, bytes)`                                                                                                                      |
| `store`                         | `__mt_st(ptr, bytes)`                                                                                                                      |
| atomic RMW / cmpxchg            | `__mt_st(ptr, bytes)`, counted as a write                                                                                                  |
| `memcpy` / `memmove` / `memset` | `__mt_range(ptr, len, read/write)`: memcpy and memmove log a read of the source and a write of the destination; memset logs only the write |
- **Access size** comes from the DataLayout, e.g. 8 bytes for an `i64`.
- **Instruction counting:** `__mt_bb` gives the instruction count. This is why IPC in the report counts **LLVM IR instructions, not x86 instructions**.
### the runtime (tracer/memtrace_rt.c)
This is linked into every benchmark as `memtrace_rt.o`. It does the real work.

**1. Phases, controlled by environment variables.** It counts instructions through `__mt_bb` and switches modes:

| Phase | Env var (study value) | Behaviour                                                                                 |
| ----- | --------------------- | ----------------------------------------------------------------------------------------- |
| SKIP  | `MT_SKIP` = 30B       | Fast-forward past program startup and initialization; loads and stores return immediately |
| WARM  | `MT_WARM` = 100M      | Fill the simulated caches; write no output                                                |
| TRACE | `MT_INSTS` = 200M     | Record misses, then write stats and `_exit(0)`                                            |
**2. Private-cache filter.** It models the core’s own caches in software:
- L1D: 48 KiB, 12-way.
- L2: 2 MiB, 16-way.
- 64 B lines, LRU, write-back with write-allocate, non-inclusive.
Only **L2 misses** leave the core, so only they reach the shared LLC/DRAM that differs between architectures. Filtering here keeps traces small. The private levels are identical for every architecture, so filtering them out up front doesn’t bias the comparison.

**3. Address translation.** Virtual pages map to “physical” frames in first-touch order. Addresses stay compact, and the frame count gives the touched footprint.

**4. Trace format.** Each L2 miss becomes one line in the SimpleO3 format:

```cpp
<bubble> <addr> [<dirty-victim addr>]
```

- `bubble` is the number of non-missing instructions since the last miss. It is how the replayed core knows how much “other work” happens between memory requests.
- A dirty eviction from L2 adds a write-back address. Write-backs caused by L1 evictions wait in a small buffer and ride on the next miss line that has room.

**5. Stats.** `<trace>.json` records:
- Loads and stores.
- L1 and L2 misses.
- Write-backs.
- Instructions traced.
- Pages touched.
- Peak RSS.
- Why the run ended (`trace_window_complete` or `program_exit`).

**Known limits:**
- IR instructions are used not x86 instructions.
- There is one 200M-instruction window per benchmark.
- Only C/C++ benchmarks are covered, since the pass runs on clang’s IR and Fortran isn’t built(the iso contained fortran as well which were not used).
- Code in uninstrumented libraries (like libc internals) isn’t traced, apart from the memcpy/memset calls the pass catches.


###### The [/memarch](https://github.com/phantomsoldierking/Super-mem/tree/main/memarch) has the ramulator simulation for the all the archtypes.
---
#### AI inference analytics
- **What:** 12 scenarios: Llama-3.1 8B/70B decode and prefill, batch and context-length sweeps, Mixtral-8x7B (MoE), ResNet-50 and DLRM.
- **How:**
    - Each model becomes a list of operators, each with its FLOPs and the bytes it moves per traffic class.
    - Each byte is placed in SRAM or DRAM according to the chip’s SRAM role.
    - Time per operator = `max(compute time, memory time) + all-reduce + overhead`. Memory time uses the **calibrated** bandwidth from layer 1, not the datasheet number.
    - The chip count is the minimum that fits the model. For SRAM-only chips this means many chips, which adds inter-chip communication.
- **Outputs:**
    - Latency per token.
    - Chips needed.
    - Throughput per chip.
    - Energy per token: memory pJ/bit, plus chips × TDP × time.
