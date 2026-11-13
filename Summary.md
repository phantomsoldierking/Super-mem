The seven memory-system archetypes are defined [here](https://github.com/phantomsoldierking/Super-mem/#1-purpose); their DRAM is simulated cycle-accurately in Ramulator 2.1.

To generate the memory trace (L2-cache misses), [/tracer](https://github.com/phantomsoldierking/Super-mem/tree/main/tracer) has a **compiler plugin** that adds hooks into the program, and a **runtime library** that those hooks call while the program runs. Together they turn a SPEC benchmark into a memory trace that Ramulator can replay.

### The LLVM pass (tracer/MemTracePass.cpp)

The LLVM pass is a pass plugin loaded with `clang -fpass-plugin=libMemTracePass.so`. It registers at the **OptimizerLastEP** point, so it runs after `-O2` has finished. The code is instrumented as optimized, which means loads that the optimizer removed are not traced.

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

### The runtime (tracer/memtrace_rt.c)

This is linked into every benchmark as `memtrace_rt.o`. It does the real work.

**1. Phases, controlled by environment variables.** It counts instructions through `__mt_bb` and switches modes:

| Phase | Env var (study value) | Behaviour                                                                                 |
| ----- | --------------------- | ----------------------------------------------------------------------------------------- |
| SKIP  | `MT_SKIP` = 30B       | Fast-forward past program startup and initialization; loads and stores return immediately |
| WARM  | `MT_WARM` = 100M      | Fill the simulated caches; write no output                                                |
| TRACE | `MT_INSTS` = 200M     | Record misses, then write stats and `_exit(0)`                                            |

**2. Private-cache filter.** It models the core's own caches in software:

- L1D: 48 KiB, 12-way.
- L2: 2 MiB, 16-way.
- 64 B lines, LRU, write-back with write-allocate, non-inclusive.

Only **L2 misses** leave the core, so only they reach the shared LLC/DRAM that differs between architectures. Filtering here keeps traces small. The private levels are identical for every architecture, so filtering them out up front doesn't bias the comparison.

**3. Address translation.** Virtual pages map to "physical" frames in first-touch order. Addresses stay compact, and the frame count gives the touched footprint.

**4. Trace format.** Each L2 miss becomes one line in the SimpleO3 format:

```text
<bubble> <addr> [<dirty-victim addr>]
```

- `bubble` is the number of non-missing instructions since the last miss. It is how the replayed core knows how much "other work" happens between memory requests.
- A dirty eviction from L2 adds a write-back address. When an L1 write-back is inserted into L2 and displaces a dirty L2 line, that line has no miss of its own; it waits in a small buffer and rides on the next miss line that has room.

**5. Stats.** `<trace>.json` records:

- Loads and stores.
- L1 and L2 misses.
- Write-backs.
- Instructions traced.
- Pages touched.
- Peak RSS.
- Why the run ended (`trace_window_complete` or `program_exit`).

**Known limits:**

- Instructions are counted as LLVM IR instructions, not x86 instructions.
- There is one 200M-instruction window per benchmark.
- Only C/C++ benchmarks are covered, since the pass runs on clang's IR and no Fortran compiler was used (the ISO also contains Fortran benchmarks, which were not built).
- Code in uninstrumented libraries (like libc internals) isn't traced, apart from the memcpy/memset calls the pass catches.

---

### memarch

[/memarch](https://github.com/phantomsoldierking/Super-mem/tree/main/memarch) drives the Ramulator simulations (DRAM calibration and SPEC trace replay) and holds the analytical AI inference model for all the archetypes.

#### AI inference analytics

- **What:** 12 scenarios (Llama-3.1 8B/70B decode and prefill, Mixtral-8x7B (MoE), ResNet-50 and DLRM), plus batch-size and context-length sweeps on Llama-3.1 8B.
- **How:**
    - Each model becomes a list of operators, each with its FLOPs and the bytes it moves per traffic class.
    - Each byte is placed in SRAM or DRAM according to the chip's SRAM role.
    - Time per operator = `max(compute time, memory time) + all-reduce + overhead`. Memory time uses the **calibrated** bandwidth from the Ramulator DRAM calibration, not the datasheet number.
    - The chip count is the minimum that fits the model. For SRAM-only chips this means many chips, which adds inter-chip communication.
- **Outputs:**
    - Latency per token.
    - Chips needed.
    - Throughput per chip.
    - Energy per token: dynamic memory and inter-chip link energy (pJ/bit), plus chips × TDP × time.