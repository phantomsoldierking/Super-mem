# Memory Architectures of AI Inference Chips: A Workload-Driven Comparison

<p class="subtitle">Project summary · {{date}}</p>

## Abstract

AI inference chips differ most in their memory systems: HBM behind a small cache (GPUs), compiler-managed
SRAM in front of GDDR6 or LPDDR5 (dataflow meshes and NPUs), or SRAM alone with no DRAM (LPU-class and
wafer-scale designs). We model seven such memory systems and evaluate them on twelve AI inference scenarios
(Llama-3.1 8B/70B decode and prefill, Mixtral-8x7B, ResNet-50, DLRM) and on memory traces of eight
SPEC CPU 2026 benchmarks. DRAM is simulated cycle-accurately with Ramulator 2.1; SPEC traces are captured
with a custom LLVM pass from benchmarks built from the official SPEC CPU 2026 kit. SRAM-resident designs cut
single-stream decode latency by 8–12x versus an HBM3 GPU, but need 76–654 chips for 8B–70B models, deliver
far less throughput per chip and per watt, and are bound by inter-chip communication. HBM remains the most
balanced design. Fast DRAM interfaces lose 55–80% of their bandwidth on random access, which matters for
KV-cache and embedding traffic. On SPEC CPU 2026 the ranking reverses: the
working sets fit in every last-level cache, IPC follows SRAM hit latency, and the AI memory systems reach
only 0.32–0.83x of a CPU + DDR5 hierarchy (geomean), with HBM4 no faster than HBM3.

## 1. Architectures

{{arch_table}}

SRAM role: **cache** = hardware LRU; **scratchpad** = compiler-pinned with DRAM overflow; **resident** = all
state in SRAM, chip count set by capacity.

## 2. Method

* **DRAM (Ramulator 2.1).** One controller per channel (FR-FCFS, open row, MOP4CLXOR mapping, refresh).
  Five access patterns matching inference traffic classes (weight streams, activation read/write streams,
  paged-KV gathers, embedding-row gathers, random lines) give the *achieved* bandwidth of each DRAM system.
* **AI inference model.** Operator graphs with FLOPs and bytes per traffic class; placement by SRAM role;
  per-op roofline `max(compute, memory) + all-reduce + overhead` on the minimum number of chips that hold the
  model state; energy from pJ/bit per memory technology and from TDP x time.
* **SPEC CPU 2026.** The supplied ISO was installed (with a `libcrypt.so.1` compatibility shim) and eight
  C/C++ rate benchmarks were built through `runcpu` with clang and an LLVM pass plugin that instruments every
  load, store and basic block. The linked runtime skips 30B instructions, warms private L1/L2 caches, and
  records the L2-miss stream of 200M instructions. Ramulator's SimpleO3 core (4-wide, 4 GHz) replays the
  traces with each architecture's SRAM as LLC and its DRAM behind it, warm-cache, on 1 and 4 cores.

## 3. Results

{{calib_table}}

{{fig:03_ai_latency_speedup|Figure 1. Step-latency speedup over GPU + HBM3 (minimum chips that hold the model).}}

{{fig:05_ai_chips|Figure 2. Chips required by capacity.}}

{{fig:07_energy|Figure 3. Energy per token: dynamic memory + link energy (top), chips x TDP x time (bottom).}}

{{fig:10_spec_ipc|Figure 4. SPEC CPU 2026 IPC relative to CPU + DDR5 when only the memory system changes.}}

## 4. Findings

1. **Decode is bandwidth-bound; prefill is compute-bound.** On DRAM systems 87–100% of decode time waits on
   memory, and speed tracks *achieved* bandwidth (HBM4 2.2x, DDR5 0.12x of HBM3). Prefill is compute-bound
   on every system that needs at most five chips; only the large SRAM-only deployments are dominated by
   all-reduces.
2. **SRAM-only designs trade chips for latency.** 8–12x lower batch-1 Llama decode latency, but 76 (8B) to 654 (70B)
   LPU-class chips or 1–4 wafers; the bottleneck becomes all-reduce latency and per-op synchronization, and
   throughput per chip falls to 1/10–1/30 of an HBM3 GPU. Dynamic memory energy is 10–50x lower, yet with
   static power counted the LPU-class system uses ~4x the energy per token of the GPU at batch 1.
3. **Capacity cliffs.** Growing KV caches (long context, large batch) and idle MoE experts force SRAM designs
   to add chips abruptly (2.3x per-chip throughput loss when a wafer system doubles); HBM degrades smoothly.
4. **Access pattern matters as much as peak bandwidth.** HBM3/HBM4 keep ~45%/35% of their bandwidth on
   random 64 B accesses and GDDR6 21%, while DDR5 keeps 86%. Paged KV caches and embedding tables therefore
   see a fraction of datasheet bandwidth on the fastest interfaces.
5. **General-purpose code wants the opposite.** On the SPEC traces, DRAM traffic is the same on every system
   for 7 of 8 benchmarks; IPC tracks LLC hit latency (geomean vs CPU + DDR5, 1 core: LPU 0.83, NPU 0.73,
   mesh 0.63, GPU 0.41, wafer 0.32), down to 0.15–0.27x on pointer-chasing omnetpp and nest. Only
   bandwidth-heavy lbm benefits from 200+ MiB of SRAM (4 cores: 0.81–0.87x, vs 0.45x for the NPU,
   whose LPDDR5 reaches 126 ns loaded latency). Ramulator's LLC miss counters include MSHR retries, so DRAM accesses are used instead.

**Limitations.** Public-spec class parameters, not vendor models; analytical on-chip model on top of
cycle-accurate DRAM; 16-bit weights; one 200M-instruction window per SPEC benchmark, C/C++ only; AI-chip SRAM latencies seen from a CPU core are assumed. Code,
configurations and the full report are in the project repository (`make all` reproduces every number).
