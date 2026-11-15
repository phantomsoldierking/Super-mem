# Memory Architectures of AI Inference Chips — Detailed Project Report

<p class="subtitle">How the memory systems of seven AI/CPU chip archetypes behave on AI inference and on SPEC CPU 2026 general-purpose code. Everything in this document is regenerated from <code>results/</code> by <code>make report</code> ({{date}}).</p>

## 0. Reading guide

This is the long-form companion to the short submission document. It explains **what was built, why, how
each tool was used (Ramulator 2.1, an LLVM pass, the SPEC CPU 2026 ISO), what the results are, how they
were checked, and where they stop being trustworthy.** Section 9 is a file-by-file map of the repository;
Section 10 lists every command needed to reproduce the results.

Summary of outcomes:

* A reproducible framework (`memarch/`) that describes a chip's memory system in one JSON file and evaluates
  it on (a) 12 AI inference scenarios and (b) memory traces of 8 SPEC CPU 2026 benchmarks.
* Cycle-accurate DRAM behaviour from Ramulator 2.1 for DDR5, LPDDR5, GDDR6, HBM3 and HBM4, measured for five
  access patterns that correspond to inference traffic classes.
* An LLVM 23 pass plugin + runtime (`tracer/`) that turns any C/C++ program into a Ramulator trace; it was
  used to build and trace SPEC CPU 2026 from the supplied ISO.
* Findings (Sections 5–7): SRAM-resident designs win batch-1 decode latency by 6–9x but need tens to hundreds
  of chips and lose on throughput per chip and per watt; HBM is the most balanced; DRAM efficiency on random
  access differs by up to 3.6x between technologies; on general-purpose code the large, slow on-chip SRAMs of AI chips
  help only bandwidth-bound programs and hurt latency-bound ones.

## 1. Goal and what existed before

**Goal.** Explore the memory architecture of AI-inference-optimized chips and compare them on different
workloads.

**Starting point.** The repository contained an ideation prototype (`simulations/`): a two-layer toy decoder
replayed on (1) 80 HBM3 channels driven tick-by-tick from Python through Ramulator's test harness and (2) a
hand-written banked-SRAM model; a Cairo-drawn PDF; and the SPEC CPU 2026 ISO, unused. Review found:

* Only two architectures and one workload family; the "Llama 8B" results replayed 1/4096 of the traffic.
* Ticking Ramulator from Python made realistic sizes infeasible (minutes per microsecond of simulated time).
* HBM3 was driven with 32 B requests and the default row-bank-column mapper, which (as measured here) reaches
  only about half of pin bandwidth on streams; the 4-node SRAM reference had a hypothetical 64 GiB of SRAM.
* No general-purpose workloads, so the "different workloads" part of the goal was not addressed.

The prototype was therefore replaced rather than extended. It is not shipped (it is superseded and its
numbers are not comparable to the new ones).

## 2. Architectures studied

Each archetype is one JSON file in `configs/architectures/`. Values are public-specification *class*
values (order of magnitude of shipping parts), not calibrated vendor models.

{{arch_table}}

The **role** of the on-chip SRAM is the key architectural distinction:

* **cache** (CPU, GPU): hardware-managed LRU. It helps data that is reused within a step (activations) but
  not a weight/KV stream that is larger than the cache — an LRU cache misses on every access of a cyclic
  stream bigger than itself.
* **scratchpad** (Tenstorrent-class mesh, Qualcomm/d-Matrix-class NPU): the compiler pins data. Activations
  are pinned first, then as much of the weights/KV cache as fits; the rest streams from DRAM. Degrades
  gracefully as the model grows.
* **resident** (Groq-class LPU, Cerebras-class wafer): there is no DRAM. All state lives in SRAM, so the
  number of chips is set by capacity.

{{fig:01_landscape|Figure 1. Capacity and bandwidth per chip of each memory level. SRAM (circles) is 10-100x faster than the DRAM behind it (squares) and 100-1000x smaller.}}

## 3. How the tools were used

### 3.1 Ramulator 2.1 (cycle-accurate DRAM)

Ramulator 2.1 is the checked-out simulator in `ramulator2/` (prebuilt `libramulator.so` and Python 3.14
bindings). The new backend (`memarch/ramulator_backend.py`) uses only Ramulator's C++ simulation loop
(`ramulator.Simulation`), never Python-level ticking, which makes it ~1000x faster than the prototype.

For each architecture it builds one controller per channel: FR-FCFS scheduler, open-row policy, refresh on,
64-entry read/write queues, cache-line channel interleaving. Problems found and fixed while integrating:

| Issue | Evidence | Fix |
|---|---|---|
| Default `RoBaRaCoCh` mapper put consecutive lines in one bank group | HBM3 stream: 51% of pin BW | `MOP4CLXOR` mapper: HBM3 and HBM4 99% |
| LPDDR5/GDDR6 channels move 32 B; 64 B lines skipped every other channel | channel 1 received 0 requests | gang 16-bit channels into 32-bit pairs (64 B per access, same total BW) |
| HBM pseudo-channels carry 32 B; SimpleO3 sends 64 B and Ramulator rejects it | `size_bytes 64 > tx_bytes 32` | 64 B payload with doubled burst length (two back-to-back BL8 bursts to the open row) |
| GDDR6 preset lacks refresh timings | `unresolved timings nRFCpb, nRFCab` | nRFCab = 210 ns, nRFCpb = 60 ns |
| SimpleO3 LLC uses `sets-1` as a bit mask; 96/50/144/210/230 MiB caches have non-power-of-two set counts | code inspection of `llc.cpp` | choose power-of-two sets and round associativity (e.g. 96 MiB = 65,536 sets x 24 ways) |
| HBM3/HBM4 controllers tick at half-CK (`tick_multiplier = 2`); the preset CK was used for the SimpleO3 clock ratio and for converting cycles to ns | HBM ran at half speed in SPEC replay; HBM latencies reported 2x | use the tick period (312/250 ps); HBM SPEC replays re-run |

**Access-pattern calibration.** Five synthetic traces, each ≥120k requests (2,000 per channel for wide
systems), are replayed with Ramulator's `LoadStoreTrace` frontend: sequential reads (weight streaming),
2:1 read/write streams (activations, KV appends), random 4 KiB pages read sequentially (paged KV cache),
random 256 B rows (embedding-bag lookups) and uniform random 64 B lines. Achieved bandwidth is the
controller-reported throughput summed over channels; it is cached in `results/cache/`.

{{calib_table}}

{{fig:02_dram_calibration|Figure 2. Achieved DRAM bandwidth as a fraction of pin bandwidth, by access pattern (Ramulator 2.1).}}

**What it shows.** DDR5 keeps 86–93% of its bandwidth for every pattern: it has many banks (32 per
sub-channel) relative to its modest data rate, so bank-level parallelism hides row misses. The fast
interfaces do not: HBM3/HBM4 stream at 99% but keep only 58%/46% of their bandwidth on random 64 B accesses,
LPDDR5 45%, and GDDR6 only 24–25% on random lines and rows because 16 banks per channel cannot cover the
row-cycle time at 14 Gb/s. For inference this means weight streaming gets 83–99% of the datasheet number,
paged-KV gathers 48–73% (DDR5 93%), and embedding lookups 25–89%.

### 3.2 The AI operator model

`memarch/workloads.py` converts each scenario of `configs/scenarios.json` into a list of operators. Each
operator records FLOPs and the bytes it moves per *traffic class*; the class selects the calibrated DRAM
bandwidth:

| Traffic class | Example | Calibrated pattern |
|---|---|---|
| weight | QKV/O/FFN/LM-head parameters, once per step | stream read |
| kv_read | full KV cache read by decode attention | 4 KiB page gather |
| kv_write / act | KV appends, activations | 2:1 read/write stream |
| gather | embedding rows (DLRM, token embedding) | 256 B row gather |

Models: Llama-3.1-8B and 70B (GQA, SwiGLU), Mixtral-8x7B (top-2 of 8 experts; the expected number of
distinct experts for *n* tokens is `E(1-(1-k/E)^n)`), ResNet-50 enumerated layer by layer, and a
DLRM-DCNv2-shaped recommender (26 tables, 250M rows of 128 fp16 values = 64 GB, 10 lookups per table).
Unit tests check parameter counts against the published models (8.03B, 70.6B, 46.7B, 25.5M; ResNet-50
8.2 GFLOP/image).

`memarch/perf.py` evaluates one step per chip:

```
chips      = min n such that (persistent state + peak activations) / n <= 0.9 x capacity per chip
t_op       = max(t_compute, t_memory) + t_collective + overhead_per_op
t_compute  = FLOPs / chips / (peak x efficiency)
t_memory   = max( SRAM bytes / (SRAM BW x eff),  sum_class DRAM bytes[class] / calibrated BW[class] )
allreduce  = 2*ceil(log2 n)*hop latency + 2(n-1)/n * payload / link BW      (all-to-all for embeddings)
energy     = bytes x pJ/bit (SRAM, DRAM, link) + 0.4 pJ/FLOP;  TDP energy = chips x TDP x step time
```

Capacity determines the chip count; tensor parallelism splits every operator over those chips and adds two
all-reduces per transformer layer. Per-operator overhead models kernel launch / synchronization (3 µs GPU,
2 µs CPU/mesh/NPU). For the two SRAM-only systems, per-op overhead and hop latency are the only knobs that
were tuned, so that Llama batch-1 decode lands near public token rates:

{{public_check}}

The model is within about 1.6x of these public figures (which themselves vary across software releases and
quantization choices). The HBM GPU reaches {{gpu_bound_pct}} of its pin-bandwidth bound (all DRAM bytes per token
over pin bandwidth); the rest is per-operator launch overhead and the KV-cache gathers, which run below
stream efficiency.

### 3.3 The SPEC CPU 2026 ISO

The supplied `cpu2026-1.0.1.iso` is SPEC CPU 2026 v1.0.1. Steps (all scripted in the `Makefile`):

1. **Extract** with `bsdtar` (no root/mount needed) into `~/spec/iso`, skipping docs, PTDaemon and
   redistributable sources.
2. **Install** with `./install.sh -d ~/spec/cpu2026 -u linux-x86_64`. The vendor toolset's `specperl`
   is linked against `libcrypt.so.1`, which this Arch-based system no longer ships; the toolset self-test
   failed. `objdump` showed it only needs `crypt@XCRYPT_2.0`, which `libcrypt.so.2` exports, so a symlink in
   `spec/.compat-lib/` on `LD_LIBRARY_PATH` fixes it without root. The installation then passed all tests.
3. **Choose benchmarks.** The host has clang/LLVM 23 but no Fortran compiler, so the 8 C/C++ rate
   benchmarks with the most varied memory behaviour were selected (`spec/benchmarks.txt`): 772.marian_r
   (neural machine translation — AI inference on a CPU), 782.lbm_r (fluid-dynamics streaming), 710.omnetpp_r
   (discrete-event simulation, pointer chasing), 777.zstd_r (compression), 708.sqlite_r (database),
   750.sealcrypto_r (homomorphic encryption), 706.stockfish_r (chess search, hash tables) and 767.nest_r
   (spiking neural-network simulation).
4. **Build** through SPEC's own harness: `spec/memtrace-llvm.cfg` sets `clang -O2 -march=x86-64-v3
   -fpass-plugin=libMemTracePass.so` and links the tracer runtime via `EXTRA_LIBS`;
   `runcpu --action=build` then `--action=setup --size=refrate` creates run directories with the reference
   inputs. Builds are unmodified SPEC sources (`spec/build_spec.sh`).
5. **Run and trace.** `spec/trace_spec.py` reads each run directory's `speccmds.cmd`, takes the first
   `refrate` workload and runs the binary in place with the tracer enabled through environment variables.
   Output correctness is not checked by `runcpu` because the process exits after the trace window; these are
   measurement runs, not reportable SPEC results.

### 3.4 The LLVM pass and tracer runtime

`tracer/MemTracePass.cpp` is a new-pass-manager plugin registered at `OptimizerLastEP`, so it instruments
the *optimized* code (register-allocated values do not appear as memory accesses). For every basic block it
inserts `__mt_bb(n)` with the block's instruction count, and before every memory operation a call with the
pointer and access size:

```cpp
if (auto *L = dyn_cast<LoadInst>(I))       IRB.CreateCall(LD, {L->getPointerOperand(), Size(L->getType())});
else if (auto *S = dyn_cast<StoreInst>(I)) IRB.CreateCall(ST, {S->getPointerOperand(), Size(S->getValueOperand()->getType())});
else if (atomic RMW / cmpxchg)            -> store of the operand size
else if (auto *MI = dyn_cast<MemIntrinsic>(I)) -> range read of the source (memcpy/memmove) + range write of the destination
```

`tracer/memtrace_rt.c` (linked into every benchmark) implements a three-phase state machine driven by
`MT_SKIP`, `MT_WARM` and `MT_INSTS`:

1. **Skip** — only counts instructions (≈700M IR instructions/s), used to get past initialization
   (30 billion instructions here).
2. **Warm** — simulates the private caches without output (100M instructions).
3. **Trace** — 200M instructions. Accesses go through a 48 KiB 12-way L1D and a 2 MiB 16-way non-inclusive
   L2 (64 B lines, write-back/write-allocate, LRU). Each L2 miss is written as a Ramulator SimpleO3 line
   `<instructions since previous line> <address> [<dirty victim address>]`. Virtual pages are mapped to
   physical frames in first-touch order, giving compact addresses and a footprint count. A JSON file records
   instruction, access, miss and write-back counts and peak RSS.

Two runtime bugs were found by review and tests and fixed before the final traces: (i) several misses in one
basic block over-counted instructions (fixed by carrying the represented-instruction count; the stream test
now matches the executed count within 1%), and (ii) dirty L2 lines displaced when an L1 write-back allocated
into L2 were dropped; they are now queued and attached to the next miss line.

### 3.5 SPEC replay on each memory system

`memarch/cpu_study.py` replays each trace with Ramulator's **SimpleO3** frontend: a 4-wide out-of-order core
with a 128-entry window at 4 GHz and a shared LLC with 32 MSHRs per core. The core is identical for all
architectures; the LLC takes the architecture's on-chip SRAM capacity and latency and the DRAM is the
architecture's Ramulator DRAM (SRAM-only chips get a small DDR5 host memory for spills). Because Ramulator
replays a trace cyclically, every point is simulated for N and 2N instructions and the difference is
reported, so the LLC is warm. Single-core runs use N = 100M; 4-core "rate" runs give each core its own copy
of the trace in a disjoint address range and use N = 10M per core.

This is a deliberate thought experiment: AI chips do not run SPEC. It asks what their memory hierarchy —
very large, slower SRAM and high-bandwidth but bank-poor DRAM — does to latency-sensitive general-purpose
memory streams, holding the core constant.

## 4. Validation performed

* **Model sizes and FLOPs** match the published models (unit tests).
* **Bandwidth physics**: calibrated bandwidth never exceeds pin bandwidth; random < stream for every DRAM
  (unit test on LPDDR5); GPU batch-1 Llama-8B throughput is {{gpu_bound_pct}} of the pin-bandwidth bound.
* **Public token rates** for SRAM-only systems within ~1.6x (table above).
* **Tracer**: a STREAM-triad kernel produces ≥90% L2 misses per L1 miss, non-zero dirty write-backs, and
  an instruction total within 1% of the executed count (test).
* **Replay**: SimpleO3 LLC geometry is power-of-two for all architectures (test); every result's DRAM
  counters are differenced between the 2N and N runs. Review found that Ramulator's SimpleO3 LLC increments
  its read-miss and read-access counters *before* the MSHR check, so MSHR merges and every cycle a request
  is retried while the MSHRs are full are counted again (slow LLCs appeared to miss up to 6x more than fast
  ones of a smaller size). Timing is unaffected; the report uses DRAM accesses per kilo-instruction, derived
  from the DRAM controllers, instead of those counters.
* **Reproducibility of the DRAM calibration**: an independent review rebuilt Ramulator from the pinned
  submodule (7bead6c) and found that the calibration first committed did not reproduce; it had been made with
  an earlier local build of the library. The calibration, AI study and figures in this report were
  regenerated with the pinned build (HBM3/HBM4 stream efficiency 74/73% became 99%, random-access efficiency
  rose by 3–13 points, DDR5 is unchanged). The bandwidth counter was cross-checked against requests served
  per controller tick. Ramulator is now built from source with `make ramulator`.
* **Accounting**: step time equals compute-bound + memory-bound + communication + overhead time for every
  architecture (test). `make test` runs 12 tests.

## 5. Results — AI inference

{{fig:03_ai_latency_speedup|Figure 3. Step-latency speedup over GPU + HBM3. Each system uses the minimum number of chips that holds the model state.}}

{{ai_overview}}

**Decode is a memory-streaming problem.** At batch 1, every generated token reads all weights and the
KV cache once. Time follows achieved memory bandwidth: CPU/DDR5 0.10x, LPDDR5 NPU 0.07x, GDDR6 mesh 0.12x
and HBM4 2.2x of the HBM3 GPU. The SRAM-resident systems are 6–8x faster for Llama-8B and 8–9x for
Llama-70B because they read weights at tens of TB/s — but they get there by spreading the model over
76 (8B) to 654 (70B) LPU-class chips or 1–4 wafers.

{{fig:05_ai_chips|Figure 4. Chips needed just to hold the model state. SRAM-only systems pay for capacity with chips.}}

{{fig:04_ai_per_chip|Figure 5. Throughput per chip relative to GPU + HBM3.}}

**Per chip, HBM wins.** Normalized per chip, the LPU-class system delivers about a twelfth of an HBM3 GPU's
tokens/s on Llama-8B and about a thirty-seventh on Llama-70B; on the LLM scenarios only GPU + HBM4 (2.0–2.7x) and the wafer, whose single "chip" holds 44 GB of
SRAM, beat it per chip. Batching widens the gap: Figure 7 shows GPU throughput per chip growing ~32x from
batch 1 to 256, while the SRAM systems must add chips to hold the growing KV cache.

{{fig:06_time_breakdown|Figure 6. Where step time goes. DRAM systems are memory-bound in decode and compute-bound in prefill; SRAM systems are bound by inter-chip communication and per-operator synchronization.}}

**The bottleneck moves rather than disappears.** For DRAM-based designs, 83–100% of decode time is spent
waiting on memory. For the LPU-class system, 68–79% of the time is all-reduce latency across hundreds of
chips and the rest per-operator synchronization; for the wafer, on-wafer synchronization dominates. Prefill
(2,048 tokens) is compute-bound on every DRAM-based system and on a single wafer, so memory architecture
matters little there; peak FLOPS decide. The exceptions are the multi-chip SRAM-only deployments (77
LPU-class chips for Llama-8B; 655 chips or 4 wafers for Llama-70B), where the all-reduces of 2,048-token
activations take 71–92% of the time.

{{fig:08_batch_sweep|Figure 7. Llama-8B decode throughput per chip as batch grows (2k context).}}

{{fig:09_context_sweep|Figure 8. Llama-8B decode step latency as the context (KV cache) grows, batch 8.}}

**Capacity cliffs.** The sweeps show the cost of capacity-bound designs: when the KV cache pushes the
wafer from one to two systems, tensor-parallel communication appears: at batch 128 the step time jumps
from 0.69 to 1.62 ms and throughput per wafer drops 2.3x; at 32k context it goes from 0.66 to 0.96 ms and
per-wafer throughput drops 2.9x. HBM systems degrade smoothly.
Long context (32k tokens, batch 8) makes the KV cache 34 GB — larger than the weights — and raises the LPU
chip count from 76 to 233.

**Mixture-of-experts** decouples bandwidth from capacity: at batch 1 Mixtral reads ~25 GB of its 93 GB per
token, so DRAM designs pay bandwidth only for active experts, but SRAM-resident designs must still provision
SRAM for all experts (432 LPU-class chips). At batch 32 nearly all experts are active.

**Non-LLM workloads.** ResNet-50 at batch 1 (51 MB of weights) fits in every scratchpad, and every system
lands within 0.7–1.3x of the GPU — memory architecture barely matters for a small CNN.
At batch 64 its peak activations (≈155 MB) plus 51 MB of weights no longer fit the 144 MB NPU SRAM, traffic
spills to LPDDR5 and it falls to 0.08x, while compute-rich designs win. DLRM at batch 2,048 is compute-bound
on the HBM GPUs and gather-bound on GDDR6/LPDDR5, whose 25%/45% row-gather efficiency over 64 GB of tables
costs them 2–4x; HBM4 doubles HBM3, and holding the tables in SRAM takes 296 LPU-class chips.

{{fig:07_energy|Figure 9. Energy per token. Top: dynamic memory + link energy. Bottom: chips x TDP x time, an upper bound that includes static power.}}

**Energy.** Memory-access energy per token is 13–16x lower on SRAM-resident systems than on the HBM3 GPU
(about 50x lower than DDR5), because SRAM costs 0.25–0.3 pJ/bit against 3.9–15 pJ/bit for DRAM. Inter-chip
traffic eats into that: counting link energy, the LPU-class system is still 11–14x better at batch 1 but
only 1.1–4x better at batch 32, when hundreds of chips exchange larger activations. Once static power is
counted (TDP x time), a 76-chip LPU-class system uses about as much energy per Llama-8B token as a CPU at
batch 1 and ~5x more than an HBM3 GPU; at batch 32 the HBM GPUs are the most energy-efficient
(0.17–0.20 J/token vs ≥0.31 J for every other design). SRAM-only designs buy latency with power.

### 5.1 Per-scenario tables

{{ai_table:llama8b_decode_b1}}

{{ai_table:llama8b_decode_b32}}

{{ai_table:llama8b_decode_longctx}}

{{ai_table:llama8b_prefill}}

{{ai_table:llama70b_decode_b1}}

{{ai_table:llama70b_decode_b32}}

{{ai_table:mixtral_decode_b1}}

{{ai_table:mixtral_decode_b32}}

{{ai_table:resnet50_b1}}

{{ai_table:resnet50_b64}}

{{ai_table:dlrm_b2048}}

## 6. Results — SPEC CPU 2026 on the same memory systems

{{spec_traces}}

**What the traces look like.** Within the traced 200M-instruction windows, seven of the eight benchmarks
touch 3–274 MiB and make 0.2–2 DRAM accesses per thousand instructions on the CPU baseline; only 782.lbm
(a 542 MiB lattice sweep, 43 accesses/kinst on 1 core, 101 on 4) is bandwidth-heavy. For those seven, the
DRAM traffic is *identical* on all seven memory systems (one exception: 4-core nest on the GPU + HBM3's 50 MiB
LLC, 3.2 instead of 2.7 accesses/kinst): their hot working sets fit in every LLC.
What differs between the architectures is therefore the latency of the LLC itself, and the IPC tracks it
almost exactly:

| LLC (on-chip SRAM) | CPU 96 MiB | LPU 232 MiB | NPU 144 MiB | Mesh 208 MiB | GPU 50/128 MiB | Wafer 1 GiB* |
|---|---:|---:|---:|---:|---:|---:|
| Hit latency (4 GHz cycles) | 56 | 120 | 160 | 240 | 480 | 800 |
| Geomean IPC vs CPU, 1 core | 1.00 | 0.83 | 0.73 | 0.63 | 0.44 | 0.32 |
| Geomean IPC vs CPU, 4 cores | 1.00 | 0.83 | 0.70 | 0.71 | 0.53/0.52 | 0.39 |

\* The wafer's 44 GiB SRAM is capped at a 1 GiB LLC in SimpleO3; nothing in these traces needs more.

**Findings.**

1. **Latency, not bandwidth, decides general-purpose code.** SPEC-style code issues dependent, irregular
   accesses with little memory-level parallelism. A large SRAM that is 120–200 ns away (GPU L2 across the
   die, wafer fabric) costs 2–7x IPC on pointer-chasing codes (omnetpp 0.28x/0.18x, nest 0.23x/0.15x on
   1 core) even though it never misses. Compute-dense codes with tiny footprints (marian 3 MiB, stockfish's
   hot set) lose least (0.5–1.0x).
2. **HBM bandwidth buys nothing here.** GPU + HBM4 has 2.5x the DRAM bandwidth of GPU + HBM3 and a 2.5x larger
   LLC, yet its IPC is the same to within 1% on every benchmark except 4-core lbm, where it is 10% *slower*
   (HBM4's lower random-access efficiency raises its loaded latency): neither DRAM bandwidth nor LLC capacity is
   the bottleneck. Even on 4-core lbm the HBM GPUs use only 61–67 GB/s of their 3–8 TB/s, because 4-wide cores
   with 32 MSHRs each cannot generate enough parallel misses. Loaded DRAM read latency is similar on HBM
   (40–50 ns) and DDR5 (28–73 ns), so the GPUs' 2x IPC deficit comes from the 120 ns L2 hit latency, not DRAM.
3. **Capacity helps only the bandwidth-heavy benchmark.** On 4-core lbm the 208–232 MiB LLCs of the mesh and
   LPU cut DRAM accesses from 101 to 32–36 per kinst and recover 0.81–0.87x of CPU IPC, while the NPU,
   whose 144 MiB does not hold the working set, sends the full stream to LPDDR5; four interleaved lbm
   streams look like a gather pattern to its 8 channels, the loaded read latency reaches 126 ns (vs 60 ns
   on DDR5) and IPC falls to 0.45x. The CPU itself is the only system that drives >100 GB/s of DDR5.
4. **The ranking is the mirror image of the AI ranking.** The memory systems that win AI decode — wide HBM
   and huge distributed SRAM — are the worst for general-purpose code, and the plain CPU + DDR5 hierarchy
   (small, fast LLC; low-latency DRAM) is the best. The LPU-class design's short SRAM latency (30 ns) makes
   it the least bad AI chip at 0.83x.


{{fig:10_spec_ipc|Figure 10. IPC relative to CPU + DDR5 for each SPEC CPU 2026 benchmark (1 and 4 cores).}}

{{fig:11_spec_memory|Figure 11. Loaded DRAM read latency at the controller and DRAM bandwidth used (4 cores).}}

### 6.1 IPC relative to CPU + DDR5, 1 core

{{spec_table:1}}

### 6.2 IPC relative to CPU + DDR5, 4 cores

{{spec_table:4}}

## 7. Conclusions

1. **Decode is a memory-bandwidth problem, prefill a compute problem.** On every DRAM-based design 83–100% of
   batch-1 decode time is spent waiting for memory, and speed tracks the bandwidth the DRAM actually
   achieves (Ramulator), not its pin rate. HBM4 is 2.2x faster than HBM3; DDR5 and LPDDR5 are 10–14x slower.
2. **SRAM-only designs buy latency with chips.** Holding the whole model in SRAM gives 6–9x lower batch-1
   latency than an HBM3 GPU, matching public Groq/Cerebras token rates within ~1.6x, but needs 76–654
   LPU-class chips or 1–4 wafers, makes all-reduce latency and per-op synchronization the bottleneck, and
   delivers 1/12–1/37 of the GPU's throughput per chip and ~5x its energy per token once static power counts.
3. **Capacity cliffs, not gradual slowdowns.** Long contexts, large batches and MoE experts grow state
   faster than SRAM grows; SRAM designs then add chips in steps and lose per-chip throughput abruptly
   (2.3x for a wafer system at batch 128), while HBM degrades smoothly.
4. **Access pattern matters as much as peak bandwidth.** HBM keeps only 46–58%, LPDDR5 45% and GDDR6 24% of its
   bandwidth on random 64 B accesses (DDR5 86%), so paged KV caches and embedding gathers see a fraction of datasheet
   bandwidth on exactly the fastest interfaces.
5. **The same memory systems are poor for general-purpose code.** On SPEC CPU 2026 traces built with the
   LLVM tracer, IPC follows LLC hit latency: geomean 0.83x (LPU) to 0.32x (wafer) of a CPU + DDR5 hierarchy,
   with HBM4 no better than HBM3. AI inference memory systems are specialised for wide, predictable streams;
   their latency costs are hidden by the workload, not removed.

**Overall:** HBM-based GPUs remain the most balanced choice — good latency, the best throughput per watt on
every LLM decode scenario, the best throughput per chip among single-die designs, and graceful scaling
with context and batch. SRAM-only systems are a latency play for
interactive single-stream decode of models that fit their aggregate SRAM. SRAM-plus-commodity-DRAM designs
(mesh, NPU) are efficient only when the model fits on-chip (ResNet-50, small batches) and fall to their
DRAM bandwidth otherwise.

## 8. Limitations

* Architecture parameters are public-spec class values; SRAM-only per-op overhead and hop latency were
  tuned to public token rates. Absolute numbers are indicative; the comparisons and trends are the result.
* The AI model is analytical on top of cycle-accurate DRAM: no on-chip network contention, SRAM bank
  conflicts, operator fusion or overlap between consecutive operators. Prefill activations are charged to
  DRAM when they do not fit, which overstates DRAM traffic for fused kernels.
* 16-bit weights throughout. int8/fp8/fp4 would halve or quarter capacity and bandwidth needs and help the
  capacity-limited SRAM designs most.
* SPEC: one 200M-instruction window of one workload per benchmark; instruction counts are LLVM IR
  instructions; SimpleO3 has no prefetcher, no I-cache and no dependences between loads; Fortran
  benchmarks are excluded; multi-core runs use 10M instructions per core, so their per-core footprint is
  smaller than in the 1-core runs (relevant only for lbm). The AI chips' SRAM latencies, as seen from a CPU
  core, are assumptions (30–200 ns); the SPEC IPC ratios scale directly with them.
* HBM is modelled with 64 B accesses (two pseudo-channel bursts); GPUs use 32 B sectors.

## 9. Repository map

| Path | Purpose |
|---|---|
| `configs/architectures/*.json` | The seven memory-system descriptions |
| `configs/models/*.json`, `configs/scenarios.json` | Models and the 12 inference scenarios |
| `memarch/ramulator_backend.py` | Ramulator system construction, calibration, SimpleO3 replay, LLC geometry |
| `memarch/workloads.py` | Operator graphs (LLM dense/MoE, ResNet-50, DLRM) |
| `memarch/perf.py` | Placement, roofline timing, collectives, energy |
| `memarch/ai_study.py`, `memarch/cpu_study.py` | The AI and SPEC studies |
| `memarch/figures.py` | All figures |
| `tracer/` | LLVM pass plugin, runtime, sanity kernel |
| `spec/` | SPEC config, build script, trace driver, benchmark list |
| `results/` | `ai/` and `spec/` (JSON + CSV), `figures/` (PNG + PDF), `cache/` (calibration cache) |
| `report/` | Templates of this document and the summary, and the builder |
| `tests/` | 12 tests (`make test`) |
| `ramulator2/` | Ramulator 2.1 checkout (external) |
| `papers/` | Ramulator and NVMExplorer papers and sources |

## 10. Reproduction

```bash
make venv                      # Python deps into .venv
make ai figures                # Ramulator calibration + AI study + sweeps (~5 min, then cached)
make spec-install              # extract the ISO and install SPEC CPU 2026 to ~/spec/cpu2026 (~5 min)
make spec-build                # instrumented clang build + run-directory setup (~20 min)
make spec-trace                # 8 traces, 200M instructions each (~3 min)
make spec                      # replay on 7 architectures x {1, 4} cores (~3 h on 12 threads)
make figures report test
```
