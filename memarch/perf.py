"""Operator-level performance and energy model for AI inference on each
memory architecture.

For each operator on each chip:

    t_op = max(t_compute, t_memory) + t_comm + overhead
    t_compute = FLOPs_chip / (peak * efficiency)
    t_memory  = max(bytes_sram / BW_sram, sum_class bytes_dram[class] / BW_dram[pattern(class)])

SRAM and DRAM are separate paths and overlap. DRAM bandwidth per traffic class
is the *achieved* bandwidth Ramulator measured for that access pattern on the
architecture's DRAM configuration, so row-buffer locality, bank conflicts,
refresh and controller queueing are reflected.

Placement depends on the SRAM role:
  cache      - hardware LRU cache: activations hit; persistent state (weights,
               KV, tables) hits only if the whole per-chip set fits, because a
               cyclic stream larger than an LRU cache misses on every access.
  scratchpad - compiler-managed: activations pinned first, then as much
               persistent state as fits; the rest streams from DRAM.
  resident   - SRAM is the only memory; chips are added until state fits.

Multi-chip execution uses tensor parallelism across the chips required for
capacity, with an all-reduce (or all-to-all for embeddings) per marked op.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .workloads import TRAFFIC_PATTERN

SRAM_EFFICIENCY = {"stream_read": 0.9, "stream_rw_2to1": 0.85, "kv_page_gather": 0.85,
                   "row_gather_256B": 0.7}
CAPACITY_HEADROOM = 0.9  # fraction of memory usable for model state
COMPUTE_PJ_PER_FLOP = 0.4  # same for every architecture, so memory effects dominate differences
PERSISTENT = ("weight", "kv_read", "gather")  # classes that read persistent state


@dataclass
class Result:
    arch: str
    workload: str
    chips: int
    step_s: float
    compute_s: float
    memory_s: float
    comm_s: float
    overhead_s: float
    sram_bytes: float
    dram_bytes: float
    comm_bytes: float
    energy_mem_j: float
    energy_comm_j: float
    energy_compute_j: float
    sram_fraction_persistent: float
    bound: str
    tokens: int
    unit: str
    tdp_w: float = 0.0
    compute_bound_s: float = 0.0
    memory_bound_s: float = 0.0

    def row(self):
        d = self.__dict__.copy()
        d["throughput_per_s"] = self.tokens / self.step_s
        d["throughput_per_chip"] = d["throughput_per_s"] / self.chips
        d["latency_ms"] = self.step_s * 1e3
        d["energy_j_per_unit"] = (self.energy_mem_j + self.energy_comm_j + self.energy_compute_j) / self.tokens
        d["memory_energy_j_per_unit"] = (self.energy_mem_j + self.energy_comm_j) / self.tokens
        d["system_power_w"] = self.chips * self.tdp_w
        # Upper bound: every chip draws TDP for the whole step (includes static/idle power).
        d["tdp_energy_j_per_unit"] = d["system_power_w"] * self.step_s / self.tokens
        d["throughput_per_kw"] = d["throughput_per_s"] / (d["system_power_w"] / 1000)
        return d


def chips_required(arch, wl):
    """Minimum chips so per-chip persistent state + activations fit."""
    cap = arch.capacity_per_chip() * CAPACITY_HEADROOM
    need = wl.resident_bytes
    act = wl.peak_activation_bytes
    for chips in range(1, 1_000_000):
        if need / chips + act / chips <= cap:
            return chips
    raise ValueError("Workload does not fit")


def allreduce_time(arch, payload, chips):
    if chips <= 1 or payload <= 0:
        return 0.0, 0.0
    ic = arch.interconnect
    hops = 2 * math.ceil(math.log2(chips))  # tree/recursive-doubling latency term
    wire = 2 * (chips - 1) / chips * payload  # ring bandwidth term, bytes per chip
    return hops * ic["hop_latency_us"] * 1e-6 + wire / (ic["link_GBps"] * 1e9), wire * chips


def alltoall_time(arch, payload, chips):
    if chips <= 1 or payload <= 0:
        return 0.0, 0.0
    ic = arch.interconnect
    wire = (chips - 1) / chips * payload / chips  # bytes each chip sends
    hops = math.ceil(math.log2(chips))
    return hops * ic["hop_latency_us"] * 1e-6 + wire / (ic["link_GBps"] * 1e9), wire * chips


def evaluate(arch, wl, calibration=None, chips=None):
    """Evaluate one workload step on `chips` chips (default: minimum that fit)."""
    chips = chips or chips_required(arch, wl)
    sram = arch.sram
    sram_bw = sram["bandwidth_GBps"] * 1e9
    peak = {"matrix": arch.compute["matrix_tflops"] * 1e12 * arch.compute["matrix_efficiency"],
            "vector": arch.compute["vector_tflops"] * 1e12 * 0.5}

    # ---- placement --------------------------------------------------------
    persistent_chip = wl.resident_bytes / chips
    act_chip = wl.peak_activation_bytes / chips
    sram_cap = sram["capacity_bytes"] * CAPACITY_HEADROOM
    role = sram["role"]
    act_on_chip = act_chip <= sram_cap
    if role == "resident":
        frac = 1.0
        act_on_chip = True
    elif role == "scratchpad":
        free = max(0.0, sram_cap - (act_chip if act_on_chip else 0.0))
        frac = min(1.0, free / persistent_chip) if persistent_chip else 1.0
    else:  # LRU cache
        frac = 1.0 if persistent_chip + act_chip <= sram_cap * 0.8 else 0.0
        act_on_chip = act_chip <= sram_cap * 0.5

    def dram_bw(cls):
        pattern = TRAFFIC_PATTERN[cls]
        return calibration["patterns"][pattern]["achieved_GBps"] * 1e9

    totals = dict(compute=0.0, memory=0.0, comm=0.0, overhead=0.0, sram=0.0, dram=0.0, wire=0.0)
    energy_mem = 0.0
    bound_time = {"compute": 0.0, "memory": 0.0}
    for op in wl.ops:
        t_c = op.flops / chips / peak[op.engine]
        sram_b, dram_t, dram_b = 0.0, 0.0, 0.0
        sram_t = 0.0
        for cls, b in op.traffic.items():
            b /= chips
            on_chip = (frac if cls in PERSISTENT else (1.0 if act_on_chip else 0.0))
            if cls == "kv_write":
                on_chip = frac
            s, d = b * on_chip, b * (1 - on_chip)
            if s:
                sram_t += s / (sram_bw * SRAM_EFFICIENCY[TRAFFIC_PATTERN[cls]])
            if d:
                if not arch.has_dram:
                    raise ValueError(f"{arch.name} has no DRAM but {cls} spilled")
                dram_t += d / dram_bw(cls)
            sram_b += s
            dram_b += d
        t_m = max(sram_t, dram_t)
        if op.comm_kind == "alltoall":
            t_x, wire = alltoall_time(arch, op.comm_bytes, chips)
        else:
            t_x, wire = allreduce_time(arch, op.comm_bytes, chips)
        totals["compute"] += t_c
        totals["memory"] += t_m
        totals["comm"] += t_x
        totals["overhead"] += arch.op_overhead_us * 1e-6
        totals["sram"] += sram_b * chips
        totals["dram"] += dram_b * chips
        totals["wire"] += wire
        bound_time["compute" if t_c >= t_m else "memory"] += max(t_c, t_m)
        energy_mem += chips * 8 * (sram_b * sram["energy_pj_per_bit"]
                                   + dram_b * (arch.dram["energy_pj_per_bit"] if arch.has_dram else 0)) * 1e-12
    step = sum(bound_time.values()) + totals["comm"] + totals["overhead"]
    energy_comm = totals["wire"] * 8 * arch.interconnect["energy_pj_per_bit"] * 1e-12
    energy_compute = wl.total("flops") * COMPUTE_PJ_PER_FLOP * 1e-12
    shares = {"compute": bound_time["compute"], "memory": bound_time["memory"],
              "communication": totals["comm"], "overhead": totals["overhead"]}
    return Result(arch.name, wl.id, chips, step, totals["compute"], totals["memory"], totals["comm"],
                  totals["overhead"], totals["sram"], totals["dram"], totals["wire"], energy_mem,
                  energy_comm, energy_compute, frac, max(shares, key=shares.get), wl.tokens, wl.unit, arch.tdp_w,
                  bound_time["compute"], bound_time["memory"])
