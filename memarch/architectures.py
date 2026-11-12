"""Memory-architecture descriptions shared by every workload model."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .paths import CONFIG_DIR

# Ramulator's SimpleO3 LLC takes its capacity as a C int, so very large SRAMs
# (wafer-scale) are capped for CPU-trace replay. The cap exceeds every traced
# SPEC footprint, so the cap does not change hit rates.
MAX_SIMPLEO3_LLC = 1 << 30


@dataclass
class Architecture:
    name: str
    label: str
    analogue: str
    philosophy: str
    compute: dict
    sram: dict
    dram: dict | None
    interconnect: dict
    tdp_w: float
    op_overhead_us: float
    spill: dict | None = None
    raw: dict = field(default_factory=dict, repr=False)

    # ---- capacities and bandwidths -------------------------------------
    @property
    def has_dram(self) -> bool:
        return self.dram is not None

    @property
    def sram_bytes(self) -> int:
        return self.sram["capacity_bytes"]

    @property
    def dram_bytes(self) -> int:
        return self.dram["capacity_bytes"] if self.dram else 0

    @property
    def sram_role(self) -> str:
        return self.sram["role"]

    def capacity_per_chip(self) -> int:
        """Bytes available to hold model state on one chip."""
        if self.sram_role == "resident":
            return self.sram_bytes
        return self.dram_bytes

    # ---- CPU-trace replay helpers ---------------------------------------
    def cpu_dram(self) -> dict:
        """DRAM behind the LLC when replaying CPU traces (spill memory for SRAM-only chips)."""
        return self.dram or self.spill

    def cpu_llc_bytes(self, cores: int) -> int:
        return min(self.sram_bytes, MAX_SIMPLEO3_LLC)

    def sram_latency_ns(self, cores: int = 1) -> float:
        return self.sram["latency_ns"]


def load_architecture(path: Path) -> Architecture:
    data = json.loads(Path(path).read_text())
    keys = {k: data.get(k) for k in Architecture.__dataclass_fields__ if k != "raw"}
    return Architecture(**keys, raw=data)


def load_all(names=None) -> list[Architecture]:
    archs = [load_architecture(p) for p in sorted((CONFIG_DIR / "architectures").glob("*.json"))]
    order = ["cpu_ddr5", "gpu_hbm3", "gpu_hbm4", "mesh_gddr6", "npu_lpddr5", "lpu_sram", "wafer_sram"]
    archs.sort(key=lambda a: order.index(a.name) if a.name in order else len(order))
    if names:
        missing = set(names) - {a.name for a in archs}
        if missing:
            raise KeyError(f"Unknown architectures: {sorted(missing)}")
        archs = [a for a in archs if a.name in names]
    return archs
