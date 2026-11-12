"""Repository paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAMULATOR_DIR = ROOT / "ramulator2"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
TRACE_DIR = ROOT / "traces"
