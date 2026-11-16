"""Unit and integration tests: python -m pytest tests/"""
import json
import subprocess
from pathlib import Path

import pytest

from memarch.architectures import load_all
from memarch.perf import chips_required, evaluate
from memarch.workloads import build, expected_active_experts, load_scenarios

ROOT = Path(__file__).resolve().parents[1]
FAKE_CALIB = {"patterns": {p: {"achieved_GBps": 1000.0} for p in
                           ("stream_read", "kv_page_gather", "row_gather_256B", "random_64B", "stream_rw_2to1")}}


def wl(**kw):
    sc = {"id": "t", "model": "llama3_1_8b", "phase": "decode", "batch": 1, "context": 2048}
    sc.update(kw)
    return build(sc)


# ---------------------------------------------------------------- workloads
def test_parameter_counts_match_published_models():
    llama8 = wl()
    assert llama8.resident["weight"] / 2 == pytest.approx(8.03e9, rel=0.01)
    llama70 = wl(model="llama3_1_70b")
    assert llama70.resident["weight"] / 2 == pytest.approx(70.55e9, rel=0.01)
    mixtral = wl(model="mixtral_8x7b")
    assert mixtral.resident["weight"] / 2 == pytest.approx(46.7e9, rel=0.01)
    resnet = build({"id": "r", "model": "resnet50", "phase": "forward", "batch": 1})
    assert resnet.resident["weight"] / 2 == pytest.approx(25.5e6, rel=0.01)
    assert resnet.total("flops") == pytest.approx(8.2e9, rel=0.03)


def test_decode_reads_every_weight_once_and_whole_kv_cache():
    w = wl(batch=4, context=1000)
    t = w.traffic()
    m = w.model
    kv_expected = m["layers"] * 4 * 1001 * 2 * m["kv_heads"] * m["head_dim"] * m["kv_bytes"]
    assert t["kv_read"] == kv_expected
    embed = m["vocab"] * m["hidden"] * m["weight_bytes"]
    assert t["weight"] == pytest.approx(w.resident["weight"] - embed)


def test_moe_expert_activation():
    assert expected_active_experts(8, 2, 1) == pytest.approx(2)
    assert expected_active_experts(8, 2, 1000) == pytest.approx(8)
    b1, b32 = wl(model="mixtral_8x7b"), wl(model="mixtral_8x7b", batch=32)
    assert b1.traffic()["weight"] < 0.4 * b32.traffic()["weight"]


def test_all_scenarios_build():
    for sc in load_scenarios():
        w = build(sc)
        assert w.ops and w.resident_bytes > 0 and w.total("flops") > 0


# ---------------------------------------------------------------- perf model
def test_chip_count_follows_capacity():
    archs = {a.name: a for a in load_all()}
    w = wl(model="llama3_1_70b")
    assert chips_required(archs["gpu_hbm3"], w) == 2
    lpu = chips_required(archs["lpu_sram"], w)
    assert lpu * archs["lpu_sram"].sram_bytes * 0.9 >= w.resident_bytes
    assert (lpu - 1) * archs["lpu_sram"].sram_bytes * 0.9 < w.resident_bytes + w.peak_activation_bytes


def test_resident_sram_has_no_dram_traffic_and_cache_streams_from_dram():
    archs = {a.name: a for a in load_all()}
    r = evaluate(archs["lpu_sram"], wl(), None)
    assert r.dram_bytes == 0 and r.sram_fraction_persistent == 1.0
    g = evaluate(archs["gpu_hbm3"], wl(), FAKE_CALIB)
    assert g.dram_bytes >= wl().traffic()["weight"]
    assert g.bound == "memory"


def test_scratchpad_holds_small_models_entirely():
    archs = {a.name: a for a in load_all()}
    resnet = build({"id": "r", "model": "resnet50", "phase": "forward", "batch": 1})
    r = evaluate(archs["npu_lpddr5"], resnet, FAKE_CALIB)
    assert r.sram_fraction_persistent == 1.0 and r.dram_bytes == 0


def test_decode_latency_grows_with_context_and_prefill_is_compute_bound():
    gpu = {a.name: a for a in load_all()}["gpu_hbm3"]
    t = [evaluate(gpu, wl(batch=8, context=c), FAKE_CALIB).step_s for c in (1024, 8192, 32768)]
    assert t[0] < t[1] < t[2]
    assert evaluate(gpu, wl(phase="prefill"), FAKE_CALIB).bound == "compute"


def test_step_time_accounting():
    for a in load_all():
        r = evaluate(a, wl(batch=8), FAKE_CALIB if a.has_dram else None)
        assert r.step_s == pytest.approx(r.compute_bound_s + r.memory_bound_s + r.comm_s + r.overhead_s)
        assert r.step_s >= max(r.compute_s, r.memory_s) / 1e3  # sanity


# ---------------------------------------------------------------- Ramulator
@pytest.mark.slow
def test_ramulator_calibration_is_physical():
    from memarch.ramulator_backend import calibrate, peak_bandwidth_GBps

    a = {x.name: x for x in load_all()}["npu_lpddr5"]
    assert peak_bandwidth_GBps(a.dram) == pytest.approx(204.8)
    c = calibrate(a.dram, patterns=["stream_read", "random_64B"], requests=20000, cache=False)
    s, r = c["patterns"]["stream_read"], c["patterns"]["random_64B"]
    assert 0.5 < s["efficiency"] <= 1.0
    assert 0.0 < r["efficiency"] < s["efficiency"]


# ---------------------------------------------------------------- tracer
@pytest.mark.slow
def test_memtrace_stream_kernel(tmp_path):
    subprocess.run(["make", "-C", str(ROOT / "tracer"), "all"], check=True, capture_output=True)
    build_dir = ROOT / "tracer/build"
    exe = tmp_path / "stream"
    subprocess.run(["clang", "-O2", f"-fpass-plugin={build_dir}/libMemTracePass.so",
                    str(ROOT / "tracer/test/stream.c"), str(build_dir / "memtrace_rt.o"), "-o", str(exe)],
                   check=True)
    trace = tmp_path / "s.trace"
    env = {"MT_OUT": str(trace), "MT_SKIP": "60000000", "MT_WARM": "1000000", "MT_INSTS": "5000000"}
    subprocess.run([str(exe)], env=env, check=True, capture_output=True)
    stats = json.loads(Path(str(trace) + ".json").read_text())
    assert stats["end_reason"] == "trace_window_complete"
    lines = trace.read_text().splitlines()
    assert len(lines) == stats["l2_misses"] > 0
    # The triad touches 3 arrays far larger than L2: almost every new line misses,
    # and the written array produces dirty write-backs.
    assert stats["l2_misses"] >= 0.9 * stats["l1d_misses"]
    assert stats["l2_dirty_writebacks"] > 0
    insts = sum(int(l.split()[0]) + 1 for l in lines)
    assert abs(insts - stats["trace_insts"]) <= 0.01 * stats["trace_insts"]


def test_simpleo3_llc_geometry_is_power_of_two():
    from memarch.ramulator_backend import llc_geometry

    for a in load_all():
        for cores in (1, 4):
            total, ways, sets = llc_geometry(a.cpu_llc_bytes(cores), cores)
            assert sets & (sets - 1) == 0 and ways >= 16
            assert (total // cores) % 1024 == 0
            assert abs(total - a.cpu_llc_bytes(cores)) <= 0.02 * a.cpu_llc_bytes(cores)


@pytest.mark.slow
def test_controller_tick_period_includes_hbm_tick_multiplier():
    """HBM3/HBM4 controllers tick at half-CK; cycle counts and the SimpleO3 clock
    ratio must use the tick period, or HBM is simulated at half speed."""
    from memarch.ramulator_backend import _ratio, dram_clock_ps

    archs = {a.name: a for a in load_all()}
    assert dram_clock_ps(archs["gpu_hbm3"].dram) == 312  # 625 ps CK / 2
    assert dram_clock_ps(archs["gpu_hbm4"].dram) == 250  # 500 ps CK / 2
    assert dram_clock_ps(archs["cpu_ddr5"].dram) == 416  # DDR5 ticks once per CK
    assert _ratio(4.0, 312) == (5, 4)  # 4 GHz core : 3.2 GHz HBM3 controller


def test_psd_slope_classifies_white_and_pink_noise():
    import numpy as np

    from memarch.access_patterns import summarise_psd, welch_psd

    rng = np.random.default_rng(0)
    white = rng.standard_normal(1 << 16)
    w = summarise_psd(*welch_psd(white))
    assert abs(w["beta"]) < 0.15 and w["psd_class"] == "stationary"
    spec = np.fft.rfft(rng.standard_normal(1 << 16))
    f = np.fft.rfftfreq(1 << 16)
    spec[1:] /= np.sqrt(f[1:])  # S(f) ~ 1/f
    spec[0] = 0
    pink = np.fft.irfft(spec)
    p = summarise_psd(*welch_psd(pink))
    assert 0.8 < p["beta"] < 1.2 and p["psd_class"] == "monofractal"


def test_trace_locality_classes(tmp_path):
    import json

    from memarch.access_patterns import LOCALITY, analyse_trace

    t = tmp_path / "x.trace"
    # next line, same page, within 1 MiB, far, then next line again
    t.write_text("\n".join(f"9 {a}" for a in [0, 64, 1024, 1 << 19, 1 << 30, (1 << 30) + 64]) + "\n")
    (tmp_path / "x.trace.json").write_text(json.dumps({"touched_pages_in_window": 4}))
    r = analyse_trace(t)
    assert [round(r["locality"][k], 2) for k in LOCALITY] == [0.4, 0.2, 0.2, 0.2]
    assert r["mpki"] == 100 and r["rereference"] == 0
