"""Architecture-independent AI inference workloads as operator lists.

Each operator records FLOPs and the bytes it moves, split by *traffic class*.
The class determines the access pattern used to look up the DRAM efficiency
measured with Ramulator (see ``ramulator_backend.PATTERNS``):

  weight      - parameters streamed once per step            -> stream_read
  kv_read     - paged KV-cache reads                          -> kv_page_gather
  kv_write    - KV-cache appends                              -> stream_rw_2to1
  act         - activations / intermediates (read + write)    -> stream_rw_2to1
  gather      - embedding-row lookups                         -> row_gather_256B

``resident`` holds the bytes that must stay in memory between steps (weights,
KV cache, embedding tables); it sets capacity requirements and chip counts.
``comm_bytes`` is the tensor-parallel all-reduce payload produced by the op.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .paths import CONFIG_DIR

TRAFFIC_PATTERN = {
    "weight": "stream_read",
    "kv_read": "kv_page_gather",
    "kv_write": "stream_rw_2to1",
    "act": "stream_rw_2to1",
    "gather": "row_gather_256B",
}


@dataclass
class Op:
    name: str
    flops: float
    engine: str  # "matrix" | "vector"
    traffic: dict = field(default_factory=dict)  # class -> bytes
    comm_bytes: float = 0.0  # all-reduce payload when tensor-parallel
    comm_kind: str = "allreduce"  # or "alltoall"

    def bytes(self, cls=None):
        return sum(self.traffic.values()) if cls is None else self.traffic.get(cls, 0)


@dataclass
class Workload:
    id: str
    model: dict
    scenario: dict
    ops: list
    resident: dict  # class -> bytes held between steps
    tokens: int  # output tokens (decode) or samples/images processed per step
    unit: str  # "token" | "image" | "sample"
    peak_activation_bytes: float  # largest live activation footprint

    @property
    def resident_bytes(self):
        return sum(self.resident.values())

    def total(self, attr):
        return sum(getattr(o, attr) for o in self.ops)

    def traffic(self):
        out = {}
        for o in self.ops:
            for k, v in o.traffic.items():
                out[k] = out.get(k, 0) + v
        return out


def load_model(name):
    return json.loads((CONFIG_DIR / "models" / f"{name}.json").read_text())


def load_scenarios():
    return json.loads((CONFIG_DIR / "scenarios.json").read_text())["scenarios"]


# ---------------------------------------------------------------------------
# Transformer decoder (dense and mixture-of-experts)
# ---------------------------------------------------------------------------
def expected_active_experts(experts, top_k, tokens):
    """Expected number of distinct experts touched by `tokens` uniformly routed tokens."""
    if experts <= 1:
        return 1.0
    return experts * (1 - (1 - top_k / experts) ** tokens)


def llm_workload(sc, m):
    h, f, L = m["hidden"], m["intermediate"], m["layers"]
    kvw = m["kv_heads"] * m["head_dim"]
    wb, kb, ab = m["weight_bytes"], m["kv_bytes"], m["act_bytes"]
    B, ctx = sc["batch"], sc["context"]
    prefill = sc["phase"] == "prefill"
    n = B * ctx if prefill else B  # tokens processed in this step
    E, k = m.get("experts", 1), m.get("top_k", 1)
    ops = []

    expert_w = 3 * h * f * wb
    attn_w = (h * (h + 2 * kvw) + h * h) * wb
    weights_total = L * (attn_w + E * expert_w + 2 * h * wb + (h * E * wb if E > 1 else 0)) + 2 * m["vocab"] * h * wb + h * wb
    kv_total = L * 2 * B * (ctx + (0 if prefill else 1)) * kvw * kb

    ops.append(Op("embedding", 0, "vector", {"gather": n * h * wb, "act": n * h * ab}))
    for layer in range(L):
        p = f"L{layer}."
        ops.append(Op(p + "norm_qkv", 2 * n * h * (h + 2 * kvw) + 5 * n * h, "matrix",
                      {"weight": h * (h + 2 * kvw) * wb + h * wb, "act": n * (h + h + 2 * kvw) * ab}))
        if prefill:
            # Flash-attention style: K/V produced on chip, causal QK^T and PV.
            ops.append(Op(p + "attention", 2 * B * ctx * ctx * h, "matrix",
                          {"kv_write": n * 2 * kvw * kb, "act": 2 * n * h * ab}))
        else:
            ops.append(Op(p + "attention", 4 * B * (ctx + 1) * h, "matrix",
                          {"kv_read": B * (ctx + 1) * 2 * kvw * kb, "kv_write": B * 2 * kvw * kb,
                           "act": 2 * n * h * ab}))
        ops.append(Op(p + "o_proj", 2 * n * h * h + n * h, "matrix",
                      {"weight": h * h * wb, "act": 2 * n * h * ab}, comm_bytes=n * h * ab))
        if E > 1:
            active = min(E, expected_active_experts(E, k, n))
            ops.append(Op(p + "router", 2 * n * h * E, "vector", {"weight": h * E * wb, "act": n * h * ab}))
            ops.append(Op(p + "moe_ffn", 2 * n * k * 3 * h * f + 6 * n * k * f, "matrix",
                          {"weight": active * expert_w + h * wb, "act": n * (2 * h + 3 * k * f) * ab},
                          comm_bytes=n * h * ab))
        else:
            ops.append(Op(p + "ffn", 2 * n * 3 * h * f + 6 * n * f + 5 * n * h, "matrix",
                          {"weight": expert_w + h * wb, "act": n * (2 * h + 3 * f) * ab}, comm_bytes=n * h * ab))
    out_rows = B  # only the last position is projected to logits
    ops.append(Op("lm_head", 2 * out_rows * h * m["vocab"], "matrix",
                  {"weight": m["vocab"] * h * wb + h * wb, "act": out_rows * (h + m["vocab"]) * ab},
                  comm_bytes=out_rows * m["vocab"] * ab))
    peak_act = n * max(3 * f if E == 1 else 3 * k * f, 4 * h) * ab
    return Workload(sc["id"], m, sc, ops, {"weight": weights_total, "kv": kv_total},
                    tokens=B if not prefill else B * ctx, unit="token", peak_activation_bytes=peak_act)


# ---------------------------------------------------------------------------
# ResNet-50 (convolutional vision model)
# ---------------------------------------------------------------------------
def resnet50_layers(image=224, classes=1000):
    """(name, cin, cout, kernel, stride, h_in) for every conv/fc of ResNet-50."""
    layers = [("conv1", 3, 64, 7, 2, image)]
    hw = image // 4  # after conv1 stride 2 and max-pool
    cin = 64
    for stage, (blocks, mid) in enumerate([(3, 64), (4, 128), (6, 256), (3, 512)]):
        out = mid * 4
        for b in range(blocks):
            stride = 2 if (b == 0 and stage > 0) else 1
            p = f"s{stage}b{b}."
            layers.append((p + "c1", cin, mid, 1, 1, hw))
            layers.append((p + "c2", mid, mid, 3, stride, hw))
            hw_out = hw // stride
            layers.append((p + "c3", mid, out, 1, 1, hw_out))
            if b == 0:
                layers.append((p + "proj", cin, out, 1, stride, hw))
            cin, hw = out, hw_out
    layers.append(("fc", 2048, classes, 1, 1, 1))
    return layers


def cnn_workload(sc, m):
    B, wb, ab = sc["batch"], m["weight_bytes"], m["act_bytes"]
    ops, weights, peak = [], 0, 0
    for name, cin, cout, kk, stride, hw in resnet50_layers(m["image"], m["classes"]):
        hw_out = max(1, hw // stride)
        w = cin * cout * kk * kk * wb
        flops = 2 * B * hw_out * hw_out * cout * cin * kk * kk
        a_in, a_out = B * hw * hw * cin * ab, B * hw_out * hw_out * cout * ab
        weights += w
        peak = max(peak, a_in + a_out)
        ops.append(Op(name, flops, "matrix", {"weight": w, "act": a_in + a_out}))
    return Workload(sc["id"], m, sc, ops, {"weight": weights}, tokens=B, unit="image",
                    peak_activation_bytes=peak)


# ---------------------------------------------------------------------------
# DLRM-style recommendation model
# ---------------------------------------------------------------------------
def dlrm_workload(sc, m):
    B, wb, ab, d = sc["batch"], m["weight_bytes"], m["act_bytes"], m["embedding_dim"]
    tables, lookups = m["tables"], m["lookups_per_table"]
    ops, mlp_w = [], 0

    def mlp(prefix, dims, n_in):
        nonlocal mlp_w
        for i, out in enumerate(dims):
            w = n_in * out * wb
            mlp_w += w
            ops.append(Op(f"{prefix}{i}", 2 * B * n_in * out, "matrix", {"weight": w, "act": B * (n_in + out) * ab}))
            n_in = out
        return n_in

    mlp("bottom_mlp", m["bottom_mlp"], m["dense_features"])
    pooled = B * tables * d * ab
    ops.append(Op("embedding_bag", B * tables * lookups * d, "vector",
                  {"gather": B * tables * lookups * d * wb, "act": pooled},
                  comm_bytes=pooled, comm_kind="alltoall"))
    x = (tables + 1) * d
    for i in range(m["cross_layers"]):
        w = x * x * wb
        mlp_w += w
        ops.append(Op(f"cross{i}", 2 * B * x * x + 3 * B * x, "matrix", {"weight": w, "act": 2 * B * x * ab}))
    mlp("top_mlp", m["top_mlp"], x)
    emb = m["total_rows"] * d * wb
    return Workload(sc["id"], m, sc, ops, {"embedding": emb, "weight": mlp_w}, tokens=B, unit="sample",
                    peak_activation_bytes=B * x * ab * 2)


def build(scenario):
    m = load_model(scenario["model"])
    fam = m["family"]
    if fam == "llm":
        return llm_workload(scenario, m)
    if fam == "cnn":
        return cnn_workload(scenario, m)
    if fam == "recsys":
        return dlrm_workload(scenario, m)
    raise ValueError(f"Unknown model family {fam}")


def all_workloads(ids=None):
    scs = load_scenarios()
    if ids:
        scs = [s for s in scs if s["id"] in ids]
    return [build(s) for s in scs]
