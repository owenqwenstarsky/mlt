"""Rough per-laptop memory estimate for `mlt train --dry-run`.

Every laptop holds the full model, so the smallest laptop sets the limit. The
numbers are estimates (activation memory in particular depends on kernels and
MLX's allocator), meant to catch configurations that clearly cannot fit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..config.schema import RunConfig
from ..paths import from_root

_DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "U8": 1, "I8": 1, "I64": 8, "F64": 8}


@dataclass
class Shape:
    n_params: int
    hidden: int
    layers: int
    vocab: int
    ffn: int
    q_dim: int
    kv_dim: int
    disk_bytes: int | None = None
    prequantized: bool = False


def _scratch_shape(cfg: RunConfig, vocab: int) -> Shape:
    a = cfg.model.arch.resolved()
    hd = a.d_model // a.n_heads
    q, kv = a.n_heads * hd, a.n_kv_heads * hd
    per_layer = a.d_model * (2 * q + 2 * kv) + 3 * a.d_model * a.d_ff + 2 * a.d_model
    n = vocab * a.d_model * (1 if a.tie_embeddings else 2) + a.n_layers * per_layer + a.d_model
    return Shape(n, a.d_model, a.n_layers, vocab, a.d_ff, q, kv)


def _read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        return json.loads(fh.read(n))


def _pretrained_shape(model_dir: Path) -> Shape:
    cfg = json.loads((model_dir / "config.json").read_text())
    cfg = {**cfg.get("text_config", {}), **cfg}
    bits = (cfg.get("quantization") or {}).get("bits")
    n = disk = 0
    for f in sorted(model_dir.glob("*.safetensors")):
        for name, info in _read_safetensors_header(f).items():
            if name == "__metadata__":
                continue
            numel = math.prod(info["shape"]) if info["shape"] else 1
            disk += numel * _DTYPE_BYTES.get(info["dtype"], 2)
            if name.endswith((".scales", ".biases")) and bits:
                continue
            n += numel * 32 // bits if (bits and info["dtype"] == "U32") else numel
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    hd = cfg.get("head_dim") or hidden // heads
    kvh = cfg.get("num_key_value_heads") or heads
    return Shape(
        n_params=n,
        hidden=hidden,
        layers=cfg["num_hidden_layers"],
        vocab=cfg["vocab_size"],
        ffn=cfg.get("intermediate_size", 4 * hidden),
        q_dim=heads * hd,
        kv_dim=kvh * hd,
        disk_bytes=disk,
        prequantized=bits is not None,
    )


def _lora_trainable(cfg: RunConfig, s: Shape, targets: list[str]) -> int:
    r = cfg.method.rank
    dims = {
        "q_proj": (s.hidden, s.q_dim),
        "k_proj": (s.hidden, s.kv_dim),
        "v_proj": (s.hidden, s.kv_dim),
        "o_proj": (s.q_dim, s.hidden),
        "gate_proj": (s.hidden, s.ffn),
        "up_proj": (s.hidden, s.ffn),
        "down_proj": (s.ffn, s.hidden),
    }
    per_layer = sum(r * sum(dims.get(t.rsplit(".", 1)[-1], (s.hidden, s.hidden))) for t in targets)
    layers = min(cfg.method.layers or s.layers, s.layers)
    return per_layer * layers


def estimate(cfg: RunConfig, model_dir: Path | None, vocab: int | None, targets: list[str], mbs: int) -> dict[str, float]:
    s = _scratch_shape(cfg, vocab) if cfg.model.is_scratch else _pretrained_shape(model_dir)
    tc, m = cfg.train, cfg.method
    wbytes = 4 if tc.precision == "fp32" else 2
    accumulate = math.ceil(cfg.train.global_batch_size / max(1, mbs)) > 1

    if m.type == "qlora":
        weights = s.disk_bytes if s.prequantized else s.n_params * (m.bits / 8 + 4 / m.group_size)
    elif s.prequantized:
        weights = s.disk_bytes
    else:
        weights = s.n_params * wbytes

    if m.type == "full":
        t = s.n_params
        state = t * wbytes + 8 * t + (4 * t if accumulate else 0) + (4 * t if tc.precision == "mixed" else 0)
    else:
        t = _lora_trainable(cfg, s, targets)
        state = t * (4 + 4 + 8 + (4 if accumulate else 0))

    L, d, f = cfg.data.seq_len, s.hidden, s.ffn
    per_layer = mbs * L * (10 * d + 4 * f) * 2
    if tc.grad_checkpoint:
        acts = s.layers * mbs * L * d * 2 + per_layer
    else:
        acts = s.layers * per_layer
    logits = mbs * L * s.vocab * 10
    total = weights + state + acts + logits
    gb = 1e9
    return {
        "params_b": s.n_params / 1e9,
        "trainable_m": t / 1e6,
        "weights_gb": weights / gb,
        "train_state_gb": state / gb,
        "activations_gb": (acts + logits) / gb,
        "total_gb": total / gb,
    }


def model_dir_for(plan_model_path: str | None) -> Path | None:
    return from_root(plan_model_path) if plan_model_path else None
