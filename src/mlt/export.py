"""Turn a run's output into a standalone Hugging Face-format model.

LoRA / QLoRA adapters are merged into the *unquantized* base model (the usual
QLoRA practice); the base is quantized again only if you ask for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.nn as nn
from mlx.utils import tree_unflatten

from .models.loading import is_adapter_dir, resolve_model_path
from .paths import from_root
from .train.checkpoint import save_full_model


def adapter_base(adapter_dir: Path) -> Path:
    meta = json.loads((adapter_dir / "mlt_adapter.json").read_text())
    base_path = meta.get("method", {}).get("base_path")
    if base_path and from_root(base_path).exists():
        return from_root(base_path)
    return resolve_model_path(meta["base_model"])


def merge_lora(model: nn.Module, dequantize: bool) -> None:
    fused = [(n, m.fuse(dequantize=dequantize)) for n, m in model.named_modules() if hasattr(m, "fuse")]
    if fused:
        model.update_modules(tree_unflatten(fused))


def export_model(
    src: Path, out: Path, dequantize: bool = False, quantize_bits: int | None = None, group_size: int = 64, log=print
) -> Path:
    from mlx_lm.utils import dequantize_model, load_adapters, load_model, quantize_model

    if is_adapter_dir(src):
        base = adapter_base(src)
        log(f"merging adapters {src} into base {base}")
        model, config = load_model(base, lazy=True)
        load_adapters(model, str(src))
        merge_lora(model, dequantize=dequantize)
        tokenizer_dir = src
    else:
        model, config = load_model(src, lazy=True)
        tokenizer_dir = src
    config = dict(config)
    if dequantize and "quantization" in config:
        model = dequantize_model(model)
        config.pop("quantization", None)
        config.pop("quantization_config", None)
    if quantize_bits is not None:
        if "quantization" in config:
            raise ValueError("model is already quantized; pass --dequantize to re-quantize")
        model, config = quantize_model(model, config, group_size=group_size, bits=quantize_bits)
    save_full_model(out, model, config, tokenizer_dir)
    log(f"exported -> {out}")
    return out
