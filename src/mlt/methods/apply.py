"""Make the right parameters trainable for full / LoRA / QLoRA.

QLoRA is LoRA on a base whose linear layers were quantized at load time (MLX
affine quantization, default 4-bit, group size 64); see `models.loading`.
"""

from __future__ import annotations

from typing import Any

import mlx.nn as nn

from ..config.schema import MethodConfig
from ..models.families.base import FamilyAdapter


def lora_targets(method: MethodConfig, family: FamilyAdapter) -> list[str]:
    return list(family.lora_targets) if method.targets == "default" else list(method.targets)


def lora_num_layers(method: MethodConfig, model: nn.Module) -> int:
    n = len(model.layers)
    return n if method.layers is None else min(method.layers, n)


def adapter_config(method: MethodConfig, family: FamilyAdapter, model: nn.Module) -> dict[str, Any]:
    """adapter_config.json in mlx-lm's format, so `mlx_lm.generate --adapter-path` works too."""
    return {
        "fine_tune_type": "lora",
        "num_layers": lora_num_layers(method, model),
        "lora_parameters": {
            "rank": method.rank,
            "scale": method.scale,
            "dropout": method.dropout,
            "keys": lora_targets(method, family),
        },
    }


def apply_method(model: nn.Module, method: MethodConfig, family: FamilyAdapter) -> None:
    if method.type == "full":
        model.unfreeze()
        return

    from mlx_lm.tuner.utils import linear_to_lora_layers

    keys = lora_targets(method, family)
    available = {name for name, m in model.layers[0].named_modules() if isinstance(m, (nn.Linear, nn.QuantizedLinear))}
    missing = [k for k in keys if k not in available]
    if missing:
        raise ValueError(
            f"LoRA targets {missing} not found in a transformer block. Linear modules "
            f"available: {sorted(available)}"
        )
    model.freeze()
    cfg = adapter_config(method, family, model)
    linear_to_lora_layers(model, cfg["num_layers"], cfg["lora_parameters"])
