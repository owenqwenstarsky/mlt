"""Model family adapters.

mlx-lm supplies the architectures; a family adapter holds the few things mlt needs
to know per family. Adding a family = one new module that builds a `FamilyAdapter`
and registers it (see `qwen.py`), plus a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Every linear projection in a standard attention + SwiGLU block.
STANDARD_LORA_TARGETS: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class FamilyAdapter:
    name: str
    # mlx-lm `model_type` values this family covers.
    model_types: tuple[str, ...]
    # Module paths, relative to a transformer block, that LoRA attaches to by default.
    lora_targets: tuple[str, ...] = STANDARD_LORA_TARGETS
    # Extra kwargs passed to `tokenizer.apply_chat_template` when rendering SFT data.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
