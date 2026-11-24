from ..registry import register
from .base import FamilyAdapter

QWEN = register(
    FamilyAdapter(
        name="qwen",
        model_types=("qwen2", "qwen3"),
        notes=(
            "Qwen2 / Qwen2.5 / Qwen3 dense models. Qwen3's template adds an empty "
            "<think></think> block to the final assistant turn; mlt trains on it as "
            "part of the reply, which matches how the model is prompted at inference."
        ),
    )
)
