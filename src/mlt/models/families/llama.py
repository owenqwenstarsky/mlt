from ..registry import register
from .base import FamilyAdapter

LLAMA = register(
    FamilyAdapter(
        name="llama",
        model_types=("llama",),
        notes="Llama 2/3 and every model pretrained from scratch with mlt (exported as llama).",
    )
)
