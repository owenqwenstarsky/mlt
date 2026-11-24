"""Size presets for the from-scratch GPT.

Parameter counts in the comments assume a 32k vocabulary with tied embeddings; the
exact count is printed at the start of every run.
"""

PRESETS: dict[str, dict] = {
    # ~10M params
    "tiny": dict(n_layers=4, d_model=256, n_heads=4, n_kv_heads=4, context_length=512),
    # ~42M params
    "small": dict(n_layers=8, d_model=512, n_heads=8, n_kv_heads=8, context_length=1024),
    # ~110M params
    "base": dict(n_layers=12, d_model=768, n_heads=12, n_kv_heads=12, context_length=1024),
    # ~340M params
    "medium": dict(n_layers=24, d_model=1024, n_heads=16, n_kv_heads=8, context_length=1024),
}


def default_ffn_dim(d_model: int, multiple_of: int = 64) -> int:
    """SwiGLU hidden size: 8/3 * d_model rounded up to a multiple of `multiple_of`."""
    hidden = int(8 * d_model / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)
