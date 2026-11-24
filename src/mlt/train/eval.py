"""Validation loss and quick text samples."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..data.batching import make_microbatches
from ..data.sampler import rank_slice, split_counts
from ..dist.world import World
from .step import StepFunctions, _to_mx


def validation_loss(
    fns: StepFunctions,
    model: nn.Module,
    val,
    world: World,
    shares: list[float],
    micro_batch_size: int,
    seq_len: int,
    max_items: int,
) -> dict[str, float]:
    """Loss over a fixed prefix of the validation set, split across ranks like training."""
    n = min(len(val), max_items)
    if n < world.size:
        return {}
    counts = split_counts(n, shares)
    sl = rank_slice(counts, world.rank)
    items = [val.get(i) for i in range(sl.start, sl.stop)]
    model.eval()
    total = mx.array(0.0, dtype=mx.float32)
    count = mx.array(0.0, dtype=mx.float32)
    for mb in make_microbatches(items, micro_batch_size, seq_len):
        ls, c = fns.eval_fn(*_to_mx(mb))
        total, count = total + ls, count + c
        mx.eval(total, count)
    model.train()
    stats = world.all_sum(mx.stack([total, count]), small=True)
    loss = stats[0].item() / max(stats[1].item(), 1.0)
    return {"val_loss": loss, "val_ppl": math.exp(min(loss, 30.0)), "val_tokens": stats[1].item()}


def greedy_sample(model: nn.Module, tokenizer: Any, prompt: str, max_tokens: int, max_len: int) -> str:
    """Simple greedy decoding without a KV cache; fine for short samples during training."""
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"] or [tokenizer.eos_token_id]
    stop = {tokenizer.eos_token_id}
    model.eval()
    out: list[int] = []
    for _ in range(max_tokens):
        window = (ids + out)[-max_len:]
        logits = model(mx.array(np.asarray([window], dtype=np.int32)))
        nxt = int(mx.argmax(logits[0, -1]).item())
        if nxt in stop:
            break
        out.append(nxt)
    model.train()
    return tokenizer.decode(out)
