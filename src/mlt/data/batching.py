"""Turn a rank's share of a global batch into micro-batches.

Each micro-batch is (inputs, targets, mask) with mask = 1 where the target is a
loss token. Text windows are already full length. Conversations are sorted by
length before chunking (cheap length bucketing) and padded to a multiple of
`pad_multiple`, which limits how many distinct shapes the compiled step sees.
"""

from __future__ import annotations

import numpy as np

Item = tuple[np.ndarray, np.ndarray | None]
MicroBatch = tuple[np.ndarray, np.ndarray, np.ndarray]


def make_microbatches(
    items: list[Item], micro_batch_size: int, seq_len: int, pad_multiple: int = 64, pad_id: int = 0
) -> list[MicroBatch]:
    if not items:
        return []
    if items[0][1] is None:
        return [_text_batch(items[i : i + micro_batch_size]) for i in range(0, len(items), micro_batch_size)]
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]), reverse=True)
    out = []
    for i in range(0, len(order), micro_batch_size):
        group = [items[j] for j in order[i : i + micro_batch_size]]
        out.append(_chat_batch(group, seq_len, pad_multiple, pad_id))
    return out


def _text_batch(items: list[Item]) -> MicroBatch:
    w = np.stack([t for t, _ in items])
    inputs, targets = w[:, :-1], w[:, 1:]
    return inputs, targets, np.ones_like(targets, dtype=np.float32)


def _chat_batch(items: list[Item], seq_len: int, pad_multiple: int, pad_id: int) -> MicroBatch:
    longest = max(len(t) for t, _ in items) - 1
    L = min(seq_len, pad_multiple * -(-longest // pad_multiple))
    B = len(items)
    inputs = np.full((B, L), pad_id, dtype=np.int64)
    targets = np.full((B, L), pad_id, dtype=np.int64)
    mask = np.zeros((B, L), dtype=np.float32)
    for b, (toks, m) in enumerate(items):
        n = min(len(toks) - 1, L)
        inputs[b, :n] = toks[:n]
        targets[b, :n] = toks[1 : n + 1]
        mask[b, :n] = m[1 : n + 1]
    return inputs, targets, mask
