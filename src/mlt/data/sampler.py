"""Global, laptop-count-independent data order.

The sample order is a function of (seed, epoch) only. Each optimizer step consumes
the next `global_batch_size` items of that order, and each rank takes a contiguous
slice of it sized by its share. Because the order never depends on how many laptops
there are, a run can resume on a different set of laptops and see the same data.
"""

from __future__ import annotations

import math

import numpy as np


class GlobalSampler:
    def __init__(self, n_items: int, global_batch_size: int, seed: int):
        if n_items < 1:
            raise ValueError("dataset is empty")
        self.n = n_items
        self.batch_size = global_batch_size
        self.seed = seed
        self._cache: dict[int, np.ndarray] = {}

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._cache:
            if len(self._cache) > 2:
                self._cache.pop(min(self._cache))
            rng = np.random.default_rng([self.seed, epoch])
            self._cache[epoch] = rng.permutation(self.n)
        return self._cache[epoch]

    def batch(self, step: int) -> np.ndarray:
        pos = step * self.batch_size + np.arange(self.batch_size, dtype=np.int64)
        epochs = pos // self.n
        out = np.empty(self.batch_size, dtype=np.int64)
        for e in np.unique(epochs):
            sel = epochs == e
            out[sel] = self._perm(int(e))[pos[sel] % self.n]
        return out

    def epoch_at(self, step: int) -> float:
        return step * self.batch_size / self.n

    def steps_for_epochs(self, epochs: float) -> int:
        return max(1, math.ceil(epochs * self.n / self.batch_size))


def split_counts(total: int, shares: list[float]) -> list[int]:
    """Split `total` items by `shares` (largest remainder), every rank getting >= 1."""
    k = len(shares)
    if total < k:
        raise ValueError(f"cannot split {total} items across {k} ranks")
    s = np.asarray(shares, dtype=np.float64)
    if (s <= 0).any():
        raise ValueError("shares must be positive")
    raw = total * s / s.sum()
    counts = np.floor(raw).astype(np.int64)
    deficit = int(total - counts.sum())  # always < k
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:deficit]] += 1
    for i in np.nonzero(counts == 0)[0]:
        counts[int(np.argmax(counts))] -= 1
        counts[i] += 1
    return [int(c) for c in counts]


def rank_slice(counts: list[int], rank: int) -> slice:
    start = sum(counts[:rank])
    return slice(start, start + counts[rank])
