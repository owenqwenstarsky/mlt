"""Read-only views over prepared data.

Both views expose `len()` and `get(i)` over a fixed list of item indices, so the
sampler never needs to know whether it is drawing text windows or conversations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HASH_MULT = 2654435761  # Knuth multiplicative hash, for a deterministic val split


def holdout_split(n: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically scatter roughly `val_fraction` of item indices into val."""
    idx = np.arange(n, dtype=np.int64)
    if val_fraction <= 0:
        return idx, idx[:0]
    bucket = (idx * _HASH_MULT) % 10_000
    is_val = bucket < int(round(val_fraction * 10_000))
    if is_val.all():
        is_val[0] = False
    if not is_val.any() and n > 1:
        is_val[-1] = True
    return idx[~is_val], idx[is_val]


class Prepared:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.kind = self.meta["kind"]
        self.tokens = np.memmap(self.path / "tokens.bin", dtype=self.meta["dtype"], mode="r")
        if self.kind == "chat":
            self.mask = np.memmap(self.path / "mask.bin", dtype=np.uint8, mode="r")
            self.offsets = np.load(self.path / "offsets.npy")
            self.first_target = np.load(self.path / "first_target.npy")


class TextWindows:
    """Non-overlapping windows of seq_len + 1 tokens over the token stream."""

    kind = "text"

    def __init__(self, tokens: np.ndarray, seq_len: int, indices: np.ndarray | None = None):
        self.tokens = tokens
        self.seq_len = seq_len
        n = (len(tokens) - 1) // seq_len
        if n < 1:
            raise ValueError(f"dataset has {len(tokens)} tokens, fewer than seq_len + 1 ({seq_len + 1})")
        self.indices = np.arange(n, dtype=np.int64) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def get(self, i: int) -> tuple[np.ndarray, None]:
        w = int(self.indices[i])
        start = w * self.seq_len
        return np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64), None

    def subset(self, sel: np.ndarray) -> TextWindows:
        return TextWindows(self.tokens, self.seq_len, self.indices[sel])


class ChatExamples:
    """Conversations truncated to seq_len + 1 tokens, with their target masks."""

    kind = "chat"

    def __init__(self, data: Prepared, seq_len: int, indices: np.ndarray | None = None):
        self.data = data
        self.seq_len = seq_len
        if indices is None:
            # Drop conversations whose first target falls outside the window.
            indices = np.nonzero(data.first_target < seq_len + 1)[0].astype(np.int64)
            if len(indices) == 0:
                raise ValueError("no conversation has a target token within seq_len")
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def length(self, i: int) -> int:
        e = int(self.indices[i])
        return min(int(self.data.offsets[e + 1] - self.data.offsets[e]), self.seq_len + 1)

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        e = int(self.indices[i])
        s = int(self.data.offsets[e])
        n = min(int(self.data.offsets[e + 1]) - s, self.seq_len + 1)
        toks = np.asarray(self.data.tokens[s : s + n], dtype=np.int64)
        mask = np.asarray(self.data.mask[s : s + n], dtype=np.uint8)
        return toks, mask

    def subset(self, sel: np.ndarray) -> ChatExamples:
        return ChatExamples(self.data, self.seq_len, self.indices[sel])


def open_dataset(path: Path, seq_len: int) -> TextWindows | ChatExamples:
    data = Prepared(path)
    if data.kind == "text":
        return TextWindows(data.tokens, seq_len)
    return ChatExamples(data, seq_len)


def train_val_views(
    train_path: Path, val_path: Path | None, val_fraction: float, seq_len: int
) -> tuple[TextWindows | ChatExamples, TextWindows | ChatExamples | None]:
    train = open_dataset(train_path, seq_len)
    if val_path is not None:
        return train, open_dataset(val_path, seq_len)
    if val_fraction > 0:
        tr, va = holdout_split(len(train), val_fraction)
        return train.subset(tr), train.subset(va)
    return train, None
