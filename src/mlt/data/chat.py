"""Render conversations with the model's chat template and build loss masks.

The mask marks which tokens are trained on (assistant replies for SFT). Spans are
found generically, without per-template parsing:

  1. Render the whole conversation once.
  2. For each assistant turn i, render messages[:i] with the generation prompt.
     If that is a prefix of the full rendering, the reply starts right after it.
  3. The reply ends where its content ends, plus an end-of-turn special token
     that immediately follows (e.g. <|im_end|>, <|eot_id|>).

Turns whose prefix does not line up (some templates rewrite earlier turns) are
skipped and counted, and conversations with no usable turn are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .sources import Message


@dataclass
class ChatStats:
    conversations: int = 0
    dropped: int = 0
    assistant_turns: int = 0
    skipped_turns: int = 0
    target_tokens: int = 0
    total_tokens: int = 0
    examples_of_skips: list[str] = field(default_factory=list)


def _special_strings(tok: Any) -> list[str]:
    specials = set(tok.get_added_vocab().keys())
    specials.update(getattr(tok, "all_special_tokens", []) or [])
    return sorted((s for s in specials if s), key=len, reverse=True)


def _render(tok: Any, messages: list[Message], add_generation_prompt: bool, kwargs: dict) -> str:
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
    )


def assistant_spans(
    tok: Any, messages: list[Message], kwargs: dict, stats: ChatStats | None = None
) -> tuple[str, list[tuple[int, int]]]:
    full = _render(tok, messages, False, kwargs)
    specials = _special_strings(tok)
    spans: list[tuple[int, int]] = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        if stats is not None:
            stats.assistant_turns += 1
        start = None
        if i > 0:
            prefix = _render(tok, messages[:i], True, kwargs)
            if full.startswith(prefix):
                start = len(prefix)
        content = m["content"]
        pos = -1
        if start is not None:
            for candidate in (content, content.strip()):
                if candidate:
                    pos = full.find(candidate, start)
                    if pos >= 0:
                        content = candidate
                        break
                else:
                    pos = start
                    break
        if start is None or pos < 0:
            if stats is not None:
                stats.skipped_turns += 1
                if len(stats.examples_of_skips) < 3:
                    stats.examples_of_skips.append(f"turn {i}: {content[:60]!r}")
            continue
        end = pos + len(content)
        for s in specials:
            if full.startswith(s, end):
                end += len(s)
                break
        spans.append((start, end))
    return full, spans


def encode_conversation(
    tok: Any,
    messages: list[Message],
    kwargs: dict | None = None,
    train_on: str = "assistant",
    stats: ChatStats | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (token_ids, is_target) or None if nothing in it is trainable."""
    kwargs = kwargs or {}
    if stats is not None:
        stats.conversations += 1
    if train_on == "all":
        full = _render(tok, messages, False, kwargs)
        ids = np.asarray(tok(full, add_special_tokens=False)["input_ids"], dtype=np.int64)
        mask = np.ones_like(ids, dtype=np.uint8)
    else:
        full, spans = assistant_spans(tok, messages, kwargs, stats)
        if not spans:
            if stats is not None:
                stats.dropped += 1
            return None
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = np.asarray(enc["input_ids"], dtype=np.int64)
        offsets = np.asarray(enc["offset_mapping"], dtype=np.int64).reshape(-1, 2)
        mask = np.zeros(len(ids), dtype=np.uint8)
        for s, e in spans:
            mask |= ((offsets[:, 0] < e) & (offsets[:, 1] > s) & (offsets[:, 1] > offsets[:, 0])).astype(np.uint8)
    # The first token is never a prediction target.
    if len(mask):
        mask[0] = 0
    if not mask.any():
        if stats is not None:
            stats.dropped += 1
        return None
    if stats is not None:
        stats.target_tokens += int(mask.sum())
        stats.total_tokens += len(ids)
    return ids, mask
