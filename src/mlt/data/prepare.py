"""Tokenize a data source into cached shards under `.mlt/data/<key>/`.

Layout of a prepared directory:
  meta.json     kind, dtype, counts, stats (written last: its presence marks completion)
  tokens.bin    all token ids, concatenated (uint16 when the vocab fits, else uint32)
  text only:    documents are separated by the eos token
  chat only:    mask.bin (uint8, 1 = loss target), offsets.npy (example boundaries),
                first_target.npy (index of each example's first target token)
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config.schema import DataSource
from ..paths import data_dir, stable_hash
from ..tokenization.load import load_tokenizer, tokenizer_fingerprint
from .chat import ChatStats, encode_conversation
from .sources import iter_conversations, iter_texts

FORMAT_VERSION = 1
_TEXT_BATCH = 256


def dataset_key(
    src: DataSource, kind: str, tokenizer_dir: Path, train_on: str, chat_kwargs: dict
) -> str:
    return stable_hash(
        {
            "v": FORMAT_VERSION,
            "kind": kind,
            "source": src.model_dump(mode="json"),
            "tokenizer": tokenizer_fingerprint(tokenizer_dir),
            "train_on": train_on if kind == "chat" else None,
            "chat_kwargs": chat_kwargs if kind == "chat" else None,
        }
    )


def _dtype_for(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size <= np.iinfo(np.uint16).max + 1 else np.dtype(np.uint32)


def prepare_dataset(
    src: DataSource,
    kind: str,
    tokenizer_dir: Path,
    train_on: str = "assistant",
    chat_kwargs: dict | None = None,
    log=print,
) -> Path:
    chat_kwargs = chat_kwargs or {}
    key = dataset_key(src, kind, tokenizer_dir, train_on, chat_kwargs)
    out = data_dir() / f"{kind}-{key}"
    if (out / "meta.json").exists():
        return out
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tok = load_tokenizer(tokenizer_dir)
    dtype = _dtype_for(len(tok))
    t0 = time.time()
    log(f"preparing {kind} data from {src.source}:{src.path} -> {out}")
    if kind == "text":
        meta = _prepare_text(src, tok, tmp, dtype)
    elif kind == "chat":
        meta = _prepare_chat(src, tok, tmp, dtype, train_on, chat_kwargs)
    else:
        raise ValueError(f"unknown data kind {kind!r}")
    meta.update(
        kind=kind,
        dtype=dtype.name,
        vocab_size=len(tok),
        format_version=FORMAT_VERSION,
        source=src.model_dump(mode="json"),
        seconds=round(time.time() - t0, 2),
    )
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(out)
    log(f"prepared {meta['n_tokens']:,} tokens in {meta['seconds']}s")
    return out


def _prepare_text(src: DataSource, tok: Any, out: Path, dtype: np.dtype) -> dict:
    eos = tok.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos token; it is needed to separate documents")
    n_tokens = n_docs = 0
    with open(out / "tokens.bin", "wb") as fh:
        batch: list[str] = []

        def flush() -> None:
            nonlocal n_tokens, n_docs
            if not batch:
                return
            enc = tok(batch, add_special_tokens=True)["input_ids"]
            parts = []
            for ids in enc:
                if not ids or ids[-1] != eos:
                    ids = list(ids) + [eos]
                parts.append(np.asarray(ids, dtype=dtype))
            arr = np.concatenate(parts)
            fh.write(arr.tobytes())
            n_tokens += len(arr)
            n_docs += len(batch)
            batch.clear()

        for text in iter_texts(src):
            batch.append(text)
            if len(batch) >= _TEXT_BATCH:
                flush()
        flush()
    if n_tokens == 0:
        raise ValueError(f"no text found in {src.path!r}")
    return {"n_tokens": n_tokens, "n_docs": n_docs}


def _prepare_chat(
    src: DataSource, tok: Any, out: Path, dtype: np.dtype, train_on: str, chat_kwargs: dict
) -> dict:
    stats = ChatStats()
    offsets = [0]
    first_target: list[int] = []
    with open(out / "tokens.bin", "wb") as ft, open(out / "mask.bin", "wb") as fm:
        for messages in iter_conversations(src):
            enc = encode_conversation(tok, messages, chat_kwargs, train_on, stats)
            if enc is None:
                continue
            ids, mask = enc
            ft.write(ids.astype(dtype).tobytes())
            fm.write(mask.tobytes())
            offsets.append(offsets[-1] + len(ids))
            first_target.append(int(np.argmax(mask)))
    if len(offsets) == 1:
        raise ValueError(
            f"no trainable conversations in {src.path!r} "
            f"({stats.conversations} read, {stats.dropped} dropped; skipped turns e.g. "
            f"{stats.examples_of_skips})"
        )
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "first_target.npy", np.asarray(first_target, dtype=np.int64))
    return {"n_tokens": offsets[-1], "n_examples": len(offsets) - 1, "stats": asdict(stats)}
