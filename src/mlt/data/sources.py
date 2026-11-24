"""Record readers for every supported data source.

Hugging Face datasets are always streamed (never fully downloaded); the tokenized
result is cached under `.mlt/data`, so a dataset is only read once per tokenizer.
"""

from __future__ import annotations

import glob
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..config.schema import DataSource
from ..paths import from_root

Message = dict[str, str]


def _expand(path: str, suffixes: tuple[str, ...]) -> list[Path]:
    p = from_root(path)
    if p.is_dir():
        files = [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix in suffixes]
    elif any(c in path for c in "*?["):
        files = [Path(f) for f in sorted(glob.glob(str(p), recursive=True)) if Path(f).is_file()]
    else:
        files = [p]
    if not files:
        raise FileNotFoundError(f"no data files found for {path!r}")
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"data file not found: {f}")
    return files


def iter_records(src: DataSource) -> Iterator[dict[str, Any]]:
    """Yield raw records (dicts) from a source, honouring `max_samples`."""
    n = 0
    for rec in _iter_raw(src):
        if src.max_samples is not None and n >= src.max_samples:
            return
        yield rec
        n += 1


def _iter_raw(src: DataSource) -> Iterator[dict[str, Any]]:
    field = src.field or "text"
    if src.source == "jsonl":
        for f in _expand(src.path, (".jsonl", ".json")):
            with open(f, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{f}:{lineno}: invalid JSON: {e}") from None
    elif src.source == "txt":
        for f in _expand(src.path, (".txt", ".md")):
            yield {field: f.read_text(encoding="utf-8")}
    elif src.source == "parquet":
        import pyarrow.parquet as pq

        for f in _expand(src.path, (".parquet",)):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=1024):
                yield from batch.to_pylist()
    elif src.source == "hf":
        from datasets import load_dataset

        ds = load_dataset(src.path, src.name, split=src.split, streaming=True)
        yield from ds
    else:  # pragma: no cover - guarded by the schema
        raise ValueError(f"unknown source {src.source!r}")


def iter_texts(src: DataSource) -> Iterator[str]:
    field = src.field or "text"
    for i, rec in enumerate(iter_records(src)):
        text = rec.get(field)
        if not isinstance(text, str):
            raise ValueError(
                f"record {i} of {src.path!r} has no string field {field!r} "
                f"(keys: {sorted(rec)})"
            )
        if text:
            yield text


def to_messages(rec: dict[str, Any], field: str) -> list[Message]:
    """Normalise a chat record to a list of {role, content} messages.

    Accepts `{field: [{role, content}, ...]}` (OpenAI style) or
    `{prompt, completion}` pairs (optionally with `system`).
    """
    if field in rec:
        msgs = rec[field]
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"field {field!r} must be a non-empty list of messages")
        out = []
        for m in msgs:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError("each message needs `role` and `content`")
            msg = {"role": str(m["role"]), "content": m["content"] or ""}
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            out.append(msg)
        return out
    if "prompt" in rec and "completion" in rec:
        out = []
        if rec.get("system"):
            out.append({"role": "system", "content": rec["system"]})
        out.append({"role": "user", "content": rec["prompt"]})
        out.append({"role": "assistant", "content": rec["completion"]})
        return out
    raise ValueError(f"chat record has neither {field!r} nor prompt/completion (keys: {sorted(rec)})")


def iter_conversations(src: DataSource) -> Iterator[list[Message]]:
    field = src.field or "messages"
    for i, rec in enumerate(iter_records(src)):
        try:
            yield to_messages(rec, field)
        except ValueError as e:
            raise ValueError(f"record {i} of {src.path!r}: {e}") from None
