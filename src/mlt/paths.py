"""Filesystem layout.

Everything mlt generates lives under the project root (the directory commands run
from, which is the same absolute path on every laptop):

  .mlt/models/<repo>        downloaded Hugging Face models
  .mlt/tokenizers/<key>     reused / trained tokenizers
  .mlt/data/<key>           tokenized dataset shards
  .mlt/cluster/             hostfile and benchmark results
  runs/<name>/              resolved config, metrics, checkpoints, final model
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(os.environ.get("MLT_ROOT", os.getcwd())).resolve()


def cache_root() -> Path:
    return project_root() / ".mlt"


def models_dir() -> Path:
    return cache_root() / "models"


def tokenizers_dir() -> Path:
    return cache_root() / "tokenizers"


def data_dir() -> Path:
    return cache_root() / "data"


def cluster_dir() -> Path:
    return cache_root() / "cluster"


def stable_hash(obj: Any, n: int = 16) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def files_hash(paths: list[Path], n: int = 16) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:n]


def relative_to_root(p: Path) -> str:
    """Store paths relative to the project root so they resolve on every laptop."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(project_root()))
    except ValueError:
        return str(p)


def from_root(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else project_root() / p
