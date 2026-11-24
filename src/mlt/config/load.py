"""Loading configs from YAML and fingerprinting them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import RunConfig

# Fields that may change between a checkpoint and its resumption without changing
# the optimisation trajectory: where it runs and how often it reports/saves.
_RESUME_IGNORED_TRAIN = {
    "log_every",
    "eval_every",
    "eval_batches",
    "save_every",
    "keep_checkpoints",
    "sample_prompts",
    "sample_max_tokens",
    "micro_batch_size",
    "compile",
    "grad_checkpoint",
}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError(f"config not found: {path}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return parse_config(raw, source=str(path))


def parse_config(raw: dict[str, Any], source: str = "<config>") -> RunConfig:
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError(f"invalid config {source}:\n" + "\n".join(lines)) from None


def dump_config(cfg: RunConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json", by_alias=True)


def trajectory_view(cfg: RunConfig) -> dict[str, Any]:
    """The parts of a config that determine the training trajectory."""
    d = dump_config(cfg)
    d.pop("cluster", None)
    d.pop("output_dir", None)
    for k in _RESUME_IGNORED_TRAIN:
        d["train"].pop(k, None)
    return d


def trajectory_hash(cfg: RunConfig) -> str:
    blob = json.dumps(trajectory_view(cfg), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def diff_views(a: dict[str, Any], b: dict[str, Any], prefix: str = "") -> list[str]:
    """Human readable list of differing keys between two nested dicts."""
    out: list[str] = []
    for k in sorted(set(a) | set(b)):
        key = f"{prefix}{k}"
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(diff_views(va, vb, key + "."))
        elif va != vb:
            out.append(f"{key}: {va!r} -> {vb!r}")
    return out
