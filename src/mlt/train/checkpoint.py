"""Resumable checkpoints and the final model export.

Only rank 0 writes. A checkpoint holds what is needed to continue the exact
trajectory: trainable weights (fp32 master weights under `mixed`), optimizer
moments and the step. The data position is derived from the step, so it does not
depend on how many laptops are used when resuming.

  runs/<name>/checkpoints/step_000500/
    weights.safetensors   optimizer.safetensors   state.json
  runs/<name>/final/      a loadable model (HF format) or LoRA adapters
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..tokenization.load import copy_tokenizer_files

CKPT_PREFIX = "step_"


def checkpoints_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoints"


def list_checkpoints(run_dir: Path) -> list[Path]:
    d = checkpoints_dir(run_dir)
    if not d.exists():
        return []
    found = [p for p in d.iterdir() if p.is_dir() and p.name.startswith(CKPT_PREFIX) and (p / "state.json").exists()]
    return sorted(found, key=lambda p: int(p.name[len(CKPT_PREFIX) :]))


def latest_checkpoint(run_dir: Path) -> Path | None:
    found = list_checkpoints(run_dir)
    return found[-1] if found else None


def save_checkpoint(
    run_dir: Path,
    step: int,
    weights: dict[str, mx.array],
    optimizer_state: dict[str, mx.array],
    state: dict[str, Any],
    keep: int,
) -> Path:
    out = checkpoints_dir(run_dir) / f"{CKPT_PREFIX}{step:07d}"
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "weights.safetensors"), weights)
    if optimizer_state:
        mx.save_safetensors(str(tmp / "optimizer.safetensors"), optimizer_state)
    (tmp / "state.json").write_text(json.dumps({**state, "step": step, "saved_at": time.time()}, indent=2))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    for old in list_checkpoints(run_dir)[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return out


def load_checkpoint(path: Path) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, Any]]:
    weights = mx.load(str(path / "weights.safetensors"))
    opt_file = path / "optimizer.safetensors"
    opt = mx.load(str(opt_file)) if opt_file.exists() else {}
    state = json.loads((path / "state.json").read_text())
    return weights, opt, state


# ------------------------------------------------------------------- exports


def save_full_model(out: Path, model: nn.Module, hf_config: dict, tokenizer_dir: Path) -> Path:
    from mlx_lm.utils import save_config, save_model

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    save_model(tmp, model)
    save_config(dict(hf_config), tmp / "config.json")
    copy_tokenizer_files(Path(tokenizer_dir), tmp)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def save_adapters(
    out: Path,
    model: nn.Module,
    adapter_cfg: dict,
    base_model: str,
    method: dict,
    tokenizer_dir: Path,
) -> Path:
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    adapters = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(tmp / "adapters.safetensors"), adapters)
    (tmp / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2))
    (tmp / "mlt_adapter.json").write_text(json.dumps({"base_model": base_model, "method": method}, indent=2))
    copy_tokenizer_files(Path(tokenizer_dir), tmp)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out
