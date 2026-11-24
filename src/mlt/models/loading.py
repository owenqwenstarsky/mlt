"""Building the model a run trains: a fresh GPT or a pretrained checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..config.schema import ArchConfig, MethodConfig
from ..paths import from_root, models_dir
from ..tokenization.load import generation_eos_ids, load_tokenizer
from .families.base import FamilyAdapter
from .gpt import GPT, GPTConfig
from .registry import get_family

DOWNLOAD_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.jinja",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "tokenizer*",
]

DTYPES = {"fp32": mx.float32, "bf16": mx.bfloat16, "mixed": mx.bfloat16}


@dataclass
class LoadedModel:
    model: nn.Module
    hf_config: dict[str, Any]  # written as config.json on export
    family: FamilyAdapter
    quantized: bool


# ------------------------------------------------------------------ resolution


def local_model_dir(repo: str) -> Path:
    return models_dir() / repo.strip("/").replace("/", "--")


def is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and not (path / "config.json").exists()


def resolve_model_path(source: str) -> Path:
    """A local checkpoint dir, or where a HF repo is (or will be) downloaded."""
    local = from_root(source)
    if local.exists():
        if is_adapter_dir(local):
            raise ValueError(
                f"{source} holds LoRA adapters, not a full model. Merge them first: "
                f"`mlt export {source} --out <dir>`, then train from <dir>."
            )
        if not (local / "config.json").exists():
            raise FileNotFoundError(f"{source} has no config.json")
        return local
    if source.startswith((".", "/")) or source.count("/") != 1:
        raise FileNotFoundError(f"model path {source!r} does not exist")
    return local_model_dir(source)


def download_model(repo: str, revision: str | None = None, log=print) -> Path:
    out = local_model_dir(repo)
    if (out / "config.json").exists() and list(out.glob("*.safetensors")):
        return out
    from huggingface_hub import snapshot_download

    log(f"downloading {repo} -> {out}")
    snapshot_download(
        repo_id=repo, revision=revision, local_dir=str(out), allow_patterns=DOWNLOAD_PATTERNS
    )
    if not list(out.glob("*.safetensors")):
        raise FileNotFoundError(f"{repo} has no .safetensors weights (only safetensors are supported)")
    return out


def read_hf_config(path: Path) -> dict[str, Any]:
    return json.loads((path / "config.json").read_text())


def family_for_path(path: Path) -> FamilyAdapter:
    return get_family(read_hf_config(path)["model_type"])


# --------------------------------------------------------------------- loading


def build_scratch(arch: ArchConfig, tokenizer_dir: Path, precision: str) -> LoadedModel:
    tok = load_tokenizer(tokenizer_dir)
    gcfg = GPTConfig.from_arch(arch, vocab_size=len(tok))
    model = GPT(gcfg)
    model.init_weights()
    if precision != "fp32":
        model.set_dtype(DTYPES[precision])
    mx.eval(model.parameters())
    eos = generation_eos_ids(tok)
    hf_config = gcfg.to_hf_config(
        bos_token_id=tok.bos_token_id, eos_token_id=eos if len(eos) > 1 else (eos[0] if eos else None)
    )
    return LoadedModel(model=model, hf_config=hf_config, family=get_family("llama"), quantized=False)


def _eval_layerwise(model: nn.Module) -> None:
    """Materialise weights one block at a time so lazy loading/quantizing keeps peak memory low."""
    layers = getattr(model, "layers", None)
    if layers:
        for layer in layers:
            mx.eval(layer.parameters())
    mx.eval(model.parameters())


def load_pretrained(
    path: Path,
    method: MethodConfig,
    precision: str,
    trust_remote_code: bool = False,
    quantize_for_qlora: bool = True,
) -> LoadedModel:
    from mlx_lm.utils import load_model

    model, config = load_model(path, lazy=True, trust_remote_code=trust_remote_code)
    family = get_family(config["model_type"])
    quantized = "quantization" in config
    if method.type == "full" and quantized:
        raise ValueError(
            f"{path} is quantized; full fine-tuning needs unquantized weights. Use "
            "method `qlora`, or start from the unquantized model."
        )
    model.set_dtype(DTYPES[precision])
    if method.type == "qlora" and not quantized and quantize_for_qlora:
        gs = method.group_size

        def predicate(_path: str, m: nn.Module) -> bool:
            return hasattr(m, "to_quantized") and m.weight.shape[-1] % gs == 0

        nn.quantize(model, group_size=gs, bits=method.bits, class_predicate=predicate)
        config = dict(config)
        config["quantization"] = {"group_size": gs, "bits": method.bits, "mode": "affine"}
        quantized = True
    _eval_layerwise(model)
    return LoadedModel(model=model, hf_config=config, family=family, quantized=quantized)
