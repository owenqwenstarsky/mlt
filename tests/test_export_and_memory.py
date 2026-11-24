import json

import mlx.core as mx

from mlt.config import parse_config
from mlt.config.schema import MethodConfig
from mlt.export import export_model
from mlt.methods.apply import adapter_config, apply_method
from mlt.models.registry import get_family
from mlt.train.checkpoint import save_adapters, save_full_model
from mlt.train.memory import estimate

from .conftest import tiny_gpt

LLAMA = get_family("llama")


def test_export_merges_adapters(root, tokenizer_dir):
    base = tiny_gpt(vocab=400, d=64)
    base_dir = save_full_model(root / "base", base, base.cfg.to_hf_config(eos_token_id=0), tokenizer_dir)
    method = MethodConfig(type="lora", rank=4, alpha=8)
    apply_method(base, method, LLAMA)
    for _, mod in base.named_modules():
        if hasattr(mod, "lora_b"):
            mod.lora_b = mx.random.normal(mod.lora_b.shape) * 0.05
    x = mx.random.randint(0, 400, (1, 12))
    expected = base(x)
    adapters = save_adapters(
        root / "adapters", base, adapter_config(method, LLAMA, base), base_model="base",
        method={**method.model_dump(mode="json"), "base_path": "base"}, tokenizer_dir=tokenizer_dir,
    )  # fmt: skip
    out = export_model(adapters, root / "merged", log=lambda *_: None)
    from mlx_lm.utils import load_model

    merged, config = load_model(out)
    assert config["model_type"] == "llama"
    assert mx.allclose(expected, merged(x), atol=2e-3, rtol=1e-2)
    assert base_dir.exists() and (out / "tokenizer.json").exists()

    q = export_model(out, root / "q4", quantize_bits=4, group_size=32, log=lambda *_: None)
    assert json.loads((q / "config.json").read_text())["quantization"]["bits"] == 4


def _cfg(method, **train):
    return parse_config({
        "name": "m", "stage": "pretraining" if method == "full" else "sft",
        "model": {"from": "scratch", "arch": {"preset": "small"}, "tokenizer": {"reuse": "x"}} if method == "full" else {"from": "a/b"},
        "method": {"type": method},
        "data": {"train": {"source": "jsonl", "path": "x"}, "seq_len": 512},
        "train": {"steps": 1, **train},
    })  # fmt: skip


def test_memory_estimate_trends(tmp_path):
    small = estimate(_cfg("full"), None, 32000, [], mbs=4)
    ckpt = estimate(_cfg("full", grad_checkpoint=True), None, 32000, [], mbs=4)
    assert 0.03 < small["params_b"] < 0.06
    assert ckpt["activations_gb"] < small["activations_gb"]

    # A fake 2-layer HF checkpoint: estimates come from safetensors headers only.
    model = tiny_gpt(vocab=400, d=64)
    d = save_full_model(tmp_path / "m", model, model.cfg.to_hf_config(), tmp_path)
    targets = list(LLAMA.lora_targets)
    lora = estimate(_cfg("lora"), d, None, targets, mbs=1)
    qlora = estimate(_cfg("qlora"), d, None, targets, mbs=1)
    full = estimate(_cfg("lora").model_copy(update={"method": MethodConfig(type="full")}), d, None, targets, mbs=1)
    assert qlora["weights_gb"] < lora["weights_gb"]
    assert lora["trainable_m"] < full["trainable_m"]
    assert full["train_state_gb"] > lora["train_state_gb"]
