import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlt.config.schema import MethodConfig
from mlt.export import merge_lora
from mlt.methods.apply import apply_method
from mlt.models.registry import get_family

from .conftest import tiny_gpt

LLAMA = get_family("llama")


def test_only_adapters_trainable_and_identity_at_init():
    m = tiny_gpt()
    x = mx.random.randint(0, 97, (2, 8))
    before = m(x)
    apply_method(m, MethodConfig(type="lora", rank=4, alpha=8), LLAMA)
    names = [k for k, _ in tree_flatten(m.trainable_parameters())]
    assert names and all(k.endswith(("lora_a", "lora_b")) for k in names)
    assert len(names) == 2 * 7 * 2  # a+b, 7 projections, 2 layers
    assert mx.allclose(before, m(x), atol=1e-6)  # lora_b starts at zero


def test_layers_and_custom_targets():
    m = tiny_gpt()
    apply_method(m, MethodConfig(type="lora", targets=["self_attn.q_proj", "self_attn.v_proj"], layers=1), LLAMA)
    names = [k for k, _ in tree_flatten(m.trainable_parameters())]
    assert all(".layers.1." in k for k in names) and len(names) == 4


def test_unknown_target_rejected():
    with pytest.raises(ValueError, match="not found"):
        apply_method(tiny_gpt(), MethodConfig(type="lora", targets=["self_attn.nope"]), LLAMA)


def _perturb_adapters(m):
    for _, mod in m.named_modules():
        if hasattr(mod, "lora_b"):
            mod.lora_b = mx.random.normal(mod.lora_b.shape) * 0.05


def test_merge_matches_unmerged():
    m = tiny_gpt()
    apply_method(m, MethodConfig(type="lora", rank=4, alpha=8), LLAMA)
    _perturb_adapters(m)
    x = mx.random.randint(0, 97, (2, 8))
    y = m(x)
    merge_lora(m, dequantize=False)
    assert not any(hasattr(mod, "lora_b") for _, mod in m.named_modules())
    # fp32 GPU matmuls on Apple silicon are not exact; compare at their precision
    assert mx.allclose(y, m(x), atol=2e-3, rtol=1e-2)


def test_qlora_adapters_on_quantized_base():
    m = tiny_gpt()
    nn.quantize(m, group_size=32, bits=4, class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and mod.weight.shape[-1] % 32 == 0)
    assert isinstance(m.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    x = mx.random.randint(0, 97, (2, 8))
    before = m(x)
    apply_method(m, MethodConfig(type="qlora", rank=4, alpha=8, group_size=32), LLAMA)
    names = [k for k, _ in tree_flatten(m.trainable_parameters())]
    assert names and all(k.endswith(("lora_a", "lora_b")) for k in names)
    assert mx.allclose(before, m(x), atol=1e-5)
    _perturb_adapters(m)
    loss = lambda model: model(x).sum()  # noqa: E731
    _, grads = nn.value_and_grad(m, loss)(m)
    assert all(mx.abs(g).sum().item() > 0 for _, g in tree_flatten(grads))
