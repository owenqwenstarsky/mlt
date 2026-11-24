import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama

from mlt.config import parse_config
from mlt.models.gpt import count_parameters
from mlt.train.checkpoint import save_full_model
from mlt.train.memory import _scratch_shape

from .conftest import tiny_gpt


def _reference(model):
    hf = model.cfg.to_hf_config()
    hf.pop("mlt_gpt")
    ref = llama.Model(llama.ModelArgs.from_dict(hf))
    ref.load_weights(tree_flatten(model.parameters()), strict=True)
    return ref


@pytest.mark.parametrize("tie,kv", [(True, 4), (False, 2), (True, 1)])
def test_matches_mlx_lm_llama(tie, kv):
    m = tiny_gpt(tie=tie, kv=kv)
    x = mx.random.randint(0, 97, (2, 16))
    assert mx.allclose(m(x), _reference(m)(x), atol=1e-5)


def test_output_shape_and_causality():
    m = tiny_gpt()
    x = mx.random.randint(0, 97, (1, 12))
    y = m(x)
    assert y.shape == (1, 12, 97)
    x2 = mx.concatenate([x[:, :6], (x[:, 6:] + 1) % 97], axis=1)
    y2 = m(x2)
    assert mx.allclose(y[:, :6], y2[:, :6], atol=1e-5)
    assert not mx.allclose(y[:, 6:], y2[:, 6:], atol=1e-5)


def test_parameter_formula_matches_model():
    cfg = parse_config({
        "name": "t", "stage": "pretraining",
        "model": {"from": "scratch", "arch": {"n_layers": 3, "d_model": 64, "n_heads": 4, "n_kv_heads": 2, "context_length": 64, "tie_embeddings": False}, "tokenizer": {"reuse": "x"}},
        "data": {"train": {"source": "jsonl", "path": "x"}, "seq_len": 32}, "train": {"steps": 1},
    })  # fmt: skip
    from mlt.models.gpt import GPT, GPTConfig

    m = GPT(GPTConfig.from_arch(cfg.model.arch, vocab_size=123))
    assert _scratch_shape(cfg, 123).n_params == count_parameters(m)


def test_export_loads_in_mlx_lm(tmp_path, tokenizer_dir):
    from mlx_lm.utils import load_model

    m = tiny_gpt(vocab=400)
    out = save_full_model(tmp_path / "final", m, m.cfg.to_hf_config(eos_token_id=0), tokenizer_dir)
    loaded, config = load_model(out)
    assert config["model_type"] == "llama"
    x = mx.random.randint(0, 400, (1, 10))
    assert mx.allclose(m(x), loaded(x), atol=1e-5)
    assert (out / "tokenizer.json").exists()
