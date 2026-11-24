"""A minimal decoder-only GPT for pretraining from scratch.

The block is the modern Llama-style design: pre-norm RMSNorm, rotary position
embeddings, SwiGLU MLP, optional grouped-query attention and tied embeddings.
Parameter names and the exported `config.json` match Hugging Face's Llama format,
so a checkpoint saved from this model loads directly in mlx-lm (and transformers /
llama.cpp) as `model_type: llama`. That is what lets our own pretrained models go
through exactly the same continued-pretraining / SFT path as a downloaded model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..config.schema import ArchConfig


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    context_length: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_arch(cls, arch: ArchConfig, vocab_size: int) -> GPTConfig:
        a = arch.resolved()
        return cls(
            vocab_size=vocab_size,
            n_layers=a.n_layers,
            d_model=a.d_model,
            n_heads=a.n_heads,
            n_kv_heads=a.n_kv_heads,
            d_ff=a.d_ff,
            context_length=a.context_length,
            rope_theta=a.rope_theta,
            norm_eps=a.norm_eps,
            tie_embeddings=a.tie_embeddings,
            init_std=a.init_std,
        )

    def to_hf_config(
        self, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None
    ) -> dict:
        cfg = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": self.vocab_size,
            "hidden_size": self.d_model,
            "intermediate_size": self.d_ff,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.context_length,
            "rms_norm_eps": self.norm_eps,
            "rope_theta": self.rope_theta,
            "rope_traditional": False,
            "tie_word_embeddings": self.tie_embeddings,
            "attention_bias": False,
            "mlp_bias": False,
            "hidden_act": "silu",
            "torch_dtype": "bfloat16",
            "mlt_gpt": asdict(self),
        }
        if bos_token_id is not None:
            cfg["bos_token_id"] = bos_token_id
        if eos_token_id is not None:
            cfg["eos_token_id"] = eos_token_id
        return cfg


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        d = cfg.d_model
        self.q_proj = nn.Linear(d, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, d, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class Backbone(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)

    def __call__(self, inputs: mx.array) -> mx.array:
        h = self.embed_tokens(inputs)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.model_type = "llama"
        self.model = Backbone(cfg)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    @property
    def layers(self) -> list[Block]:
        return self.model.layers

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        if cache is not None:
            raise ValueError("GPT is a training model; export it and use mlx-lm for KV-cached generation")
        h = self.model(inputs)
        if self.cfg.tie_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def init_weights(self) -> None:
        """GPT-2 style init: N(0, std) everywhere, residual projections scaled by depth."""
        std = self.cfg.init_std
        resid_std = std / math.sqrt(2 * self.cfg.n_layers)
        new = []
        for name, p in tree_flatten(self.parameters()):
            if name.endswith("norm.weight"):
                new.append((name, mx.ones_like(p)))
            elif name.endswith(("o_proj.weight", "down_proj.weight")):
                new.append((name, mx.random.normal(p.shape) * resid_std))
            else:
                new.append((name, mx.random.normal(p.shape) * std))
        self.load_weights(new)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.trainable_parameters() if trainable_only else model.parameters()
    return sum(v.size for _, v in tree_flatten(params))
