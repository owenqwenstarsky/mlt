from __future__ import annotations

import json
import os
import random
from pathlib import Path

import mlx.core as mx
import pytest

from mlt.models.gpt import GPT, GPTConfig
from mlt.tokenization.train_bpe import train_bpe

WORDS = "the cat sat on a mat and the dog ran in the park while birds sang songs about tall green trees".split()


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MLT_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test (set MLT_NETWORK_TESTS=1)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("MLT_ROOT", str(tmp_path))
    return tmp_path


def make_corpus(path: Path, n_docs: int = 200, words: int = 50, seed: int = 0) -> Path:
    rng = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(n_docs):
            f.write(json.dumps({"text": " ".join(rng.choice(WORDS) for _ in range(words))}) + "\n")
    return path


def make_chat(path: Path, n: int = 100, seed: int = 0) -> Path:
    rng = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(n):
            a, b = rng.randint(0, 9), rng.randint(0, 9)
            msgs = [{"role": "user", "content": f"add {a} {b}"}, {"role": "assistant", "content": f"{a + b}"}]
            if rng.random() < 0.5:
                msgs = [{"role": "system", "content": "you add numbers"}] + msgs
                msgs += [{"role": "user", "content": "and double it"}, {"role": "assistant", "content": f"{2 * (a + b)}"}]
            f.write(json.dumps({"messages": msgs}) + "\n")
    return path


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory) -> Path:
    """A tiny byte-level BPE tokenizer with the ChatML template (built offline)."""
    out = tmp_path_factory.mktemp("tok") / "bpe"
    rng = random.Random(0)
    texts = [" ".join(rng.choice(WORDS) for _ in range(40)) for _ in range(200)]
    texts += [f"add {a} {b} = {a + b} you add numbers and double it" for a in range(10) for b in range(10)]
    train_bpe(texts, out, vocab_size=400, min_frequency=1)
    return out


@pytest.fixture(scope="session")
def tokenizer(tokenizer_dir):
    from mlt.tokenization.load import load_tokenizer

    return load_tokenizer(tokenizer_dir)


def tiny_gpt(vocab: int = 97, layers: int = 2, d: int = 64, heads: int = 4, kv: int = 4, tie: bool = True, seed: int = 0) -> GPT:
    mx.random.seed(seed)
    cfg = GPTConfig(vocab_size=vocab, n_layers=layers, d_model=d, n_heads=heads, n_kv_heads=kv, d_ff=128, context_length=64, tie_embeddings=tie)
    m = GPT(cfg)
    m.init_weights()
    mx.eval(m.parameters())
    return m
