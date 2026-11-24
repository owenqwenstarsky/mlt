"""Tokenizer resolution and loading.

Every tokenizer mlt uses ends up as a local Hugging Face tokenizer directory, so
data preparation, training and export all go through one code path.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..config.schema import DataSource, TokenizerConfig
from ..paths import files_hash, from_root, stable_hash, tokenizers_dir
from .chat_template import CHATML_TEMPLATE, END_OF_TEXT, IM_END, IM_START

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


def load_tokenizer(path: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


def tokenizer_fingerprint(path: str | Path) -> str:
    path = Path(path)
    files = [path / f for f in TOKENIZER_FILES if (path / f).exists()]
    if not files:
        raise FileNotFoundError(f"no tokenizer files in {path}")
    return files_hash(files)


def copy_tokenizer_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in TOKENIZER_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)


def ensure_chat_support(tok: Any) -> bool:
    """Give a tokenizer without a chat template ChatML tokens and template.

    Returns True if the tokenizer was modified. Done before a from-scratch model is
    built, so the vocabulary already contains the chat tokens and the model can go
    straight into SFT later.
    """
    changed = False
    if tok.eos_token is None:
        tok.add_tokens([END_OF_TEXT], special_tokens=True)
        tok.eos_token = END_OF_TEXT
        changed = True
    if not getattr(tok, "chat_template", None):
        vocab = tok.get_vocab()
        missing = [t for t in (IM_START, IM_END) if t not in vocab]
        if missing:
            tok.add_tokens(missing, special_tokens=True)
        tok.chat_template = CHATML_TEMPLATE
        changed = True
    return changed


def generation_eos_ids(tok: Any) -> list[int]:
    """Token ids that should stop generation: eos plus an end-of-turn token if present."""
    ids = []
    if tok.eos_token_id is not None:
        ids.append(int(tok.eos_token_id))
    vocab = tok.get_vocab()
    for t in (IM_END, "<|eot_id|>", "<|end|>"):
        if t in vocab and vocab[t] not in ids:
            ids.append(int(vocab[t]))
    return ids


def resolve_pretraining_tokenizer(
    cfg: TokenizerConfig, train_source: DataSource, log=print
) -> Path:
    """Materialise the tokenizer for a from-scratch run under `.mlt/tokenizers`."""
    if cfg.train is not None:
        corpus = cfg.train.corpus or train_source
        key = stable_hash({"train": cfg.train.model_dump(mode="json"), "corpus": corpus.model_dump()})
        out = tokenizers_dir() / f"bpe-{key}"
        if (out / "tokenizer.json").exists():
            return out
        from ..data.sources import iter_texts
        from .train_bpe import train_bpe

        log(f"training BPE tokenizer (vocab {cfg.train.vocab_size}) -> {out}")
        tmp = out.with_name(out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        train_bpe(
            iter_texts(corpus),
            tmp,
            vocab_size=cfg.train.vocab_size,
            min_frequency=cfg.train.min_frequency,
            max_docs=cfg.train.max_docs,
        )
        tmp.rename(out)
        return out

    source = cfg.reuse if cfg.reuse is not None else str(from_root(cfg.path))
    ident: dict[str, str] = {"source": source}
    if cfg.path is not None:  # local files can change; key on their content
        ident["content"] = tokenizer_fingerprint(source)
    key = stable_hash(ident)
    safe = source.strip("/").replace("/", "--")[-60:]
    out = tokenizers_dir() / f"reuse-{safe}-{key}"
    if (out / "tokenizer_config.json").exists():
        return out
    log(f"loading tokenizer {source}")
    tok = load_tokenizer(source)
    ensure_chat_support(tok)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tok.save_pretrained(str(tmp))
    tmp.rename(out)
    return out
