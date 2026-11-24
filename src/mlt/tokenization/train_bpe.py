"""Train a byte-level BPE tokenizer and save it in Hugging Face format."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from .chat_template import CHATML_TEMPLATE, END_OF_TEXT, IM_END, IM_START

SPECIAL_TOKENS = [END_OF_TEXT, IM_START, IM_END]


def _limit(texts: Iterable[str], max_docs: int | None) -> Iterator[str]:
    for i, t in enumerate(texts):
        if max_docs is not None and i >= max_docs:
            return
        yield t


def train_bpe(
    texts: Iterable[str],
    out_dir: Path,
    vocab_size: int,
    min_frequency: int = 2,
    max_docs: int | None = None,
    show_progress: bool = False,
) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tok.train_from_iterator(_limit(texts, max_docs), trainer)

    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=END_OF_TEXT)
    fast.chat_template = CHATML_TEMPLATE
    out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out_dir))
    return out_dir
