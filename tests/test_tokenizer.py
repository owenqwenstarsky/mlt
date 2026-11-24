from mlt.tokenization.chat_template import IM_END, IM_START
from mlt.tokenization.load import (
    ensure_chat_support,
    generation_eos_ids,
    load_tokenizer,
    tokenizer_fingerprint,
)
from mlt.tokenization.train_bpe import train_bpe


def test_trained_bpe_roundtrip_and_chat(tokenizer):
    text = "the cat sat on a mat, 42 times!"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(ids) == text
    vocab = tokenizer.get_vocab()
    assert IM_START in vocab and IM_END in vocab and tokenizer.eos_token == "<|endoftext|>"
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
    assert rendered == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert generation_eos_ids(tokenizer) == [vocab["<|endoftext|>"], vocab[IM_END]]


def test_ensure_chat_support_adds_tokens(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(["hello world"] * 10, trainers.BpeTrainer(vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok = PreTrainedTokenizerFast(tokenizer_object=t)
    n = len(tok)
    assert ensure_chat_support(tok)
    assert len(tok) == n + 3 and tok.eos_token == "<|endoftext|>" and tok.chat_template
    assert not ensure_chat_support(tok)


def test_fingerprint_changes_with_content(tmp_path):
    a = train_bpe(["aaa bbb"] * 20, tmp_path / "a", vocab_size=300, min_frequency=1)
    b = train_bpe(["ccc ddd eee"] * 20, tmp_path / "b", vocab_size=300, min_frequency=1)
    assert tokenizer_fingerprint(a) != tokenizer_fingerprint(b)
    assert tokenizer_fingerprint(a) == tokenizer_fingerprint(a)
    assert load_tokenizer(a).decode(load_tokenizer(a)("aaa")["input_ids"]) == "aaa"
