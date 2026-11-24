import os

import pytest

from mlt.data.chat import ChatStats, encode_conversation


def _decoded_targets(tok, ids, mask):
    """Decode each contiguous run of target tokens."""
    runs, cur = [], []
    for i, m in zip(ids, mask, strict=True):
        if m:
            cur.append(int(i))
        elif cur:
            runs.append(tok.decode(cur))
            cur = []
    if cur:
        runs.append(tok.decode(cur))
    return runs


def test_single_turn_masks_only_reply(tokenizer):
    msgs = [{"role": "user", "content": "add 2 3"}, {"role": "assistant", "content": "5"}]
    ids, mask = encode_conversation(tokenizer, msgs)
    assert _decoded_targets(tokenizer, ids, mask) == ["5<|im_end|>"]


def test_multi_turn_and_system(tokenizer):
    msgs = [
        {"role": "system", "content": "you add numbers"},
        {"role": "user", "content": "add 2 3"},
        {"role": "assistant", "content": "5"},
        {"role": "user", "content": "and double it"},
        {"role": "assistant", "content": "10"},
    ]
    stats = ChatStats()
    ids, mask = encode_conversation(tokenizer, msgs, stats=stats)
    assert _decoded_targets(tokenizer, ids, mask) == ["5<|im_end|>", "10<|im_end|>"]
    assert stats.assistant_turns == 2 and stats.skipped_turns == 0


def test_train_on_all(tokenizer):
    msgs = [{"role": "user", "content": "add 2 3"}, {"role": "assistant", "content": "5"}]
    ids, mask = encode_conversation(tokenizer, msgs, train_on="all")
    assert mask[0] == 0 and mask[1:].all()


def test_no_assistant_is_dropped(tokenizer):
    stats = ChatStats()
    assert encode_conversation(tokenizer, [{"role": "user", "content": "hi"}], stats=stats) is None
    assert stats.dropped == 1


class _RewritingTemplate:
    """A template that renders earlier assistant turns differently from the latest one
    (like Qwen3's think blocks): earlier replies are not a prefix of the full text."""

    def __init__(self, tok):
        self.tok = tok
        self.tmpl = (
            "{%- for m in messages %}<|im_start|>{{ m.role }}\n"
            "{%- if m.role == 'assistant' and not loop.last %}[old]{% endif %}"
            "{{ m.content }}<|im_end|>\n{%- endfor %}"
            "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
        )

    def __getattr__(self, name):
        return getattr(self.tok, name)

    def apply_chat_template(self, messages, **kw):
        return self.tok.apply_chat_template(messages, chat_template=self.tmpl, **kw)

    def __call__(self, *a, **kw):
        return self.tok(*a, **kw)


def test_template_rewrites_are_skipped_not_mislabelled(tokenizer):
    tok = _RewritingTemplate(tokenizer)
    msgs = [
        {"role": "user", "content": "add 1 1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "add 2 2"},
        {"role": "assistant", "content": "4"},
    ]
    stats = ChatStats()
    ids, mask = encode_conversation(tok, msgs, stats=stats)
    targets = _decoded_targets(tokenizer, ids, mask)
    assert targets[-1] == "4<|im_end|>"
    assert all("add" not in t for t in targets)


@pytest.mark.network
def test_qwen3_template():
    from mlt.tokenization.load import load_tokenizer

    tok = load_tokenizer(os.environ.get("MLT_QWEN_TOKENIZER", "Qwen/Qwen3-0.6B"))
    msgs = [
        {"role": "user", "content": "add 2 3"},
        {"role": "assistant", "content": "5"},
        {"role": "user", "content": "double"},
        {"role": "assistant", "content": "10"},
    ]
    stats = ChatStats()
    ids, mask = encode_conversation(tok, msgs, stats=stats)
    targets = _decoded_targets(tok, ids, mask)
    assert targets[0] == "5<|im_end|>"
    assert targets[-1].endswith("10<|im_end|>") and "<think>" in targets[-1]
    assert stats.skipped_turns == 0
