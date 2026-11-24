import json

import numpy as np
import pytest

from mlt.config.schema import DataSource
from mlt.data.batching import make_microbatches
from mlt.data.prepare import prepare_dataset
from mlt.data.sampler import GlobalSampler, rank_slice, split_counts
from mlt.data.shards import holdout_split, open_dataset, train_val_views
from mlt.data.sources import iter_conversations, iter_texts, to_messages

from .conftest import make_chat, make_corpus


def test_split_counts():
    assert split_counts(10, [0.5, 0.5]) == [5, 5]
    assert split_counts(10, [0.7, 0.3]) == [7, 3]
    assert sum(split_counts(33, [0.2, 0.5, 0.3])) == 33
    assert split_counts(3, [0.98, 0.01, 0.01]) == [1, 1, 1]
    assert split_counts(7, [1.0]) == [7]
    with pytest.raises(ValueError):
        split_counts(1, [0.5, 0.5])


def test_sampler_deterministic_and_independent_of_world_size():
    s = GlobalSampler(n_items=37, global_batch_size=8, seed=3)
    batches = [s.batch(i) for i in range(10)]
    assert all(len(b) == 8 for b in batches)
    first_epoch = np.concatenate(batches)[:37]
    assert sorted(first_epoch.tolist()) == list(range(37))  # each item once per epoch
    assert np.array_equal(GlobalSampler(37, 8, 3).batch(7), batches[7])
    assert not np.array_equal(GlobalSampler(37, 8, 4).batch(0), batches[0])
    for shares in ([1.0], [0.5, 0.5], [0.6, 0.1, 0.3]):
        counts = split_counts(8, shares)
        joined = np.concatenate([batches[4][rank_slice(counts, r)] for r in range(len(shares))])
        assert np.array_equal(joined, batches[4])
    assert s.steps_for_epochs(1.0) == 5


def test_holdout_split():
    tr, va = holdout_split(1000, 0.1)
    assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == 1000
    assert 60 < len(va) < 140
    tr2, va2 = holdout_split(1000, 0.1)
    assert np.array_equal(va, va2)
    assert len(holdout_split(10, 0.0)[1]) == 0


def test_text_prepare_windows_and_cache(root, tokenizer_dir):
    make_corpus(root / "c.jsonl", n_docs=30)
    src = DataSource(source="jsonl", path="c.jsonl", field="text")
    path = prepare_dataset(src, "text", tokenizer_dir, log=lambda *_: None)
    meta = json.loads((path / "meta.json").read_text())
    assert meta["n_docs"] == 30 and meta["dtype"] == "uint16"
    mtime = (path / "meta.json").stat().st_mtime
    assert prepare_dataset(src, "text", tokenizer_dir, log=lambda *_: None) == path
    assert (path / "meta.json").stat().st_mtime == mtime  # cached, not rebuilt
    ds = open_dataset(path, seq_len=16)
    toks = np.fromfile(path / "tokens.bin", dtype=np.uint16)
    assert len(ds) == (len(toks) - 1) // 16
    w, _ = ds.get(2)
    assert np.array_equal(w, toks[32:49])
    (inp, tgt, mask), = make_microbatches([ds.get(0), ds.get(1)], 2, 16)
    assert inp.shape == (2, 16) and np.array_equal(inp[:, 1:], tgt[:, :-1]) and mask.all()


def test_chat_prepare_and_batching(root, tokenizer_dir):
    make_chat(root / "chat.jsonl", n=20)
    src = DataSource(source="jsonl", path="chat.jsonl", field="messages")
    path = prepare_dataset(src, "chat", tokenizer_dir, log=lambda *_: None)
    meta = json.loads((path / "meta.json").read_text())
    assert meta["n_examples"] == 20 and meta["stats"]["skipped_turns"] == 0
    train, val = train_val_views(path, None, 0.2, seq_len=128)
    assert len(train) + len(val) == 20 and len(val) > 0
    items = [train.get(i) for i in range(3)]
    batches = make_microbatches(items, 2, 128, pad_multiple=8)
    assert [b[0].shape[0] for b in batches] == [2, 1]
    for inp, _tgt, _mask in batches:
        assert inp.shape[1] % 8 == 0
    # mask is shifted with the targets: mask[t] belongs to target token t+1
    toks, m = items[0]
    inp, tgt, mask = make_microbatches([items[0]], 1, 128, pad_multiple=8)[0]
    n = len(toks) - 1
    assert np.array_equal(mask[0, :n], m[1:].astype(np.float32)) and not mask[0, n:].any()


def test_chat_truncation_drops_examples_without_targets(root, tokenizer_dir):
    make_chat(root / "chat.jsonl", n=10)
    path = prepare_dataset(DataSource(source="jsonl", path="chat.jsonl", field="messages"), "chat", tokenizer_dir, log=lambda *_: None)
    from mlt.data.shards import Prepared

    first = Prepared(path).first_target
    assert first.min() < first.max()
    full = open_dataset(path, seq_len=512)
    short = open_dataset(path, seq_len=int(first.min()))
    assert 0 < len(short) < len(full) == len(first)
    with pytest.raises(ValueError, match="no conversation"):
        open_dataset(path, seq_len=2)


def test_sources(root):
    (root / "docs").mkdir()
    (root / "docs" / "a.txt").write_text("hello")
    (root / "docs" / "b.txt").write_text("world")
    assert list(iter_texts(DataSource(source="txt", path="docs"))) == ["hello", "world"]
    (root / "p.jsonl").write_text(json.dumps({"prompt": "q", "completion": "a", "system": "s"}) + "\n")
    convs = list(iter_conversations(DataSource(source="jsonl", path="p.jsonl", field="messages")))
    assert [m["role"] for m in convs[0]] == ["system", "user", "assistant"]
    with pytest.raises(ValueError):
        to_messages({"x": 1}, "messages")
    make_corpus(root / "c.jsonl", n_docs=5)
    assert len(list(iter_texts(DataSource(source="jsonl", path="c.jsonl", max_samples=3)))) == 3
