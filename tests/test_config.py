import pytest

from mlt.config import ConfigError, parse_config, trajectory_hash
from mlt.config.schema import ArchConfig


def base(**over):
    cfg = {
        "name": "t",
        "stage": "sft",
        "model": {"from": "Qwen/Qwen3-0.6B"},
        "method": {"type": "lora"},
        "data": {"train": {"source": "jsonl", "path": "x.jsonl"}},
        "train": {"steps": 10},
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


def pretrain(**over):
    defaults = dict(
        stage="pretraining",
        model={"from": "scratch", "arch": {"preset": "tiny"}, "tokenizer": {"reuse": "gpt2"}},
        method={"type": "full"},
        data={"train": {"source": "jsonl", "path": "x.jsonl"}, "seq_len": 256},
    )
    return base(**{**defaults, **over})


def test_stage_defaults():
    sft = parse_config(base())
    assert sft.train.precision == "bf16" and sft.train.weight_decay == 0.0
    assert sft.data.train.field == "messages" and sft.data_kind == "chat"
    pt = parse_config(pretrain())
    assert pt.train.precision == "mixed" and pt.train.weight_decay == 0.1
    assert pt.data.train.field == "text" and pt.data_kind == "text"
    assert pt.model.arch.n_layers == 4 and pt.model.arch.d_ff % 64 == 0


@pytest.mark.parametrize(
    "raw, msg",
    [
        (base(stage="pretraining"), "starts from `scratch`"),
        (pretrain(method={"type": "lora"}), "method.type must be `full`"),
        (base(model={"from": "scratch"}), "not `scratch`"),
        (base(stage="continued_pretraining", model={"from": "x/y", "arch": {"preset": "tiny"}}), "come from the checkpoint"),
        (base(train={"steps": 1, "epochs": 1}), "exactly one of `steps` or `epochs`"),
        (base(train={}), "exactly one of `steps` or `epochs`"),
        (base(stage="continued_pretraining", data={"train": {"source": "jsonl", "path": "x"}, "train_on": "all"}), "train_on"),
        (base(train={"steps": 1, "precision": "mixed"}), "precision `mixed`"),
        (pretrain(data={"train": {"source": "jsonl", "path": "x"}, "seq_len": 4096}), "exceeds arch.context_length"),
        (base(cluster={"hosts": [{"host": "a"}, {"host": "b"}], "share": [0.5, 0.6]}), "sum to 1"),
        (base(cluster={"hosts": [{"host": "a"}, {"host": "a"}]}), "duplicate"),
        (base(model={"from": "scratch", "tokenizer": {"reuse": "a", "path": "b"}}), "exactly one of"),
        (base(unknown=1), "Extra inputs"),
    ],
)
def test_rejects(raw, msg):
    with pytest.raises(ConfigError, match=msg):
        parse_config(raw)


def test_arch_resolution():
    a = ArchConfig(n_layers=2, d_model=96, n_heads=6).resolved()
    assert a.n_kv_heads == 6 and a.d_ff == 256 and a.context_length == 1024
    with pytest.raises(ValueError):
        ArchConfig(n_layers=2, d_model=100, n_heads=3).resolved()
    with pytest.raises(ValueError, match="unknown preset"):
        ArchConfig(preset="huge").resolved()


def test_micro_batch_sizes():
    cfg = parse_config(base(train={"steps": 1, "micro_batch_size": 3}, cluster={"hosts": [{"host": "a", "micro_batch_size": 4}, {"host": "b"}]}))
    assert cfg.micro_batch_sizes() == [4, 3]


def test_trajectory_hash_ignores_where_and_how_often():
    h = trajectory_hash(parse_config(base()))
    assert h == trajectory_hash(parse_config(base(train={"steps": 10, "log_every": 1, "save_every": 5, "micro_batch_size": 7})))
    assert h == trajectory_hash(parse_config(base(cluster={"hosts": [{"host": "a"}, {"host": "b"}]})))
    assert h != trajectory_hash(parse_config(base(train={"steps": 10, "lr": 1e-3})))
