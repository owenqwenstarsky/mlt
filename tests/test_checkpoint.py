import shutil

import mlx.core as mx
import pytest

from mlt.config import parse_config
from mlt.dist.world import single
from mlt.plan import build_plan
from mlt.train.checkpoint import list_checkpoints, load_checkpoint
from mlt.train.loop import ResumeError, train

from .conftest import make_chat, make_corpus

QUIET = dict(log_every=100)


def _pretrain_cfg(tokenizer_dir, out, **train_over):
    return parse_config({
        "name": "ckpt", "stage": "pretraining", "output_dir": out,
        "model": {"from": "scratch", "arch": {"n_layers": 2, "d_model": 64, "n_heads": 4, "context_length": 32},
                  "tokenizer": {"path": str(tokenizer_dir)}},
        "data": {"train": {"source": "jsonl", "path": "c.jsonl"}, "seq_len": 32},
        "train": {"steps": 10, "global_batch_size": 4, "micro_batch_size": 2, "lr": 1e-3, "warmup_steps": 2, **QUIET, **train_over},
    })  # fmt: skip


def _final_weights(run_dir):
    return load_checkpoint(list_checkpoints(run_dir)[-1])[0]


def _cut_back_to(run_dir, step):
    for p in list_checkpoints(run_dir):
        if int(p.name.split("_")[1]) > step:
            shutil.rmtree(p)
    shutil.rmtree(run_dir / "final")


# On the GPU, MLX accumulates embedding gradients with atomic adds, so two runs are
# not bit-identical; in fp32 the noise is ~1e-7, in bf16 it is visible. fp32 must
# match tightly; bf16/mixed must stay within a few Adam steps' worth of noise.
@pytest.mark.parametrize("precision,atol", [("fp32", 1e-5), ("mixed", 5e-4), ("bf16", 5e-3)])
def test_resume_equals_uninterrupted(root, tokenizer_dir, precision, atol):
    make_corpus(root / "c.jsonl", n_docs=60)
    a = _pretrain_cfg(tokenizer_dir, "runs_a", precision=precision)
    pa = build_plan(a, log=lambda *_: None)
    train(pa, single())

    b = _pretrain_cfg(tokenizer_dir, "runs_b", precision=precision, save_every=4)
    pb = build_plan(b, log=lambda *_: None)
    train(pb, single())
    _cut_back_to(pb.run_dir, 4)
    train(pb, single(), resume=True)

    wa, wb = _final_weights(pa.run_dir), _final_weights(pb.run_dir)
    assert wa.keys() == wb.keys()
    for k in wa:
        assert mx.allclose(wa[k], wb[k], atol=atol), (k, mx.abs(wa[k] - wb[k]).max().item())
    assert (pb.run_dir / "final" / "config.json").exists()


def test_resume_refuses_changed_config(root, tokenizer_dir):
    make_corpus(root / "c.jsonl", n_docs=60)
    cfg = _pretrain_cfg(tokenizer_dir, "runs", save_every=4)
    plan = build_plan(cfg, log=lambda *_: None)
    train(plan, single())
    changed = build_plan(_pretrain_cfg(tokenizer_dir, "runs", save_every=4, lr=5e-3), log=lambda *_: None)
    with pytest.raises(ResumeError, match="train.lr"):
        train(changed, single(), resume=True)


def test_resume_without_checkpoint_errors(root, tokenizer_dir):
    make_corpus(root / "c.jsonl", n_docs=60)
    plan = build_plan(_pretrain_cfg(tokenizer_dir, "runs"), log=lambda *_: None)
    with pytest.raises(ResumeError, match="no checkpoint"):
        train(plan, single(), resume=True)


def test_lora_sft_resume(root, tokenizer_dir):
    make_corpus(root / "c.jsonl", n_docs=60)
    train(build_plan(_pretrain_cfg(tokenizer_dir, "runs"), log=lambda *_: None), single())
    make_chat(root / "chat.jsonl", n=40)

    def sft(out, **over):
        return build_plan(parse_config({
            "name": "sft", "stage": "sft", "output_dir": out, "model": {"from": "runs/ckpt/final"},
            "method": {"type": "lora", "rank": 4, "alpha": 8},
            "data": {"train": {"source": "jsonl", "path": "chat.jsonl"}, "seq_len": 64},
            "train": {"steps": 6, "global_batch_size": 4, "micro_batch_size": 2, "lr": 1e-2, **QUIET, **over},
        }), log=lambda *_: None)  # fmt: skip

    pa = sft("sft_a")
    train(pa, single())
    pb = sft("sft_b", save_every=3)
    train(pb, single())
    _cut_back_to(pb.run_dir, 3)
    train(pb, single(), resume=True)
    wa, wb = _final_weights(pa.run_dir), _final_weights(pb.run_dir)
    assert all(k.endswith(("lora_a", "lora_b")) for k in wa)
    for k in wa:
        assert mx.allclose(wa[k], wb[k], atol=1e-6), k
    assert (pb.run_dir / "final" / "adapters.safetensors").exists()
