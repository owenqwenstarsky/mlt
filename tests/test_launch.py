import json

import pytest

from mlt.config import parse_config
from mlt.dist.launch import (
    apply_rank_order,
    launch_command,
    rank_order,
    read_hostfile,
    write_run_hostfile,
)
from mlt.plan import RunPlan


def _cfg(hosts):
    return parse_config({
        "name": "t", "stage": "sft", "model": {"from": "a/b"}, "method": {"type": "lora"},
        "data": {"train": {"source": "jsonl", "path": "x"}}, "train": {"steps": 1, "global_batch_size": 8},
        "cluster": {"hosts": [{"host": h} for h in hosts], "share": "equal"},
    })  # fmt: skip


def _hostfile(tmp_path, ring):
    path = tmp_path / "hostfile.json"
    path.write_text(json.dumps({"backend": "ring", "envs": [], "hosts": [{"ssh": h, "ips": [f"10.0.0.{i}"]} for i, h in enumerate(ring)]}))
    return path


def test_rank_order_rotates_ring_to_put_this_laptop_first(tmp_path):
    cfg = _cfg(["me", "b", "c"])
    _, entries = read_hostfile(_hostfile(tmp_path, ["b", "me", "c"]))
    order = rank_order(cfg, entries)
    # ring b -> me -> c, rotated to start at me: me, c, b (neighbours preserved)
    assert [cfg.cluster.hosts[i].host for i in order] == ["me", "c", "b"]


def test_rank_order_rejects_mismatched_hosts(tmp_path):
    _, entries = read_hostfile(_hostfile(tmp_path, ["me", "x"]))
    with pytest.raises(ValueError, match="differ"):
        rank_order(_cfg(["me", "b"]), entries)


def test_missing_hostfile_explains(tmp_path):
    with pytest.raises(FileNotFoundError, match="mlt cluster setup"):
        read_hostfile(tmp_path / "nope.json")


def test_run_hostfile_and_plan_order(tmp_path):
    cfg = _cfg(["me", "b"])
    meta, entries = read_hostfile(_hostfile(tmp_path, ["b", "me"]))
    order = rank_order(cfg, entries)
    out = write_run_hostfile(meta, entries, cfg, order, tmp_path / "run.json")
    hosts = json.loads(out.read_text())["hosts"]
    assert hosts[0]["ssh"] == "127.0.0.1" and hosts[0]["ips"] == ["10.0.0.1"]
    assert hosts[1]["ssh"] == "b"
    plan = RunPlan(config={}, model_path=None, tokenizer_dir="t", train_data="d", val_data=None,
                   hosts=["me", "b"], shares=[0.7, 0.3], micro_batch_sizes=[4, 2], family="qwen")  # fmt: skip
    apply_rank_order(plan, order)
    assert plan.hosts == ["me", "b"] and plan.shares == [0.7, 0.3]


def test_user_prefixed_hosts(tmp_path):
    cfg = _cfg(["owen@me", "owen@b"])
    _, entries = read_hostfile(_hostfile(tmp_path, ["owen@b", "owen@me"]))
    assert rank_order(cfg, entries) == [0, 1]
    assert entries[0].transfer_address == "owen@10.0.0.0"


def test_launch_command(root, tmp_path):
    cmd = launch_command(tmp_path / "h.json", ["-m", "mlt.worker", "plan.json"])
    assert cmd[0].endswith("mlx.launch") and "--backend" in cmd and cmd[-3:] == ["-m", "mlt.worker", "plan.json"]
    assert f"MLT_ROOT={root}" in cmd


def test_prepare_cluster_without_sync(root, tokenizer_dir):
    from mlt.dist.launch import prepare_cluster
    from mlt.plan import PLAN_FILE, build_plan

    from .conftest import make_corpus

    make_corpus(root / "c.jsonl", n_docs=20)
    (root / ".mlt" / "cluster").mkdir(parents=True)
    _hostfile(root / ".mlt" / "cluster", ["b", "me"])
    cfg = parse_config({
        "name": "cl", "stage": "pretraining",
        "model": {"from": "scratch", "arch": {"n_layers": 1, "d_model": 32, "n_heads": 2, "context_length": 32},
                  "tokenizer": {"path": str(tokenizer_dir)}},
        "data": {"train": {"source": "jsonl", "path": "c.jsonl"}, "seq_len": 32},
        "train": {"steps": 1, "global_batch_size": 4},
        "cluster": {"hosts": [{"host": "me", "micro_batch_size": 4}, {"host": "b", "micro_batch_size": 1}], "share": [0.75, 0.25]},
    })  # fmt: skip
    plan = build_plan(cfg, log=lambda *_: None)
    hostfile = prepare_cluster(cfg, plan, sync=False, log=lambda *_: None)
    saved = RunPlan.load(plan.run_dir / PLAN_FILE)
    assert saved.hosts == ["me", "b"] and saved.shares == [0.75, 0.25] and saved.micro_batch_sizes == [4, 1]
    assert json.loads(hostfile.read_text())["hosts"][0]["ssh"] == "127.0.0.1"
