from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mlt.cli import app
from mlt.config import load_config

from .conftest import make_corpus

EXAMPLES = sorted((Path(__file__).parent.parent / "configs" / "examples").glob("*.yaml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.name


def test_train_dry_run_and_local_training(root, tokenizer_dir):
    make_corpus(root / "c.jsonl", n_docs=40)
    cfg = {
        "name": "cli", "stage": "pretraining",
        "model": {"from": "scratch", "arch": {"n_layers": 1, "d_model": 32, "n_heads": 2, "context_length": 32},
                  "tokenizer": {"path": str(tokenizer_dir)}},
        "data": {"train": {"source": "jsonl", "path": "c.jsonl"}, "seq_len": 32},
        "train": {"steps": 3, "global_batch_size": 2, "log_every": 1},
    }  # fmt: skip
    path = root / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runner = CliRunner()
    res = runner.invoke(app, ["train", str(path), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "memory estimate" in res.output and not (root / "runs" / "cli" / "final").exists()
    res = runner.invoke(app, ["train", str(path)])
    assert res.exit_code == 0, res.output
    assert "step 3/3" in res.output and (root / "runs" / "cli" / "final" / "config.json").exists()


def test_bad_config_reports_cleanly(root):
    path = root / "bad.yaml"
    path.write_text(yaml.safe_dump({"name": "x", "stage": "sft", "model": {"from": "scratch"}, "data": {"train": {"source": "jsonl", "path": "x"}}, "train": {"steps": 1}}))
    res = CliRunner().invoke(app, ["train", str(path)])
    assert res.exit_code == 2 and "not `scratch`" in res.output


def test_doctor_local_json(root):
    res = CliRunner().invoke(app, ["doctor", "--local", "--json"])
    assert res.exit_code == 0 and '"mlx"' in res.output
