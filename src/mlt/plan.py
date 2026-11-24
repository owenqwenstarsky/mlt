"""Resolve a config into a concrete run plan.

The coordinator (rank 0's laptop) builds the plan once: it downloads the model,
resolves or trains the tokenizer and tokenizes the data. The plan records where
those artifacts live (relative to the project root) and how each optimizer step's
batch is split across laptops. Workers only read the plan, so they never touch the
network or redo any preparation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config.load import dump_config, parse_config
from .config.schema import RunConfig
from .data.prepare import prepare_dataset
from .models.loading import download_model, family_for_path, resolve_model_path
from .paths import cluster_dir, from_root, relative_to_root
from .tokenization.load import resolve_pretraining_tokenizer

PLAN_FILE = "plan.json"


@dataclass
class RunPlan:
    config: dict
    model_path: str | None
    tokenizer_dir: str
    train_data: str
    val_data: str | None
    hosts: list[str]
    shares: list[float]
    micro_batch_sizes: list[int]
    family: str

    @property
    def run_config(self) -> RunConfig:
        return parse_config(self.config)

    @property
    def run_dir(self) -> Path:
        cfg = self.run_config
        return from_root(cfg.output_dir) / cfg.name

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> RunPlan:
        return cls(**json.loads(Path(path).read_text()))


def resolve_shares(cfg: RunConfig, log=print) -> list[float]:
    if cfg.cluster is None:
        return [1.0]
    n = len(cfg.cluster.hosts)
    share = cfg.cluster.share
    if isinstance(share, list):
        return list(share)
    if share == "auto":
        bench = cluster_dir() / "bench.json"
        if bench.exists():
            result = json.loads(bench.read_text())
            speeds = result.get("tokens_per_sec", {})
            names = [h.host for h in cfg.cluster.hosts]
            if all(name in speeds for name in names):
                if result.get("config") != cfg.name:
                    log(f"share: auto -> using benchmark measured for config {result.get('config')!r}")
                total = sum(speeds[name] for name in names)
                return [speeds[name] / total for name in names]
        log("share: auto -> no benchmark for every host yet (run `mlt cluster bench`); splitting equally")
    return [1.0 / n] * n


def build_plan(cfg: RunConfig, log=print) -> RunPlan:
    model_path: Path | None = None
    if cfg.model.is_scratch:
        tokenizer_dir = resolve_pretraining_tokenizer(cfg.model.tokenizer, cfg.data.train, log=log)
        family_name = "llama"
        chat_kwargs: dict = {}
    else:
        model_path = resolve_model_path(cfg.model.from_)
        if not (model_path / "config.json").exists():
            model_path = download_model(cfg.model.from_, cfg.model.revision, log=log)
        tokenizer_dir = model_path
        family = family_for_path(model_path)
        family_name = family.name
        chat_kwargs = family.chat_template_kwargs

    kind = cfg.data_kind
    train_data = prepare_dataset(cfg.data.train, kind, tokenizer_dir, cfg.data.train_on, chat_kwargs, log=log)
    val_data = None
    if cfg.data.val is not None:
        val_data = prepare_dataset(cfg.data.val, kind, tokenizer_dir, cfg.data.train_on, chat_kwargs, log=log)

    hosts = [h.host for h in cfg.cluster.hosts] if cfg.cluster else ["local"]
    return RunPlan(
        config=dump_config(cfg),
        model_path=relative_to_root(model_path) if model_path else None,
        tokenizer_dir=relative_to_root(tokenizer_dir),
        train_data=relative_to_root(train_data),
        val_data=relative_to_root(val_data) if val_data else None,
        hosts=hosts,
        shares=resolve_shares(cfg, log=log),
        micro_batch_sizes=cfg.micro_batch_sizes(),
        family=family_name,
    )


def plan_artifacts(plan: RunPlan) -> list[str]:
    """Paths (relative to the project root) every laptop needs before training."""
    paths = [plan.tokenizer_dir, plan.train_data]
    if plan.val_data:
        paths.append(plan.val_data)
    if plan.model_path:
        paths.append(plan.model_path)
    return sorted(set(paths))
