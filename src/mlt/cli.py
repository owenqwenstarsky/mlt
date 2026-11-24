"""The `mlt` command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from .config.load import ConfigError, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
cluster_app = typer.Typer(no_args_is_help=True, help="Thunderbolt cluster setup and benchmarking.")
tokenizer_app = typer.Typer(no_args_is_help=True, help="Tokenizer utilities.")
app.add_typer(cluster_app, name="cluster")
app.add_typer(tokenizer_app, name="tokenizer")

ConfigArg = Annotated[Path, typer.Argument(help="Run config (YAML).", exists=True, dir_okay=False)]


def _config(path: Path):
    try:
        return load_config(path)
    except ConfigError as e:
        typer.secho(str(e), fg="red", err=True)
        raise typer.Exit(2) from None


def _plan(cfg):
    from .plan import build_plan

    return build_plan(cfg, log=lambda m: typer.echo(m))


def _is_multi_host(cfg) -> bool:
    return cfg.cluster is not None and len(cfg.cluster.hosts) > 1


# --------------------------------------------------------------------- train


@app.command()
def train(
    config: ConfigArg,
    resume: Annotated[bool, typer.Option(help="Continue from the run's latest checkpoint.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Prepare everything and print the memory estimate, then stop.")] = False,
    sync: Annotated[bool, typer.Option(help="Copy code, data and model to the other laptops first.")] = True,
) -> None:
    """Train: pretraining, continued_pretraining or sft (set by `stage` in the config)."""
    cfg = _config(config)
    plan = _plan(cfg)
    _print_memory(cfg, plan)
    if dry_run:
        return
    if _is_multi_host(cfg):
        from .dist.launch import launch_module, prepare_cluster
        from .plan import PLAN_FILE

        hostfile = prepare_cluster(cfg, plan, sync=sync, log=typer.echo)
        args = [str(plan.run_dir / PLAN_FILE)] + (["--resume"] if resume else [])
        raise typer.Exit(launch_module(hostfile, "mlt.worker", args, log=typer.echo))

    from .dist.world import single
    from .plan import PLAN_FILE
    from .train.loop import train as run_train

    plan.save(plan.run_dir / PLAN_FILE)
    run_train(plan, single(), resume=resume)


def _print_memory(cfg, plan) -> None:
    from .dist.cluster import local_info
    from .methods.apply import lora_targets
    from .models.registry import get_family
    from .paths import cluster_dir, from_root
    from .tokenization.load import load_tokenizer
    from .train.memory import estimate, model_dir_for

    vocab = len(load_tokenizer(from_root(plan.tokenizer_dir))) if cfg.model.is_scratch else None
    family = get_family("llama") if cfg.model.is_scratch else None
    if family is None:
        from .models.loading import family_for_path

        family = family_for_path(model_dir_for(plan.model_path))
    targets = lora_targets(cfg.method, family)
    known = {}
    hosts_file = cluster_dir() / "hosts.json"
    if hosts_file.exists():
        known = json.loads(hosts_file.read_text())
    local = local_info()
    typer.echo("memory estimate per laptop (rough):")
    for i, host in enumerate(plan.hosts):
        est = estimate(cfg, model_dir_for(plan.model_path), vocab, targets, plan.micro_batch_sizes[i])
        limit = known.get(host, {}).get("gpu_working_set_gb") or (local["gpu_working_set_gb"] if i == 0 else None)
        verdict = ""
        if limit:
            verdict = "  OK" if est["total_gb"] < 0.9 * limit else "  WARNING: likely exceeds"
            verdict += f" (GPU working set {limit:.1f} GB)"
        typer.echo(
            f"  {host}: weights {est['weights_gb']:.2f} + optimizer/grads {est['train_state_gb']:.2f} "
            f"+ activations {est['activations_gb']:.2f} = {est['total_gb']:.2f} GB{verdict}"
        )


# ------------------------------------------------------------------- prepare


@app.command()
def prepare(config: ConfigArg) -> None:
    """Download the model, resolve the tokenizer and tokenize the data (cached in .mlt/)."""
    cfg = _config(config)
    plan = _plan(cfg)
    typer.echo(json.dumps({k: v for k, v in plan.__dict__.items() if k != "config"}, indent=2))


@tokenizer_app.command("train")
def tokenizer_train(config: ConfigArg) -> None:
    """Train (or materialise a reused) tokenizer for a pretraining config."""
    from .tokenization.load import resolve_pretraining_tokenizer

    cfg = _config(config)
    if cfg.model.tokenizer is None:
        typer.secho("this config does not define model.tokenizer (only pretraining does)", fg="red", err=True)
        raise typer.Exit(2)
    path = resolve_pretraining_tokenizer(cfg.model.tokenizer, cfg.data.train, log=typer.echo)
    typer.echo(str(path))


# ------------------------------------------------------------------- cluster


@cluster_app.command("setup")
def cluster_setup(
    hosts: Annotated[str, typer.Option(help="Comma-separated ssh names; this laptop first.")],
    auto_setup: Annotated[bool, typer.Option(help="Let mlx configure the Thunderbolt bridge (runs sudo on each laptop).")] = False,
) -> None:
    """Discover the Thunderbolt topology and write .mlt/cluster/hostfile.json."""
    from .dist.cluster import run_setup

    raise typer.Exit(run_setup([h.strip() for h in hosts.split(",") if h.strip()], auto_setup, log=typer.echo))


@cluster_app.command("bench")
def cluster_bench(config: ConfigArg, sync: bool = True) -> None:
    """Measure each laptop's throughput for this config; `share: auto` then uses it."""
    cfg = _config(config)
    plan = _plan(cfg)
    if not _is_multi_host(cfg):
        typer.secho("config has no multi-host `cluster` section", fg="red", err=True)
        raise typer.Exit(2)
    from .dist.launch import launch_module, prepare_cluster
    from .plan import PLAN_FILE

    hostfile = prepare_cluster(cfg, plan, sync=sync, log=typer.echo)
    raise typer.Exit(launch_module(hostfile, "mlt.dist.bench", [str(plan.run_dir / PLAN_FILE)], log=typer.echo))


# -------------------------------------------------------------------- doctor


@app.command()
def doctor(
    config: Annotated[Path | None, typer.Argument(help="Config whose cluster.hosts to check.")] = None,
    local: Annotated[bool, typer.Option(help="Only report this laptop.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Check versions, code and GPU memory on every laptop."""
    from .dist.cluster import compare, local_info, remote_info
    from .paths import cluster_dir

    info = local_info()
    if local:
        typer.echo(json.dumps(info) if as_json else "\n".join(f"{k}: {v}" for k, v in info.items()))
        return
    infos = {"(this laptop)": info}
    if config is not None:
        cfg = _config(config)
        if cfg.cluster is not None:
            for i, h in enumerate(cfg.cluster.hosts):
                infos[h.host] = info if i == 0 else remote_info(h.host)
            hosts_file = cluster_dir() / "hosts.json"
            hosts_file.parent.mkdir(parents=True, exist_ok=True)
            hosts_file.write_text(json.dumps({h: i for h, i in infos.items() if not h.startswith("(")}, indent=2))
    if as_json:
        typer.echo(json.dumps(infos, indent=2))
    else:
        for h, i in infos.items():
            typer.echo(f"{h}:")
            for k, v in i.items():
                typer.echo(f"  {k}: {v}")
    problems = compare({h: i for h, i in infos.items() if not h.startswith("(")} or infos)
    for p in problems:
        typer.secho(f"problem: {p}", fg="yellow")
    ws = info.get("gpu_working_set_gb")
    typer.echo(
        f"note: macOS lets the GPU use ~{ws} GB here. mlt never changes this; raising it "
        "(sudo sysctl iogpu.wired_limit_mb=...) is possible but leaves less memory for macOS."
    )
    if problems:
        raise typer.Exit(1)


# -------------------------------------------------------------------- export


@app.command()
def export(
    src: Annotated[Path, typer.Argument(help="A run's final/ dir (model or adapters) or any model dir.", exists=True)],
    out: Annotated[Path, typer.Option(help="Output directory.")],
    dequantize: bool = False,
    quantize: Annotated[int | None, typer.Option(help="Quantize the exported model to this many bits.")] = None,
    group_size: int = 64,
) -> None:
    """Merge LoRA adapters into the base model and write a Hugging Face-format folder."""
    from .export import export_model

    export_model(src, out, dequantize=dequantize, quantize_bits=quantize, group_size=group_size, log=typer.echo)


@app.command()
def chat(
    model: Annotated[Path, typer.Argument(help="Model dir or adapter dir (e.g. runs/<name>/final).", exists=True)],
    max_tokens: int = 512,
    system: str | None = None,
) -> None:
    """Chat with a trained model to sanity-check it."""
    from .chat import repl

    repl(model, max_tokens, system)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
