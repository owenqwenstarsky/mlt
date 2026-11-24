"""Measure each laptop's training throughput and the link's all-reduce bandwidth.

Runs under `mlx.launch` (python -m mlt.dist.bench <plan.json>). Each rank builds the
run's model, times forward+backward on synthetic batches at its own micro-batch
size, then all ranks time an all_sum. Rank 0 writes `.mlt/cluster/bench.json`; with
`share: auto`, training splits each batch in proportion to the measured speeds.
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from ..dist.world import init_world
from ..methods.apply import apply_method
from ..models.loading import build_scratch, load_pretrained
from ..paths import cluster_dir, from_root
from ..plan import RunPlan
from ..train.step import StepFunctions, enable_grad_checkpointing, local_gradients


def throughput(plan: RunPlan, rank: int, steps: int = 5) -> float:
    cfg = plan.run_config
    tc = cfg.train
    if cfg.model.is_scratch:
        lm = build_scratch(cfg.model.arch, from_root(plan.tokenizer_dir), tc.precision)
    else:
        lm = load_pretrained(from_root(plan.model_path), cfg.method, tc.precision)
    apply_method(lm.model, cfg.method, lm.family)
    if tc.grad_checkpoint:
        enable_grad_checkpointing(lm.model)
    lm.model.train()
    fns = StepFunctions(lm.model, tc.compile)
    mbs, L = plan.micro_batch_sizes[rank], cfg.data.seq_len
    vocab = lm.hf_config["vocab_size"]
    rng = np.random.default_rng(rank)
    w = rng.integers(0, vocab, size=(mbs, L + 1))
    mb = (w[:, :-1], w[:, 1:], np.ones((mbs, L), dtype=np.float32))
    local_gradients(fns, [mb])  # warm up / compile
    t0 = time.perf_counter()
    for _ in range(steps):
        local_gradients(fns, [mb])
    return steps * mbs * L / (time.perf_counter() - t0)


def allreduce_gbps(world, mb: int = 256, reps: int = 5) -> float:
    x = mx.ones((mb * 1024 * 1024 // 2,), dtype=mx.bfloat16)
    mx.eval(world.all_sum(x))
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(world.all_sum(x))
    dt = (time.perf_counter() - t0) / reps
    return x.nbytes * 8 / dt / 1e9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("plan")
    args = p.parse_args()
    world = init_world()
    plan = RunPlan.load(from_root(args.plan))
    tps = throughput(plan, world.rank)
    speeds = world.all_gather(mx.array([tps], dtype=mx.float32)).reshape(-1).tolist()
    gbps = allreduce_gbps(world) if world.distributed else 0.0
    if world.is_main:
        result = {
            "config": plan.run_config.name,
            "tokens_per_sec": dict(zip(plan.hosts, speeds, strict=True)),
            "micro_batch_sizes": dict(zip(plan.hosts, plan.micro_batch_sizes, strict=True)),
            "allreduce_gbit_per_sec": gbps,
            "measured_at": time.time(),
        }
        out = cluster_dir() / "bench.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        total = sum(speeds)
        for h, s in zip(plan.hosts, speeds, strict=True):
            print(f"{h}: {s:,.0f} tok/s -> share {s / total:.3f}")
        if world.distributed:
            print(f"all_sum bandwidth (algorithmic): {gbps:.1f} Gbit/s")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
