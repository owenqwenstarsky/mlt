"""The training loop that every rank runs."""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import get_total_parameters

from ..config.load import diff_views, trajectory_hash, trajectory_view
from ..data.batching import make_microbatches
from ..data.sampler import GlobalSampler, rank_slice, split_counts
from ..data.shards import train_val_views
from ..dist.reduce import broadcast_flat
from ..dist.world import World
from ..methods.apply import adapter_config, apply_method
from ..models.gpt import count_parameters
from ..models.loading import build_scratch, load_pretrained
from ..paths import from_root
from ..plan import RunPlan
from ..tokenization.load import load_tokenizer
from .checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_adapters,
    save_checkpoint,
    save_full_model,
)
from .eval import greedy_sample, validation_loss
from .metrics import Metrics
from .optim import AdamW, lr_at
from .step import (
    StepFunctions,
    apply_update,
    enable_grad_checkpointing,
    local_gradients,
    synchronise,
)


class ResumeError(RuntimeError):
    pass


def _set_trainable(model, flat: dict[str, mx.array], dtype: mx.Dtype | None = None) -> None:
    if dtype is not None:
        flat = {k: v.astype(dtype) for k, v in flat.items()}
    model.update(tree_unflatten(list(flat.items())))


def train(plan: RunPlan, world: World, resume: bool = False) -> Path | None:
    cfg = plan.run_config
    tc = cfg.train
    run_dir = plan.run_dir
    metrics = Metrics(run_dir if world.is_main else None, world.is_main)
    log = metrics.log

    if world.size != len(plan.hosts):
        raise RuntimeError(f"plan is for {len(plan.hosts)} host(s) but {world.size} rank(s) are running")

    mx.random.seed(tc.seed)
    tokenizer_dir = from_root(plan.tokenizer_dir)

    # ---------------------------------------------------------------- model
    if cfg.model.is_scratch:
        lm = build_scratch(cfg.model.arch, tokenizer_dir, tc.precision)
    else:
        lm = load_pretrained(
            from_root(plan.model_path), cfg.method, tc.precision, cfg.model.trust_remote_code
        )
    model = lm.model
    apply_method(model, cfg.method, lm.family)
    if tc.grad_checkpoint:
        enable_grad_checkpointing(model)
    model.train()

    trainable = dict(tree_flatten(model.trainable_parameters()))
    model_dtype = next(iter(trainable.values())).dtype
    master = {k: v.astype(mx.float32) for k, v in trainable.items()} if tc.precision == "mixed" else None
    optimizer = AdamW(tc.beta1, tc.beta2, tc.eps, tc.weight_decay)
    optimizer.init(master if master is not None else trainable)

    # --------------------------------------------------------------- resume
    start_step, tokens_seen, found = 0, 0, 0
    if resume and world.is_main:
        ckpt = latest_checkpoint(run_dir)
        if ckpt is None:
            raise ResumeError(f"--resume: no checkpoint in {run_dir / 'checkpoints'}")
        weights, opt_state, state = load_checkpoint(ckpt)
        if state.get("trajectory_hash") != trajectory_hash(cfg):
            diffs = diff_views(state.get("trajectory", {}), trajectory_view(cfg))
            raise ResumeError(
                f"config changed since {ckpt.name}; resuming would not continue the same run:\n  "
                + "\n  ".join(diffs or ["(unknown difference)"])
            )
        if master is not None:
            master = weights
            _set_trainable(model, master, model_dtype)
        else:
            _set_trainable(model, weights)
        optimizer.load_state(opt_state, state["step"])
        start_step, tokens_seen, found = state["step"], state.get("tokens_seen", 0), 1
        log(f"resuming from {ckpt}")

    if world.distributed:
        # Rank 0 is the source of truth: identical weights/optimizer state everywhere.
        info = world.all_sum(mx.array([found, start_step, tokens_seen], dtype=mx.float32), small=True)
        found, start_step, tokens_seen = (int(x) for x in info.tolist())
        source = master if master is not None else dict(tree_flatten(model.trainable_parameters()))
        source = broadcast_flat(source, world)
        if master is not None:
            master = source
            _set_trainable(model, master, model_dtype)
        else:
            _set_trainable(model, source)
        if found:
            optimizer.load_state(broadcast_flat(optimizer.state(), world), start_step)
        mx.eval(model.trainable_parameters(), optimizer.m, optimizer.v)

    # ----------------------------------------------------------------- data
    seq_len = cfg.data.seq_len
    train_ds, val_ds = train_val_views(
        from_root(plan.train_data),
        from_root(plan.val_data) if plan.val_data else None,
        cfg.data.val_fraction,
        seq_len,
    )
    B = tc.global_batch_size
    sampler = GlobalSampler(len(train_ds), B, tc.seed)
    total_steps = tc.steps if tc.steps is not None else sampler.steps_for_epochs(tc.epochs)
    counts = split_counts(B, plan.shares)
    my_slice = rank_slice(counts, world.rank)
    mbs = plan.micro_batch_sizes[world.rank]

    n_params, n_train = get_total_parameters(model), count_parameters(model, trainable_only=True)
    log(f"run {cfg.name}: stage={cfg.stage} method={cfg.method.type} precision={tc.precision}")
    log(f"params {n_params / 1e6:.1f}M, trainable {n_train / 1e6:.2f}M ({100 * n_train / max(n_params, 1):.2f}%)")
    log(f"data: {len(train_ds):,} train items" + (f", {len(val_ds):,} val items" if val_ds else ""))
    log(
        f"{world.size} rank(s); per-step batch split {counts} (micro-batch sizes "
        f"{plan.micro_batch_sizes}); steps {start_step}->{total_steps}"
    )
    if world.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    fns = StepFunctions(model, tc.compile)
    comm_dtype = mx.bfloat16 if (cfg.cluster and cfg.cluster.comm_dtype == "bf16") else mx.float32
    tokenizer = load_tokenizer(tokenizer_dir) if (world.is_main and tc.sample_prompts) else None
    max_len = cfg.model.arch.context_length if cfg.model.is_scratch else seq_len

    def checkpoint(step: int) -> None:
        if world.is_main:
            weights = master if master is not None else dict(tree_flatten(model.trainable_parameters()))
            path = save_checkpoint(
                run_dir, step, weights, optimizer.state(),
                {"trajectory_hash": trajectory_hash(cfg), "trajectory": trajectory_view(cfg),
                 "tokens_seen": tokens_seen, "world_size": world.size},
                keep=tc.keep_checkpoints,
            )  # fmt: skip
            log(f"saved {path}")

    def evaluate(step: int) -> None:
        if val_ds is not None:
            res = validation_loss(
                fns, model, val_ds, world, plan.shares, mbs, seq_len, tc.eval_batches * B
            )
            if res:
                log(f"step {step}: val_loss {res['val_loss']:.4f} ppl {res['val_ppl']:.2f}")
                metrics.record("eval", step=step, **res)
        if tokenizer is not None:
            for p in tc.sample_prompts:
                text = greedy_sample(model, tokenizer, p, tc.sample_max_tokens, max_len)
                log(f"sample> {p!r} -> {text!r}")
                metrics.record("sample", step=step, prompt=p, text=text)

    # ----------------------------------------------------------------- loop
    win = {"loss": 0.0, "n": 0.0, "inputs": 0, "compute": 0.0, "comm": 0.0, "t0": time.time(), "steps": 0}
    mx.reset_peak_memory()
    step = start_step
    for step in range(start_step, total_steps):
        t0 = time.perf_counter()
        idx = sampler.batch(step)[my_slice]
        items = [train_ds.get(int(i)) for i in idx]
        microbatches = make_microbatches(items, mbs, seq_len)
        win["inputs"] += sum(min(len(t) - 1, seq_len) for t, _ in items)

        grads, ls, n = local_gradients(fns, microbatches)
        t1 = time.perf_counter()
        grads, g_loss, g_n = synchronise(grads, ls, n, world, comm_dtype)
        mx.eval(grads, g_loss, g_n)
        t2 = time.perf_counter()
        lr = lr_at(step, total_steps, tc)
        norm, master = apply_update(model, optimizer, grads, lr, tc.grad_clip, master)
        del grads

        win["loss"] += g_loss.item()
        win["n"] += g_n.item()
        win["compute"] += t1 - t0
        win["comm"] += t2 - t1
        win["steps"] += 1
        done = step + 1
        last = done == total_steps

        if done % tc.log_every == 0 or last:
            local = mx.array(
                [win["compute"], win["comm"], float(win["inputs"]), mx.get_peak_memory() / 1e9],
                dtype=mx.float32,
            )
            per_rank = world.all_gather(local).tolist()
            elapsed = time.time() - win["t0"]
            global_inputs = sum(r[2] for r in per_rank)
            tokens_seen += int(global_inputs)
            loss = win["loss"] / max(win["n"], 1.0)
            log(
                f"step {done}/{total_steps} loss {loss:.4f} lr {lr:.2e} gnorm {norm.item():.3f} "
                f"tok/s {global_inputs / max(elapsed, 1e-9):,.0f} epoch {sampler.epoch_at(done):.2f}"
            )
            if world.distributed:
                for r, (comp, comm, toks, mem) in enumerate(per_rank):
                    log(
                        f"  [{plan.hosts[r]}] compute {comp / win['steps']:.2f}s/step "
                        f"sync+wait {comm / win['steps']:.2f}s/step {toks / max(comp, 1e-9):,.0f} tok/s "
                        f"peak {mem:.1f}GB"
                    )
            metrics.record(
                "train", step=done, loss=loss, lr=lr, grad_norm=norm.item(),
                tokens_per_sec=global_inputs / max(elapsed, 1e-9), tokens_seen=tokens_seen,
                ranks=[dict(zip(("compute_s", "sync_s", "tokens", "peak_gb"), r, strict=True)) for r in per_rank],
            )  # fmt: skip
            win = {"loss": 0.0, "n": 0.0, "inputs": 0, "compute": 0.0, "comm": 0.0, "t0": time.time(), "steps": 0}
            mx.reset_peak_memory()

        if tc.eval_every and (done % tc.eval_every == 0 or last):
            evaluate(done)
        elif last and val_ds is not None:
            evaluate(done)
        if tc.save_every and done % tc.save_every == 0 and not last:
            checkpoint(done)

    # ---------------------------------------------------------------- final
    final_dir = None
    if start_step < total_steps:
        checkpoint(total_steps)
    if world.is_main:
        final_dir = run_dir / "final"
        if cfg.method.type == "full":
            save_full_model(final_dir, model, lm.hf_config, tokenizer_dir)
        else:
            save_adapters(
                final_dir,
                model,
                adapter_config(cfg.method, lm.family, model),
                base_model=cfg.model.from_,
                method={**cfg.method.model_dump(mode="json"), "base_path": plan.model_path},
                tokenizer_dir=tokenizer_dir,
            )
        log(f"final model -> {final_dir}")
    world.barrier()
    metrics.close()
    return final_dir
