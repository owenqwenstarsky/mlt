"""Loss, local gradient accumulation and the synchronised optimizer step.

The loss is a token SUM (not a mean). Each rank sums losses and gradients over its
micro-batches; one all_sum combines gradients, loss and token counts; dividing by
the global token count gives exactly the gradient a single machine would compute
for the whole global batch, regardless of how unevenly it was split.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from ..data.batching import MicroBatch
from ..dist.reduce import sum_flat
from ..dist.world import World
from .optim import AdamW, clip_by_global_norm


def loss_sum(model: nn.Module, inputs: mx.array, targets: mx.array, mask: mx.array):
    logits = model(inputs).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, targets, reduction="none")
    return (ce * mask).sum(), mask.sum()


def enable_grad_checkpointing(model: nn.Module) -> None:
    from mlx_lm.tuner.trainer import grad_checkpoint

    grad_checkpoint(model.layers[0])


class StepFunctions:
    def __init__(self, model: nn.Module, compile: bool):
        vg = nn.value_and_grad(model, loss_sum)

        def train_fn(inputs, targets, mask):
            (ls, n), grads = vg(model, inputs, targets, mask)
            return ls, n, grads

        def eval_fn(inputs, targets, mask):
            return loss_sum(model, inputs, targets, mask)

        if compile:
            state = [model.state, mx.random.state]
            train_fn = mx.compile(train_fn, inputs=state, outputs=state)
            eval_fn = mx.compile(eval_fn, inputs=state)
        self.train_fn = train_fn
        self.eval_fn = eval_fn


def _to_mx(mb: MicroBatch) -> tuple[mx.array, mx.array, mx.array]:
    inputs, targets, mask = mb
    return mx.array(inputs.astype(np.int32)), mx.array(targets.astype(np.int32)), mx.array(mask)


def local_gradients(
    fns: StepFunctions, microbatches: list[MicroBatch]
) -> tuple[dict[str, mx.array], mx.array, mx.array]:
    """Sum gradients, loss and target-token count over this rank's micro-batches.

    With a single micro-batch the gradients stay in the parameter dtype; with more,
    they are accumulated in fp32.
    """
    acc: dict[str, mx.array] | None = None
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_n = mx.array(0.0, dtype=mx.float32)
    for mb in microbatches:
        ls, n, grads = fns.train_fn(*_to_mx(mb))
        flat = dict(tree_flatten(grads))
        if acc is None:
            acc = flat if len(microbatches) == 1 else {k: v.astype(mx.float32) for k, v in flat.items()}
        else:
            acc = {k: acc[k] + flat[k] for k in acc}
        total_loss = total_loss + ls
        total_n = total_n + n
        mx.eval(acc, total_loss, total_n)
    if acc is None:
        raise RuntimeError("rank received no micro-batches")
    return acc, total_loss, total_n


@dataclass
class StepResult:
    loss: float
    target_tokens: float
    grad_norm: float


def synchronise(
    grads: dict[str, mx.array],
    loss_total: mx.array,
    n_total: mx.array,
    world: World,
    comm_dtype: mx.Dtype | None,
) -> tuple[dict[str, mx.array], mx.array, mx.array]:
    """All-reduce gradient sums and counts; return gradients normalised per global token."""
    summed = sum_flat(grads, world, comm_dtype=comm_dtype, out_dtype=None)
    stats = world.all_sum(mx.stack([loss_total, n_total]), small=True)
    g_loss, g_n = stats[0], stats[1]
    inv = 1.0 / mx.maximum(g_n, 1.0)
    return {k: g * inv.astype(g.dtype) for k, g in summed.items()}, g_loss, g_n


def apply_update(
    model: nn.Module,
    optimizer: AdamW,
    grads: dict[str, mx.array],
    lr: float,
    grad_clip: float | None,
    master: dict[str, mx.array] | None,
) -> tuple[mx.array, dict[str, mx.array] | None]:
    """Clip, step the optimizer and write the new weights into the model.

    With `master` (mixed precision) the optimizer updates fp32 master weights and
    the model receives a bf16 copy.
    """
    grads, norm = clip_by_global_norm(grads, grad_clip)
    if master is not None:
        master = optimizer.update(master, grads, lr)
        model_dtype = next(iter(dict(tree_flatten(model.trainable_parameters())).values())).dtype
        new = {k: v.astype(model_dtype) for k, v in master.items()}
        mx.eval(master)
    else:
        params = dict(tree_flatten(model.trainable_parameters()))
        new = optimizer.update(params, grads, lr)
    model.update(tree_unflatten(list(new.items())))
    mx.eval(model.trainable_parameters(), optimizer.m, optimizer.v)
    return norm, master
