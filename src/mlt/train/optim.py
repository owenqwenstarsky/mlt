"""AdamW over flat parameter dicts, LR schedules and gradient clipping.

The optimizer keeps its moments in fp32 regardless of the parameter dtype, so bf16
weights still get full-precision optimizer state. It works on flat dicts keyed by
parameter path, which makes checkpointing and broadcasting trivial.
"""

from __future__ import annotations

import math

import mlx.core as mx

from ..config.schema import TrainConfig


def lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    peak = cfg.lr
    floor = peak * cfg.min_lr_ratio
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return peak * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return peak
    span = max(1, total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, (step - cfg.warmup_steps) / span))
    if cfg.schedule == "linear":
        return peak - (peak - floor) * progress
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


@mx.compile
def _adamw_kernel(p, g, m, v, lr, b1, b2, eps, wd, bc1, bc2):
    g = g.astype(mx.float32)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * mx.square(g)
    update = (m / bc1) / (mx.sqrt(v / bc2) + eps)
    p32 = p.astype(mx.float32)
    p32 = p32 * (1 - lr * wd) - lr * update
    return p32.astype(p.dtype), m, v


class AdamW:
    def __init__(self, beta1: float, beta2: float, eps: float, weight_decay: float):
        self.beta1, self.beta2, self.eps, self.weight_decay = beta1, beta2, eps, weight_decay
        self.step = 0
        self.m: dict[str, mx.array] = {}
        self.v: dict[str, mx.array] = {}

    @staticmethod
    def decays(name: str, p: mx.array) -> bool:
        """Decay matrices (linear weights, embeddings); never norms or biases."""
        return p.ndim >= 2

    def init(self, params: dict[str, mx.array]) -> None:
        self.m = {k: mx.zeros(p.shape, dtype=mx.float32) for k, p in params.items()}
        self.v = {k: mx.zeros(p.shape, dtype=mx.float32) for k, p in params.items()}
        self.step = 0

    def update(
        self, params: dict[str, mx.array], grads: dict[str, mx.array], lr: float
    ) -> dict[str, mx.array]:
        self.step += 1
        b1, b2 = self.beta1, self.beta2
        scal = {
            "lr": mx.array(lr, dtype=mx.float32),
            "b1": mx.array(b1, dtype=mx.float32),
            "b2": mx.array(b2, dtype=mx.float32),
            "eps": mx.array(self.eps, dtype=mx.float32),
            "bc1": mx.array(1 - b1**self.step, dtype=mx.float32),
            "bc2": mx.array(1 - b2**self.step, dtype=mx.float32),
        }
        wd_on = mx.array(self.weight_decay, dtype=mx.float32)
        wd_off = mx.array(0.0, dtype=mx.float32)
        new = {}
        for k, p in params.items():
            wd = wd_on if self.decays(k, p) else wd_off
            new[k], self.m[k], self.v[k] = _adamw_kernel(
                p, grads[k], self.m[k], self.v[k],
                scal["lr"], scal["b1"], scal["b2"], scal["eps"], wd, scal["bc1"], scal["bc2"],
            )  # fmt: skip
        return new

    def state(self) -> dict[str, mx.array]:
        out = {f"m.{k}": v for k, v in self.m.items()}
        out.update({f"v.{k}": v for k, v in self.v.items()})
        return out

    def load_state(self, state: dict[str, mx.array], step: int) -> None:
        self.m = {k[2:]: v for k, v in state.items() if k.startswith("m.")}
        self.v = {k[2:]: v for k, v in state.items() if k.startswith("v.")}
        self.step = step


def global_norm(grads: dict[str, mx.array]) -> mx.array:
    total = mx.array(0.0, dtype=mx.float32)
    for g in grads.values():
        total = total + mx.sum(mx.square(g.astype(mx.float32)))
    return mx.sqrt(total)


def clip_by_global_norm(
    grads: dict[str, mx.array], max_norm: float | None
) -> tuple[dict[str, mx.array], mx.array]:
    norm = global_norm(grads)
    if max_norm is None:
        return grads, norm
    scale = mx.minimum(mx.array(1.0), max_norm / (norm + 1e-6))
    return {k: g * scale.astype(g.dtype) for k, g in grads.items()}, norm
