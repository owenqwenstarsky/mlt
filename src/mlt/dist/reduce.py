"""Gradient and parameter collectives.

Gradients are flattened, optionally cast to bf16 (halving traffic over the
Thunderbolt link), packed into large buckets and summed with one `all_sum` per
bucket. Few large messages are much faster than many small ones over TCP.

The packing is separate from the collective so it can be tested in memory by
simulating ranks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from .world import World

DEFAULT_BUCKET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Slot:
    name: str
    shape: tuple[int, ...]
    dtype: mx.Dtype
    bucket: int
    offset: int
    size: int


def pack(
    flat: dict[str, mx.array], dtype: mx.Dtype | None, bucket_bytes: int = DEFAULT_BUCKET_BYTES
) -> tuple[list[mx.array], list[Slot]]:
    """Concatenate arrays (in sorted-name order) into 1-D buckets of at most ~bucket_bytes.

    dtype=None keeps each array's dtype and only buckets arrays of the same dtype together.
    """
    slots: list[Slot] = []
    buckets: list[list[mx.array]] = []
    sizes: list[int] = []
    bucket_dtypes: list[mx.Dtype] = []
    for name in sorted(flat):
        a = flat[name]
        target = dtype or a.dtype
        nbytes = a.size * target.size
        if (
            not buckets
            or bucket_dtypes[-1] != target
            or (sizes[-1] > 0 and (sizes[-1] * target.size + nbytes) > bucket_bytes)
        ):
            buckets.append([])
            sizes.append(0)
            bucket_dtypes.append(target)
        slots.append(Slot(name, tuple(a.shape), a.dtype, len(buckets) - 1, sizes[-1], a.size))
        buckets[-1].append(a.reshape(-1).astype(target))
        sizes[-1] += a.size
    return [mx.concatenate(b) if len(b) > 1 else b[0] for b in buckets], slots


def unpack(
    buckets: list[mx.array], slots: list[Slot], dtype: mx.Dtype | None = None
) -> dict[str, mx.array]:
    out = {}
    for s in slots:
        a = buckets[s.bucket][s.offset : s.offset + s.size].reshape(s.shape)
        out[s.name] = a.astype(dtype or s.dtype)
    return out


def sum_flat(
    flat: dict[str, mx.array],
    world: World,
    comm_dtype: mx.Dtype | None = mx.bfloat16,
    out_dtype: mx.Dtype | None = mx.float32,
    bucket_bytes: int = DEFAULT_BUCKET_BYTES,
    reduce_fn: Callable[[mx.array], mx.array] | None = None,
) -> dict[str, mx.array]:
    """Sum a flat dict of arrays across ranks; result cast to `out_dtype`."""
    if world.size == 1 and reduce_fn is None:
        return {k: v.astype(out_dtype or v.dtype) for k, v in flat.items()}
    reduce_fn = reduce_fn or world.all_sum
    buckets, slots = pack(flat, comm_dtype, bucket_bytes)
    summed = [reduce_fn(b) for b in buckets]
    return unpack(summed, slots, out_dtype)


def broadcast_flat(flat: dict[str, mx.array], world: World) -> dict[str, mx.array]:
    """Make every rank hold rank 0's values (sum of rank 0's values and zeros elsewhere)."""
    if world.size == 1:
        return flat
    contrib = flat if world.is_main else {k: mx.zeros_like(v) for k, v in flat.items()}
    return sum_flat(contrib, world, comm_dtype=None, out_dtype=None)
