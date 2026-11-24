"""The distributed world: one process per laptop.

`mlx.launch` starts each rank with MLX_RANK and MLX_HOSTFILE set; without them we
run as a single process and every collective is a no-op.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass
class World:
    group: Any
    rank: int
    size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.size > 1

    def all_sum(self, x: mx.array, small: bool = False) -> mx.array:
        if self.size == 1:
            return x
        stream = mx.cpu if small else None
        return mx.distributed.all_sum(x, group=self.group, stream=stream)

    def all_gather(self, x: mx.array) -> mx.array:
        """Gather a small array from every rank: result has a leading axis of size `size`."""
        if self.size == 1:
            return x[None]
        return mx.distributed.all_gather(x[None], group=self.group, stream=mx.cpu)

    def barrier(self) -> None:
        mx.eval(self.all_sum(mx.array(1.0), small=True))


def single() -> World:
    return World(group=None, rank=0, size=1)


def init_world() -> World:
    if os.environ.get("MLX_RANK") is None:
        return single()
    group = mx.distributed.init(strict=True)
    return World(group=group, rank=group.rank(), size=group.size())
