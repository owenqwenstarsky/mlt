"""Entry point each rank runs under `mlx.launch`: python -m mlt.worker <plan.json> [--resume]."""

from __future__ import annotations

import argparse

from .dist.world import init_world
from .paths import from_root
from .plan import RunPlan
from .train.loop import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("plan")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    world = init_world()
    plan = RunPlan.load(from_root(args.plan))
    train(plan, world, resume=args.resume)


if __name__ == "__main__":
    main()
