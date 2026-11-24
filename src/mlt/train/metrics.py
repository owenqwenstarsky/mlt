"""Console + JSONL metrics (rank 0 only)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


class Metrics:
    def __init__(self, run_dir: Path | None, enabled: bool):
        self.enabled = enabled
        self.fh = None
        if enabled and run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.fh = open(run_dir / "metrics.jsonl", "a", buffering=1)

    def log(self, msg: str) -> None:
        if self.enabled:
            print(msg, file=sys.stdout, flush=True)

    def record(self, kind: str, **values: Any) -> None:
        if self.fh is not None:
            self.fh.write(json.dumps({"kind": kind, "time": time.time(), **values}) + "\n")

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None
