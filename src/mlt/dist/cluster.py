"""Cluster setup and health checks."""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..paths import cluster_dir, files_hash, project_root


def setup_command(hosts: list[str], auto_setup: bool, output: Path) -> list[str]:
    """`mlx.distributed_config` finds the Thunderbolt topology, assigns bridge IPs and
    writes a ring hostfile. With --auto-setup it applies the network config itself,
    which runs sudo on every laptop (you will be prompted)."""
    exe = Path(sys.executable).parent / "mlx.distributed_config"
    cmd = [
        str(exe),
        "--hosts", ",".join(hosts),
        "--over", "thunderbolt",
        "--backend", "ring",
        "--output-hostfile", str(output),
    ]  # fmt: skip
    if auto_setup:
        cmd.append("--auto-setup")
    return cmd


def run_setup(hosts: list[str], auto_setup: bool, log=print) -> int:
    out = cluster_dir() / "hostfile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = setup_command(hosts, auto_setup, out)
    log("$ " + shlex.join(cmd))
    rc = subprocess.call(cmd)
    if rc == 0 and out.exists():
        log(f"hostfile written to {out}")
    return rc


def local_info() -> dict[str, Any]:
    import mlx.core as mx

    root = project_root()
    src = sorted((root / "src" / "mlt").rglob("*.py")) if (root / "src" / "mlt").exists() else []
    lock = root / "uv.lock"
    dev = mx.device_info()
    return {
        "hostname": platform.node(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": md.version("mlx"),
        "mlx_lm": md.version("mlx-lm"),
        "device": dev.get("device_name"),
        "memory_gb": round(dev.get("memory_size", 0) / 1e9, 1),
        "gpu_working_set_gb": round(dev.get("max_recommended_working_set_size", 0) / 1e9, 1),
        "code_hash": files_hash(src) if src else None,
        "lock_hash": files_hash([lock]) if lock.exists() else None,
        "root": str(root),
    }


def remote_info(ssh: str) -> dict[str, Any]:
    root = project_root()
    cmd = f"cd {shlex.quote(str(root))} && .venv/bin/python -m mlt.cli doctor --local --json"
    res = subprocess.run(["ssh", ssh, cmd], capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        return {"error": (res.stderr or res.stdout).strip()[-500:] or f"ssh exit {res.returncode}"}
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"error": f"unexpected output: {res.stdout[-300:]}"}


def compare(infos: dict[str, dict[str, Any]]) -> list[str]:
    """Problems that would break or skew a distributed run."""
    problems = []
    ok = {h: i for h, i in infos.items() if "error" not in i}
    for h, i in infos.items():
        if "error" in i:
            problems.append(f"{h}: {i['error']}")
    for key in ("mlx", "mlx_lm", "python", "code_hash", "lock_hash", "root"):
        values = {h: i.get(key) for h, i in ok.items()}
        if len(set(values.values())) > 1:
            problems.append(f"{key} differs: {values}")
    return problems
