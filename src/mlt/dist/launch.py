"""Start a run on every laptop.

  1. The coordinator (this laptop, listed first in `cluster.hosts`) builds the plan.
  2. Code, the plan and every artifact it references are copied to the other laptops
     with rsync over the Thunderbolt link, and `uv sync --frozen` runs there.
  3. `mlx.launch` starts one rank per laptop using the ring backend.

Rank order follows the hostfile written by `mlt cluster setup`, rotated so this
laptop is rank 0. Rotating keeps the Thunderbolt ring's neighbours intact, which
matters with three or more laptops.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config.schema import RunConfig
from ..paths import from_root, project_root
from ..plan import PLAN_FILE, RunPlan, plan_artifacts

RSYNC_EXCLUDES = [".venv", ".mlt", "runs", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"]


@dataclass
class HostEntry:
    ssh: str
    ips: list[str]
    raw: dict

    @property
    def user_prefix(self) -> str:
        return self.ssh.split("@", 1)[0] + "@" if "@" in self.ssh else ""

    @property
    def name(self) -> str:
        return self.ssh.split("@", 1)[-1]

    @property
    def transfer_address(self) -> str:
        """Prefer the Thunderbolt IP for bulk copies; fall back to the ssh name."""
        return f"{self.user_prefix}{self.ips[0]}" if self.ips else self.ssh


def read_hostfile(path: Path) -> tuple[dict, list[HostEntry]]:
    if not path.exists():
        raise FileNotFoundError(
            f"hostfile {path} not found. Run `mlt cluster setup --hosts <this-laptop>,<other>` first."
        )
    data = json.loads(path.read_text())
    hosts = data["hosts"] if isinstance(data, dict) else data
    meta = data if isinstance(data, dict) else {"hosts": hosts}
    return meta, [HostEntry(ssh=h["ssh"], ips=list(h.get("ips") or []), raw=h) for h in hosts]


def rank_order(cfg: RunConfig, entries: list[HostEntry]) -> list[int]:
    """Indices into cfg.cluster.hosts, in rank order (hostfile ring order, rotated)."""
    assert cfg.cluster is not None
    names = [h.host for h in cfg.cluster.hosts]
    by_name = {n: i for i, n in enumerate(names)}
    hf_names = [e.name for e in entries]
    unknown = set(hf_names) ^ {n.split("@", 1)[-1] for n in names}
    if unknown:
        raise ValueError(
            f"cluster.hosts {names} and hostfile hosts {hf_names} differ ({sorted(unknown)}); "
            "re-run `mlt cluster setup` with the same hosts"
        )
    strip = {n.split("@", 1)[-1]: by_name[n] for n in names}
    ring = [strip[n] for n in hf_names]
    start = ring.index(0)  # this laptop (first in cluster.hosts) becomes rank 0
    return ring[start:] + ring[:start]


def apply_rank_order(plan: RunPlan, order: list[int]) -> RunPlan:
    plan.hosts = [plan.hosts[i] for i in order]
    plan.shares = [plan.shares[i] for i in order]
    plan.micro_batch_sizes = [plan.micro_batch_sizes[i] for i in order]
    return plan


def write_run_hostfile(meta: dict, entries: list[HostEntry], cfg: RunConfig, order: list[int], out: Path) -> Path:
    names = [cfg.cluster.hosts[i].host.split("@", 1)[-1] for i in order]
    by_name = {e.name: e for e in entries}
    hosts = []
    for rank, name in enumerate(names):
        h = dict(by_name[name].raw)
        if rank == 0:
            h["ssh"] = "127.0.0.1"  # mlx.launch starts 127.0.0.1 locally instead of over ssh
        hosts.append(h)
    data = {k: v for k, v in meta.items() if k != "hosts"}
    data["hosts"] = hosts
    data.setdefault("backend", "ring")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return out


def _run(cmd: list[str], log, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + shlex.join(cmd))
    return subprocess.run(cmd, check=check)


def sync_to_hosts(plan: RunPlan, entries: list[HostEntry], cfg: RunConfig, order: list[int], log=print) -> None:
    root = project_root()
    remote_names = [cfg.cluster.hosts[i].host.split("@", 1)[-1] for i in order[1:]]
    by_name = {e.name: e for e in entries}
    plan_rel = Path(cfg.output_dir) / cfg.name / PLAN_FILE
    for name in remote_names:
        e = by_name[name]
        addr = e.transfer_address
        dirs = {str(Path(p).parent) for p in plan_artifacts(plan)} | {str(plan_rel.parent)}
        mkdirs = " ".join(shlex.quote(str(root / d)) for d in sorted(dirs))
        _run(["ssh", e.ssh, f"mkdir -p {shlex.quote(str(root))} {mkdirs}"], log)
        excludes = [f"--exclude={x}" for x in RSYNC_EXCLUDES]
        _run(["rsync", "-a", *excludes, f"{root}/", f"{addr}:{root}/"], log)
        for rel in plan_artifacts(plan):
            _run(["rsync", "-a", f"{root / rel}/", f"{addr}:{root / rel}/"], log)
        _run(["rsync", "-a", str(root / plan_rel), f"{addr}:{root / plan_rel}"], log)
        uv = (
            "for u in uv $HOME/.local/bin/uv /opt/homebrew/bin/uv; do "
            "command -v $u >/dev/null 2>&1 && exec $u sync --frozen --quiet; done; "
            "echo 'uv not found on PATH' >&2; exit 1"
        )
        _run(["ssh", e.ssh, f"cd {shlex.quote(str(root))} && {uv}"], log)


def launch_command(hostfile: Path, module_args: list[str]) -> list[str]:
    root = project_root()
    launcher = Path(sys.executable).parent / "mlx.launch"
    return [
        str(launcher),
        "--hostfile", str(hostfile),
        "--backend", "ring",
        "--cwd", str(root),
        "--python", str(root / ".venv" / "bin" / "python"),
        "--env", f"MLT_ROOT={root}",
        "--",
        *module_args,
    ]  # fmt: skip


def prepare_cluster(cfg: RunConfig, plan: RunPlan, sync: bool, log=print) -> Path:
    """Order ranks, write the plan + run hostfile, sync laptops. Returns the run hostfile."""
    meta, entries = read_hostfile(from_root(cfg.cluster.hostfile))
    order = rank_order(cfg, entries)
    apply_rank_order(plan, order)
    plan.save(plan.run_dir / PLAN_FILE)
    hostfile = write_run_hostfile(meta, entries, cfg, order, plan.run_dir / "hostfile.json")
    log(f"rank order: {plan.hosts}")
    if sync:
        sync_to_hosts(plan, entries, cfg, order, log=log)
    return hostfile


def launch_module(hostfile: Path, module: str, args: list[str], log=print) -> int:
    cmd = launch_command(hostfile, ["-m", module, *args])
    log("$ " + shlex.join(cmd))
    return subprocess.call(cmd)
