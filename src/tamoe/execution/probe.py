"""Non-destructive local and SSH host inventory collector."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import ctypes
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tamoe.utils.atomic_io import atomic_write_json, atomic_write_text

DEFAULT_HOSTS = (
    "doraemon02",
    "doraemon03",
    "doraemon04",
    "doraemon15",
    "doraemon19",
    "doraemon20",
)
TAMOE_ENVIRONMENT_NAMES = (
    "TAMOE_PROJECT_ROOT",
    "TAMOE_DATA_ROOT",
    "TAMOE_RUN_ROOT",
    "TAMOE_CACHE_ROOT",
    "TAMOE_LOCAL_SCRATCH",
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


def _run(command: Sequence[str], timeout: int = 15) -> CommandResult:
    try:
        result = subprocess.run(
            list(command), text=True, capture_output=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            returncode=None,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            timed_out=True,
        )
    except OSError as exc:
        return CommandResult(returncode=None, stdout="", stderr=f"{type(exc).__name__}: {exc}")
    return CommandResult(result.returncode, result.stdout.strip(), result.stderr.strip())


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # environment audit must preserve broken imports
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }


def _windows_memory() -> dict[str, int] | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {
        "total_bytes": status.total_physical,
        "available_bytes": status.available_physical,
    }


def probe_local(project_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(project_root)
    return {
        "ssh_reachable": True,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu_logical": os.cpu_count(),
        "memory": _windows_memory(),
        "python": {"version": sys.version, "executable": sys.executable},
        "torch": _torch_info(),
        "nvidia_gpu": asdict(
            _run(
                (
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version",
                    "--format=csv,noheader,nounits",
                )
            )
        ),
        "nvidia_processes": asdict(
            _run(
                (
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                )
            )
        ),
        "cuda_nvcc": asdict(_run(("nvcc", "--version"))),
        "slurm": {"sbatch": shutil.which("sbatch"), "sinfo": shutil.which("sinfo")},
        "paths": {
            "project_root": {
                "path": str(project_root),
                "exists": project_root.exists(),
                "total_bytes": usage.total,
                "free_bytes": usage.free,
            }
        },
        "tamoe_environment": {name: os.environ.get(name) for name in TAMOE_ENVIRONMENT_NAMES},
    }


REMOTE_PAYLOAD = r"""
import json, os, platform, shutil, socket, subprocess, sys
def run(command, timeout=10):
    try:
        value=subprocess.run(command,shell=True,text=True,capture_output=True,timeout=timeout)
        return {"returncode":value.returncode,"stdout":value.stdout.strip(),"stderr":value.stderr.strip(),"timed_out":False}
    except subprocess.TimeoutExpired as exc:
        return {"returncode":None,"stdout":"","stderr":str(exc),"timed_out":True}
    except Exception as exc:
        return {"returncode":None,"stdout":"","stderr":f"{type(exc).__name__}: {exc}","timed_out":False}
names=("TAMOE_PROJECT_ROOT","TAMOE_DATA_ROOT","TAMOE_RUN_ROOT","TAMOE_CACHE_ROOT","TAMOE_LOCAL_SCRATCH")
paths={"cwd":os.getcwd()}
for name in names:
    if os.environ.get(name): paths[name]=os.environ[name]
path_stats={}
for name,value in paths.items():
    try:
        stat=os.stat(value); usage=shutil.disk_usage(value)
        path_stats[name]={"path":value,"exists":True,"device":stat.st_dev,"inode":stat.st_ino,"total_bytes":usage.total,"free_bytes":usage.free}
    except Exception as exc:
        path_stats[name]={"path":value,"exists":False,"error":f"{type(exc).__name__}: {exc}"}
torch_result=run("cd /tmp && python3 -I -c 'import json,torch; print(json.dumps({\"available\":True,\"version\":torch.__version__,\"cuda_version\":torch.version.cuda,\"cuda_available\":torch.cuda.is_available(),\"gpu_count\":torch.cuda.device_count()}))'")
torch_info=json.loads(torch_result["stdout"]) if torch_result["returncode"] == 0 else {"available":False,"error":torch_result["stderr"] or torch_result["stdout"]}
payload={"ssh_reachable":True,"hostname":socket.gethostname(),"fqdn":socket.getfqdn(),"os":platform.platform(),"architecture":platform.machine(),"cpu_logical":os.cpu_count(),"memory":run("free -b"),"python":{"version":sys.version,"executable":sys.executable},"torch":torch_info,"nvidia_gpu":run("nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version --format=csv,noheader,nounits"),"nvidia_processes":run("nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits"),"nvidia_smi":run("nvidia-smi"),"cuda_nvcc":run("nvcc --version"),"nccl":run("ldconfig -p 2>/dev/null | grep libnccl"),"slurm":{"sbatch":run("command -v sbatch"),"sinfo":run("sinfo --version")},"paths":path_stats,"tamoe_environment":{name:os.environ.get(name) for name in names},"shared_mounts":run("findmnt -rn -t nfs,nfs4 -o TARGET,SOURCE,FSTYPE")}
print(json.dumps(payload,sort_keys=True))
"""


def probe_remote(host: str, timeout: int) -> dict[str, Any]:
    encoded = base64.b64encode(REMOTE_PAYLOAD.encode("utf-8")).decode("ascii")
    remote_command = f"echo {encoded} | base64 -d | python3 -I -"
    result = _run(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=12",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=8",
            "-o",
            "ServerAliveCountMax=2",
            host,
            remote_command,
        ),
        timeout=timeout,
    )
    if result.returncode != 0:
        return {
            "ssh_reachable": False,
            "probe_status": "TIMEOUT" if result.timed_out else "FAILED",
            "error": result.stderr or result.stdout or "SSH probe failed without output",
        }
    try:
        payload = json.loads(result.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "ssh_reachable": True,
            "probe_status": "INVALID_OUTPUT",
            "error": f"{type(exc).__name__}: {exc}",
            "raw_output_tail": result.stdout[-2000:],
        }
    payload["probe_status"] = "SUCCEEDED"
    return payload


def _gpu_availability(host: dict[str, Any]) -> dict[str, int]:
    output = host.get("nvidia_gpu", {}).get("stdout", "")
    free_values: list[int] = []
    idle_count = 0
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 6:
            try:
                total = int(fields[3])
                free = int(fields[4])
                utilization = int(fields[5])
            except ValueError:
                continue
            free_values.append(free)
            if utilization == 0 and free >= int(total * 0.98):
                idle_count += 1
    return {"idle_gpu_count": idle_count, "max_free_gpu_mib": max(free_values, default=-1)}


def _shared_path_status(hosts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    successful = {
        name: value for name, value in hosts.items() if value.get("probe_status") == "SUCCEEDED"
    }
    home_observations = {
        name: value.get("paths", {}).get("cwd") for name, value in successful.items()
    }
    sources = {}
    for _name, value in successful.items():
        output = value.get("shared_mounts", {}).get("stdout", "")
        for line in output.splitlines():
            fields = line.split()
            if len(fields) >= 2:
                sources.setdefault(fields[0], set()).add(fields[1])
    common_mounts = {
        mount: sorted(values)
        for mount, values in sources.items()
        if len(values) == 1
        and all(mount in h.get("shared_mounts", {}).get("stdout", "") for h in successful.values())
    }
    all_reachable = len(successful) == len(hosts)
    project_roots = {
        value.get("tamoe_environment", {}).get("TAMOE_PROJECT_ROOT")
        for value in successful.values()
    }
    project_roots.discard(None)
    project_roots.discard("")
    confirmed_project_root = (
        next(iter(project_roots)) if all_reachable and len(project_roots) == 1 else None
    )
    return {
        "all_hosts_reachable": all_reachable,
        "common_mounts_on_reachable_hosts": common_mounts,
        "home_observations": home_observations,
        "confirmed_project_root": confirmed_project_root,
        "status": "CONFIRMED" if confirmed_project_root else "UNRESOLVED",
        "reason": (
            "All hosts expose one identical configured TAMOE_PROJECT_ROOT."
            if confirmed_project_root
            else "No identical configured project root was observed on all six hosts."
        ),
    }


def collect_inventory(
    hosts: Sequence[str], project_root: Path, *, timeout: int = 45
) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(hosts), 6)) as executor:
        futures = {executor.submit(probe_remote, host, timeout): host for host in hosts}
        remote = {
            futures[future]: future.result() for future in concurrent.futures.as_completed(futures)
        }
    remote = {host: remote[host] for host in hosts}
    ranked = sorted(
        (
            {"host": host, **_gpu_availability(value)}
            for host, value in remote.items()
            if value.get("probe_status") == "SUCCEEDED"
        ),
        key=lambda item: (item["idle_gpu_count"], item["max_free_gpu_mib"]),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "probe_policy": "non-destructive; no process termination or environment mutation",
        "local": probe_local(project_root),
        "hosts": remote,
        "shared_path": _shared_path_status(remote),
        "gpu_availability_ranking": ranked,
        "first_end_to_end_host": ranked[0]["host"] if ranked else None,
        "scheduler_decision": (
            "ssh_dispatcher_pending_path_confirmation" if remote else "unavailable"
        ),
    }


def render_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# M0 environment inventory",
        "",
        f"Generated: `{inventory['generated_at_utc']}`",
        "",
        "This report records non-destructive observations only. No process was terminated and no system environment was modified.",
        "",
        "## Host summary",
        "",
        "| Host | SSH/probe | OS | CPU | GPU summary | PyTorch | Slurm |",
        "|---|---|---|---:|---|---|---|",
    ]
    for name, host in inventory["hosts"].items():
        status = host.get("probe_status", "UNKNOWN")
        if status != "SUCCEEDED":
            error = str(host.get("error", "unknown error")).replace("|", "\\|")
            lines.append(f"| {name} | {status}: {error} | — | — | — | — | — |")
            continue
        gpu = host.get("nvidia_gpu", {}).get("stdout", "").replace("\n", "<br>").replace("|", "\\|")
        torch = host.get("torch", {})
        torch_text = (
            torch.get("version")
            if torch.get("available")
            else f"unavailable: {torch.get('error', 'unknown')}"
        )
        torch_text = str(torch_text).replace("\n", "<br>").replace("|", "\\|")
        slurm = "yes" if host.get("slurm", {}).get("sbatch", {}).get("returncode") == 0 else "no"
        lines.append(
            f"| {name} | {status} | {host.get('os', '—')} | {host.get('cpu_logical', '—')} | {gpu or 'none'} | {torch_text} | {slurm} |"
        )
    local = inventory["local"]
    lines.extend(
        [
            "",
            "## Windows development host",
            "",
            f"- Host: `{local.get('hostname')}`",
            f"- OS: `{local.get('os')}`",
            f"- Logical CPUs: `{local.get('cpu_logical')}`",
            f"- Python: `{local.get('python', {}).get('version')}`",
            f"- PyTorch: `{local.get('torch')}`",
            f"- GPU: `{local.get('nvidia_gpu', {}).get('stdout', 'unavailable')}`",
            "",
            "## Shared path decision",
            "",
            f"- Status: **{inventory['shared_path']['status']}**",
            f"- Reason: {inventory['shared_path']['reason']}",
            f"- Common mounts on reachable hosts: `{inventory['shared_path']['common_mounts_on_reachable_hosts']}`",
            "- `configs/hosts.env` remains local and its root paths remain empty until all six hosts can be checked.",
            "",
            "## Scheduler and first-run decision",
            "",
            "- Slurm was not detected on successfully probed hosts; a conservative SSH dispatcher is the applicable future scheduler.",
            f"- Provisional least-disruptive host by idle-GPU count, then free memory: `{inventory.get('first_end_to_end_host')}`.",
            "- No Linux training is authorized until `TAMOE_PROJECT_ROOT`, data, run, and cache roots are confirmed across all six hosts.",
            "",
            "Full command outputs, process lists, memory, driver/CUDA, disk, mount, and path observations are preserved in `reports/environment_inventory.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", default=",".join(DEFAULT_HOSTS))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=45)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    hosts = tuple(host.strip() for host in arguments.hosts.split(",") if host.strip())
    inventory = collect_inventory(
        hosts, arguments.project_root.resolve(), timeout=arguments.timeout
    )
    atomic_write_json(arguments.output_json, inventory)
    atomic_write_text(arguments.output_markdown, render_markdown(inventory))
    print(json.dumps({"status": "SUCCEEDED", "hosts": list(hosts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
