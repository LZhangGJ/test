# M0 milestone report

## Outcome

M0 code bootstrap and the Windows CPU synthetic smoke test are complete. The
non-destructive host audit succeeded on five of six Linux hosts. Shared NFS
mounts are consistent on the reachable hosts, but the project root remains
unresolved because `doraemon19` accepts TCP connections without completing an
SSH handshake and no common `TAMOE_PROJECT_ROOT` is configured.

No Linux training or multi-host dispatch was started.

## Files changed

- `pyproject.toml`: package metadata, runtime dependencies, pytest and Ruff policy.
- `src/tamoe/`: typed config, reproducibility, atomic I/O, JSONL logging, run IDs,
  synthetic episodes, tiny backbone/experts, analysis-only oracle, smoke runner,
  and non-destructive host probe.
- `tests/`: eight M0 unit and end-to-end tests.
- `scripts/`: Windows/Linux smoke entry points and host-probe entry point.
- `configs/m0_smoke.json`: resolved smoke-test inputs.
- `reports/environment_inventory.json`: complete machine-readable observations.
- `reports/environment_inventory.md`: human-readable host and path decision.
- `README.md`: exact M0 developer commands.

The local-only `configs/hosts.env` was created with empty root variables and is
ignored by Git. No data, checkpoint, cache, model weight, secret, or run output
is included in this milestone.

## Design decisions

- The router-facing research code is not implemented in M0.
- Configuration is a validated frozen dataclass with a stable SHA-256 hash.
- Every smoke run uses a unique immutable run directory, atomic JSON writes,
  a JSONL event log, and a completion marker.
- The episode oracle is explicitly named and serialized as analysis-only.
- The remote probe uses `python3 -I` so files in a user's home directory cannot
  shadow Python standard-library modules.
- A shared project root is accepted only when an identical configured path is
  observed on all six hosts; common mounts alone do not authorize a path.
- Slurm was not detected, so a conservative SSH dispatcher is the future
  scheduler, after the shared root and a single-host run are validated.

## Exact commands

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\ruff.exe check . --fix
.venv\Scripts\ruff.exe format .
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m compileall -q -f src tests scripts
powershell -ExecutionPolicy Bypass -File scripts\windows_smoke.ps1 -Python .\.venv\Scripts\python.exe -OutputRoot runs
.venv\Scripts\python.exe scripts\probe_hosts.py --hosts doraemon02,doraemon03,doraemon04,doraemon15,doraemon19,doraemon20 --output-json reports\environment_inventory.json --output-markdown reports\environment_inventory.md --project-root . --timeout 55
```

SSH configuration diagnostics also used `doraemon03` to resolve the internal
addresses of `doraemon15` and `doraemon19`. The local SSH config now uses
`ProxyJump doraemon03` for those two hosts; a recoverable pre-edit backup is at
`C:\Users\DHU_Z\.ssh\config.codex-backup-20260805`.

## Verification

- pytest: **8 passed in 10.59 seconds**.
- Ruff: **all checks passed**.
- compileall: **passed**.
- Windows CPU smoke: **SUCCEEDED**.
- Support/query overlap: **none**.
- One optimizer step: **completed**.
- Atomic completion marker: **written**.

## Experiment and resources

- Experiment ID: `M0_ENV`.
- Run ID: `M0_synthetic_cpu_smoke_b0264e8659bf_20260805T052536Z`.
- Host: Windows `lzhang`.
- Compute: CPU; `num_workers=0`; PyTorch `2.13.0+cpu`.
- Local physical GPU observed but not used: RTX 5070 Ti 16 GB.
- Run directory (ignored by Git):
  `runs/M0/M0_synthetic_cpu_smoke_b0264e8659bf_20260805T052536Z`.

## Audit findings and uncertainty

- `doraemon02`: 4 × Quadro P6000; probe succeeded; default Python has no PyTorch.
- `doraemon03`: 2 × RTX 3090; probe succeeded; default Python has no PyTorch.
- `doraemon04`: 4 × GTX 1080 Ti; probe succeeded; default Python has no PyTorch.
- `doraemon15`: 4 × Quadro RTX 8000; probe succeeded through `doraemon03`;
  PyTorch `1.10.0+cu102` is importable.
- `doraemon19`: TCP port 22 is reachable from `doraemon03`, but the SSH
  handshake times out; environment remains unknown.
- `doraemon20`: 8 × A100 80 GB; probe succeeded; several GPUs are occupied by
  other users and the default isolated Python has no PyTorch.
- Provisional first end-to-end host: `doraemon15`, which had four idle GPUs at
  audit time. This is an observation, not a reservation.
- `/homes/lzhang` has the same NFS inode on all five reachable hosts. The
  obvious `/homes/lzhang/test` path is unrelated medical data and must not be
  overwritten. `/homes/lzhang/wacv` is a different GitHub repository.
- The first dependency installation attempt was blocked by the sandbox network
  policy; the approved retry completed inside the isolated `.venv`.

## Gate and next action

- Gate 1: **NOT STARTED**; M1–M3 evidence does not yet exist.
- M0 engineering: **PASS**.
- Multi-host execution readiness: **BLOCKED** by the unresponsive
  `doraemon19` SSH handshake and the absence of an approved shared project root.
- The next safe action is to confirm an unused shared project root after
  `doraemon19` becomes auditable; then run one end-to-end job on one idle GPU.
