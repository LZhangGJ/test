# M2 milestone report

## Outcome

M2 implementation and the local frozen-ResNet expert smoke test are complete.
The code supports a configurable frozen backbone, content-keyed feature cache,
equal-architecture residual adapters, a shared expert, an equal-parameter
single-adapter baseline, one source-group expert per meta-training group, an
episode-local prototype head, resource accounting, atomic checkpoints, strict
configuration validation, and resume.

## Design decisions

- Default formal backbone: frozen ResNet-18 with official ImageNet-1K V1
  weights; the tiny backbone remains available for CPU tests.
- Every expert uses the same residual bottleneck architecture and rank.
- Shared expert schedule: task-balanced round robin over all meta-train tasks.
- Equal-parameter single baseline: pooled random schedule over the identical
  tasks, training steps, architecture, and rank.
- Source experts: one per task group; the Organ group expert trains over all
  three views without splitting them.
- The episode classifier has no global class head. It builds normalized local
  class prototypes from each support set.
- Feature-cache keys include the source NPZ SHA-256, task/split, backbone,
  weights, preprocessing version, code version, sampling limit, and seed.
- Checkpoints include the exact config/hash, optimizer, step, feature-cache
  keys, final metrics, and model state. A mismatched config fails loudly.
- Capacity and compute accounting report total, trainable, and per-query
  activated parameters separately. Adapter FLOPs are an explicit approximation.

## Files

- `src/tamoe/models/backbones.py`
- `src/tamoe/models/episodic_head.py`
- `src/tamoe/experts/adapters.py`
- `src/tamoe/experts/training.py`
- `src/tamoe/experts/bank.py`
- `src/tamoe/data/feature_cache.py`
- `src/tamoe/metrics/resources.py`
- `configs/m2_pilot.json`
- `scripts/m2_smoke.py`
- M2 model, training, resume, bank, and resource tests under `tests/`
- `reports/m2_smoke.json` and `reports/m2_smoke.md`

## Exact commands

```text
.venv\Scripts\python.exe scripts\m2_smoke.py --data-root data\medmnist --cache-root cache\m2 --run-root runs --config configs\m2_pilot.json --report-json reports\m2_smoke.json --report-markdown reports\m2_smoke.md --project-root .
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m compileall -q -f src tests scripts
```

## Smoke results

- Experiment ID: `M2_SMOKE`.
- Host/device: Windows `lzhang`, CPU.
- Data: balanced 100-sample BreastMNIST frozen-feature subset.
- Backbone: ResNet-18 / ImageNet-1K V1; embedding dimension 512.
- Adapter rank: 16; trainable parameters: 16,384.
- Activated parameters per query including backbone: 11,192,896.
- Approximate adapter FLOPs per query: 32,768.
- Training: 10 steps; final accuracy 0.75; final loss 0.4559.
- Cache key:
  `11c994b70bf90665b8a4ba50b3cd8f384fb6883312019d6f411317c6c42efd3e`.
- Saved and loaded expert outputs were exactly equal within `torch.allclose`.
- Checkpoint/cache/run payloads remain ignored by Git.

## Verification and Gate status

- pytest: **19 passed**.
- Ruff: **passed**.
- compileall: **passed**.
- M2 engineering: **PASS**.
- Gate 1: **NOT STARTED** until the capacity-matched expert bank is trained and
  evaluated across held-out tasks on Linux.
- Next action: commit/push M2, synchronize the shared checkout, prepare the
  preregistered MedMNIST data/cache, and execute the M3 Gate 1 pilot on one
  locked `doraemon15` GPU.
