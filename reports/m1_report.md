# M1 milestone report

## Outcome

M1 is complete. The initial MedMNIST v2 single-label registry, group-aware task
splits, deterministic N-way/K-shot sampling, support/query disjointness checks,
saved episode indices, variable label spaces, and task-ID leakage boundary are
implemented and tested. A real BreastMNIST episode smoke test succeeded.

## Protocol decisions

- The registry contains exactly the ten preregistered compatible tasks.
- ChestMNIST and RetinaMNIST are excluded from this nominal single-label path.
- OrganAMNIST, OrganCMNIST, and OrganSMNIST share `group_id=organmnist` and
  cannot cross meta-train/meta-validation/meta-test boundaries.
- Split seeds `0, 1, 6, 36` provide four independent deterministic splits and
  collectively place every registered task in meta-test at least once.
- Each task split uses five training groups, one validation group, and two test
  groups; the Organ group expands to three tasks after splitting.
- Episode labels are remapped locally, allowing binary, 4-way, 5-way, all-way,
  and other variable label spaces.
- Audit indices and query labels remain outside `RouterInput`. The router sees
  only support images, support labels, and query images.
- Dataset download is a separate command and produces a checksum manifest;
  training code never implicitly downloads data.

## Files

- `src/tamoe/data/medmnist_tasks.py`
- `src/tamoe/data/task_splits.py`
- `src/tamoe/data/medmnist_dataset.py`
- `src/tamoe/episodes/sampler.py`
- `configs/m1_medmnist.json`
- `configs/task_splits/medmnist_seed{0,1,6,36}.json`
- `scripts/generate_task_splits.py`
- `scripts/prepare_medmnist.py`
- `scripts/m1_episode_smoke.py`
- M1 registry, split, sampler, and leakage tests under `tests/`
- `reports/m1_breastmnist_manifest.json`
- `reports/m1_episode_smoke.json`
- `reports/m1_episode_report.md`

## Exact commands

```text
.venv\Scripts\python.exe scripts\generate_task_splits.py --config configs\m1_medmnist.json --output-directory configs\task_splits
.venv\Scripts\python.exe scripts\prepare_medmnist.py --root data\medmnist --tasks breastmnist --size 28 --manifest reports\m1_breastmnist_manifest.json
.venv\Scripts\python.exe scripts\m1_episode_smoke.py --root data\medmnist --task breastmnist --config configs\m1_medmnist.json --output-json reports\m1_episode_smoke.json --output-markdown reports\m1_episode_report.md
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m compileall -q -f src tests scripts
```

## Results

- Experiment ID: `M1_SMOKE`.
- Host: Windows `lzhang`, CPU, `num_workers=0`.
- Dataset: BreastMNIST train split, 546 samples (147/399 class counts).
- Conditions: 1-, 2-, 5-, 10-, and 16-shot.
- Pilot support resamples: 3 per shot condition, 15 episodes total.
- Every episode had disjoint support/query indices.
- Exact support/query indices and episode hashes are in
  `reports/m1_episode_smoke.json`.
- Raw downloaded NPZ data remains under ignored `data/medmnist` and is not
  committed.

## Gate status

- M1: **PASS**.
- Gate 1: **NOT STARTED**; M2 expert training and M3 oracle evidence are still
  required.
- Next action: implement the frozen representation, equal-architecture expert
  bank, episode-local head, resource accounting, cache validation, and resume
  support required by M2.
