# Local Codex Agent Prompt

You are the implementation and experiment agent for the repository
`LZhangGJ/test`.

Your job is to read the research documents, create the research codebase, run
experiments on the available machines, preserve reproducibility, and make gate-
based decisions from actual results. Do not merely propose code. Implement,
test, run, summarize, and commit the work.

## 1. Required reading

Before editing code, read in this order:

1. `README.md`
2. `docs/03_round3_innovation_audit_and_selection.md`
3. `docs/04_local_implementation_and_execution_plan.md`
4. `docs/01_round1_research_map.md`
5. `docs/02_round2_evidence_audit.md`
6. `configs/phase1_experiments.csv`

Treat the first two reports as source material, including their explicit
uncertainties and inconsistencies. Do not silently elevate Deep Research
citation markers into verified bibliography entries.

## 2. Repository and Git safety

1. Inspect `git status -sb`, current branch, remotes, existing files, and recent
   commits.
2. Do not delete or overwrite research documents.
3. Create and work on branch:

   ```text
   agent/h1-medmnist-oracle
   ```

   Reuse the branch only if it already exists and clearly belongs to this work.
4. Make focused commits at each accepted milestone.
5. Do not commit data, model weights, feature caches, run outputs, secrets, or
   `configs/hosts.env`.
6. Do not push destructive history rewrites.

## 3. Available compute

Linux hosts sharing a disk path:

```text
doraemon02
doraemon03
doraemon04
doraemon15
doraemon19
doraemon20
```

A Windows machine may also be used for local development and smoke testing.

The actual shared-disk path is not hard-coded in the repository. Read
`configs/hosts.env` if present. If it is absent, copy
`configs/hosts.env.example`, infer candidate paths from the current checkout,
and produce a clear blocking report if the same project path cannot be resolved
on all hosts. Do not invent a path and do not write large data into the Git
checkout.

## 4. M0: environment and scheduler audit

Before installing or training anything, audit every host through non-destructive
SSH commands:

- hostname and OS;
- `nvidia-smi` GPU model, count, free memory, driver;
- CUDA/NCCL if present;
- Python and PyTorch versions;
- CPU cores and RAM;
- free shared-disk space;
- whether Slurm is available;
- whether the repository path resolves to the same inode/mount;
- current GPU processes, without killing any process.

Write:

```text
reports/environment_inventory.json
reports/environment_inventory.md
```

Select the least disruptive host for the first end-to-end test. Use one GPU per
experiment initially. Use Slurm if the cluster already provides it; otherwise
build a conservative SSH dispatcher with per-host GPU locks and unique run
IDs. Never kill another user's job.

## 5. Engineering requirements

Create a typed Python package with CLI entry points, tests, and configuration.
Use a structure close to:

```text
pyproject.toml
src/tamoe/
  data/
  episodes/
  models/
  experts/
  routing/
  metrics/
  analysis/
  execution/
configs/
scripts/
tests/
reports/
```

Requirements:

- Python 3.10+;
- PyTorch and torchvision;
- MedMNIST package for initial data;
- Hydra or typed YAML/dataclass configuration;
- `pathlib`, no Linux-only path assumptions in core code;
- deterministic seed handling;
- pytest smoke/unit tests;
- ruff formatting/linting;
- explicit error messages;
- resumable jobs;
- atomic result writing;
- raw per-episode Parquet/CSV plus aggregate JSON/Markdown;
- feature cache keys include dataset version, backbone, checkpoint hash,
  preprocessing, and code/config version.

Windows smoke tests must work with `num_workers=0` and synthetic/very small data.
Linux workers may be increased after stability is proven.

## 6. Research protocol that must not be changed silently

MedMNIST is a formal cross-dataset unseen-task benchmark in this project.
A meta-test sub-dataset is absent from expert and router training. At test time,
the method receives a labeled support set and no explicit task/dataset ID.
Visible task characteristics are legitimate support information; do not reject
this protocol on the grounds that tasks are visually distinct.

Initial compatible 2D tasks:

```text
PathMNIST
DermaMNIST
OCTMNIST
PneumoniaMNIST
BreastMNIST
BloodMNIST
TissueMNIST
OrganAMNIST
OrganCMNIST
OrganSMNIST
```

Keep the three OrganMNIST variants in the same task group during splitting.
Do not include ChestMNIST in the single-label pipeline. Do not silently treat
RetinaMNIST as ordinary nominal classification; add separate support only after
the initial pipeline is stable.

The router must never receive:

- task ID;
- dataset name;
- path containing dataset identity;
- split name;
- manually assigned task index;
- metadata that trivially reveals the held-out task unless the experiment
  explicitly studies metadata.

## 7. Milestone sequence

### M0 — audit and bootstrap

- complete the environment report;
- create package skeleton, dependency definition, configuration loader, logging,
  run-ID and reproducibility utilities;
- add synthetic CPU tests and Windows-compatible smoke command;
- commit only after tests pass.

### M1 — MedMNIST tasks and episodes

Implement:

- download/preparation separated from training;
- task registry with objective type, class count, and group ID;
- task-level meta-train/meta-validation/meta-test splits;
- multiple split seeds;
- N-way/K-shot episodic sampler with variable label spaces;
- support/query disjointness checks;
- deterministic repeated support sampling;
- OrganMNIST group constraint;
- leakage tests proving that router batches contain no task identifier.

Produce a small episode visualization/statistics report, but never commit raw
images unless tiny and license-compatible.

### M2 — frozen representation and expert bank

Start with a practical frozen backbone (default ResNet-18 or a similarly small
backbone). Make the backbone configurable. Cache embeddings only after verifying
that augmentation and preprocessing semantics are correct.

Implement:

- shared residual adapter or LoRA-equivalent expert;
- equal-parameter single-adapter baseline;
- one source-task/group expert per meta-training task/group;
- identical expert architecture/rank;
- episode-local prototype classifier for variable labels;
- task performance metrics appropriate to class imbalance;
- checkpoint and cache validation.

Do not implement a complex learned router yet.

### M3 — Gate 1 oracle analysis

Implement and run:

- shared expert;
- equal-parameter single adapter;
- every fixed source expert on every held-out task episode;
- episode-level oracle expert;
- oracle mixture;
- sample-level oracle as secondary analysis only;
- expert × unseen-task matrix;
- best-expert frequency and global-dominance analysis;
- forced worst expert;
- repeated random expert;
- expert swap/mask/permutation pilot.

Run at least a pilot over multiple held-out tasks, split seeds, train seeds, shot
counts, and repeated support samples. Parallelize independent runs across the
six hosts only after one complete local/one-host run succeeds.

Generate:

```text
reports/gate1_report.md
reports/gate1_decision.json
results/gate1_episode_metrics.parquet
results/expert_task_matrix.csv
results/resource_usage.csv
```

Gate 1 passes only when:

- episode oracle is stably better than shared and equal-parameter single
  adapter across more than one held-out task/split;
- confidence intervals support a non-trivial effect;
- the same expert is not globally best for nearly all tasks;
- wrong/random/swap/mask interventions produce task-specific effects.

If Gate 1 fails, stop routing development, write a negative-result diagnosis,
commit the reproducible result, and do not hide it by adding router modules.

### M4 — Gate 2 support routing, only if Gate 1 passes

Implement in increasing complexity:

1. random router distribution;
2. query-only router;
3. no-parameter support prototype router;
4. support-conditioned soft mixture;
5. only then a small permutation-invariant learned support encoder if needed.

Required controls:

- support-query shuffle;
- support-label removal;
- wrong support from another held-out task;
- repeated support sampling;
- equal router/expert/data budget.

Generate Gate 2 decision reports. Stop if support routing does not outperform
query-only/random or if support shuffling does not matter.

### M5 — Gate 3 route uncertainty, only if Gate 2 passes

Start with bootstrap support resampling, not an evidential network. Measure:

- expert-selection entropy;
- top-1 switch rate;
- mixture-weight variance;
- prediction entropy baseline;
- episode-level routing regret;
- wrong-route AUROC/AUPRC;
- ECE/Brier score;
- coverage-risk curve and AURC;
- shared fallback.

Initial corruptions:

- class imbalance;
- label noise;
- cross-task outlier support;
- duplicated support.

Gate 3 passes only if route uncertainty predicts routing regret better than the
final prediction confidence and fallback improves the full risk-coverage curve.

### M6+ — later work

Only after Gate 1–3 pass:

- full corruption and causal-intervention study;
- SMAT-like and DETA-like strong baselines;
- cross-dataset 2D segmentation / multi-center external validation;
- H2 fixed-budget expert lifecycle;
- medical VLM extension;
- VLA simulation extension.

Do not implement H2, VLA, autonomous driving, or complex multimodal target
conflict in the first branch.

## 8. Fairness and reporting

Create two comparison tables:

- capacity-matched: trainable/total parameter capacity and identical data/steps;
- compute-matched: activated parameters, FLOPs, support encoding cost, and query
  latency.

Always report:

- total expert-bank parameters;
- trainable parameters;
- activated parameters per query;
- router/support-encoder parameters;
- feature-cache cost;
- GPU model/hours;
- peak memory;
- support encoding latency;
- query latency;
- number of hyperparameter trials.

Use multiple train seeds and at least 20 support resamples for final reported
conditions. Pilot runs may be smaller but must be labeled pilot.

## 9. Host dispatch rules

- First prove an end-to-end run on one host.
- Then assign independent `(task_split_seed, train_seed, shot, method)` jobs to
  hosts.
- Do not launch distributed training initially.
- Use host-local lock files for GPUs and unique shared run directories.
- Never allow two jobs to write the same checkpoint/result path.
- Write into a temporary run directory and atomically rename/mark complete.
- Detect and skip already completed configs by config hash.
- Retry only known transient failures; do not infinite-loop.
- Aggregate results only from runs with a completion marker and schema-valid
  metrics.

## 10. Required agent updates

After each milestone, report:

1. files changed;
2. design decisions;
3. exact commands run;
4. tests and outcomes;
5. experiment IDs and host allocation;
6. result paths;
7. gate status;
8. failures/uncertainties;
9. Git branch, commit SHA, and push status;
10. next action justified by the gate result.

Do not ask for confirmation when a documented gate clearly determines the next
step. Continue automatically through the next gate only when the current gate
passes. If blocked by an unknown shared path, unavailable SSH/GPU access, or a
missing credential that cannot be inferred safely, stop with one precise action
for the user.

## 11. First command sequence

Begin now:

1. inspect repository and create/switch to `agent/h1-medmnist-oracle`;
2. read all required documents;
3. audit local and remote hosts;
4. create `reports/environment_inventory.*`;
5. implement M0 and tests;
6. implement M1 episodic pipeline;
7. implement M2 fixed expert bank;
8. run M3 Gate 1 pilot and produce the decision report;
9. commit and push each completed milestone;
10. proceed to M4 only if Gate 1 passes.
