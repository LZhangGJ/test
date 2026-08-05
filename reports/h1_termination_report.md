# H1 learned-routing termination report

## Final decision

H1 terminates at **Gate 2 FAIL**. Gate 1R authorized only the soft/selective
M4 branch, and the preregistered primary `support_conditioned_soft_mixture`
did not outperform the required random, query-only, shuffled-support, and
label-removed controls with stable split-level confidence intervals. M5/Gate 3
is therefore not authorized and no uncertainty head is added.

## Evidence sequence

### Protected canonical Gate 1

The protected canonical experiment `M3_GATE1_CANONICAL_20260805T070831Z`
remains **FAIL** and its artifacts are unchanged. Oracle headroom and global
diversity passed, but task-specific intervention evidence failed: zero
task/split groups passed the full intervention suite, versus two required.
The largest global-most-used mask effect was 0.0053, below the 0.01 material
threshold.

### M3A exploratory audit

M3A did not change the canonical decision. It found positive hierarchical
oracle gaps over shared (0.0298, 95% CI [0.0221, 0.0475]) and single (0.0317,
95% CI [0.0202, 0.0483]), but also weak identity stability: 52.78% of episodes
had top-1/top-2 margin at most 0.01, 18.52% had an exact accuracy tie, and the
mean cross-condition Spearman correlation was 0.262. These observations
motivated fresh Gate 1R only and remained exploratory.

### Confirmatory Gate 1R

Experiment `GATE1R_CONFIRMATORY_20260805T104912Z_a17e972` produced
**PASS_SOFT** on 360 fresh episodes.

- Every shared A1–A3 requirement passed.
- Split 6 oracle–shared gap was 0.0455, 95% CI [0.0374, 0.0544]; split 36 was
  0.0201, 95% CI [0.0117, 0.0292].
- Split 6 oracle–single gap was 0.0394, 95% CI [0.0314, 0.0468]; split 36 was
  0.0154, 95% CI [0.0094, 0.0226].
- Maximum primary-path frequency was 0.1944 and seven paths exceeded 5% usage.
- Forced-worst, random-expected, and wrong-task/assignment controls passed on
  all four tasks.
- Hard identity failed because median within-task rank Spearman was 0.2309
  (required at least 0.30) and exact-tie rate was 0.2528 (required below 0.20).

This confirms useful bank-level diversity and oracle headroom but rejects a
stable hard-routing identity claim.

### Gate 2 support value

Experiment `GATE2_SOFT_20260805T114653Z_82ced99` produced **FAIL** on 720
episodes (two fresh splits, four tasks, three train seeds, three shot counts,
and 20 support resamples).

| Method | Mean accuracy | Mean macro-F1 | Mean NLL |
|---|---:|---:|---:|
| Router-candidate oracle (analysis only) | 0.5869 | 0.5798 | 1.2196 |
| Capacity-matched single | 0.5583 | 0.5498 | 1.2389 |
| Shared | 0.5519 | 0.5434 | 1.2401 |
| Shared fallback | 0.5522 | 0.5439 | 1.2328 |
| Random expected | 0.5500 | 0.5420 | 1.2539 |
| Wrong support task | 0.5501 | 0.5415 | 1.2501 |
| Support-query shuffle | 0.5497 | 0.5413 | 1.2512 |
| Support prototype weighting | 0.5482 | 0.5405 | 1.2475 |
| Support-conditioned soft mixture | 0.5469 | 0.5391 | 1.2462 |
| Support label removal | 0.5449 | 0.5370 | 1.2403 |
| Query-only weighting | 0.5448 | 0.5368 | 1.2401 |

The primary soft mixture was 0.0031 below random expected and 0.0050 below
shared on aggregate. Its oracle-gap recovery relative to shared was -0.1415.
Against random expected, its split means were -0.0016 (split 6) and -0.0045
(split 36), with both 95% intervals crossing zero. Against query-only, mean
deltas were positive but both split intervals crossed zero. Shuffling support
did not reliably hurt performance; split 36 instead favored shuffled support.
Removing labels also caused no stable split-level loss.

Failed Gate 2 criteria:

- `random_expected_each_split_mean`
- `random_expected_each_split_ci`
- `random_expected_positive_tasks`
- `query_only_weighting_each_split_ci`
- `support_query_shuffle_each_split_mean`
- `support_query_shuffle_each_split_ci`
- `support_label_removal_each_split_mean`
- `support_label_removal_each_split_ci`

## Hypothesis disposition

- **H1a, bank value:** supported only in the Gate 1R soft sense. Fresh oracle
  headroom, multiple useful paths, and intervention effects exist.
- **Hard identity stability:** rejected. Rank stability and exact-tie criteria
  failed.
- **H1b, incremental support value:** rejected for the implemented
  metadata-free no-parameter weighting family. Support did not beat the
  required controls or respond causally to shuffle/label removal.
- **H1c, routing-specific uncertainty:** not tested because Gate 2 did not
  authorize M5. No conclusion about uncertainty prediction is claimed.

## Remaining bank value

The fixed bank may still have ensemble or calibration value: the analysis-only
oracle remains 0.0351 above shared, and fixed 50% shared fallback improved NLL
from 1.2462 to 1.2328 while matching shared accuracy. This is not evidence for
deployable support routing. Any future ensemble/calibration study must use a
separate protocol that does not derive deployment weights from query labels.

## Resource and reproducibility record

| Stage | Experiment | Host / GPU | Wall time | End-to-end GPU-hours |
|---|---|---|---:|---:|
| Canonical Gate 1 | `M3_GATE1_CANONICAL_20260805T070831Z` | doraemon15 / RTX 8000 cuda:0 | 133.4 s | 0.0371 |
| Gate 1R | `GATE1R_CONFIRMATORY_20260805T104912Z_a17e972` | doraemon15 / RTX 8000 cuda:1 | 2271.8 s | 0.6310 |
| Gate 2 | `GATE2_SOFT_20260805T114653Z_82ced99` | doraemon15 / RTX 8000 cuda:1 | 2757.0 s | 0.7658 |
| **Total recorded formal runs** |  |  | **5162.2 s** | **1.4340** |

Recorded bank training/reference time totals 0.1433 GPU-hours across these
formal stages. Gate 1R and Gate 2 decisions passed their schema checks; the
final repository suite has 38 passing tests and Ruff is clean.

## Recommendation

Terminate the current H1 learned-routing claim. Do not add a learned set
encoder or complex uncertainty network after this failure. A later,
separately preregistered project may investigate fixed-bank calibration,
ensembling, other medical task families, or H2 expert lifecycle questions.
None is automatically authorized here, and the project does not proceed to
VLM, VLA, autonomous driving, or H2 in this branch.
