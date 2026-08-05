# Gate 1R confirmatory report

- Frozen outcome: **PASS_SOFT**
- Experiment ID: `GATE1R_CONFIRMATORY_20260805T104912Z_a17e972`
- Fresh episodes: `360`
- Confirmatory split sample size: `2` (seeds 6 and 36).
- Runtime commit: `a17e9725d749497a42eeddd797366f0429181a49`
- Preregistration: `gate1r-preregistered-v1` at `761d1b4e0004e2636e3d4e21cc6cd95e07196774`
- Query labels are used only by the explicitly analysis-only oracle and intervention diagnostics; no reported oracle is deployable routing.

## Frozen criteria

### shared_requirements

- `A1_each_split_oracle_shared_mean`: **PASS**
- `A1_each_split_oracle_single_mean`: **PASS**
- `A1_each_split_hierarchical_ci`: **PASS**
- `A1_positive_tasks_oracle_shared`: **PASS**
- `A1_positive_tasks_oracle_single`: **PASS**
- `A2_maximum_best_frequency`: **PASS**
- `A2_minimum_frequent_paths`: **PASS**
- `A2_task_modal_diversity`: **PASS**
- `A3_forced_worst`: **PASS**
- `A3_random_expected`: **PASS**
- `A3_assignment`: **PASS**
- `A3_task_dependent_pattern`: **PASS**

### pass_hard_additional_requirements

- `B1_episode_margin`: **PASS**
- `B2_train_seed_modal_consistency`: **PASS**
- `B3_ranking_spearman`: **FAIL**
- `B4_exact_tie_rate`: **FAIL**

## Identity diagnostics

- Maximum primary path frequency: `0.1944`
- Exact accuracy tie rate: `0.2528`
- Assignment permutation and wrong-task assignment are mathematically redundant here because every split has exactly two held-out tasks; both are reported.

## Protocol consequence

All shared requirements passed but at least one hard-identity requirement failed. Only the preregistered soft/selective M4 branch is authorized.
