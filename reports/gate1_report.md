# Gate 1 fixed-expert oracle pilot

- Decision: **FAIL**
- Experiment ID: `M3_GATE1_CANONICAL_20260805T070831Z`
- Commit: `ca4cf41c19e3a00d76d26227d88d451967c964d1`
- Host / device: `doraemon15` / `cuda:0`
- Episodes: `108`
- Split seeds: `[0, 1]`
- Train seeds: `[0, 1]`
- Shots: `[1, 5, 16]`
- Query-label use is analysis-only for episode oracle, convex oracle mixture, sample oracle, and forced-worst controls.

## Preregistered criteria

- `multiple_task_split_oracle_gap`: **PASS**
- `no_global_expert_dominance`: **PASS**
- `stable_split_oracle_gap`: **PASS**
- `task_specific_intervention_effects`: **FAIL**

## Global dominance

- Maximum episode-best frequency: `0.2130`
- Frequencies: `{'shared': 0.21296296296296297, 'single': 0.1574074074074074, 'source_breastmnist': 0.10185185185185185, 'source_dermamnist': 0.027777777777777776, 'source_octmnist': 0.1388888888888889, 'source_organmnist': 0.027777777777777776, 'source_pathmnist': 0.1111111111111111, 'source_pneumoniamnist': 0.1574074074074074, 'source_tissuemnist': 0.06481481481481481}`

## Split-level paired oracle gaps (accuracy)

| Baseline | Split | Mean | 95% CI | N |
|---|---:|---:|---:|---:|
| shared | 0 | 0.0375 | [0.0239, 0.0528] | 36 |
| shared | 1 | 0.0259 | [0.0210, 0.0309] | 72 |
| single | 0 | 0.0347 | [0.0233, 0.0479] | 36 |
| single | 1 | 0.0301 | [0.0248, 0.0356] | 72 |

## Negative-result diagnosis

- Failed criteria: `['task_specific_intervention_effects']`.
- The largest task/split mean loss from masking the globally most-used expert was `0.0053`, below the preregistered `0.0100` material-effect threshold.
- Consequently, `0` task/split groups passed the complete forced-worst/random/swap/mask/permutation intervention suite; at least `2` were required.
- The positive oracle gap supports expert diversity, but this pilot does not support the full preregistered causal-intervention claim needed to justify learned routing.

## Stop decision

Gate 1 did not satisfy every preregistered criterion. Learned routing development is stopped; this report preserves the negative pilot rather than adding router complexity.
