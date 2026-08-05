# Gate 2 support-routing report

- Outcome: **FAIL**
- Experiment: `GATE2_SOFT_20260805T114653Z_82ced99`
- Gate 1R authorization: `PASS_SOFT`; hard top-1 is diagnostic only.
- Primary method: `support_conditioned_soft_mixture`.
- Router inputs contain tensors only and receive no task/dataset identity.

## Frozen Gate 2 criteria

- `random_expected_each_split_mean`: **FAIL**
- `random_expected_each_split_ci`: **FAIL**
- `random_expected_positive_tasks`: **FAIL**
- `query_only_weighting_each_split_mean`: **PASS**
- `query_only_weighting_each_split_ci`: **FAIL**
- `query_only_weighting_positive_tasks`: **PASS**
- `support_query_shuffle_each_split_mean`: **FAIL**
- `support_query_shuffle_each_split_ci`: **FAIL**
- `support_query_shuffle_positive_tasks`: **PASS**
- `support_label_removal_each_split_mean`: **FAIL**
- `support_label_removal_each_split_ci`: **FAIL**
- `support_label_removal_positive_tasks`: **PASS**

## Core results

- Primary mean accuracy: `0.5469`
- Primary mean NLL: `1.2462`
- Oracle-gap recovery over shared: `-0.1415`

## Decision consequence

At least one support-value requirement failed. Learned routing terminates and M5 is not run.
