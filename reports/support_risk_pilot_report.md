# Support-risk routing pilot

- Frozen outcome: **STOP_FIXED_BANK_ROUTING**
- Experiment: `SUPPORT_RISK_20260805T154030Z_dfb35df`
- This is a new falsifiable hypothesis, not a reinterpretation or rescue of Gate 2.
- Exact Gate 2 checkpoints are loaded read-only; no router, encoder, or uncertainty head is trained.
- Primary method was frozen before results: `shrinkage_support_risk_with_shared_fallback`.

## Frozen Go criteria

- `accuracy_improvement_over_capacity_matched_single`: **FAIL**
- `paired_aggregate_ci_lower_bound`: **FAIL**
- `oracle_gap_recovery`: **FAIL**
- `positive_task_count`: **FAIL**
- `positive_both_fresh_splits`: **FAIL**
- `support_ablation_evidence`: **FAIL**
- `nll_no_worse_than_shared`: **FAIL**

## Core values

- Primary accuracy: `0.5981`
- Capacity-matched single accuracy: `0.6000`
- Accuracy improvement: `-0.0019`
- Oracle-gap recovery: `-0.0650`
- Positive tasks: `1/4`
- Positive splits: `0/2`
- Passing support ablations: `[]`

## Aggregate metrics

| Method | Accuracy | Macro-F1 | NLL | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| oracle_analysis_only | 0.6291 | 0.6246 | 1.1456 | 0.2371 | 0.5811 |
| leave_one_out_support_risk_soft_mixture | 0.6028 | 0.5981 | 1.1535 | 0.2248 | 0.5866 |
| shrinkage_support_risk_mixture | 0.6026 | 0.5980 | 1.1545 | 0.2254 | 0.5870 |
| capacity_matched_single | 0.6000 | 0.5956 | 1.1519 | 0.2210 | 0.5850 |
| shrinkage_support_risk_with_shared_fallback | 0.5981 | 0.5935 | 1.1446 | 0.2178 | 0.5829 |
| support_shuffle | 0.5977 | 0.5931 | 1.1453 | 0.2179 | 0.5832 |
| wrong_task_support | 0.5976 | 0.5930 | 1.1453 | 0.2183 | 0.5832 |
| leave_one_out_support_risk_top1 | 0.5975 | 0.5924 | 1.1375 | 0.2088 | 0.5808 |
| support_label_removal | 0.5963 | 0.5914 | 1.1439 | 0.2157 | 0.5840 |
| shared | 0.5948 | 0.5903 | 1.1443 | 0.2117 | 0.5824 |
| random_expected | 0.5931 | 0.5884 | 1.1733 | 0.2215 | 0.5956 |
| naive_support_loss_top1 | 0.5922 | 0.5877 | 1.1430 | 0.2054 | 0.5820 |
| original_support_prototype | 0.5890 | 0.5851 | 1.1532 | 0.2061 | 0.5870 |
| original_support_soft_mixture | 0.5879 | 0.5837 | 1.1552 | 0.2054 | 0.5878 |

## Paired hierarchical 95% intervals vs capacity-matched single

Deltas are method minus control; negative NLL, ECE, and Brier deltas are better.

| Method | Accuracy delta | Macro-F1 delta | NLL delta | ECE delta | Brier delta |
|---|---:|---:|---:|---:|---:|
| naive_support_loss_top1 | -0.0078 [-0.0199, +0.0004] | -0.0079 [-0.0205, +0.0006] | -0.0089 [-0.0451, +0.0160] | -0.0156 [-0.0538, +0.0082] | -0.0029 [-0.0180, +0.0090] |
| leave_one_out_support_risk_top1 | -0.0025 [-0.0163, +0.0086] | -0.0033 [-0.0170, +0.0072] | -0.0144 [-0.0458, +0.0109] | -0.0121 [-0.0486, +0.0102] | -0.0042 [-0.0175, +0.0087] |
| leave_one_out_support_risk_soft_mixture | +0.0028 [-0.0047, +0.0116] | +0.0025 [-0.0048, +0.0110] | +0.0016 [-0.0280, +0.0368] | +0.0039 [-0.0084, +0.0186] | +0.0017 [-0.0092, +0.0142] |
| shrinkage_support_risk_mixture | +0.0026 [-0.0043, +0.0111] | +0.0023 [-0.0047, +0.0102] | +0.0026 [-0.0277, +0.0382] | +0.0045 [-0.0081, +0.0190] | +0.0021 [-0.0091, +0.0150] |
| shrinkage_support_risk_with_shared_fallback | -0.0019 [-0.0077, +0.0034] | -0.0021 [-0.0080, +0.0030] | -0.0073 [-0.0243, +0.0118] | -0.0032 [-0.0150, +0.0058] | -0.0021 [-0.0090, +0.0048] |

## Primary accuracy deltas by split and task

| Scope | Mean delta | Paired hierarchical 95% CI |
|---|---:|---:|
| split 6 | -0.0014 | [-0.0117, +0.0064] |
| split 36 | -0.0024 | [-0.0096, +0.0026] |
| task 6:breastmnist | -0.0060 | [-0.0183, +0.0042] |
| task 6:octmnist | +0.0032 | [-0.0034, +0.0098] |
| task 36:bloodmnist | -0.0003 | [-0.0051, +0.0044] |
| task 36:pathmnist | -0.0045 | [-0.0153, +0.0038] |

## Support ablation accuracy drops

Positive values mean the frozen primary outperformed the ablated support control.

| Ablation | Mean drop | Paired hierarchical 95% CI |
|---|---:|---:|
| support_shuffle | +0.0004 | [-0.0005, +0.0013] |
| support_label_removal | +0.0019 | [-0.0069, +0.0100] |
| wrong_task_support | +0.0006 | [-0.0005, +0.0019] |

## Decision consequence

At least one required criterion failed. Fixed-bank support-risk routing stops; no model complexity is added after this result. The terminated H1 project remains unchanged.
