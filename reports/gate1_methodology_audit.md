# M3A canonical Gate 1 methodology audit

## Protection and scope

- Canonical experiment: `M3_GATE1_CANONICAL_20260805T070831Z`
- Canonical decision remains **FAIL** and is not recomputed or reinterpreted as PASS.
- No model was trained and no fresh task split was evaluated.
- Canonical schema validation: `{'decision_status': 'FAIL', 'episode_count': 108, 'row_count': 2592, 'matrix_consistent': True, 'schema_valid': True}`
- Source SHA-256: `{'gate1_episode_metrics.parquet': '1ddf550772924d684f9f66bfcc96ff1ee4d4f0df7a9dadfc1d60f92669a8bc59', 'expert_task_matrix.csv': '5f729762cec976f19bbbaad4e32533f144467c603e7c7e06453bfc0d5a8ec190', 'gate1_decision.json': '9f55f5caea46247cd92f6da268795e20d94c363134c5a1e29bf810936465db16'}`

## Oracle margins and near ties

- Mean top-1 minus top-2 accuracy margin: `0.0147`.
- Accuracy margin <= 0.01: `52.78%` of episodes.
- Exact top-accuracy tie: `18.52%` of episodes.
- Mean best-versus-second loss margin: `0.0392`.
- Canonical expert identity uses stored episode-oracle selections. For methodology audit rankings, equal accuracy is resolved by lower query loss and then expert name.
- Stored canonical identity differs from that explicit loss tie-break on `13`/`108` episodes. Oracle accuracy is unchanged, but identity-frequency, conditional-mask, and assignment diagnostics can be tie-sensitive.

## Mask and leave-one-out semantics

- The global highest-frequency canonical oracle expert is `shared`.
- `global_most_used_conditional` deletes that expert only on episodes where it was the stored canonical episode oracle, then selects the best remaining expert using query-label accuracy (lower query loss and name break ties).
- This estimates the global expert's marginal indispensability and expert-bank redundancy; it is not, by itself, a sufficient test of task specificity.
- Leave-one-out, global/task-modal conditional masks, canonical `swap`, `mask_most_used`, and `permuted_task_assignment` are analysis-only, query-label-derived diagnostics.
- Global conditional mask summary: `{'task_split_count': 6, 'conditional_episode_count': 23, 'weighted_mean_accuracy_drop': 0.01136637770611307, 'task_split_ci_low_above_zero_count': 1}`.
- Task-modal conditional mask summary: `{'task_split_count': 6, 'conditional_episode_count': 36, 'weighted_mean_accuracy_drop': 0.011830322444438934, 'task_split_ci_low_above_zero_count': 5}`.
- Unconditional leave-one-out summary: `{'task_split_expert_count': 42, 'maximum_mean_accuracy_drop': 0.01111111044883728, 'median_mean_accuracy_drop': 0.0013888879782623714, 'ci_low_above_zero_count': 10}`.

## Pairwise and task-level effects

- Pairwise summary: `{'task_split_pair_count': 126, 'mean_accuracy_tie_rate': 0.09656084656084656, 'accuracy_ci_excludes_zero_count': 30}`.
- Oracle comparison summary: `{'random_mean': {'positive_mean_task_count': 6, 'ci_low_above_zero_task_count': 6, 'task_count': 6, 'minimum_mean_accuracy_advantage': 0.020580807576576862, 'maximum_mean_accuracy_advantage': 0.053611109157403324}, 'shared': {'positive_mean_task_count': 6, 'ci_low_above_zero_task_count': 6, 'task_count': 6, 'minimum_mean_accuracy_advantage': 0.02272727092107137, 'maximum_mean_accuracy_advantage': 0.04861111111111111}, 'single': {'positive_mean_task_count': 6, 'ci_low_above_zero_task_count': 6, 'task_count': 6, 'minimum_mean_accuracy_advantage': 0.019444440801938374, 'maximum_mean_accuracy_advantage': 0.05000000364250607}}`.

## Bootstrap hierarchy

Episode bootstrap, task-cluster, split-cluster, train-seed-cluster, and nested split/task/train-seed/episode bootstrap intervals are reported separately in the JSON. The two-split interval is necessarily coarse and must not be treated as a large-sample CI.

## Functional diversity versus near-tie selection

- Oracle-shared hierarchical mean/CI: `{'mean': 0.029794352887957183, 'ci_low': 0.02213889258241074, 'ci_high': 0.04748481201111439, 'split_count': 2, 'task_cluster_count': 6, 'train_seed_cluster_count': 12, 'episode_count': 108}`.
- Oracle-single hierarchical mean/CI: `{'mean': 0.03166328839681767, 'ci_low': 0.020219782322507214, 'ci_high': 0.04826388991851774, 'split_count': 2, 'task_cluster_count': 6, 'train_seed_cluster_count': 12, 'episode_count': 108}`.
- Positive aggregate headroom coexists with frequent near ties. The bank therefore shows functional diversity at aggregate level, while many individual oracle identities are replaceable or tie-sensitive.
- Rank-stability summary: `{'mean_pairwise_spearman': 0.2621638548224532, 'minimum_pairwise_spearman': -0.7142857142857144, 'mean_level_modal_consistency': 0.611111111111111}`.

## Hard routing versus soft mixture

- Mixture minus hard-oracle accuracy: `{'mean': -0.024960787483939418, 'std': 0.024545357298904724, 'ci_low': -0.02981424049853727, 'ci_high': -0.020543206431385543, 'n': 108}`.
- Hard-oracle minus mixture loss: `{'mean': 0.09486476911438836, 'std': 0.10123412978385456, 'ci_low': 0.07684038981657337, 'ci_high': 0.11485741261944726, 'n': 108}`.
- Diagnosis: Hard query-label oracle selection remains the accuracy ceiling, while the convex oracle mixture has materially lower query NLL. Given frequent accuracy near ties and unstable expert ranks, any later feasibility work should prioritize support-conditioned soft mixtures for robustness, but must separately verify accuracy. Both quantities here are analysis-only query-label ceilings, not deployable routers.

## Canonical decision

M3A is explanatory only. It does not alter the original Gate 1 FAIL, does not authorize M4, and does not use split 6 or 36.
