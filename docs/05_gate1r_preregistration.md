# Gate 1R confirmatory preregistration

**Protocol ID:** `gate1r-confirmatory-v1`  
**Frozen configuration:** `configs/m3_gate1r_confirmatory.json`  
**Required annotated tag:** `gate1r-preregistered-v1`  
**Allowed confirmatory outcomes:** `PASS_HARD`, `PASS_SOFT`, or `FAIL`

This document must be committed, pushed, and annotated-tagged before any model is
trained or evaluated with task split 6 or 36. After the tag is pushed, the fresh
splits, seeds, endpoints, candidate sets, identity rules, thresholds, and outcome
rules are immutable.

## 1. Protected historical evidence

The protected canonical experiment is
`M3_GATE1_CANONICAL_20260805T070831Z`; its original decision is **FAIL**. Its
parquet, CSV, experiment ID, decision JSON, and negative-result report will not be
modified or reinterpreted. Task splits 0 and 1 are historical evidence only and
are excluded from the Gate 1R decision.

M3A was exploratory/post-hoc analysis, not confirmatory evidence. It found:

- oracle-shared hierarchical gap 0.0298, 95% CI [0.0221, 0.0475];
- oracle-single hierarchical gap 0.0317, 95% CI [0.0202, 0.0483];
- oracle-random hierarchical gap 0.0316, 95% CI [0.0245, 0.0516];
- 52.78% of episodes had top-1/top-2 accuracy margin at most 0.01;
- 18.52% had an exact top-accuracy tie;
- stored canonical identity differed from explicit loss-tiebreak identity in
  13/108 episodes;
- mean cross-condition Spearman rank correlation was 0.262 and the minimum was
  -0.714;
- global-most-used conditional masking had weighted mean accuracy drop 0.0114,
  but only 1/6 task-splits had CI lower bound above zero;
- median leave-one-expert-out accuracy drop was 0.0014; and
- the query-label-optimized oracle mixture lost 0.0250 accuracy relative to the
  hard oracle while improving NLL by 0.0949.

The fixed interpretation is: aggregate oracle headroom exists, but hard expert
identity exhibits near ties, tie-break sensitivity, and rank instability.
Global-most-used masking measures marginal indispensability or redundancy, not a
necessary condition for task specificity. Soft mixtures may have NLL value, but
the observed mixture is only a query-label-derived analysis ceiling. Fresh data
alone will decide `PASS_HARD`, `PASS_SOFT`, or `FAIL`; M3A cannot change the
canonical FAIL.

## 2. Confirmatory hypotheses

Gate 1R tests three ordered claims:

1. **Headroom:** on fresh tasks, a query-label analysis oracle over deployable
   router candidates has reproducible accuracy headroom over shared, equal-size
   single-adapter, and exact random-candidate expectations.
2. **Functional diversity:** multiple candidate paths are materially useful and
   task-dependent interventions cause heterogeneous performance losses.
3. **Identity stability:** conditional on the first two claims, expert identity
   is either stable enough for hard routing (`PASS_HARD`) or only sufficiently
   useful for soft/selective combination (`PASS_SOFT`).

Failure of any shared confirmatory requirement produces `FAIL` and terminates H1
learned routing.

## 3. Fresh task splits and immutable run grid

Only the following pre-existing group-aware splits are confirmatory:

| Split | Split hash | Fresh meta-test tasks |
|---:|---|---|
| 6 | `e2aca00b68c72b67eb6f9e9b0b0c53210908312204c357e505919f129c80031b` | breastmnist, octmnist |
| 36 | `9a8fb6bb52f30fdd02e16519d3013b250b7e1dcca1093a346bb8407b40125051` | bloodmnist, pathmnist |

The grid is fixed to train seeds `[2, 3, 4]`, shots `[1, 5, 16]`, ten support
resamples, 20 desired queries per class, 10 repeated-random draws, and 10,000
bootstrap replicates. Support and query samples are disjoint and drawn without
replacement in an episode-local contiguous label space. The canonical feasible
query rule is retained: actual queries per class are
`min(20, minimum class count - shots)`.

## 4. Frozen model and training protocol

Gate 1R reuses the canonical implementation without changing:

- frozen ResNet-18 with ImageNet-1K V1 weights;
- `rgb_0_1_imagenet_norm_v1` preprocessing and content-keyed feature caching;
- residual bottleneck LayerNorm/GELU adapters, rank 16;
- 200 expert-training steps, 5-shot, 5 queries/class, at most 5-way;
- AdamW, learning rate 0.001, weight decay 0.0001;
- prototype temperature 0.1;
- maximum 500 training and 200 evaluation samples per class; and
- final-step checkpoint selection with configuration-hash validation.

## 5. Candidate sets and oracle scopes

For each split, instantiate the expert bank from its meta-training groups.

- `router_candidate_set = {shared, all source_* experts}`.
- `baseline_only_set = {single}`.
- `router_candidate_oracle` selects only from the router candidate set and is
  the primary Gate 1R oracle.
- `full_analysis_oracle` may also include `single`; it is secondary and cannot
  affect the outcome.
- `specialist_only_oracle` selects only from `source_*` experts and describes
  specialization; it cannot affect the outcome.

The equal-parameter `single` adapter is never a future router-selectable path.

## 6. Deterministic oracle identity

The primary accuracy oracle uses this exact order:

1. find maximum query accuracy;
2. retain paths whose accuracy differs from the maximum by at most `1e-12`;
3. among retained paths choose minimum query NLL;
4. if NLL remains exactly tied, choose lexicographically smallest expert name.

All primary best-path frequencies, modal paths, conditional masks, and ranking
stability use this identity. The minimum-NLL oracle is reported separately.
Also report exact accuracy tie rate, top-1/top-2 accuracy margin, and the
epsilon-optimal candidate set. A candidate is epsilon-optimal when its accuracy
is within `epsilon_accuracy = 0.01` of the episode maximum.

## 7. Endpoints and paired estimands

The primary metric is accuracy. The primary paired episode estimands are:

\[
\Delta_{shared}=A(router\ candidate\ oracle)-A(shared),
\]

\[
\Delta_{single}=A(router\ candidate\ oracle)-A(single),
\]

\[
\Delta_{random}=A(router\ candidate\ oracle)-A(random\ expected).
\]

`random_expected` is the exact arithmetic mean of all router-candidate
accuracies within each episode. It does not depend on finite random draws.
Ten repeated-random simulations are retained only as a secondary sanity check.

Secondary endpoints are macro F1, query NLL, oracle-mixture NLL, top-1/top-2
margin, epsilon-optimal set size, best-path entropy, ranking stability,
leave-one-expert-out loss, and conditional-mask loss. Oracle mixture remains an
analysis-only, query-label-derived ceiling and is not deployable.

## 8. Hierarchical statistics

Report episode-level paired differences, task means, split means, train-seed
means, and support-resample variability. Ordinary episode bootstrap intervals
must be reported but cannot alone determine the confirmatory outcome.

The global hierarchical bootstrap performs 10,000 replicates and, with
replacement, samples in this order:

1. split;
2. task within sampled split;
3. train seed within sampled task; and
4. support episode within sampled train seed.

For each fresh split, resample task, then train seed, then support episode. For
each task-level intervention criterion, resample train seed, then support
episode. Report percentile 95% intervals. The split-level sample size is two and
must be stated explicitly.

## 9. Intervention definitions

All interventions below are analysis-only causal diagnostics because they use
query labels directly or use task-modal identities derived from query labels.

### Core interventions

- `forced_worst`: lowest-accuracy router candidate; highest NLL and then name
  break an accuracy tie.
- `random_expected`: exact mean accuracy over router candidates.
- `repeated_random`: ten deterministic seeded candidate draws, secondary only.
- `task_modal_swap`: replace a task's modal primary path with the next
  lexicographically ordered router candidate in that split's bank, wrapping
  cyclically.
- `task_assignment_permutation`: within each split, sort its two held-out task
  names and cyclically assign each task the other task's modal primary path.
- `wrong_task_expert_assignment`: use the modal primary path from the other
  held-out task in the same split. Because both tasks share one trained bank,
  this path is always available. With two tasks this may equal the assignment
  permutation, and that redundancy must be reported rather than hidden.

### Diagnostic-only interventions

`global_most_used_mask`, `conditional_global_mask`,
`task_modal_conditional_mask`, `leave_one_expert_out`,
`source_expert_leave_one_out`, expert redundancy, and oracle mixture are
reported but cannot individually fail Gate 1R.

For conditional masks, restrict to episodes where the masked path is the
primary accuracy oracle, remove it, and apply the same deterministic oracle rule
to the remaining candidates. This measures marginal indispensability and
redundancy.

## 10. Shared requirements for PASS_HARD or PASS_SOFT

All requirements in A1-A3 are necessary.

### A1. Oracle headroom

For each of splits 6 and 36:

- mean `delta_oracle_shared > 0.01`;
- mean `delta_oracle_single > 0.01`; and
- the per-split hierarchical 95% CI lower bound is above zero for both.

At least three of four fresh tasks must have positive mean
`delta_oracle_shared`, and at least three must have positive mean
`delta_oracle_single`.

### A2. No global path dominance

Across primary router-candidate oracle identities:

- maximum best frequency must be strictly below 0.50;
- at least three paths must each have frequency at least 0.05; and
- all four tasks may not share one modal path.

Specialist-only frequencies are reported separately; shared is a legal primary
candidate.

### A3. Core intervention evidence

A task passes an intervention when mean accuracy drop is strictly above 0.01
and its task-hierarchical 95% CI lower bound is above zero.

- `forced_worst`: at least 3/4 tasks pass.
- `random_expected`: at least 2/4 tasks pass.
- assignment evidence: either `task_assignment_permutation` or
  `wrong_task_expert_assignment` must pass on at least 2/4 tasks.
- task-dependent pattern: for at least one of those two assignment
  interventions, maximum minus minimum task mean drop must be strictly above
  0.005.

## 11. PASS_HARD identity requirements

After all shared requirements pass, `PASS_HARD` additionally requires every
condition below:

1. more than 0.01 top-1/top-2 accuracy margin in at least 50% of episodes;
2. in at least 3/4 tasks, the task modal primary path agrees for at least two of
   three train seeds;
3. median within-task expert-ranking Spearman correlation across shot/support
   conditions is at least 0.30; and
4. exact accuracy tie rate is strictly below 0.20.

## 12. Outcome algorithm and authorization

- If every shared requirement and every hard-identity requirement passes,
  output `PASS_HARD`.
- If every shared requirement passes but any hard-identity requirement fails,
  output `PASS_SOFT`.
- If any shared requirement fails, output `FAIL`.

`PASS_HARD` authorizes M4 random-expected, query-only, support-prototype, hard
top-1, top-k, and soft-mixture studies in that order. `PASS_SOFT` authorizes only
random-expected, query-only, support-prototype weighting, support-conditioned
soft mixture, and shared fallback; hard top-1 cannot be the main direction.
`FAIL` stops M4 and requires `reports/h1_termination_report.md`.

## 13. Analysis-only query-label usage

At minimum, the following are analysis-only: router-candidate, full-analysis,
specialist-only, and minimum-NLL oracles; oracle mixture; sample oracle;
forced-worst; task-modal swap; task assignment permutation; wrong-task
assignment; and all conditional mask/leave-one-out diagnostics. None may receive
query labels in a deployable routing implementation.

## 14. Required outputs and provenance

The run must produce exactly the required Gate 1R artifact family:

- `reports/gate1r_report.md`
- `reports/gate1r_decision.json`
- `results/gate1r_episode_metrics.parquet`
- `results/gate1r_expert_task_matrix.csv`
- `results/gate1r_intervention_effects.csv`
- `results/gate1r_ranking_stability.csv`
- `results/gate1r_epsilon_optimal_sets.csv`
- `results/gate1r_leave_one_expert_out.csv`
- `results/gate1r_resource_usage.csv`

The decision JSON must contain every criterion boolean and raw value, frozen
thresholds, candidate sets, identity rules, preregistration commit and tag,
runtime Git SHA, clean-tree flag, task-split hashes, configuration hash,
host/GPU, elapsed time, GPU hours, test count, and schema version.

## 15. Bug and amendment procedure

On a genuine bug, stop immediately before using further fresh results. Create an
amendment document that describes the bug, its impact on any generated results
and on canonical evidence, and the exact code/config difference. Commit and
push the amendment and create a new annotated preregistration tag. Never move,
delete, or overwrite the old tag; never silently fix and rerun.

## 16. Freeze sequence

Before fresh evaluation: run the preregistration validator, Ruff, and pytest;
commit with `Preregister confirmatory Gate 1R protocol`; push the branch; create
annotated tag `gate1r-preregistered-v1` with message
`Freeze Gate 1R protocol before fresh split evaluation`; and push that tag.
Runtime records must prove the preregistration commit/tag, runtime SHA, and a
clean working tree.
