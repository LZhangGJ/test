# Target-aware MoE × Few-shot Adaptation
## Third-stage innovation audit, research selection, and falsifiable experiment plan

## 1. Purpose and source handling

This document consolidates the first-round research map, the second-round
evidence audit, the third-stage Deep Research result supplied by the researcher,
and the subsequent implementation discussion.

The original reports are preserved unchanged in this repository. Where those
reports disagree, this document records the disagreement rather than silently
rewriting the source. In particular, publication status and mechanism claims
must be checked against official proceedings before use in a paper.

## 2. Final direction ranking

| Rank | Direction | Role | Decision |
|---:|---|---|---|
| 1 | H1: unreliable-support, uncertainty-aware selective expert adaptation | Primary | Implement now |
| 2 | H2: fixed-budget expert reuse/update/spawn/merge/evict lifecycle | Backup | Start only after H1 gates pass |
| 3 | H4: few-shot VLA skill insertion without full router retraining | Long-term | Simulation first, not initial experiment |
| 4 | H3: multi-source target conflict-aware routing | Extension | Use as H1 stress test, not standalone generic fusion work |

The central research question is not whether an adapter, LoRA, support encoder,
or MoE layer can be added. It is:

> When task identity is not given, does a small support set contain enough
> evidence to justify expert specialization; when that evidence is unreliable,
> can the system detect routing risk and choose a safer degree of specialization?

## 3. Literature anchors and novelty boundaries

### 3.1 Strict and near-strict anchors

- **SMAT, ICML 2024:** the strongest direct precedent for a support set
  generating task-specific sparse expert interpolation. Therefore, “support
  enters expert composition” is not a standalone novelty claim.
- **CME-MoE, CVPR 2026:** the strongest precedent for few-shot class/domain/
  hybrid increments combined with conditional expert reuse and meta-expansion.
  Therefore, “few-shot reuse or expand” alone is not a sufficient claim.
- **R2-T2, ICML 2025:** a close test-time rerouting precedent using neighboring
  correctly predicted samples, but not the same as an episodic support-defined
  unseen task.

### 3.2 Continual and medical expert systems

- MoE-Adapters / MoE-Adapters++: continual VLM adaptation and dynamic adapter
  selection, but not strict support-conditioned episodic adaptation.
- Sparse Spectral LoRA: medical VLM routed LoRA and continual adaptation, but
  support does not define the router.
- Low-Rank MoE for continual medical segmentation: task/class expert expansion
  with task-library assumptions, no fixed-budget merge/evict protocol.
- MoIE continual TTA: non-learned similarity routing and incremental experts;
  close to “add experts without full router retraining,” but it is unlabeled
  domain adaptation rather than semantic few-shot support.
- MAST-Pro: prompts and medical experts, but prompt information must not be
  assumed to directly enter the router without formula-level verification.

### 3.3 VLA and autonomous-driving anchors

- MoIRA: frozen semantic routing among registered VLA agents; does not learn a
  new skill from few demonstrations.
- PoCo: policy composition without a unified support-conditioned learned router.
- OpenVLA and Octo: strong VLA adaptation backbones, but not expert-routing
  methods.
- MoEActok: skill-aware action-token experts, but not few-shot skill insertion.
- DriveMoE: official CVPR 2026 work with scene-specialized vision experts and
  skill-specialized action experts. It does not provide the proposed
  support-conditioned new-skill adaptation protocol.

### 3.4 What is already highly homogeneous

The following cannot be the main paper contribution by themselves:

- task token or task embedding into an MLP router;
- standard top-k gate plus load-balancing loss;
- one adapter/LoRA per known task;
- replacing a dense FFN with sparse MoE;
- mixture of adapters or mixture of LoRA without a new problem protocol;
- freezing a backbone and training only router/PEFT parameters;
- support prototype followed by a generic softmax gate;
- transferring a standard MoE to medical segmentation without new clinical or
  deployment evidence;
- mechanically combining prompt, adapter, LoRA, and MoE modules.

## 4. Falsifiable hypotheses

### H1a — Expert-value hypothesis

A fixed expert bank contains stable, task-dependent capability differences on
held-out tasks. The episode-level oracle expert or oracle mixture should
outperform a shared expert and an equal-parameter single adapter. Different
held-out tasks should not all prefer the same expert.

**Falsification:** oracle gap is negligible; one expert is globally best;
wrong/random/swap/mask interventions do not create task-specific losses.

### H1b — Support-value hypothesis

Without task ID, the labeled support set provides routing information beyond
what can be inferred from the query alone.

Expected ordering:

```text
support-conditioned router > query-only router > random router
```

**Falsification:** support routing does not beat query-only; shuffled
support-query pairs do not reduce performance; removing support labels has no
meaningful effect; learned routing lies inside the random-router distribution.

### H1c — Selective-specialization hypothesis

Instability or disagreement induced by support resampling/corruption predicts
routing regret. A shared fallback or reduced specialization improves
coverage–risk when support evidence is weak.

**Falsification:** route uncertainty is near random for high-regret episodes;
final prediction entropy is equally good or better; fallback does not improve
the risk–coverage curve; adding independent support does not reduce uncertainty
or regret.

### H2 — Fixed-budget lifecycle hypothesis

Under a fixed total expert budget, support-conditioned reuse/update/spawn/merge/
replace decisions outperform expand-only and one-task-one-expert strategies in
long-term performance, forgetting, and redundancy.

This hypothesis is not started until H1 establishes that experts have causal,
routable functional differences.

### H3 — Target-conflict extension

When text, labels, support images, goal images, or metadata conflict, the system
should change expert selection, specialization strength, fallback, or
abstention. Generic multimodal conflict detection is already studied; the
remaining novelty must be route-specific and intervention-validated.

### H4 — Few-shot VLA skill insertion

Language plus a few demonstrations should determine reuse/compose/spawn/reject
for skill experts without full router retraining, while preserving old skills
and supporting rollback. A simple “language router + several OpenVLA LoRAs” is
not sufficiently distinct from existing work.

## 5. MedMNIST as a formal unseen-task benchmark

MedMNIST remains an explicit project dataset, not merely a code smoke test.
Each compatible sub-dataset is treated as a task. At meta-test time, the entire
task has been absent from expert/router training; the system receives a labeled
support set and must classify its query set without explicit task or dataset ID.

This is intentionally a **cross-dataset unseen-task** setting. Visible
cross-dataset characteristics do not invalidate the setting; they are part of
the information from which the task must be inferred. However, the study must
avoid claiming that this alone proves fine-grained cross-hospital clinical
adaptation.

### Initial compatible 2D tasks

- PathMNIST
- DermaMNIST
- OCTMNIST
- PneumoniaMNIST
- BreastMNIST
- BloodMNIST
- TissueMNIST
- OrganAMNIST
- OrganCMNIST
- OrganSMNIST

The three OrganMNIST views should be grouped during task splitting to reduce
source-family leakage. ChestMNIST (multi-label) and RetinaMNIST (ordinal) should
be implemented as separate protocols rather than silently mapped into the
single-label episodic classifier.

### Required task split protocol

- multiple task-level split seeds;
- disjoint meta-train, meta-validation, and meta-test task sets;
- no task ID, dataset name, file path, or split-name input to the router;
- repeated support sampling per held-out task;
- 1-, 2-, 5-, 10-, and 16-shot conditions;
- episode-local classifier compatible with variable label spaces;
- raw per-episode metrics retained.

## 6. First expert-bank experiment

Start with a frozen backbone and interpretable, fixed experts:

- one shared adapter/LoRA trained on all meta-training tasks;
- one source-task or source-task-group expert per meta-training task/group;
- identical expert rank/architecture;
- frozen backbone;
- episode-local prototype or linear head;
- no learned uncertainty model.

This isolates whether there is anything useful to route. Symmetric jointly
trained experts are deferred because collapse can obscure whether a negative
result comes from the expert bank or the router.

## 7. Oracle definitions

### Episode-level oracle expert

For a complete query set, select the expert with the lowest query loss or
highest task metric. This is the main diagnostic upper bound because the router
receives an episode-level support set.

### Oracle mixture

Post-hoc optimize mixture weights on query labels. This estimates the upper
bound of continuous expert composition and is analysis-only.

### Sample-level oracle

Select a separate expert per query sample. This is a stronger, less realistic
upper bound and must not be used as the primary oracle-gap denominator.

### Routing regret

For an episode:

```text
routing_regret = loss(selected expert, query) - loss(episode oracle, query)
```

For higher-is-better metrics, report the equivalent metric gap with consistent
sign conventions.

## 8. Minimum baseline matrix

1. frozen zero-shot/fixed-feature baseline;
2. shared expert only;
3. equal-parameter single adapter/LoRA;
4. fixed task-ID expert as an analysis upper bound;
5. random router with repeated draws;
6. query-only router;
7. support prototype router;
8. support-conditioned soft mixture;
9. episode-level oracle expert;
10. oracle mixture;
11. final-prediction-confidence fallback;
12. later, SMAT-like and DETA-like baselines after the basic signal is proven.

Both capacity-matched and compute-matched comparisons are required. No single
baseline family can simultaneously match all total, activated, and trainable
parameters; the report must show each quantity explicitly.

## 9. Initial support stress tests

Keep the first stress-test matrix constrained:

- class imbalance;
- label flips;
- cross-task outlier samples;
- duplicated samples that inflate nominal shot count.

After positive results, extend to missing classes, mixed domains, text/support
conflict, adversarial descriptions, and request-more-support policies.

## 10. Route uncertainty before learned uncertainty heads

Use bootstrap resampling of the support set and measure:

- expert-selection frequency entropy;
- top-1 expert switching rate;
- variance of mixture weights;
- disagreement across support resamples.

Compare these with final prediction entropy. A learned uncertainty head is only
justified if these low-parameter statistics reveal routing-specific signal but
are insufficiently accurate.

A generic fallback form is:

```text
output = alpha(support) * specialized_mixture
       + (1 - alpha(support)) * shared_expert
```

The contribution is not this formula. The contribution must be evidence that
support reliability controls the value and risk of specialization.

## 11. Causal specialization tests

Required interventions include:

- forced wrong routing;
- random routing;
- expert swapping;
- expert permutation;
- masking the most-used expert;
- masking the shared expert;
- cross-task expert transfer;
- support-query shuffle;
- freeze-router/train-experts and freeze-experts/train-router;
- merge tests and redundancy tests after H2 begins.

Activation heatmaps, load balance, and t-SNE are descriptive only. Functional
specialization requires task-specific intervention losses and predictable
oracle gaps.

## 12. Decision gates

### Gate 1 — expert bank value

Pass only when oracle advantage is stable across multiple task splits/support
samples, the best expert varies across tasks, and wrong-route interventions
produce task-specific losses.

### Gate 2 — support value

Pass only when support routing beats query-only and random routing, and support
shuffle/label removal damages performance.

### Gate 3 — routing uncertainty

Pass only when uncertainty predicts high routing regret better than final
prediction entropy and fallback improves the full risk–coverage curve.

A failed gate terminates or narrows the claim. The agent must not hide a failed
gate by adding modules.

## 13. External validation sequence

1. MedMNIST cross-dataset unseen-task episodes;
2. cross-dataset 2D medical segmentation;
3. multi-center or multi-protocol medical classification/segmentation;
4. medical VLM with text/support conflict;
5. H2 fixed-budget lifecycle;
6. LIBERO or comparable simulation for H4;
7. autonomous-driving offline tasks before closed-loop evaluation.

## 14. Compute and execution strategy

Use the six Linux hosts for embarrassingly parallel task-split/seed/config
runs. Do not begin with distributed data-parallel training. Shared-disk writes
must use unique run directories and atomic finalization. Feature extraction can
be cached once per backbone/data/preprocessing version. Windows is supported
for synthetic tests, dataset sampling checks, metrics, and small CPU/GPU runs.

## 15. Immediate deliverable

The first result must be:

- expert × held-out-task performance matrix;
- episode-level oracle gap by task, shot, split, and seed;
- shared and equal-parameter single-adapter comparisons;
- expert best-frequency and global-dominance analysis;
- initial wrong/random/swap/mask interventions;
- machine-generated Gate 1 PASS/FAIL report.

Only after this deliverable supports H1a should the project implement learned
support routing.
