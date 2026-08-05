# Target-aware MoE × Few-shot Adaptation

This repository is the research and execution hub for a staged study of
**target-aware mixture-of-experts, few-shot adaptation, VLM/VLA, medical
imaging, and autonomous/robotic systems**.

The project is intentionally organized around falsifiable gates. The first
coding objective is not to build a complex MoE. It is to determine whether a
fixed expert bank contains usable task-specific capability, whether a few-shot
support set provides routing information beyond the query itself, and whether
route uncertainty predicts the regret of selecting the wrong expert.

## Current decision

Primary direction:

> Selective expert adaptation under unreliable support: for an unseen task
> without task ID, decide whether the support evidence justifies selecting or
> mixing specialized experts, falling back to a shared expert, requesting more
> support, or abstaining.

Research hypotheses are separated into:

- **H1a — expert value:** an episode-level oracle can outperform a shared or
  equal-parameter single adapter, and the best expert varies by task.
- **H1b — support value:** a support-conditioned router outperforms query-only
  and random routing on held-out tasks.
- **H1c — selective specialization:** route uncertainty predicts routing regret
  and supports useful fallback or abstention.
- **H2 — fixed-budget lifecycle:** only after H1 is supported, study reuse,
  update, spawn, merge, replacement, and eviction under a fixed expert budget.
- **H3 — target conflict:** retain as a stress test of H1 rather than an
  independent generic multimodal-fusion paper.
- **H4 — VLA skill insertion:** retain as a high-cost, long-term extension.

## Repository map

| Path | Purpose |
|---|---|
| `docs/01_round1_research_map.md` | Original broad research map and representative-paper table |
| `docs/02_round2_evidence_audit.md` | Original second-round evidence audit and reclassification |
| `docs/03_round3_innovation_audit_and_selection.md` | Consolidated innovation hypotheses, novelty boundaries, and experimental decisions |
| `docs/04_local_implementation_and_execution_plan.md` | Detailed implementation, gates, repository design, multi-host execution, and Windows support |
| `AGENT_PROMPT.md` | Main prompt for the local Codex agent |
| `configs/hosts.env.example` | Six-host shared-disk configuration template |
| `configs/phase1_experiments.csv` | Initial experiment matrix and expected outputs |
| `prompts/archive/` | Earlier generated reports/prompts preserved for traceability |

## Compute environment

Linux hosts sharing one disk path:

- `doraemon02`
- `doraemon03`
- `doraemon04`
- `doraemon15`
- `doraemon19`
- `doraemon20`

A Windows workstation can be used for CPU/small-data smoke tests and local
editing. The Linux implementation must not hard-code the shared mount path.
Copy `configs/hosts.env.example` to `configs/hosts.env`, set the actual paths,
and keep the latter untracked.

## MedMNIST position

MedMNIST is retained as a formal **cross-dataset unseen-task** benchmark, not
only as a smoke test. A held-out sub-dataset is absent from expert/router
training; at evaluation time the model receives a labeled support set but no
explicit task or dataset ID. The model must infer the target task from support
content. Multiple task-level splits and support resampling are required.

The first implementation should cover the compatible 2D single-label tasks.
Multi-label and ordinal tasks must be handled in separate protocols rather than
silently forced into the same prototypical-classification code path.

## Mandatory gate order

1. **Gate 1 — expert bank value**
   - shared and equal-parameter single-adapter baselines;
   - source-task expert bank;
   - expert × held-out-task matrix;
   - episode-level oracle and oracle mixture;
   - wrong/random/swap/mask interventions.
2. **Gate 2 — support value**
   - random, query-only, support-prototype, and support-soft-mixture routers;
   - support–query shuffle and support-label removal;
   - no learned uncertainty model before Gate 2 passes.
3. **Gate 3 — route uncertainty**
   - bootstrap support resampling first;
   - compare against final prediction entropy;
   - shared fallback and coverage–risk evaluation;
   - corruption tests: imbalance, label noise, cross-task outlier, duplicate
     support.
4. **External validation / H2 decision**
   - move to cross-dataset or multi-center medical classification/segmentation;
   - start expert lifecycle only if Gate 1–3 evidence is positive.

## Non-negotiable controls

- no task ID, dataset name, filesystem path, split name, or manually encoded
  source identifier may enter the router;
- report total, trainable, and activated parameters separately;
- use both capacity-matched and compute-matched comparisons;
- use multiple training seeds and repeated support sampling;
- preserve raw per-episode outputs, not only aggregate means;
- random, oracle, wrong-route, shuffle, swap, mask, and shared-only controls are
  required;
- do not interpret activation heatmaps as causal expert specialization;
- stop at a failed gate instead of hiding a negative result with extra modules.

## Starting the local agent

Give the local Codex agent the contents of `AGENT_PROMPT.md`. It should create a
working branch, audit all six hosts and the shared path, implement M0–M3 first,
run the Gate 1 pilot, and continue automatically only when the documented gate
criteria pass.

## Citation and metadata note

The two raw Deep Research reports preserve session-specific citation markers.
Those markers are useful for traceability inside the originating session but
are not a stable bibliography. Before paper writing, replace them with official
proceedings links, DOI/arXiv identifiers, and a checked BibTeX library. Known
metadata corrections and uncertainty are recorded in the third-stage report.
