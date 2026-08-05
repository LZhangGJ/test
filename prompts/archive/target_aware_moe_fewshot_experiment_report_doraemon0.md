# Target-aware MoE × Few-shot Adaptation
## 研究实施报告与 doraemon0 本地开发规范

**文档状态：** 实验立项与实现规格（Phase 0–1）  
**主机：** `doraemon0`  
**主要研究方向：** 不可靠 few-shot support 下的选择性专家适配  
**首个实验域：** MedMNIST2D 未见任务分类  
**后续验证域：** 跨中心/跨数据集医学分类与分割  

---

## 1. 文档目的

本报告将前三阶段文献调研和创新性审计收敛为一套可直接编码、运行和证伪的实验计划。它用于指导本地 Codex 在 `doraemon0` 上建立代码仓库、实现基线、运行实验并生成可复核结果。

当前不以“完成一套复杂模型”为目标，而以依次回答三个基础问题为目标：

1. **专家库是否存在可利用的任务相关差异？**
2. **few-shot support 是否提供了 query 本身之外的有效路由信息？**
3. **support 可靠性或路由不确定性是否能预测错误专家造成的额外损失，并指导回退或拒识？**

任意前置问题未通过时，不继续增加后续模块。

---

## 2. 研究结论与方向排序

### 2.1 主推荐方向 H1

**Selective Expert Adaptation under Unreliable Support**

研究无 task ID 的未见任务中，few-shot support 在类别不平衡、标签噪声、异常样本、重复样本或 support-query 分布偏移下，是否仍能可靠地决定专家选择、软组合或共享路径回退。

主贡献应落在以下两个层级：

- **Level 1：研究问题与评估协议**  
  不可靠 support 如何影响专家特异化决策、routing regret、coverage-risk 和拒识。
- **Level 2：学习目标与决策机制**  
  如何校准错误路由风险，以及何时选择专家、降低特异化、使用 shared expert 或拒识。

以下内容本身不构成核心创新：support encoder、LoRA、adapter、top-k MoE、uncertainty head、共享专家或 bootstrap。

### 2.2 备用方向 H2

**固定预算下的专家复用、扩展、合并与淘汰。**

只有在 H1 证明专家具有稳定的功能差异、support 能恢复部分 oracle gap 后，才进入 H2。否则生命周期管理只是在管理冗余参数。

### 2.3 扩展方向 H3

**多源 target 冲突。**

文本、support、目标图像或元数据冲突暂不单独立项，作为 H1 的后续压力测试。只有冲突对专家路由产生稳定、可校准且具有任务特异性的损害时，再考虑独立扩展。

### 2.4 高风险长期方向 H4

**few-shot VLA 新技能专家插入与安全回滚。**

该方向保留为医学实验得到正结果后的迁移方向，不进入首轮实现范围。

---

## 3. 文献边界与新颖性约束

本项目必须以以下直接先例为边界：

| 工作 | 已覆盖内容 | 本项目不能重复声称的创新 | 本项目仍可研究的差异 |
|---|---|---|---|
| SMAT，ICML 2024 | support-conditioned sparse expert interpolation、few-shot meta-tuning、OOD tasks | “support 直接控制专家组合” | support 质量、错误路由风险、fallback、拒识、route-specific calibration |
| CME-MoE，CVPR 2026 | few-shot class/domain/hybrid incremental learning、expert reuse 与 meta-expansion | “few-shot 数据决定专家复用或扩展” | 固定预算、merge/evict、support 可靠性、route abstention |
| DETA，ICCV 2023 | noisy support 下的 few-shot task adaptation 与去噪 | “发现 support 噪声会损害 few-shot adaptation” | 噪声如何导致错误参数路径，以及 route-specific risk |
| R2-T2，ICML 2025 | 利用测试邻域正确样本调整多模态 MoE routing weights | “测试时使用邻近样本重路由” | episode-level task support、unseen-task routing、fallback/abstention |
| MoE-Adapters / MoE-Adapters++ | VLM 持续学习中的动态 adapter experts | “动态 adapter 专家或持续扩展” | strict support-defined unseen task、route calibration、固定预算 |
| Sparse Spectral LoRA，CVPR 2026 | 医学 VLM routed LoRA 与持续适配 | “医学 VLM 中使用 routed LoRA experts” | strict few-shot support、错误路由、拒识和因果专门化 |
| DriveMoE，CVPR 2026 | 自动驾驶 VLA 的 scene-specialized vision MoE 与 skill-specialized action MoE | “场景/技能驱动 VLA 专家” | few-shot 新技能插入、router 冻结、拒识与安全回滚 |

**新颖性冻结表述：**

> 本项目不主张发明 support-conditioned MoE，而研究在无 task ID 的未见医学任务中，support 证据不可靠时专家特异化是否仍然成立，以及 route uncertainty 能否预测并降低相对 oracle expert 的 routing regret。

---

## 4. MedMNIST 的实验定位

### 4.1 采用原则

MedMNIST 明确保留为本项目的首个正式实验数据集，而不只作为 smoke test。

本项目将每个 MedMNIST 子数据集视为一个独立任务。完整 hold-out 子数据集在 meta-training 和 router 训练阶段不可见，因此测试时需要从 support 推断一个未见任务。模型通过图像外观和 support 分布识别任务属于允许的信息利用，而不是协议泄漏；这正是 task-free support-conditioned routing 所要评价的能力。

必须遵守：

- router 不接收数据集名称、task ID 或手工任务编号；
- 测试任务的所有图像在 meta-training、专家训练和 router 训练阶段不可见；
- query 标签只用于最终评价和 oracle 分析；
- support 标签允许进入 prototype、support encoder 和路由机制；
- `support-query shuffle` 必须作为关键反事实实验；
- 具有共同原始来源的任务族应整体分组，避免源病例或原始样本重叠，而不是禁止模型从任务外观中推断任务。

### 4.2 首轮纳入任务

v0 优先纳入能够统一使用 episodic prototype classification 的二分类或多分类任务：

- `pathmnist`
- `dermamnist`
- `octmnist`
- `pneumoniamnist`
- `breastmnist`
- `bloodmnist`
- `tissuemnist`
- `organamnist`
- `organcmnist`
- `organsmnist`

v0 暂不纳入：

- `chestmnist`：多标签任务，需单独的 prototype/metric 定义；
- `retinamnist`：有序分类，需单独评价 ordinal structure。

它们在 v1 中作为任务类型泛化扩展加入。

### 4.3 Task group 约束

以下三个任务视为同一原始任务族，不跨 meta-train/validation/test 分散：

```yaml
organ_family:
  - organamnist
  - organcmnist
  - organsmnist
```

这项约束防止原始来源重叠，不改变“通过 support 推断未见任务”的研究设定。

### 4.4 评估层级

1. **开发折（MVP）**：一个 hold-out task group，完成全部代码和 Gate 1–3。
2. **核心结果**：至少 5 个 hold-out task/task-group folds。
3. **完整 MedMNIST 结果**：对所有 8 个独立 task groups 执行 leave-one-group-out。
4. **外部验证**：随后增加跨中心分类或跨数据集分割，验证结论不只依赖 MedMNIST。

---

## 5. 预注册研究假设

### H1a：Expert-value hypothesis

> 在未见任务的 query set 上，episode-level oracle expert 应稳定优于 shared expert 和等参数 single adapter；否则专家库不存在值得路由的功能差异。

#### 证伪条件

- oracle expert 与 shared/single adapter 的差异接近零；
- 所有专家在所有测试任务上的排序基本相同，即只有一个全局最强专家；
- 专家交换、错误路由和屏蔽不产生任务特异损失。

### H1b：Support-value hypothesis

> 不提供 task ID 时，support-conditioned router 应优于 random router 和 query-only router；打乱 support-query 对应后性能应明显下降。

#### 证伪条件

- support router 不优于 query-only router；
- support-query shuffle 后性能不下降；
- 移除 support 标签后结果不变；
- support router 的优势完全由 task 数据规模或简单图像统计解释，且 query-only 可达到同样结果。

### H1c：Selective-specialization hypothesis

> support 引起的路由不确定性应预测相对 episode-level oracle 的 routing regret；在高风险 episode 上回退到 shared expert 或降低特异专家权重，应改善 coverage-risk。

#### 证伪条件

- route uncertainty 对高 routing regret 的 AUROC/AUPRC 接近随机；
- uncertainty 只预测最终分类错误，不预测错误专家的额外损失；
- fallback 不优于始终选择 learned expert；
- 最终预测 entropy 与 route-specific uncertainty 同样有效。

---

## 6. 首轮模型设计：先隔离研究问题

### 6.1 总体架构

首轮使用冻结视觉 backbone 和 embedding-level residual adapters，不直接实现复杂的 Transformer 内部 LoRA-MoE。

原因：

- 可以预计算所有 frozen features，大幅降低实验成本；
- 更容易隔离“专家差异、support 路由和 uncertainty”三个问题；
- 避免内部 MoE 训练不稳定或专家坍缩混淆研究结论；
- 若研究假设成立，再替换为 LoRA/adapter-in-backbone。

### 6.2 默认 backbone

按资源从低到高支持：

```yaml
backbones:
  default: torchvision/resnet18_imagenet1k
  secondary: timm/vit_small_patch16_224
  optional_medical: configurable_external_encoder
```

MVP 默认使用 ImageNet 预训练 ResNet-18，输出冻结 embedding。代码必须允许通过配置替换 backbone。

### 6.3 Expert bank

首轮采用**两阶段固定专家库**：

1. 每个 meta-train task group 训练一个相同结构的 residual feature adapter；
2. 在全部 meta-train tasks 上训练一个 shared adapter；
3. 测试未见任务时，所有 experts 和 backbone 均冻结；
4. router 根据 support 决定选择或组合 experts。

推荐 adapter：

```text
LayerNorm(d)
→ Linear(d, bottleneck)
→ GELU
→ Dropout
→ Linear(bottleneck, d)
→ residual addition
```

所有专家使用相同 bottleneck、初始化方式和训练预算。

### 6.4 可变类别数处理

不使用固定 task-specific classifier。每个 episode 在 expert 变换后的 embedding 上构造 prototype classifier：

\[
\mu_c^{(e)} = \frac{1}{|S_c|}\sum_{(x_i,y_i=c)\in S} g_e(f(x_i)),
\]

\[
p(y=c\mid x,e,S) \propto
\exp\{-d(g_e(f(x)),\mu_c^{(e)})/\tau\}.
\]

因此不同 MedMNIST 任务可以具有不同类别数量，测试任务不需要预先训练分类头。

### 6.5 Episode 模式

代码必须支持两种模式：

- `fixed_way`：`N=min(config.n_way, task_num_classes)`；用于快速且标准化比较；
- `all_way`：使用任务全部类别；用于最终 task-level 结果。

默认 MVP：

```yaml
episode:
  mode: fixed_way
  n_way: 5
  k_shot: [1, 2, 5, 10, 16]
  q_query_per_class: 10
```

二分类任务自动使用 2-way。

---

## 7. Router 与 Oracle 定义

### 7.1 必须实现的 router

1. `shared_only`
2. `random_top1`
3. `random_soft`
4. `query_only_mlp`
5. `support_prototype_nearest`
6. `support_soft_mixture`
7. `support_set_encoder`（Gate 2 通过后实现）
8. `uncertainty_fallback`（Gate 3 阶段实现）

### 7.2 Episode-level oracle expert

主 oracle 在整个 query set 上选择表现最佳专家：

\[
e^*_{episode}=
\arg\min_e \frac{1}{|Q|}
\sum_{(x,y)\in Q}\mathcal L(f_e(x;S),y).
\]

这是主要 routing upper bound。

### 7.3 Oracle mixture

使用 query 标签事后优化专家混合权重，仅用于判断 soft composition 的理论上界，不作为可部署方法。

### 7.4 Sample-level oracle

允许每个 query 选择不同专家，只作补充强上界，不用于主 OGR 指标。

### 7.5 Routing regret

\[
R(S,Q)=
L(f_{\hat e(S)},Q)-L(f_{e^*_{episode}},Q).
\]

若使用 soft mixture，则与 oracle mixture 或 episode oracle 分别报告。

### 7.6 Oracle-gap recovery

\[
OGR=
\frac{P_{learned}-P_{shared}}
{P_{oracle}-P_{shared}}.
\]

当分母接近零时，不报告 OGR，并直接判定 Gate 1 不成立。

---

## 8. 最小 baseline 矩阵

首轮先实现以下基线，不立即复现所有相关论文：

| ID | Baseline | 目的 |
|---|---|---|
| B00 | frozen backbone + prototype | 适配前基础性能 |
| B01 | shared adapter | shared path 基线 |
| B02 | equal-parameter single adapter | 排除多专家总参数优势 |
| B03 | task-ID fixed expert（分析上限） | 已知任务时的专家上限 |
| B04 | random top-1 router | 判断 learned routing 是否有效 |
| B05 | random soft router | soft mixture 的随机基线 |
| B06 | query-only router | 判断 support 是否提供额外信息 |
| B07 | support prototype nearest router | 最简单 support 路由 |
| B08 | support-conditioned soft mixture | 判断稀疏选择是否必要 |
| B09 | episode-level oracle expert | 专家可路由价值上限 |
| B10 | oracle mixture | 软组合上限 |
| B11 | prediction-confidence fallback | 判断 route uncertainty 是否有独立价值 |

Gate 1–2 通过后再实现：

- SMAT-like sparse support interpolation；
- DETA-like support denoising；
- learned permutation-invariant support encoder；
- calibrated route-risk estimator。

---

## 9. Support corruption 与压力测试

第一轮仅实现四类 corruption，避免矩阵失控：

### C1 类别不平衡

- balanced；
- 70/30；
- 80/20；
- 单类占 90%；
- 在 `all_way` 模式下缺少一个类别。

### C2 标签噪声

- 5%；
- 10%；
- 20%；
- 40% label flip。

### C3 跨任务 outlier

在 support 中混入其他 task 的图像，比例：

- 10%；
- 20%；
- 40%。

保持 outlier 标签可配置：随机错误标签或新建 unknown 标签（后者为扩展）。

### C4 重复样本

保持表面 support size 不变，但降低独立样本数：

- 无重复；
- 50% 重复；
- 每类仅一个独立样本重复 K 次。

必须保留每个 episode 的 corruption 元数据，确保结果可以按实际有效样本数聚合。

---

## 10. 首轮不确定性：无参数或低参数优先

### 10.1 Bootstrap route uncertainty

对 support 进行 B 次有放回重采样：

```yaml
uncertainty:
  bootstrap_samples: 30
```

对每个重采样 support 计算路由分布，输出：

- top-1 expert switch rate；
- expert-selection entropy；
- mixture-weight variance；
- Jensen–Shannon divergence；
- shared-vs-specialized margin variance。

### 10.2 Shared fallback

最小版本：当 route-risk score 高于 validation 阈值时使用 shared expert。

第二版本：连续调节专家特异化强度：

\[
f(x,S)=\alpha(S)\sum_e p(e\mid S)f_e(x)
+[1-\alpha(S)]f_{shared}(x).
\]

阈值和 `alpha` 映射只能在 validation tasks 上选择，不得用 test task 标签调节。

### 10.3 拒识

首轮的拒识单位为 **episode**，不是单个 query：

- 接受：使用特定专家或 mixture；
- 回退：使用 shared expert；
- abstain：不输出 episode 结果，仅用于 selective-risk 分析。

query-level 拒识放到后续扩展。

---

## 11. 评价指标

### 11.1 任务性能

- accuracy；
- macro-F1；
- AUROC（适用时）；
- per-task / per-fold performance；
- worst-task；
- lower-decile episode performance。

### 11.2 Few-shot 稳定性

每个 task × shot × corruption 至少执行：

- MVP：20 次 support 重采样；
- 正式结果：50 次 support 重采样；
- 至少 5 个训练种子。

报告：

- mean；
- standard deviation；
- bootstrap 95% CI；
- worst-support；
- 10th percentile；
- seed variance。

### 11.3 Routing

- episode oracle gap；
- routing regret；
- OGR；
- top-1 router accuracy（相对 episode oracle）；
- top-k oracle recall；
- route entropy；
- wrong-route detection AUROC/AUPRC；
- forced-wrong-routing performance drop。

### 11.4 Calibration 与 selective prediction

- ECE；
- Brier score；
- coverage-risk curve；
- AURC；
- risk@80%、90%、95% coverage；
- route uncertainty 与 routing regret 的 Spearman/Pearson 相关；
- prediction entropy 与 routing regret 的同类相关，作为对照。

### 11.5 效率

分开报告：

- backbone 参数；
- expert-bank 总参数；
- 单次激活参数；
- router 参数；
- support encoder 参数；
- support 编码时间；
- query latency；
- peak VRAM；
- feature-cache 大小；
- 每个完整 fold 的 GPU-hours。

---

## 12. 因果专门化实验

必须实现以下干预，而不是只画激活热图：

1. `forced_best_oracle`
2. `forced_worst_expert`
3. `random_router_repeated`
4. `expert_swap`
5. `global_expert_permutation`
6. `mask_most_used_expert`
7. `mask_shared_expert`
8. `support_query_shuffle`
9. `support_label_removal`
10. `fixed_router_train_experts`
11. `fixed_experts_train_router`
12. `expert_merge_simulation`

支持“专家具有任务相关能力”至少需要：

- oracle experts 在不同 tasks 上不同；
- learned router 能恢复明显 oracle gap；
- forced-wrong、swap、mask 产生任务特异的损失；
- 结果在多个 folds、shots 和 support 重采样下稳定；
- 等参数 single adapter 不能解释提升。

以下证据不充分：

- t-SNE 分群；
- 专家使用频率不同；
- 负载均衡更好；
- 激活热图不同。

---

## 13. 三个决策门与 kill criteria

### Gate 1：Expert bank 是否有价值

通过条件建议：

- episode oracle 相对 shared/equal-parameter single adapter 在多个 test tasks 上稳定为正；
- 95% CI 不跨零；
- 不同 test tasks 的最佳专家不完全相同；
- wrong-route 干预产生任务特异损失。

立即停止条件：

- oracle gap 基本为零；
- 一个专家在所有任务中全局最强；
- wrong/random/swap 与正确路由差异很小。

### Gate 2：Support 是否有额外价值

通过条件建议：

```text
support_router > query_only_router > random_router
```

且：

- support-query shuffle 明显降低性能；
- 移除 support labels 会损害路由；
- 支持在多个 unseen tasks 上成立。

立即停止或改题条件：

- support router 不优于 query-only；
- shuffle 后不下降；
- support 只在一个 task 有效。

### Gate 3：Route uncertainty 是否有独立价值

通过条件建议：

- uncertainty 对 high-regret episodes 的 AUROC 明显高于 0.5；
- 优于 final prediction entropy；
- fallback 改善整个 coverage-risk curve；
- 增加有效 support 后 uncertainty 和 regret 整体下降。

修改假设条件：

- uncertainty 只预测最终分类错误：改为 general selective adaptation；
- 仅在 domain-like tasks 有效：收缩为 domain-aware support adaptation；
- soft mixture 有效而 sparse top-1 无效：放弃 sparse routing 主张。

### H2 启动门槛

只有同时满足以下条件才进入专家生命周期：

1. Gate 1–3 全部通过；
2. 部分 experts 在多个 tasks 上冗余；
3. shared expert 无法替代全部 specialized experts；
4. learned router 能稳定恢复至少一部分 oracle gap。

---

## 14. 公平性控制

不强求所有 baseline 在一张表中同时匹配全部资源，而分两组比较。

### 14.1 Capacity-matched

匹配：

- 可训练参数；
- optimizer steps；
-训练 episode 数；
- backbone；
- 数据增强；
- 超参数搜索预算。

### 14.2 Compute-matched

匹配：

- 每个 query 激活参数；
- forward FLOPs；
- 激活 expert 数；
- support 编码开销；
- batch size 或有效吞吐。

每个实验必须记录：

```yaml
resource_accounting:
  total_parameters:
  trainable_parameters:
  active_parameters_per_query:
  backbone_flops:
  router_flops:
  expert_flops:
  support_encoding_ms:
  query_latency_ms:
  peak_vram_mb:
```

---

## 15. doraemon0 环境审计

Codex 开始编码前必须运行并保存：

```bash
hostname
uname -a
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
free -h
df -h
python3 --version
git --version
```

保存到：

```text
artifacts/system/doraemon0_environment.txt
```

不得预设 GPU 数量、CUDA 版本或显存。根据实际环境自动选择：

- 单 GPU；
- 多 GPU；
- CPU smoke test。

---

## 16. 推荐代码仓库结构

```text
target-aware-moe-fewshot/
├── README.md
├── pyproject.toml
├── requirements.lock
├── Makefile
├── configs/
│   ├── base.yaml
│   ├── datasets/
│   │   ├── medmnist2d.yaml
│   │   └── medmnist_folds.yaml
│   ├── models/
│   │   ├── resnet18_feature_adapter.yaml
│   │   └── vit_small_feature_adapter.yaml
│   ├── routers/
│   │   ├── random.yaml
│   │   ├── query_only.yaml
│   │   ├── support_prototype.yaml
│   │   ├── support_soft.yaml
│   │   └── uncertainty_fallback.yaml
│   └── experiments/
│       ├── e00_smoke.yaml
│       ├── e01_expert_oracle.yaml
│       ├── e02_support_value.yaml
│       ├── e03_corruption.yaml
│       └── e04_uncertainty_fallback.yaml
├── src/
│   └── tamoe/
│       ├── __init__.py
│       ├── data/
│       │   ├── medmnist.py
│       │   ├── episodes.py
│       │   ├── folds.py
│       │   └── corruptions.py
│       ├── models/
│       │   ├── backbones.py
│       │   ├── adapters.py
│       │   ├── expert_bank.py
│       │   └── prototypes.py
│       ├── routing/
│       │   ├── base.py
│       │   ├── random.py
│       │   ├── query_only.py
│       │   ├── support_prototype.py
│       │   ├── support_soft.py
│       │   ├── oracle.py
│       │   └── uncertainty.py
│       ├── training/
│       │   ├── train_experts.py
│       │   ├── train_router.py
│       │   └── loops.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── routing_metrics.py
│       │   ├── interventions.py
│       │   └── selective_risk.py
│       ├── utils/
│       │   ├── reproducibility.py
│       │   ├── logging.py
│       │   ├── checkpoints.py
│       │   └── system.py
│       └── cli.py
├── scripts/
│   ├── audit_system.sh
│   ├── download_medmnist.py
│   ├── cache_features.py
│   ├── run_experiment.py
│   ├── run_fold_matrix.py
│   └── aggregate_results.py
├── tests/
│   ├── test_episode_sampler.py
│   ├── test_task_folds.py
│   ├── test_corruptions.py
│   ├── test_oracle_router.py
│   ├── test_routing_regret.py
│   └── test_smoke_pipeline.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── feature_cache/
├── checkpoints/
├── runs/
└── artifacts/
    ├── system/
    ├── tables/
    ├── figures/
    └── reports/
```

---

## 17. 工程实现要求

### 17.1 技术栈

建议：

- Python 3.11；
- PyTorch；
- torchvision；
- timm；
- medmnist；
- numpy / scipy；
- pandas；
- scikit-learn；
- pydantic 或 dataclass 配置校验；
- PyYAML；
- TensorBoard；
- pytest；
- ruff；
- mypy（核心模块）。

优先使用 `uv`；若主机没有 `uv`，使用标准 `venv`。不得因为环境工具缺失中断工作，应选择可用方案并记录。

### 17.2 CLI 优先

Notebook 仅用于临时分析，不作为主训练入口。所有实验必须可由 CLI 重现：

```bash
python -m tamoe.cli cache-features --config configs/experiments/e00_smoke.yaml
python -m tamoe.cli train-experts --config configs/experiments/e01_expert_oracle.yaml
python -m tamoe.cli evaluate-router --config configs/experiments/e02_support_value.yaml
python -m tamoe.cli run-matrix --config configs/experiments/e03_corruption.yaml
python -m tamoe.cli aggregate --runs-dir runs/
```

### 17.3 Feature cache

由于 v0 backbone 冻结，优先缓存 base embeddings：

- 使用 `float16` 或 `bfloat16` 存储；
- 保存 sample ID、task、split、label、原始 index；
- cache key 包含 backbone 名称、权重版本、输入分辨率和 preprocessing hash；
- 发现 hash 不一致时禁止静默复用旧 cache。

### 17.4 复现性

每个 run 保存：

```text
config_resolved.yaml
git_commit.txt
environment.txt
seed.txt
metrics.json
episode_metrics.parquet
routing_outputs.npz
resource_usage.json
stdout.log
stderr.log
```

固定：

- Python seed；
- NumPy seed；
- PyTorch seed；
- DataLoader generator；
- episode sampler seed。

### 17.5 错误处理

- 不允许 silent exception；
- 数据不足以构造 episode 时给出明确错误；
- OOM 时保存失败配置和显存信息；
- 检查 NaN/Inf；
- checkpoint 包含 optimizer、scheduler、epoch 和 RNG states；
- 支持从中断点恢复。

---

## 18. Codex 开发里程碑

### M0：环境与仓库

交付：

- 环境审计文件；
- Python 环境；
- 仓库结构；
- `pytest` 和 `ruff` 可运行；
- README 中写明命令。

验收：

```bash
pytest -q
ruff check .
python -m tamoe.cli --help
```

### M1：MedMNIST 与 episodic sampler

交付：

- 下载、缓存和读取 v0 tasks；
- task-group folds；
- fixed-way/all-way episode；
- support/query 分离；
- 4 类 corruption；
- 单元测试。

验收：

- episode 中无样本重复泄漏，除非主动启用 duplicate corruption；
- meta-test task 不出现在 training cache index 中；
- 每个 corruption 的实际比例与配置一致。

### M2：Frozen features、prototype 与 experts

交付：

- backbone feature cache；
- shared adapter；
- source-task experts；
- episodic prototype loss；
- checkpoints。

验收：

- 单任务专家能在其训练任务上优于未训练 adapter；
- shared expert 可在所有 meta-train tasks 上工作；
- 所有 experts 参数量一致。

### M3：Gate 1 / Oracle 实验

交付：

- episode-level oracle；
- oracle mixture；
- shared、single adapter、random；
- expert performance matrix；
- wrong-route/swap/mask。

验收：

- 自动生成 Gate 1 报告；
- 若 Gate 1 失败，停止实现 learned router，并输出失败原因。

### M4：Gate 2 / Support routing

交付：

- query-only router；
- support prototype router；
- support soft mixture；
- support-query shuffle；
- support label removal。

验收：

- 自动比较 support vs query-only vs random；
- 自动输出 Gate 2 判定。

### M5：Gate 3 / Uncertainty 与 fallback

仅 Gate 2 通过后执行。

交付：

- bootstrap route uncertainty；
- prediction entropy baseline；
- shared fallback；
- coverage-risk / AURC；
- high-regret detection。

验收：

- route uncertainty 与 prediction uncertainty 分开报告；
- 自动输出 Gate 3 判定。

### M6：完整 folds 与统计

交付：

- 至少 5 个 hold-out groups；
- 5 个 seeds；
- 20–50 次 support resampling；
- 统一结果表；
- bootstrap CI；
- resource accounting。

### M7：外部验证或 H2 决策

- Gate 1–3 通过：选择外部医学数据验证；
- 同时存在专家冗余：启动 H2 reuse/spawn 第一版；
- Gate 失败：根据预注册规则缩小问题或终止。

---

## 19. 首轮实验配置建议

```yaml
experiment:
  name: e01_medmnist_unseen_task_oracle
  seed: 42

data:
  benchmark: medmnist2d
  included_tasks:
    - pathmnist
    - dermamnist
    - octmnist
    - pneumoniamnist
    - breastmnist
    - bloodmnist
    - tissuemnist
    - organamnist
    - organcmnist
    - organsmnist
  grouped_tasks:
    organ_family:
      - organamnist
      - organcmnist
      - organsmnist
  holdout_group: octmnist
  validation_group: dermamnist
  image_size: 224
  channels: 3

episode:
  mode: fixed_way
  n_way: 5
  k_shot: 5
  q_query_per_class: 10
  episodes_per_eval: 100
  support_resamples: 20

model:
  backbone: resnet18
  pretrained: imagenet1k
  freeze_backbone: true
  feature_dim: auto
  adapter:
    type: residual_mlp
    bottleneck_dim: 64
    dropout: 0.1
  expert_strategy: one_per_meta_train_task_group
  include_shared_expert: true

training:
  precision: auto
  batch_size: auto
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  max_steps: 5000
  early_stopping_patience: 10
  num_workers: auto

routing:
  methods:
    - shared_only
    - equal_parameter_single
    - random_top1
    - random_soft
    - episode_oracle
    - oracle_mixture

logging:
  output_dir: runs/e01_medmnist_unseen_task_oracle
  tensorboard: true
  save_episode_outputs: true
```

M4 配置追加：

```yaml
routing:
  methods:
    - query_only_mlp
    - support_prototype_nearest
    - support_soft_mixture
  interventions:
    - support_query_shuffle
    - support_label_removal
```

M5 配置追加：

```yaml
uncertainty:
  method: bootstrap_support
  bootstrap_samples: 30
  scores:
    - top1_switch_rate
    - route_entropy
    - mixture_weight_variance
  fallback:
    type: shared_expert
    threshold_selection: validation_tasks_only
```

---

## 20. Codex 行为约束

Codex 必须：

1. 先检查当前目录和已有 Git 仓库，避免覆盖用户代码；
2. 先审计 `doraemon0` 环境，再选择依赖和 batch size；
3. 每个 milestone 独立可运行、可测试；
4. 不在 smoke test 通过前发起长时间训练；
5. 不静默改变研究协议；
6. 所有自动降级必须写入日志；
7. 先实现最简单 baseline，再实现 learned router；
8. Gate 失败时停止下游模块并生成失败报告；
9. 不把任务名称或 task ID 传入 router；
10. 不把 query labels 用于 learned router 或 uncertainty；
11. oracle 结果必须明确标为不可部署分析上限；
12. 不将专家使用频率当作专门化证据；
13. 所有结果表必须包含 mean、std、CI、seed 和 support sampling 数；
14. 每次大型 run 前打印预计实验数量和资源预算；
15. 代码、配置和结果文件均使用英文命名，报告可以使用中文。

---

## 21. 当前不实施的内容

为了控制变量，Phase 0–1 不实施：

- Transformer block 内部 LoRA-MoE；
- 动态新增、合并或淘汰专家；
- 3D 医学分割；
- 医学 VQA；
- VLA 或自动驾驶；
- 多源 target 冲突模型；
- conformal routing；
- 复杂 evidential uncertainty；
- end-to-end backbone fine-tuning。

这些内容只有在 Gate 1–3 得到正结果后按优先级加入。

---

## 22. 最终交付物

Codex 完成 Phase 0–1 后应提交：

1. 可运行代码仓库；
2. `doraemon0` 环境审计；
3. MedMNIST 数据和 feature-cache 校验报告；
4. experts 与 shared adapter checkpoints；
5. Gate 1–3 的结构化结果；
6. 所有 baseline 与干预实验结果；
7. `results_summary.csv`；
8. `episode_metrics.parquet`；
9. `routing_outputs.npz`；
10. 自动生成的 `research_gate_report.md`；
11. 资源消耗报告；
12. 明确的继续、修改或终止建议。

---

## 23. 首个运行目标

本项目的第一个有效结果不是“新模型超过 SOTA”，而是完成以下判断：

> 在完整 hold-out 的 MedMNIST 未见任务上，固定专家库是否存在稳定的 episode-level oracle gap；support-conditioned router 是否比 query-only 和 random router 更好地恢复这一 gap；support 受污染时，bootstrap route uncertainty 是否能预测相对 oracle 的 routing regret，并指导 shared fallback。

只有这个判断成立，才进入更复杂结构、外部医学数据和专家生命周期。

---

## 24. 关键参考工作

- Chen et al., **Unleashing the Power of Meta-tuning for Few-shot Generalization Through Sparse Interpolated Experts (SMAT)**, ICML 2024.
- Li et al., **Few-Shot Hybrid Incremental Learning: Continually Learning under Data Scarcity and Task Uncertainty (CME-MoE)**, CVPR 2026.
- Zhang et al., **DETA: Denoised Task Adaptation for Few-Shot Learning**, ICCV 2023.
- Li et al., **R2-T2: Re-Routing in Test-Time for Multimodal Mixture-of-Experts**, ICML 2025.
- Yu et al., **Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters**, CVPR 2024.
- Nejatimanzari et al., **Sparse Spectral LoRA: Routed Experts for Medical VLMs**, CVPR 2026.
- Yang et al., **DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving**, CVPR 2026.
- Yang et al., **MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification**, Scientific Data 2023.

