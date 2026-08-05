# Target-aware MoE × Few-shot Adaptation
## 本地实现与实验执行报告

**文档用途**：本报告面向本地 Codex，用于在现有工作区内完成代码设计、实现、测试、实验调度和结果汇总。它不是最终论文草稿，也不要求一次性实现所有候选创新。

**当前主线**：在无 task ID 的未见任务或未见域上，研究不可靠 few-shot support 如何影响专家特异化决策；通过 episode-level oracle、support-conditioned routing、route uncertainty、shared fallback 和专家因果干预，判断模型何时应该选择专家、软组合专家，或避免特异化。

**计算资源**：

- Linux：`doraemon02`、`doraemon03`、`doraemon04`、`doraemon15`、`doraemon19`、`doraemon20`
- 上述 Linux 主机共享同一磁盘路径，但具体共享根目录不得在代码中硬编码。
- Windows 本机：用于代码编辑、单元测试、CPU/small-GPU smoke test、结果检查；核心代码必须跨平台。

---

# 1. 当前研究结论

## 1.1 方向优先级

| 优先级 | 方向 | 当前决策 |
|---|---|---|
| 主线 | **H1：不可靠 support 下的 support-conditioned selective expert adaptation** | 立即实现最小可证伪实验 |
| 备用 | **H2：固定专家预算下的 reuse–update–expand–merge 生命周期** | 仅在 H1 证明专家差异与 support 路由有效后启动 |
| 长期 | **H4：few-shot VLA 新技能插入、冻结通用 router 与安全回滚** | 暂不实现，保留接口扩展能力 |
| 扩展协议 | **H3：多源 target 冲突下的路由与拒识** | 作为 H1 后期压力测试，不单独立项 |

## 1.2 新颖性边界

以下内容本身不能作为主要创新：

- support encoder；
- LoRA/adapter experts；
- top-k 或 softmax router；
- uncertainty head；
- shared expert；
- 医学图像上使用 MoE；
- prompt/adapter/LoRA/MoE 的机械组合。

主线真正需要验证的是：

1. **Support value**：support 是否包含 query-only router 没有的任务信息；
2. **Routing-specific uncertainty**：support 可靠性是否能预测错误专家选择，而不只是预测最终分类错误；
3. **Selective specialization**：support 证据不足时，降低特异化、回退 shared expert 或拒识，是否减少风险；
4. **Causal specialization**：专家是否形成了可干预、可复现的功能差异。

---

# 2. MedMNIST 的正式定位

MedMNIST 保留为本研究的**正式追加实验数据集**，而不是仅作为 smoke test。

本项目认可如下任务定义：

> 将不同 MedMNIST 子数据集视为不同视觉医学任务；在测试时不提供数据集 ID 或 task ID，仅通过少量有标签 support 推断一个训练阶段未出现的任务，并在该任务的 query 上完成预测。

不同子数据集的模态、解剖区域和图像统计差异并不使该问题无效；它们构成跨任务推断信号。论文表述必须准确限定为：

- **cross-dataset unseen-task inference**；
- **task-ID-free episodic adaptation**；
- 不将该结果单独等同于细粒度跨医院或跨设备泛化。

## 2.1 MedMNIST 初始协议

### 数据范围

- 首先使用 MedMNIST v2 的 2D 单标签分类任务；
- 多标签任务单独建立后续协议，不与单标签任务共享同一输出头和主要指标；
- 所有图像统一尺寸、通道数和归一化流程；
- router 不读取数据集名称、数据集索引、文件路径、原始 split 名称或其他显式 task metadata。

### Task-level meta split

由于可用任务数量有限，采用**重复任务级留出**，而不是单次固定划分：

- 每个 task-split seed 将可用任务划分为 meta-train、meta-validation、meta-test；
- 建议初始比例为 7/2/其余测试任务，实际数量按单标签任务数自动调整；
- 至少使用 4 个独立 task-split seeds；
- 要求每个合格任务至少一次出现在 meta-test；
- 后续增加 leave-one-task-out 作为稳健性检查。

### Episodic protocol

每个 episode 包含：

- 一个当前未知的任务 `t`；
- support set `S_t`；
- query set `Q_t`；
- support 标签可见；
- query 标签仅在训练阶段计算 meta-loss，或在测试阶段用于离线评价；
- task ID、dataset ID 不可进入模型。

建议同时测试：

- 二分类 episode；
- 对类别数足够的任务测试 5-way episode；
- 对类别数不足的任务使用 all-way episode；
- `K ∈ {1, 2, 5, 10, 16}`；
- query 每类数量固定或在样本不足时按协议缩减。

### 可变标签空间

不同任务的标签空间不同，因此第一版不使用固定全局分类头。采用：

- backbone/expert 输出 embedding；
- support 生成 episode-local class prototypes；
- query 通过距离或轻量 episode classifier 预测；
- 专家作用于 representation，而不是绑定固定类别头。

这使未见任务能够仅凭 support 定义局部预测空间。

---

# 3. 主线研究假设

## H1a：专家可用性与 Oracle Gap

**假设**：固定专家库中存在随任务变化的功能差异，episode-level oracle expert 显著优于 shared expert 和等参数 single adapter。

**反证条件**：

- oracle expert 与 shared/single-adapter 的差异接近零；
- 同一个专家在几乎所有任务上都最好；
- 专家交换、遮蔽、错误路由不产生任务特异影响。

若 H1a 不成立，立即停止 router 与 uncertainty 开发。

## H1b：Support Value

**假设**：无 task ID 时，support-conditioned router 能比 query/input-dependent router 和 random router 恢复更多 oracle gap。

**反证条件**：

- support router 不优于 query-only router；
- 打乱 support-query 对应后性能不下降；
- 移除 support 标签后性能不变；
- 重复同一样本与提供独立样本效果相同。

## H1c：Routing Uncertainty 与 Selective Specialization

**假设**：support 重采样、标签扰动、缺类或跨任务混合引起的 route instability 能预测 routing regret；高风险 episode 使用 shared fallback、软组合或拒识，可改善 coverage-risk。

**反证条件**：

- uncertainty 只能预测最终分类错误，不能预测 routing regret；
- prediction entropy 与 route uncertainty 效果相同或更好；
- fallback 不优于继续使用特异专家；
- 增加 support 后 uncertainty 与 regret 不共同下降。

---

# 4. 实验总体阶段

## Phase 0：环境、数据和可复现性

目标：确认六台 Linux 主机和 Windows 环境，建立跨平台代码骨架。

交付：

- 每台主机的 GPU、显存、驱动、CUDA、Python、PyTorch、磁盘信息；
- 共享路径与本地 scratch 检测结果；
- 可重复安装环境；
- synthetic episode 的 CPU 单元测试；
- MedMNIST 下载、缓存、校验和与 episode sampler。

## Phase 1：固定专家库与 Oracle Gap

目标：只回答专家是否有值得路由的差异。

步骤：

1. 冻结 backbone；
2. 为每个 meta-train source task 训练同结构、同 rank 的 adapter/LoRA expert；
3. 训练一个在全部 meta-train tasks 上共享的 shared expert；
4. 所有 experts 冻结；
5. 在 meta-validation/meta-test episode 上计算：
   - shared expert；
   - 每个单 expert；
   - episode-level oracle expert；
   - oracle convex mixture；
   - equal-parameter single adapter。

**Gate 1**：oracle gap 不成立则停止主线。

## Phase 2：最简单的 Support Router

目标：只回答 support 是否提供额外路由信息。

实现：

- random router；
- query/input-dependent router；
- support prototype router；
- support-conditioned soft mixture；
- shared-only；
- episode-level oracle。

**Gate 2**：support router 必须稳定优于 random 和 query-only router，并在 support shuffle 后显著下降。

## Phase 3：无参数或低参数 Route Uncertainty

优先实现 bootstrap，不立即增加复杂 uncertainty network。

对 support 做有放回重采样：

\[
S^{(1)},\ldots,S^{(B)}
\]

得到多次路由分布：

\[
p(e\mid S^{(b)}).
\]

初始 uncertainty 指标：

- top-1 expert switch rate；
- route entropy；
- mixture-weight variance；
- bootstrap disagreement；
- selected-vs-second expert margin。

定义 shared fallback：高 uncertainty 时降低 specific experts 权重或完全使用 shared expert。

**Gate 3**：route uncertainty 必须预测 routing regret，并优于 final prediction entropy。

## Phase 4：Support Corruption 与因果干预

第一轮仅实施四种 corruption：

1. 类别不平衡；
2. 标签翻转；
3. 其他任务/数据集 outlier；
4. 重复样本导致的虚假 shot 增长。

随后再扩展：

- 缺类；
- support-query shift；
- 文本/support 冲突；
- 元数据错误；
- 对抗描述。

## Phase 5：第二数据族复现

H1 在 MedMNIST 上通过后，至少选择一种：

- 同一任务、不同医院的医学分类；
- Kvasir/CVC 等跨数据集 2D 分割；
- 多中心前列腺 MRI；
- 医学 VLM/VQA。

第二数据族用于判断机制是否超出 MedMNIST 跨任务设置。

## Phase 6：决定是否进入 H2

只有同时满足以下条件才进入专家生命周期：

- oracle gap 明显；
- support router 恢复稳定的 oracle gap；
- wrong routing 有任务特异代价；
- 部分 experts 存在可预测冗余；
- shared expert 无法替代全部 experts。

---

# 5. 模型与训练设计

## 5.1 Backbone

第一版支持至少两种配置：

- 小型 CNN/ResNet：用于快速调试；
- ViT-S 或同等级视觉 encoder：用于正式原型。

要求：

- backbone 完全冻结；
- 预训练权重和版本写入 resolved config；
- core API 不绑定具体 backbone；
- Windows CPU 能使用 tiny/synthetic backbone 完成单元测试。

## 5.2 Expert

初始 expert 使用 adapter 或 LoRA：

- 所有 experts 结构一致；
- rank、插入层、初始化一致；
- 每个 expert 独立保存；
- shared expert 单独训练；
- 训练后进入 router 阶段时默认冻结。

不要在 Phase 1 同时联合学习 expert 和 router，以免无法区分 expert failure 与 router failure。

## 5.3 Episode-local classifier

默认使用 prototypical classifier：

\[
c_k=\frac{1}{|S_k|}\sum_{(x_i,y_i=k)\in S}z_i
\]

query 使用 cosine 或 normalized Euclidean distance 预测。

应支持：

- 每个 expert 单独生成 embedding/prototype；
- soft mixture 后生成 embedding；
- mixed logits；
- 可配置 temperature。

## 5.4 Router

统一接口：

```python
route = router(
    support_embeddings,
    support_labels,
    query_embeddings=None,
    task_metadata=None,
)
```

返回：

```python
RouteOutput(
    weights,          # [num_experts]
    selected_indices,
    confidence,
    uncertainty,
    diagnostics,
)
```

实现顺序：

1. `RandomRouter`；
2. `QueryOnlyRouter`；
3. `SupportPrototypeRouter`；
4. `SupportSoftRouter`；
5. `BootstrapUncertaintyWrapper`；
6. `SharedFallbackPolicy`；
7. learned set encoder 仅在简单路由有信号后实现。

## 5.5 两类 Oracle

### Episode-level oracle

对完整 query set 选择平均损失最低或任务指标最高的 expert。它是主要 oracle。

### Oracle mixture

使用 query 标签离线优化专家凸组合，仅用于诊断软组合上界。

sample-level oracle 可以实现，但只能作为额外分析，不作为主要 oracle-gap recovery 分母。

## 5.6 Routing regret

对损失定义：

\[
R_{loss}=L(f_{selected},Q)-L(f_{oracle},Q).
\]

对越高越好的指标定义：

\[
R_{metric}=M(f_{oracle},Q)-M(f_{selected},Q).
\]

Oracle-gap recovery：

\[
OGR=\frac{M_{router}-M_{shared}}
{M_{oracle}-M_{shared}}.
\]

当 oracle gap 小于预设 epsilon 时，不报告 OGR 或明确标记为不可解释。

---

# 6. 第一轮基线

| ID | 基线 | 目的 |
|---|---|---|
| B00 | Frozen backbone / no adaptation | 基础下限 |
| B01 | Shared expert | 判断特异专家是否必要 |
| B02 | Equal-parameter single adapter/LoRA | 排除总参数优势 |
| B03 | Per-source fixed expert | 建立专家性能矩阵 |
| B04 | Random router，多次采样 | 判断学习路由是否超出随机 |
| B05 | Query/input-dependent router | 判断 support 是否有额外信息 |
| B06 | Support prototype router | 判断简单 support 统计是否足够 |
| B07 | Support-conditioned soft mixture | 判断软组合是否优于硬选择 |
| B08 | Episode-level oracle expert | 路由可利用上界 |
| B09 | Oracle convex mixture | 专家组合上界 |
| B10 | Prediction-entropy fallback | 判断 route uncertainty 是否独立有用 |

第二阶段再加入：

- SMAT-like support sparse interpolation；
- noisy-support denoising + router；
- learned support set encoder；
- calibrated abstention；
- H2 生命周期基线。

---

# 7. 评价指标

## 7.1 任务性能

按任务类型选择：

- Accuracy；
- macro/micro F1；
- AUROC；
- balanced accuracy；
- Dice/IoU/NSD；
- worst-task / worst-group performance。

## 7.2 Few-shot 稳定性

- 每个条件至少 20 次 support resampling；正式结果建议 50 次；
- 至少 5 个训练随机种子；
- mean、std、bootstrap 95% CI；
- worst-support；
- 10th percentile；
- support sensitivity slope。

## 7.3 路由指标

- episode-level router accuracy；
- top-k oracle recall；
- routing regret；
- oracle-gap recovery；
- expert entropy；
- wrong-route detection AUROC/AUPRC；
- Brier score；
- route ECE；
- coverage-risk curve；
- AURC；
- risk@80/90/95% coverage。

## 7.4 效率

分别报告：

- expert bank 总参数；
- 可训练参数；
- 单次激活参数；
- backbone/expert/router/support encoder FLOPs；
- support 编码时间；
- 单 query 延迟；
- 峰值显存；
- expert 加载/切换时间。

公平比较分为：

- **capacity-matched**：匹配可训练参数和训练预算；
- **compute-matched**：匹配单次激活参数、FLOPs 和推理路径。

不强行声称一组 baseline 能同时匹配所有指标。

---

# 8. 专家因果干预

必须实现统一 intervention API：

```python
InterventionConfig(
    mode="none|random|force|swap|permute|mask|shuffle_support|oracle",
    expert_a=None,
    expert_b=None,
    permutation=None,
)
```

至少包括：

1. forced worst expert；
2. random routing distribution；
3. expert swapping；
4. global expert permutation；
5. mask most-used expert；
6. mask shared expert；
7. shuffle support-query pairing；
8. remove support labels；
9. duplicate support without increasing unique samples；
10. freeze router/train experts；
11. freeze experts/train router；
12. episode oracle。

能够支持“功能专门化”的证据必须同时包含：

- oracle experts 在不同任务上发生变化；
- learned router 能利用该差异；
- wrong route/swap/mask 产生任务特异损失；
- 结果跨 task splits、support resampling 和 seeds 稳定。

激活热图、t-SNE、使用率和负载均衡只能作为描述性材料。

---

# 9. 预注册停止条件

## Gate 1：停止全部 router 工作

满足任一项：

- episode oracle 相比 shared 或 equal-parameter single adapter 的效应很小，且两个以上 task split 的 CI 覆盖零；
- 同一 expert 在绝大多数任务上始终最好；
- expert swap/mask/wrong routing 不产生任务特异影响。

## Gate 2：停止 support-conditioned 路由

满足任一项：

- support router 不优于 query-only router；
- learned router 落在 random router 分布内；
- shuffle support-query 后性能不下降；
- 移除 support 标签后效果不变。

## Gate 3：停止 routing uncertainty 主张

满足任一项：

- uncertainty 对 high-regret episode 的 AUROC 接近随机；
- final prediction entropy 同等或更好；
- fallback 不改善 coverage-risk；
- 更多独立 support 不降低 uncertainty/regret；
- uncertainty 主要由类别难度解释，而不是 route mismatch。

## 进入大规模实验的条件

- 至少两个独立 task-split 或数据族上，oracle gap 稳定为正；
- support router 显著优于 random 和 query-only；
- learned router 恢复有意义比例的 oracle gap；
- 至少三种 support corruption 下优势未完全消失；
- wrong-route detection 和 coverage-risk 改善稳定；
- 额外延迟和显存可接受；
- 无 task ID。

---

# 10. 代码仓库结构

Codex 应尽量适配现有仓库。若当前工作区没有相关结构，创建：

```text
target-aware-moe-fs/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── .gitattributes
├── configs/
│   ├── data/
│   │   ├── medmnist.yaml
│   │   └── synthetic.yaml
│   ├── model/
│   │   ├── resnet_adapter.yaml
│   │   └── vit_lora.yaml
│   ├── router/
│   │   ├── random.yaml
│   │   ├── query_only.yaml
│   │   ├── support_prototype.yaml
│   │   └── support_soft.yaml
│   └── experiment/
│       ├── phase1_oracle.yaml
│       ├── phase2_router.yaml
│       └── phase3_uncertainty.yaml
├── src/target_moe/
│   ├── data/
│   │   ├── medmnist_tasks.py
│   │   ├── episodic_sampler.py
│   │   ├── task_splits.py
│   │   └── corruptions.py
│   ├── models/
│   │   ├── backbones.py
│   │   ├── experts.py
│   │   ├── episodic_head.py
│   │   ├── routers.py
│   │   ├── uncertainty.py
│   │   └── fallback.py
│   ├── training/
│   │   ├── train_expert.py
│   │   ├── train_router.py
│   │   └── loops.py
│   ├── evaluation/
│   │   ├── oracle.py
│   │   ├── routing_metrics.py
│   │   ├── interventions.py
│   │   └── evaluate.py
│   ├── orchestration/
│   │   ├── host_probe.py
│   │   ├── job_manifest.py
│   │   ├── dispatcher.py
│   │   └── collector.py
│   └── utils/
│       ├── config.py
│       ├── reproducibility.py
│       ├── atomic_io.py
│       └── logging.py
├── scripts/
│   ├── probe_hosts.py
│   ├── prepare_medmnist.py
│   ├── train_experts.py
│   ├── evaluate_oracle.py
│   ├── train_router.py
│   ├── evaluate_router.py
│   ├── launch_grid.py
│   ├── collect_results.py
│   ├── linux_smoke.sh
│   └── windows_smoke.ps1
├── tests/
│   ├── test_episode_sampler.py
│   ├── test_no_task_id_leakage.py
│   ├── test_router_shapes.py
│   ├── test_oracle.py
│   ├── test_corruptions.py
│   └── test_cross_platform_paths.py
└── artifacts/
    ├── host_inventory/
    ├── task_splits/
    ├── manifests/
    └── reports/
```

---

# 11. 配置与运行记录规范

每个 run 必须写入独立目录：

```text
runs/<study>/<experiment_id>/
├── config_resolved.yaml
├── env.json
├── git_state.json
├── stdout.log
├── stderr.log
├── metrics.jsonl
├── summary.json
├── checkpoints/
└── artifacts/
```

`experiment_id` 建议：

```text
H1A_medmnist_<backbone>_<expertset>_<router>_split<S>_seed<N>_<gitshort>
```

必须记录：

- hostname；
- GPU index/name/memory；
- CUDA_VISIBLE_DEVICES；
- Python/PyTorch/CUDA 版本；
- git commit、branch、dirty state；
- 完整配置；
- task split hash；
- support sample IDs；
- 随机种子；
- 数据版本和校验信息。

日志默认使用本地 JSONL/CSV/TensorBoard；外部在线服务为可选，不得成为运行依赖。

---

# 12. 六台 Linux 主机的使用方案

## 12.1 主机列表

```text
doraemon02
doraemon03
doraemon04
doraemon15
doraemon19
doraemon20
```

## 12.2 共享磁盘原则

不得假设具体共享根目录。使用环境变量：

```bash
TAMOE_PROJECT_ROOT
TAMOE_DATA_ROOT
TAMOE_RUN_ROOT
TAMOE_CACHE_ROOT
TAMOE_LOCAL_SCRATCH
```

Codex 首先探测当前仓库位置和可用共享路径，并生成 `.env.example`。真实路径只写入用户本地 `.env`，不得提交仓库。

建议：

- code、configs、只读 datasets、run summaries 放共享磁盘；
- 每个 host 使用独立 cache 子目录，避免多个进程争用同一缓存锁；
- 如存在本地 NVMe/scratch，将临时 batch cache 和解压文件放 host-local 路径；
- checkpoint 与结果用原子写入后移动到共享 run 目录；
- 不允许多个任务写同一 checkpoint 或同一 metrics 文件。

## 12.3 环境探测

实现：

```bash
python scripts/probe_hosts.py \
  --hosts doraemon02,doraemon03,doraemon04,doraemon15,doraemon19,doraemon20 \
  --output artifacts/host_inventory/hosts.json
```

每台主机收集：

- SSH 可达性；
- hostname；
- OS；
- GPU 数量、型号、总显存、空闲显存、利用率；
- NVIDIA driver；
- CUDA runtime；
- Python；
- PyTorch；
- 可用磁盘；
- 当前 scheduler：Slurm/无 scheduler；
- 共享路径是否一致可见。

探测失败不得阻塞本机开发；将失败写入 inventory。

## 12.4 调度策略

优先顺序：

1. 若检测到 Slurm，生成并使用 Slurm job scripts；
2. 否则实现 SSH dispatcher；
3. 初始实验使用“单 GPU 单 run”，通过多主机并行 seeds/task splits；
4. 暂不使用跨主机 distributed training；
5. 只有单 run 确实需要多 GPU 时才使用 `torchrun`。

SSH dispatcher 要求：

- 使用 job manifest（JSON/CSV）；
- 按 GPU 空闲显存分配；
- 设置 `CUDA_VISIBLE_DEVICES`；
- 每台主机每张 GPU 使用 host-local `flock` 或 lock file；
- 支持 `PENDING/RUNNING/SUCCEEDED/FAILED/CANCELLED`；
- 支持重复启动时跳过已完成 run；
- 支持失败重试但保留原日志；
- 不依赖 root 权限；
- 不自动杀死其他用户进程。

## 12.5 初始并行分配

Phase 1/2 的主要并行维度：

- task-split seed；
- training seed；
- backbone；
- expert type；
- router baseline；
- corruption level。

不应在代码尚未通过 Gate 1 前提交完整网格。

推荐：

- Windows/一台 Linux：开发和 smoke test；
- 一台 Linux GPU：完整单 seed end-to-end；
- 两台 Linux：重复性测试；
- Gate 1 通过后再扩展到六台主机。

---

# 13. Windows 本机支持

核心 Python 代码必须：

- 使用 `pathlib.Path`；
- 不依赖 Bash 路径语法；
- 不依赖 `fork`；
- DataLoader worker 数可配置，Windows smoke test 默认 `num_workers=0`；
- 支持 CPU synthetic test；
- 若存在本机 CUDA，可运行小型 MedMNIST episode；
- 提供 `scripts/windows_smoke.ps1`。

Windows smoke test 应完成：

1. import；
2. synthetic dataset；
3. episode sampler；
4. backbone forward；
5. expert switching；
6. oracle computation；
7. one optimizer step；
8. summary.json 写入。

---

# 14. Codex 实施里程碑

## M0：仓库与环境审计

交付：

- 当前仓库结构说明；
- 不破坏现有代码的改动计划；
- host inventory；
- Python 环境文件；
- Windows/Linux smoke scripts。

完成条件：所有单元测试和 synthetic smoke test 通过。

## M1：MedMNIST task split 与 episodic pipeline

交付：

- task-level split generation；
- split JSON 与 hash；
- support/query sampler；
- no-task-ID leakage test；
- 2-way、5-way/all-way 可配置；
- 单标签和多标签任务隔离。

完成条件：固定 seed 可重复得到完全相同 episode IDs。

## M2：Backbone、experts 和 episode head

交付：

- frozen backbone；
- adapter/LoRA expert abstraction；
- shared expert；
- per-source expert training；
- prototypical head；
- capacity/compute accounting。

完成条件：单个 expert 训练、保存、加载和 episode evaluation 正常。

## M3：Oracle 分析

交付：

- expert × task performance matrix；
- episode oracle；
- oracle mixture；
- shared/single-adapter baseline；
- Gate 1 report。

完成条件：自动生成 `gate1_summary.md/json`，明确 PASS/FAIL。

**在 M3 之前，不实现复杂 learned router。**

## M4：Router baselines

交付：

- random；
- query-only；
- support prototype；
- support soft mixture；
- support shuffle/remove-label diagnostics；
- Gate 2 report。

## M5：Bootstrap uncertainty 与 fallback

交付：

- bootstrap route distribution；
- uncertainty metrics；
- prediction entropy baseline；
- shared fallback；
- coverage-risk；
- Gate 3 report。

## M6：Support corruption 与 interventions

交付：

- imbalance；
- label flip；
- cross-task outlier；
- duplicate support；
- wrong-route/swap/mask/permutation；
- 完整因果分析报告。

## M7：第二数据族

只有 Gate 1–3 通过后开始。

---

# 15. Codex 工作规则

1. 先读取当前工作区，不覆盖已有实现。
2. 所有路径可配置，不硬编码用户共享磁盘。
3. 先写测试，再写最小实现。
4. 每个里程碑独立可运行、可回滚。
5. 不在 Gate 1 前构建复杂 uncertainty model。
6. 不在 Gate 2 前跑大规模 corruption grid。
7. 不在 Gate 3 前启动 H2。
8. 不自动使用六台主机进行大规模运行；先完成单机验证。
9. 不删除旧 checkpoints、runs 或他人文件。
10. 不自动终止其他用户 GPU 进程。
11. 每次 run 保存 resolved config、环境和 git state。
12. 失败必须保留 stdout/stderr 和异常栈。
13. 结果汇总按 task split、support resampling 和 seed 分层，不只报单一均值。
14. 所有 oracle 使用 query 标签必须在代码和输出中标为 analysis-only。
15. 任何可能泄漏 task ID 的字段均需单元测试。
16. Windows 和 Linux 的核心行为应一致。
17. 每完成一个里程碑，生成：改动文件、运行命令、测试结果、未解决问题和下一步建议。
18. 不在没有实验支持时自动把模块命名为“创新方法”。

---

# 16. 第一批实际运行矩阵

第一批只验证 M0–M3：

| Exp | 数据 | Backbone | Expert bank | Router | Seeds | 目的 |
|---|---|---|---|---|---:|---|
| P00 | synthetic | tiny CNN | 3 synthetic experts | none | 1 | pipeline test |
| P01 | MedMNIST subset | ResNet18 | shared only | none | 1 | data/head test |
| P02 | MedMNIST meta-train | ResNet18 | per-source adapters | none | 3 | expert training stability |
| P03 | MedMNIST held-out tasks | ResNet18 | fixed experts + shared | episode oracle | 3 | Gate 1 |
| P04 | MedMNIST held-out tasks | ViT-S/configurable | fixed experts + shared | episode oracle | 3 | backbone robustness |

只有 P03/P04 显示 oracle gap 后运行：

| Exp | Router | Support | 目的 |
|---|---|---|---|
| R00 | random | clean | 随机分布 |
| R01 | query-only | clean | query baseline |
| R02 | support prototype | clean | support value |
| R03 | support soft mixture | clean | soft composition |
| R04 | support prototype | shuffled | leakage/causality |
| R05 | support prototype | labels removed | label value |

只有 R02/R03 通过 Gate 2 后运行 bootstrap uncertainty。

---

# 17. 最终项目目标

初始项目成功不以“得到一个复杂 MoE 模型”为标准，而以是否能清楚回答以下问题为标准：

1. MedMNIST 未见任务中是否存在可路由的专家差异？
2. support 是否比 query-only 输入提供额外任务信息？
3. support corruption 是否导致可测量的 routing regret？
4. route uncertainty 是否能够识别这种错误？
5. shared fallback 或降低特异化是否改善风险？
6. 专家差异能否经 swap、mask、wrong route 等干预得到因果支持？
7. 结论能否在第二数据族复现？

只有以上链条成立，才将 H1 整理为论文级方法，并进入 H2 或 VLM/VLA 扩展。
