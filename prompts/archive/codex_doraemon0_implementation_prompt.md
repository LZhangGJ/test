# Prompt for local Codex on doraemon0

请读取同目录下的：

`target_aware_moe_fewshot_experiment_report_doraemon0.md`

并严格依照该报告，在当前 Linux 主机 `doraemon0` 上实现和运行 Phase 0–1。

## 总目标

建立一个可复现的 PyTorch 代码库，在 MedMNIST2D 完整 hold-out 未见任务协议中，依次验证：

1. 固定 expert bank 是否存在 episode-level oracle gap；
2. support-conditioned router 是否优于 random 和 query-only router；
3. bootstrap route uncertainty 是否预测 routing regret，并能指导 shared fallback。

MedMNIST 子数据集被视为独立任务。完整 hold-out task/task-group 在 meta-training、expert training 和 router training 中不可见。模型通过 support 图像推断任务属于协议允许行为。router 不得接收 task ID、数据集名称或手工任务编号。

## 执行规则

1. 首先检查当前目录、Git 状态和已有文件，禁止覆盖未知用户代码。
2. 运行报告第 15 节的环境审计命令，将结果保存到 `artifacts/system/doraemon0_environment.txt`。
3. 根据实际 GPU、CUDA、内存和磁盘选择环境，不假设 GPU 型号或数量。
4. 若当前目录不是合适仓库，则创建 `target-aware-moe-fewshot` 子目录；若已有同名仓库，先检查并续作。
5. 优先使用 `uv`；不可用时使用 `python -m venv .venv`。
6. 先完成 M0、M1 和单元测试，再进行任何训练。
7. frozen backbone 的 v0 必须优先缓存 features，以降低运行成本。
8. 先实现 embedding-level residual adapters，不实现内部 Transformer LoRA-MoE。
9. 按报告中的 Gate 顺序执行；Gate 失败后停止后续模块并生成失败报告，不通过堆叠模块掩盖负结果。
10. 所有命令、配置、日志、checkpoint 和统计必须可复现。
11. 不要只给方案；直接编码、测试并运行当前机器可承受的实验。
12. 大型矩阵开始前，先输出 run 数量和资源预算到日志。
13. 如完整矩阵成本过高，先完成一个 hold-out fold 的所有 Gate，再扩展折数；不得降低关键基线或因果干预。
14. 运行中遇到错误时先定位、修复和添加回归测试，然后继续。
15. 每个 milestone 完成后进行一次有意义的 Git commit；不要提交数据、feature cache、checkpoint 或大型运行产物。

## 第一批必须完成的内容

### M0

- 仓库结构；
- Python 环境；
- `pyproject.toml`；
- CLI；
- pytest、ruff；
- 系统审计。

### M1

- MedMNIST v0 十个分类任务；
- `organamnist/organcmnist/organsmnist` task-group 约束；
- fixed-way 和 all-way episodic sampler；
- support/query 严格分离；
- imbalance、label noise、cross-task outlier、duplicate corruptions；
- 数据和 sampler 单元测试。

### M2

- frozen ResNet-18 backbone；
- feature cache；
- residual feature adapters；
- one expert per meta-train task group；
- shared expert；
- prototype classifier；
- expert checkpoints。

### M3

- frozen backbone baseline；
- shared expert；
- equal-parameter single adapter；
- random top-1/soft；
- episode-level oracle；
- oracle mixture；
- forced worst expert；
- swap/mask/permutation；
- Gate 1 自动报告。

若 Gate 1 通过，再完成：

### M4

- query-only router；
- support prototype nearest router；
- support soft mixture；
- support-query shuffle；
- support-label removal；
- Gate 2 自动报告。

若 Gate 2 通过，再完成：

### M5

- bootstrap support uncertainty；
- prediction entropy 对照；
- shared fallback；
- routing regret；
- OGR；
- coverage-risk、AURC；
- Gate 3 自动报告。

## 首个开发配置

使用报告第 19 节配置作为起点：

- holdout task：`octmnist`；
- validation task：`dermamnist`；
- ResNet-18 frozen backbone；
- 224×224，三通道；
- 5-way（不足5类时使用全部类）；
- 5-shot；
- 每类10个 query；
- 一个 source-task expert 对应一个 meta-train task group；
- 一个 shared expert；
- 先跑 100 episodes；
- smoke test 可缩小为每类少量样本和 5 episodes。

## 质量要求

- Python 类型标注；
- 核心数据结构使用 dataclass 或 pydantic；
- 不允许 silent fallback；
- 不允许 query 标签进入 learned router；
- oracle 清楚标记为分析上限；
- 每个结果包含 seed、fold、shot、support sample IDs 和 corruption 元数据；
- 统计至少保存 mean、std 和 bootstrap CI；
- 不把专家使用频率或 t-SNE 当作功能专门化证据；
- capacity-matched 和 compute-matched 结果分开报告。

## 最终输出

在仓库中生成：

- `artifacts/reports/research_gate_report.md`
- `artifacts/tables/results_summary.csv`
- `artifacts/tables/episode_metrics.parquet`
- `artifacts/tables/resource_usage.csv`
- `artifacts/system/doraemon0_environment.txt`

并在最后汇报：

1. 完成了哪些 milestone；
2. 运行了哪些命令；
3. Gate 1–3 是否通过；
4. 最重要的正面或负面结果；
5. 尚未完成的内容及明确原因；
6. 下一条最合理的实验命令。
