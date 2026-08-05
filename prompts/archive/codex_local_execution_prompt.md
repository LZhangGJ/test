# 给本地 Codex 的启动指令

请阅读同目录的：

`target_aware_moe_local_implementation_report.md`

你的任务是**只完成 M0–M3**，不要提前实现复杂 uncertainty model、专家生命周期或 VLA。

## 当前资源

Linux 主机：

- doraemon02
- doraemon03
- doraemon04
- doraemon15
- doraemon19
- doraemon20

这些主机共享同一磁盘路径。也可以在 Windows 本机运行。不要硬编码共享路径；使用环境变量和 `.env.example`。

## 第一阶段目标

实现并运行：

1. 工作区与环境审计；
2. 六台 Linux 主机的 GPU/环境探测；
3. Windows/Linux smoke test；
4. MedMNIST task-level meta split；
5. 无 task ID 的 episodic support/query sampler；
6. frozen backbone；
7. shared adapter/LoRA expert；
8. per-source-task fixed experts；
9. episode-local prototypical classifier；
10. episode-level oracle expert；
11. oracle mixture；
12. equal-parameter single adapter baseline；
13. Gate 1 自动报告。

## MedMNIST 研究定义

MedMNIST 是正式实验数据集之一。不同子数据集被视为不同任务。测试时不提供 dataset ID/task ID，仅提供少量有标签 support，并在训练阶段未出现的 held-out task 上预测 query。

优先使用 2D 单标签任务；多标签任务单独处理。采用重复 task-level meta split，并确保每个合格任务至少一次进入 meta-test。

输出层使用 episode-local prototypes，以支持不同任务的可变标签空间。

## 实施约束

- 先检查现有仓库，不覆盖已有代码。
- 所有代码可配置、可测试、跨平台。
- 核心路径使用 `pathlib`。
- 所有 run 使用独立目录并保存 resolved config、env、git state、metrics 和日志。
- query 标签用于 oracle 时必须标记 `analysis_only=true`。
- 编写 `test_no_task_id_leakage.py`，确保 router/model 不能读取 dataset/task ID、路径或 split 名称。
- Gate 1 前不要实现 learned support router。
- 不要自动占用所有主机或杀死其他用户进程。
- 优先单机端到端验证，再并行多个 seeds/task splits。
- 若存在 Slurm，优先生成 Slurm scripts；否则实现非破坏性的 SSH dispatcher。
- 共享磁盘上不得并发写同一文件；使用唯一 run ID、原子写入和 lock。

## Gate 1

比较：

- frozen/no-adaptation；
- shared expert；
- equal-parameter single adapter/LoRA；
- 每个固定 expert；
- episode-level oracle expert；
- oracle convex mixture。

自动生成 expert × task performance matrix 和 `gate1_summary.md/json`。

Gate 1 PASS 至少要求：

- oracle 相比 shared/single-adapter 存在稳定正 gap；
- 最佳 expert 随任务变化；
- 结果在多个 task splits、support resampling 和 seeds 上可复现。

若不满足，明确输出 FAIL 和原因，不要继续堆叠 router。

## 每个里程碑完成后报告

请输出：

1. 修改或新增的文件；
2. 关键设计决定；
3. 运行命令；
4. 测试结果；
5. 实验结果路径；
6. 当前 Git 状态/commit；
7. 发现的问题；
8. 下一步建议。

从 M0 开始执行。
