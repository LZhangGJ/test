# Target-aware MoE × Few-shot Adaptation

本仓库保存两轮文献调研综合材料，以及第三阶段“候选创新假设生成、定向查重、可证伪实验设计与研究方向筛选”的完整交付物。

## 核心结论

第三阶段将研究空间收敛为四个可证伪假设：

1. **H1（主推荐，91/100）**：面向无 task ID、不平衡、含噪和跨域 support set 的 uncertainty-aware support-conditioned expert routing。
2. **H2（备用，84/100）**：固定专家预算下的 reuse–expand–merge–evict 生命周期管理。
3. **H3（78/100）**：多源 target 冲突感知的专家路由、可信度校准与拒识。
4. **H4（65/100）**：无需完整重训 router 的 few-shot VLA 技能插入、安全拒识与回滚。

## 文件

| 文件 | 内容 |
|---|---|
| `Target-aware_MoE_Few-shot_两轮调研综合总结.docx` | 第一、二轮文献调研的综合总结与证据边界 |
| `phase3_research_report.md` | 第三阶段研究报告 |
| `phase3_candidate_hypotheses.csv` | 四个候选创新假设的机器可读表 |
| `phase3_candidate_scorecard.csv` | 候选方向评分表 |
| `phase3_experiment_matrix.csv` | 165 行实验矩阵 |
| `phase3_literature_dedup_matrix.csv` | 32 篇关键文献的定向查重矩阵 |
| `phase3_research_plan.json` | 完整机器可读研究计划 |
| `phase3_research_bundle.zip` | 第三阶段交付物压缩包 |
| `SHA256SUMS.txt` | 文件完整性校验值 |

## 建议实施顺序

1. 先以 oracle/random/wrong router 验证专家库是否存在可利用的路由上限。
2. 在 MedMNIST-v2 上完成低成本 smoke test。
3. 使用 WILDS Camelyon17 验证跨医院域偏移。
4. 通过 kill criteria 后扩展至 2D 医学分割、医学 VLM、专家生命周期与 VLA。

研究材料中的创新性判断应继续回到原始论文、附录、正式 proceedings 和项目页面定向核验，避免使用未经核验的“首次”表述。
