# 📊 实验结果索引

本目录按实验阶段归档可复核的测量结果、机制检查与可视化图表，区分**本地教学机制验证**与**实验室 V100 真实模型实测**。每项结果均注明实验条件与适用边界，完整环境与版本快照位于 [`environment/`](../environment/README.md)。

## 一览总表

| 阶段 | 实验主题 | 运行环境 | 核心结论 | 详细报告 |
|---|---|---|---|---|
| **Week 1** | 基础机制验证（4 项） | CPU / Linux TCP | Attention、增量 KV、分页 COW 与双进程 Mooncake TCP 传输均验证通过（误差 $\le 5.55 \times 10^{-16}$） | [Week 1 索引](week1/README.md) |
| **Week 1** | 真实模型 DRAM 保存与重启恢复 | 单卡 V100 + `1Cat-vLLM` + `Mooncake` | 512-token 输入冷保存 `14 MiB`（32 块），仅重启 vLLM 后从 DRAM 恢复 `13.5625 MiB`（31 块），16 个输出 token 逐位一致 | [部署验收报告](week1/deployment/README.md) |
| **Week 2–3** | **任务 A（1~5）：主存回读 vs GPU 重算** | 单卡 V100 + `Qwen2.5-1.5B` FP16 + 同机 TCP | 64–1536 tokens（36 对配对样本）内，同机 TCP 回读中位数 TTFT 均慢于重算（多花 `36.8 ~ 93.9 ms`）；116 块 KV 搬运逐字节无损，瓶颈在于 1.5B 重算快、TCP 带宽约 600 MB/s 及 V100 小-q 标量回退 | [任务 A 报告](week2-3/task-a/README.md) |

---

## 分阶段产物清单

<details>
<summary><strong>Week 1 · 机制验证与单卡 V100 功能验收</strong></summary>

| 子目录 | 涵盖内容 | 关键数值 |
|---|---|---|
| [transformer/](week1/transformer/README.md) | 手写 Attention vs SDPA、因果掩码、形状与短训练更新 | 最大误差 `1.11e-16`；40 步 loss `2.685 → 0.063` |
| [kv-cache/](week1/kv-cache/README.md) | 增量 KV 等价性、生成尾差、前缀身份与 Radix 匹配、容量线性缩放 | 最大误差 `5.55e-16`；改前缀/模型 ID 命中归零 |
| [paged-kv/](week1/paged-kv/README.md) | 分页块表、物理块共享、写时复制（COW）与资源回收 | 共享后块数 `4 → 3`；全部物理块完整回收 |
| [mooncake-transfer/](week1/mooncake-transfer/README.md) | 双进程 Mooncake Transfer Engine TCP 传输与续算 | 传输 `3541 B`（含 `768 B` KV），续算最大误差 `2.22e-16` |
| [deployment/](week1/deployment/README.md) | V100 真实模型 `Qwen2.5-1.5B` 冷保存与重启后 DRAM 恢复 | 保存 `14 MiB`、恢复 `13.5625 MiB`（31/31 DRAM key），16/16 token 一致 |

</details>

<details open>
<summary><strong>Week 2–3 · 任务 A（1~5）：主存回读与 GPU 重算性能对比</strong></summary>

| 文件 / 目录 | 内容说明 |
|---|---|
| [README.md](week2-3/task-a/README.md) | 完整分析报告：实验设置、主对比图表、时间拆解、V100 算子路径归因与 35/36 数值边界说明 |
| [ttft.png](week2-3/task-a/ttft.png) | 64–1536 tokens 下 GPU 重算与 DRAM 回读的客户端 TTFT 中位数与四分位区间折线图 |
| [summary.csv](week2-3/task-a/summary.csv) | 6 档前缀长度（每档 6 对）的统计汇总（Q1/中位数/Q3、配对差值、GET/Lookup 耗时与服务端残差） |
| [samples.csv](week2-3/task-a/samples.csv) | 36 对正式回访的逐样本原始测量记录与输出 token IDs |
| [checks.json](week2-3/task-a/checks.json) | 36 对路径审计、GPU 本地 0 命中、DRAM 来源校验与 35/36 输出一致性记录 |
| [analysis.json](week2-3/task-a/analysis.json) | 结构化结论、关键成本表与局限性摘要 |
| [diagnostics.json](week2-3/task-a/diagnostics.json) | 独立非计时诊断（64/256/1536）：116 块（50.75 MiB，约 53.2 MB）KV 逐字节无损证明、全词表 logits 误差与注意力路由耗时 |
| [numerical-check.json](week2-3/task-a/numerical-check.json) | `256 / seed=53 / rep=1` 唯一不一致样本的 Top-10 logprobs、KV 往返与全词表归因 |
| [fixed-granularity/](week2-3/task-a/fixed-granularity/) | `max_num_batched_tokens=64` 补充对照（6/6 输出全一致，独立于主 TTFT 统计） |

</details>

---

## 仓库与服务器文件分工

本仓库保留可直接阅读的分析报告、汇总表、校验结论和图表；模型权重、原始完整服务日志（`serve.log`、`master.log`、`owner.log`）以及 160+ MiB 的诊断二进制快照保留在实验室服务器工作区。具体目录映射见 [服务器目录说明](../docs/02-服务器目录说明.md)。
