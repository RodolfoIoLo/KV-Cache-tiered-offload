# Week 1 · 基础机制验证与单卡功能验收

本目录归档项目第一阶段的五组验证结果，前四组为本地 CPU / 双进程 TCP 机制检查，第五组为实验室单卡 V100 上的真实模型部署与 DRAM 恢复验收。

| 分类 | 核心验证问题 | 关键实测结果 |
|---|---|---|
| [Transformer](transformer/README.md) | 手写 Attention 与 SDPA 是否一致、因果掩码是否隔离未来、反向更新是否生效 | 最大误差 `1.11e-16`；40 步 loss `2.685 → 0.063` |
| [KV Cache](kv-cache/README.md) | 增量计算与完整重算是否等价、前缀身份与分支是否隔离、容量公式是否吻合 | 最大误差 `5.55e-16`；改前缀/模型 ID 命中为 `0` |
| [Paged KV](paged-kv/README.md) | 分页块表、物理块共享、写时复制（COW）与释放回收是否正确 | 共享后物理块 `4 → 3`；全部物理块完整回收 |
| [Mooncake Transfer](mooncake-transfer/README.md) | 双进程能否经 Transfer Engine TCP 传输 KV 并直接续算 | 传输 `3541 B`（含 `768 B` KV），续算最大误差 `2.22e-16` |
| [真实模型 DRAM 恢复](deployment/README.md) | V100 上 `Qwen2.5-1.5B` 能否保存前缀到独立 DRAM 并在重启后恢复一致输出 | 保存 `14 MiB`、恢复 `13.5625 MiB`（31/31 DRAM key），16/16 token 一致 |

> 前四项为轻量机制校验，用于确认数学等价性、内存所有权与传输协议路径，不代表 GPU 生产框架的吞吐或延迟性能；端到端受控性能对照见 [任务 A 实验报告](../week2-3/task-a/README.md)。
