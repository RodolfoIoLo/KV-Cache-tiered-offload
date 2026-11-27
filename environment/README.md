# ⚙️ 运行环境与配置快照（`environment/`）

本目录记录各阶段实验实际使用的软件版本、硬件规格、依赖列表与关键启动配置，用于复现实验结果与核对结论边界。

---

## 环境清单

| 适用范围 | 信息快照 | 依赖清单 | 核心软硬件配置 |
|---|---|---|---|
| **本地 CPU 机制验证**<br/>（Transformer / KV Cache / Paged KV） | [cpu-info.json](cpu-info.json) | [cpu-requirements.txt](cpu-requirements.txt) | Windows AMD64，Python `3.13.9`，`torch 2.9.1+cpu` |
| **Linux 双进程 TCP 传输验证**<br/>（Mooncake Transfer Engine） | [mooncake-linux-info.json](mooncake-linux-info.json) | [mooncake-linux-requirements.txt](mooncake-linux-requirements.txt) | Ubuntu 22.04.5 x86_64，Python `3.10.12`，`torch 2.9.1+cpu`，`mooncake 0.3.14`（非 CUDA 版，TCP） |
| **实验室单卡 V100 部署验收**<br/>（真实模型保存与重启恢复） | [lab-v100-info.json](lab-v100-info.json) | [lab-v100-requirements.txt](lab-v100-requirements.txt) | Ubuntu 20.04 x86_64，`Tesla V100S-PCIE-32GB`（驱动 `535.230.02` + `cuda-compat 570.211.01`），Python `3.12.12`，`torch 2.10.0+cu128`，`1Cat-vLLM v1.5.3`（`cb97f379...`），`mooncake 0.3.14`，`Qwen2.5-1.5B-Instruct` FP16 |
| **任务 A（1~5）实机性能对照**<br/>（主存回读 vs GPU 重算） | [task-a-info.json](task-a-info.json) | 沿用 [lab-v100-requirements.txt](lab-v100-requirements.txt)（另加 `matplotlib 3.11.2`） | 同上 V100 环境，锁定 `160` GPU 物理块（`2560` tok）、`4 GiB` Owner DRAM、同机 TCP、`64–1536` tok 合成前缀，含全部主实验与诊断复现命令 |

---

## 说明

- 本地 CPU / ECS 的非 CUDA 环境与实验室 V100 GPU 环境严格隔离，互不混用依赖。
- 实验室服务器上的兼容性组件（私有 CUDA 12.8 编译器、`cuda-compat 570.211.01`、`protoc 33.5`、`rdma-core v56.1`）均位于项目独立目录，未修改宿主机全局驱动或系统库；目录结构见 [服务器目录说明](../docs/02-服务器目录说明.md)。
