# 🛠️ 实验脚本说明（`scripts/`）

本目录包含项目的全部可运行脚本，按阶段分为 **Week 1 基础机制验证（4 个）** 与 **任务 A 单机单卡实机评测（6 个）**。

---

## 1. Week 1 · 基础机制验证脚本

用于脱离大型框架、独立验证核心数学与内存管理机制（不代表 GPU 推理吞吐或延迟性能）。

| 脚本 | 作用 | 运行环境 | 常用运行方式 |
|---|---|---|---|
| [week1_transformer_demo.py](week1_transformer_demo.py) | 手写 Attention vs SDPA、因果掩码、张量形状与短训练更新检查 | CPU（[cpu-info.json](../environment/cpu-info.json)） | `python scripts/week1_transformer_demo.py --mode attention`（可选 `basics` / `attention` / `decoder` / `train-step`） |
| [week1_kv_cache_demo.py](week1_kv_cache_demo.py) | 增量 KV 等价性、生成尾差、前缀身份隔离、Radix 匹配与容量线性缩放 | CPU（依赖 `week1_transformer_demo.py`） | `python scripts/week1_kv_cache_demo.py --mode equivalence`（可选 `equivalence` / `prefix` / `radix` / `size`） |
| [week1_paged_kv_demo.py](week1_paged_kv_demo.py) | PagedAttention 物理块池、块边界分配、共享前缀、写时复制（COW）与回收 | CPU | `python scripts/week1_paged_kv_demo.py` |
| [week1_mooncake_transfer_demo.py](week1_mooncake_transfer_demo.py) | 双进程通过 Mooncake Transfer Engine（TCP）交接前缀 KV 并续算 | Linux + Mooncake（[mooncake-linux-info.json](../environment/mooncake-linux-info.json)） | 按脚本 `--role sender` / `--role receiver` 分别启动两个进程 |

---

## 2. 任务 A · 单机单卡主存回读 vs GPU 重算脚本

面向实验室 V100 服务器（`1Cat-vLLM v1.5.3` + `Mooncake 0.3.14` + `Qwen2.5-1.5B-Instruct` FP16，环境见 [task-a-info.json](../environment/task-a-info.json)），六个脚本职责清晰、串联协作：

| 脚本 | 角色与核心职责 | 常用调用方式 |
|---|---|---|
| [task_a_run.sh](task_a_run.sh) | **服务端编排与安全清理入口**。检查端口、记录启动前 GPU 状态；`store` 模式拉起 Master + 4 GiB Owner；启动锁定 160 GPU 块的单实例 vLLM；写入 `run.json` 并调用压测客户端；退出时用 `pidfd` 精准清理本轮进程并写入 `status.json`。支持 `TASK_A_DIAGNOSTICS=1`、`TASK_A_ROUTING=1`、`TASK_A_BATCH_TOKENS`。 | `bash scripts/task_a_run.sh recompute /path/to/out --lengths 64,128,256,512,1024,1536 --repetitions 3 --warmups 1 --seed 53` |
| [task_a_benchmark.py](task_a_benchmark.py) | **流式压测与 `/metrics` 审计客户端**。按种子构造「Build → 3×Pressure → Revisit」序列；流式 SSE 计时客户端 TTFT；每条请求前后读取 `/metrics`，等待异步 PUT/GET 收敛后断言本地命中、外部回读、计算 token 数与字节数全部符合预期，逐条落盘 `measurements.jsonl`。 | 由 `task_a_run.sh` 自动调用，也可通过 `--api` 对已启动服务独立运行 |
| [task_a_report.py](task_a_report.py) | **配对校验、统计汇总与绘图脚本**。核验成对 `recompute` / `store` 目录的配置、`prompt_digest` 与 `serve.log` DRAM 层日志；排除 warmup，计算中位数、IQR、配对差值、GET/Lookup 耗时与服务端残差，输出 `samples.csv`、`summary.csv`、`checks.json` 及 `ttft.png`。 | `python scripts/task_a_report.py --pair <compute_dir> <read_dir> --out <out_dir> --plot` |
| [task_a_diagnostics.py](task_a_diagnostics.py) | **非计时 Worker 诊断插件**（通过 `--worker-extension-cls` 注入）。Hook 模型 `forward`/`compute_logits`、`flash_v100` 路由统计与 Mooncake 收发线程：记录 CUDA Event 前向耗时与注意力路由、保存全词表 logits、并在 `before_put`/`after_get` 用 `cudaMemcpy` 抓取 GPU 显存块原始字节。 | 设置 `TASK_A_DIAGNOSTICS=1` 后由 `task_a_run.sh` 自动注入（仅用于独立诊断，不计入主计时） |
| [task_a_check_diagnostics.py](task_a_check_diagnostics.py) | **诊断快照离线核对脚本**。逐字节比对 `before_put` 与 `after_get` 的 KV 快照，计算 `read` vs `recompute` 的全词表 logits 误差、Top-3 候选及首步前向耗时和路由，生成 `diagnostics.json`。 | `python scripts/task_a_check_diagnostics.py --read <diag_read> --compute <diag_compute> --output diagnostics.json` |
| [task_a_routing.py](task_a_routing.py) | **可选长度选路连接器（任务 A 第 6 步预留）**。继承 `MooncakeStoreConnector`，在 Master 查到真实连续命中长度后，按白名单决定走 DRAM 回读还是退回 GPU 重算。 | 设置 `TASK_A_ROUTING=1` 并传入 `--read-prefix-lengths` 启用 |
