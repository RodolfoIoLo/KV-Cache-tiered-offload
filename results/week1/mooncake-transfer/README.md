# 双进程 Mooncake Transfer Engine TCP KV 交接检查

## 验证问题

在单台 Linux 主机上启动两个独立进程（发送端与接收端），验证能否通过 `Mooncake Transfer Engine`（TCP 传输）把前缀 KV 字节从发送端内存搬运到接收端，并让接收端基于恢复的 KV 直接续算后续 token。

对应程序：[week1_mooncake_transfer_demo.py](../../../scripts/week1_mooncake_transfer_demo.py)，环境见 [mooncake-linux-info.json](../../../environment/mooncake-linux-info.json)。

## 实测结果（[checks.json](checks.json)）

| 检查项 | 实测结果 | 说明 |
|---|---|---|
| 传输通道分工 | 控制面：本地 TCP socket；数据面：`Mooncake TransferEngine (TCP)` | 真实 KV 载荷通过 Transfer Engine 内存注册与写入完成，不经控制通道 |
| 传输体积 | 总载荷 `3541 B`（含 `768 B` KV 张量） | 覆盖 `3` 个前缀 token，接收端续算 `2` 个后续 token |
| KV 逐字节一致性 | `kv_equal = true` | 接收端恢复的 KV 张量与发送端严格一致 |
| 续算数值误差 | `logits_max_error = 2.2204e-16`，`next_ids = [9]` | 与未拆分直接推理在 FP64 机器精度内一致 |

> 本次验证证明了跨进程 Transfer Engine TCP 传输与续算语义正确，属于 CPU/TCP 功能检查，不代表跨节点或 RDMA 带宽表现。
