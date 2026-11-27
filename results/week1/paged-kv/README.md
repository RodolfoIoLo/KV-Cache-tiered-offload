# 分页 KV 块池、共享与写时复制（COW）检查

## 验证问题

用固定块大小（`block_size = 4`）的 CPU 物理块池，验证 PagedAttention 的三项核心内存管理语义：逻辑块到物理块的映射与边界分配、跨请求共享前缀时的物理块复用、以及分支追加时的写时复制（Copy-on-Write）与完整回收。

对应程序：[week1_paged_kv_demo.py](../../../scripts/week1_paged_kv_demo.py)，环境见 [cpu-info.json](../../../environment/cpu-info.json)。

## 实测结果（[checks.json](checks.json)）

| 检查项 | 结果 | 说明 |
|---|---|---|
| 块边界分配（`block_size = 4`） | 长度 `3 / 4 / 5` → `1 / 1 / 2` 块 | 刚好满块不额外分配，超出 1 token 申请新块 |
| 前缀共享节省块数 | 未共享 `4` 块 → 共享后 `3` 块 | 请求 A（6 tok，块 `[0, 1]`）与请求 B（5 tok，块 `[0, 2]`）共享物理块 `0` |
| 写时复制（COW） | `copy_on_write = true` | `fork` 后共享尾块 `0`，追加时分支 A 分配新尾块 `2`，分支 B 保留块 `0`，互不覆盖 |
| 数学一致性与回收 | `passed = true`, `all_blocks_reclaimed = true` | 分页 gather 后的 Attention 输出与连续 KV 参考一致；请求结束后所有物理块归还空闲池 |

## 复现命令

```bash
python scripts/week1_paged_kv_demo.py
```
