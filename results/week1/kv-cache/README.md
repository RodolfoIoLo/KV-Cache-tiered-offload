# 增量 KV、前缀复用和容量检查

## 检查什么

在同一个固定权重的小型 CPU decoder 上，对照完整重算与缓存增量计算；进一步检查前缀身份、分支隔离、Radix 匹配及逻辑 KV 字节数。程序见 [week1_kv_cache_demo.py](../../../scripts/week1_kv_cache_demo.py)，依赖 [Transformer demo](../../../scripts/week1_transformer_demo.py)。

## 已有输出

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 增量与完整重算 logits | 最大误差 `5.551115123125783e-16` | 固定输入下两条计算路径数值相符 |
| 逐步缓存长度 | `1,2,3,4,5,6` | 每次追加一个 token 后位置与缓存正确增长 |
| 两个 token 一起追加 | 与参考一致 | mask / offset 不仅适用于单 token |
| 保存与读回缓存 | KV 一致 | 教学张量的本地无损往返成立 |
| 4-token prompt 生成 8 tokens | 末尾缓存长度 11 | 最后被选出的 token 尚未再次作为输入 |
| 两条后缀分支 | 都命中 3-token 前缀，且互不污染 | 共享前缀与分支状态被区分 |
| 错误前缀 / 错误模型身份 | 命中长度均为 0 | 不能仅凭缓存形状相同就复用 |
| Radix 边中间匹配 | 匹配长度 3 | 压缩边支持在边内结束匹配 |
| 实际张量字节数 | 与公式一致 | 长度或 batch 翻倍，payload 翻倍；FP64 改 FP32，payload 减半 |

完整输出、Radix 树结构和各组容量数值见 [checks.json](checks.json)。其中 `16 / 80 / 4 GiB` 是对应模型结构假设下的公式计算，未申请这些大小的张量，也不是可运行上下文长度的承诺。

## 这些结果还不是什么

本地 `torch.save` / `torch.load` 往返只检查状态能否保存和恢复，没有测 GPU 卸载、NVMe 设备读取或 I/O 重叠。Radix 也是教学索引，不包含完整服务系统的调度、并发和淘汰实现。真实 Mooncake DRAM 路径另见 [部署验证](../deployment/README.md)。

## 复查方式与边界

在仓库根目录使用对应 CPU 环境运行：

```bash
python scripts/week1_kv_cache_demo.py --mode equivalence
python scripts/week1_kv_cache_demo.py --mode prefix
python scripts/week1_kv_cache_demo.py --mode radix
python scripts/week1_kv_cache_demo.py --mode size
```

当前归档整理自已有输出，没有重新执行；原输出未记录对应源码提交指纹。修改过代码或依赖后，应重新运行并核对，不把已有文件当成新版本的验收。
