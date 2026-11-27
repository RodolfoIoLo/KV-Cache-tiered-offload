# KV 前缀从 GPU 保存到 DRAM，并在重启后恢复

## 验证的问题

推理进程退出后，之前计算的前缀 KV 能否留在独立进程的 DRAM 中，由新推理进程取回并继续生成相同的 token？

这是一项功能与正确性验证。它没有测量加速比，也没有验证活跃长序列超出 GPU 容量后的分层卸载。

## 方法

使用 Qwen2.5-1.5B-Instruct、源码构建的 1Cat-vLLM 和 Mooncake Store，在单张 V100 上顺序运行三阶段：

1. **all-GPU**：不接入外部缓存，得到参考输出。
2. **cold**：接入新建的空 DRAM Store，处理同一个输入，等待 KV 保存完成。
3. **restore**：只重启 vLLM，保留 Store，再处理完全相同的输入，排除旧 vLLM 的 GPU 本地前缀命中。

三阶段均为 FP16、512 输入 tokens、16 输出 tokens、贪心生成、单请求、16-token KV 块。Owner 提供 1 GiB DRAM，传输使用同机 TCP，未开启 SSD。模型、软件修订及详细配置见 [环境记录](../../../environment/lab-v100-info.json)；请求构造和执行链路见 [工程解析](../../../notes/05-从真实工程场景认识项目.md)。

## 结果

| 检查 | 实测结果 |
|---|---|
| cold 成功保存的 KV | 14 MiB，即 14680064 bytes |
| restore 成功取回的 KV | 13.5625 MiB，即 14221312 bytes |
| restore 数据来源 | 31 个 DRAM key，31 成功，0 失败；SSD key 为 0 |
| 生成 token IDs | cold、restore 均与 all-GPU 的 16 个位置完全一致 |
| cold 所选 token logprob 最大误差 | 0 |
| restore 所选 token logprob 最大误差 | `1.9046710804104805e-05`，低于本次 `1e-3` 阈值 |
| 独立 CUDA 指针传输检查 | GPU → 独立进程 8 MiB DRAM → GPU，内容逐字节一致 |

模型有 28 层、2 个 KV 头、head dimension 128。FP16 KV 为每 token 28 KiB，每 16-token 块跨层共 448 KiB，因此保存字节数对应 32 个完整块。

当前 connector 在完全命中时保留最后一个 16-token 块重新计算，以获得采样所需的末位置输出。恢复的是 496 个位置，正好对应 31 块与实测的 13.5625 MiB；差额不是一次加载失败。

## 结论与边界

这次验证支持：**在固定模型与请求下，Mooncake DRAM 中的前缀 KV 能跨越 vLLM 进程重启被取回，并被真实模型用于继续生成。**

输出对照目前只覆盖一条输入的 16 个生成 token 及其所选 logprobs；没有比较完整词表 logits，也没有覆盖块边界、并发、取消和故障恢复。保存阶段与 baseline 的 GPU 前缀缓存设置也有区别，所以不能把这三次功能检查直接用作性能对照。

活跃序列溢出、SSD 回读、跨机 PD 和计算 / I/O 重叠仍需各自的实验。这里没有给出尚未测量的吞吐、延迟或显存容量收益。
