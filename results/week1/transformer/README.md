# 小型 Transformer 的计算检查

## 检查什么

使用固定初始化的小型 CPU decoder，把手写 attention 与 PyTorch SDPA 对照，并检查因果关系、输出形状和一次短训练中的参数更新。程序见 [week1_transformer_demo.py](../../../scripts/week1_transformer_demo.py)，环境见 [CPU 记录](../../../environment/cpu-info.json)。

## 已有输出

| 检查 | 结果 | 说明 |
|---|---|---|
| 手写 attention 对照 SDPA | 最大误差 `1.1102230246251565e-16` | 使用同一组输入和 mask 比较 |
| 改变未来位置 | 过去输出保持不变 | 检查 causal mask 的语义 |
| 多头 attention 输出 | `[1,4,8]` | split、计算、merge 后恢复隐藏维度 |
| cross-attention 权重 | `[1,2,3,5]` | Query 长度 3，Key / Value 长度 5，无须相同 |
| decoder logits | `[1,4,16]` | 批量 1、输入 4 个位置、词表 16 |
| 40 次短训练更新 | loss 从 `2.6853453249551325` 降到 `0.0626180258364998` | 参数确实变化；只验证反向传播与更新链路 |

数值、第一头的完整注意力权重和生成 IDs 保存在 [checks.json](checks.json)。其中还保留基础张量操作的检查，便于对照矩阵布局和 softmax。

这些结果说明实现中的若干数学与执行关系成立。模型未经过真实语料训练，生成的整数序列不具有语言质量含义，loss 下降也不构成泛化能力证明。

## 复查方式与边界

在仓库根目录，使用对应 CPU 环境分别运行：

```bash
python scripts/week1_transformer_demo.py --mode basics
python scripts/week1_transformer_demo.py --mode attention
python scripts/week1_transformer_demo.py --mode decoder
python scripts/week1_transformer_demo.py --mode train-step
```

当前归档整理自已有输出，没有重新运行；原输出未记录对应的源码提交指纹。若当前代码或依赖已有变化，应重新核对，不能仅凭旧文件中的 `passed` 认定新实现通过。
