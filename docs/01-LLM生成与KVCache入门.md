# LLM 生成与 KV Cache 入门

KV Cache 是语言模型推理时保存的中间状态。理解它，需要沿着一条计算链路：文字 → token → 向量 → 因果 attention → 自回归生成 → 历史 K/V 复用 → 内存与传输开销。

本章以 Python 数组、矩阵运算和一个可以手算的例子解释这条链路。更完整的张量、内存与 I/O 推导见 [前置知识](02-KVCache计算与存储原理.md)。

## 1. 文字怎样进入模型

Tokenizer 把文字转换成 token ID。下面仅表示数据形式，不是真实分词结果：

```python
token_ids = [41, 206, 17]
```

ID 是词表索引，编号大小不代表语义距离。一个汉字不必等于一个 token，空格、特殊符号和输入模板也会影响分词结果。

Embedding 表把 ID 映射为向量。如果词表大小是 `V`、向量维度是 `D`，表的形状就是 `(V, D)`。概念上相当于从二维数组中取出对应行：

```python
# 概念代码，embedding_table 表示已经加载的模型参数。
vectors = [embedding_table[token_id] for token_id in token_ids]
```

向量经过多层计算，形成结合上下文的信息表示，称为 hidden state。模型权重是在训练中学习的参数；推理使用这些参数处理新输入。KV 则是针对输入产生的中间状态，不属于新训练的权重。

位置信息用来区分 token 顺序。因此，相同的 token 出现在不同位置或前文之后，可能产生不同状态。

## 2. 自回归生成怎样进行

模型根据已有 token 预测下一个 token。输出映射为词表中的候选 token 生成分数，称为 logits；按照贪心或采样等规则选择一个 token，再将其作为后续计算的输入。

```text
已有输入 → 模型 → 候选 token 分数 → 选择下一个 token
          ↑                                  ↓
          └──────── 新 token 接回输入 ─────────┘
```

处理已知输入的阶段称为 **prefill**；逐步生成后续 token 的阶段称为 **decode**。prefill 的最后一个位置可以直接用于选择首个输出 token。

## 3. Attention 中 Q、K、V 各是什么

输入向量 `x` 经过三组线性变换：

```text
q = x Wq     用于匹配上下文的 Query
k = x Wk     各位置提供的匹配特征 Key
v = x Wv     按匹配权重参与汇总的 Value
```

Query 与允许访问位置的 Key 计算匹配分数，经过缩放和 softmax 转换成权重，再对 Value 加权求和：

\[
O = \operatorname{softmax}(QK^\top/\sqrt d + M)V.
\]

`M` 是 mask，用来屏蔽不允许访问的位置。Q/K/V 都是数值向量；这里的 Key 不是普通数据库中的字符串键，Value 也不是原始文本。[Transformer 原论文](https://arxiv.org/abs/1706.03762)

### 一个数值例子

```text
q  = [1, 0]
k1 = [1, 0]    v1 = [10, 0]
k2 = [0, 1]    v2 = [ 0,10]
k3 = [1, 1]    v3 = [10,10]
```

点积得到 `[1,0,1]`，除以 `sqrt(2)` 后做 softmax，权重约为 `[0.401,0.198,0.401]`。输出约为 `[8.02,5.99]`，是三个 Value 的加权和。

以下代码可以直接复算：

```python
import math

q = [1, 0]
keys = [[1, 0], [0, 1], [1, 1]]
values = [[10, 0], [0, 10], [10, 10]]
scores = [sum(a*b for a, b in zip(q, k)) / math.sqrt(2) for k in keys]
exps = [math.exp(s - max(scores)) for s in scores]
weights = [x / sum(exps) for x in exps]
output = [sum(w*v[j] for w, v in zip(weights, values)) for j in range(2)]
print(weights, output)
```

## 4. 因果性为什么允许缓存

因果 attention 要求每个位置只能访问自身及之前的位置。句尾追加 token，不会改变历史位置可访问的信息。

在模型权重、前缀、位置配置兼容，且采用推理模式的条件下，历史位置的输入与各层状态不会因后缀追加而改变。于是，每层历史 K/V 都可以保存，供新位置使用。

以 `[A,B,C]` 为输入：

```text
Prefill：生成 A/B/C 在各层的 K/V，并预测首个输出 D。
下一步：消费 D，生成 D 的 K/V；D 的 Q 访问 A/B/C/D 的 K/V。
再下一步：消费 E，继续复用历史 KV，并追加 E 的 K/V。
```

一个 token 被选中，不等于它的各层 KV 已经生成；通常需要在后续 forward 消费该 token 时才产生对应 KV。

**缓存省掉历史 K/V 的重复计算，新 Query 仍需要读取历史 K/V。** 历史 Q 通常无需持久保留，因为新位置使用自己的 Query。

## 5. 容量为什么会成为限制

普通 dense MHA/GQA/MQA 的 KV 数据量可按下式估算：

```text
字节/token = 2 × 层数 × KV head 数 × head 维度 × 元素字节数
总数据量  = 字节/token × 所有序列的缓存 token 数之和
```

例如 `32 层、32 KV heads、head dim 128、每元素 2 bytes`，每 token 为 512 KiB，32,768 token 为 16 GiB。该数值是结构假设下的容量计算，不代表任意模型支持 32K 上下文。

GPU 显存还需要容纳权重、激活和运行时缓冲。上下文更长、活跃序列更多时，KV 会限制可承载的请求量。

## 6. 卸载与恢复解决什么问题

将部分 KV 放入主机 DRAM 或 SSD，可以扩大有效缓存容量。代价是 GPU 需要这些数据时，必须经过相应 I/O 与传输路径恢复。

两类场景需要分开理解：

| 场景 | 访问特点 | 主要权衡 |
|---|---|---|
| 会话间保存与前缀复用 | 新请求到来时恢复兼容前缀 | 恢复旧 KV 与重算前缀哪个更便宜 |
| 活跃 decode 工作集卸载 | 普通完整注意力每步都访问历史 KV | 每 token 的读取量、带宽与等待 |

把旧会话保存到 SSD，并不自动说明单个超显存活跃请求可以高效生成。技术分类和完整生命周期见 [知识全景](03-KVCache系统全景.md)。

## 7. 缓存语义怎样验证

完整前缀重算与增量 KV 应在固定模型、输入和执行条件下进行比较。先对固定 token 序列比较每步 logits，再比较贪心生成的 token ID；恢复前缀缓存后重复相同检查。

无损序列化往返只能检验状态保存与恢复，不能代表 SSD 卸载性能。容量公式和假设带宽计算也应标明为理论估算，其中 GB/s 与 Gbit/s 需要区分。

## 8. 概念检查

- token ID、embedding、模型权重、隐藏状态和 KV 分别表示什么？
- Q/K 怎样决定权重，V 怎样形成输出？
- 历史 K/V 为什么可以复用，又为什么仍需读取？
- 前缀复用与活跃工作集卸载，分别改变哪一段数据路径？

相关论文的机制与图表导航见 [文献导读](04-文献阅读导引.md)。
