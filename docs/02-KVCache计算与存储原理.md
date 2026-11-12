# KV Cache 计算与存储原理

本章按计算原理、缓存管理、数据搬运和性能测量组织，适用于具有 Python 基础、希望理解 KV Cache 系统的读者。公式针对常见 decoder-only、dense causal attention；特殊架构另外注明。

## 1. 张量、矩阵与模型推理

### 1.1 最低限度的数学

向量是有序的一组数，矩阵乘法把输入向量投影到新的表示空间。一个 token 的隐藏表示记作 `x ∈ R^D`，线性层 `xW` 可以生成不同用途的表示。点积 `q·k` 是加权求和，数值越大通常意味着该 Query 对该 Key 的匹配分数越高。

softmax 将分数转换为非负且总和为 1 的权重：

\[
\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}.
\]

实现时先减去最大分数以降低指数溢出的风险；这不改变数学结果。最终输出是 Value 的加权和。它不是“查找一个最相似的 token”，而通常是对所有允许位置做连续加权。

张量是多维数组。常见维度有 batch、sequence、head、head dimension。shape 相同不保证含义相同：`[B,H,T,d]` 和 `[B,T,H,d]` 的数据布局与内核契约不同，搬运时不能只核对总字节数。

### 1.2 token 与自回归

Tokenizer 把字符串变成整数 token ID。一个汉字不必等于一个 token，两个看似相同的字符串也可能由于空格、模板或特殊符号产生不同 ID。

语言模型根据此前 token 的条件分布选出下一个 token：

\[
p(x_1,\ldots,x_n) = \prod_{t=1}^{n}p(x_t\mid x_{<t}).
\]

“选出”可以是贪心 argmax，也可以是采样。生成出的 token 又成为下一步输入，因此普通自回归的各生成步骤有顺序依赖。一次 forward 通常输出 next-token logits；prefill 的最后一个位置可以用于选择首个生成 token，不需要额外执行一次完整 prompt forward。

**与课题的关系**：缓存 key 要基于模型实际使用的 token 与执行语义；正确性要比较 token ID 和 logits，不能只看回答文本“差不多”。

### 1.3 Transformer 中必须了解的部件

常见 decoder block 包含归一化、因果 self-attention、残差连接和逐 token 的前馈网络。层与层串联：第 `l` 层输出成为下一层输入。模型权重在正常推理期间固定，KV 则是请求相关的动态状态。

2017 年原始论文讨论 encoder-decoder Transformer；理解自回归 KV Cache 需要重点关注 decoder 的 masked self-attention。[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**自测**：能说清 token ID、hidden state、模型权重、attention score、K/V 各是什么，哪些随请求变化。

## 2. 从 self-attention 推导 KV Cache

### 2.1 每一层在算什么

先忽略 batch 与多头。对长度 `T`、隐藏维度 `D` 的输入 `X`：

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
\]
\[
O=\operatorname{softmax}(QK^\top/\sqrt{d}+M)V.
\]

`M` 是因果 mask：位置 `t` 只能看到 `j≤t` 的位置，未来位置的分数被屏蔽。`QKᵀ` 可以形成 `T×T` 的分数矩阵，但实际高效内核不一定把整个矩阵写回显存。

在 decode 的新位置 `t`，只需生成当前的 `q_t,k_t,v_t`，然后：

\[
o_t=\operatorname{softmax}\left(q_t[K_1;\ldots;K_t]^\top/\sqrt d\right)[V_1;\ldots;V_t].
\]

历史 K/V 被复用，新的 K/V 被追加。每层都有自己的投影和隐藏状态，因此不能只缓存第一层的 K/V，也不能只缓存最终层输出。

### 2.2 为什么历史 K/V 不需要重新计算

固定模型、固定位置、固定前缀，在推理模式且无随机 dropout 时：

1. 第一层历史位置的输入 embedding 不会因为句尾追加 token 而变化。
2. 因果 mask 使历史位置看不到新追加的 token，所以历史 attention 输出不变。
3. 逐 token 归一化、前馈层与残差处理不会把未来信息引入历史位置。
4. 对层数作同样的归纳，后续各层历史位置的隐藏状态与 K/V 也不变。

因此 KV Cache 是利用因果性做的计算复用，不是模型“背下了知识”，也不是每轮训练出来的新参数。

### 2.3 为什么通常不缓存 Q

当前 Query 用来询问历史。未来位置计算 attention 时使用的是未来自己的 Query，历史 Query 不再需要参与这些新位置的输出。保留历史 K/V 足以支持新的 attention。

这里不讨论某些特殊内核为了调度保留的临时 Q 缓冲；那与持久的历史 KV cache 是不同层面的对象。

### 2.4 缓存节省了什么，没有省掉什么

没有缓存时，每生成一步都可以重新计算完整前缀，历史 token 的投影、前馈和历史位置的 attention 都会被重复执行。使用缓存后，每步通常只执行新 token 的各层计算，再读取历史 K/V。

对单个普通 dense attention 层，新位置与长度 `T` 的历史交互仍需要约 `O(Td)` 的 attention 工作；历史越长，这项工作和读取量仍然增长。prefill 中所有 prompt 位置之间的 dense attention 则通常有 `O(T²d)` 的计算量。完整模型还包括线性投影、MLP 等，不能把 attention 的复杂度当作整个服务的运行时间。

**验证思路**：逐层跟踪新 K/V 追加，在相同模型与固定 token 输入下比较完整前缀重算与增量结果；检查位置、mask 和缓存长度是否一致。

**自测**：如果修改了 prompt 中间一个 token，从哪个位置之后的 KV 可能失效？答案是该位置及其之后，各层的后续状态都可能受影响。

## 3. Prefill 与 Decode：两种不同的执行形态

### 3.1 Prefill

处理一段已知输入，许多 token 可以在因果 mask 下并行计算。矩阵较大时通常较容易利用 GPU 算力，长 prompt 会占用较长时间。prefill 结束后已经有输入 token 的各层 KV。

命中缓存时，系统可以只对未命中的后缀进行增量 prefill，但它仍需要访问已恢复的前缀 KV。是否划算，取决于省掉的计算和新增的数据读取。

### 3.2 Decode

每个活跃请求通常每步生成一个新 token。单请求的小矩阵计算未必能充分利用 GPU；多请求 batch 可以摊薄权重读取与 kernel 开销。长上下文时读取 KV 的开销也会很突出。

常见说法“prefill 偏计算，decode 偏内存”是一个分析起点，不能代替 profiler：batch、模型结构、量化、上下文、并行方式都会改变瓶颈。

### 3.3 连续批处理与 Chunked Prefill

静态 batch 要等待整批结束；连续批处理允许完成的请求退出、新请求进入。它使吞吐更高，也意味着缓存状态会频繁分配、引用、释放。

Chunked prefill 把长 prompt 拆成调度片段，让长 prefill 和 decode 更灵活地交错；它不是把 Transformer 的因果依赖删除，也不是 KV 的物理存储块。**调度 chunk、缓存 block、SSD I/O 请求、网络 slice 可以是不同大小**。

**与课题的关系**：更早知道下一批请求是谁，才更容易做有效预取；长 prefill 与后台写回也可能挤占 decode 所需资源。

## 4. KV 的 shape 与容量预算

### 4.1 通用公式

设层数为 `L`，KV head 数为 `H_kv`，每个 head 维度为 `d_h`，每元素字节数为 `s`，请求 `i` 的已缓存长度为 `T_i`。对常规各层相同、全历史 dense attention：

\[
S_{token}=2LH_{kv}d_hs,
\qquad
M_{logical}=S_{token}\sum_iT_i.
\]

`2` 代表 K 和 V。长度相同且没有共享时，`sum T_i=B×T`。若 K/V 精度不同，则把 `2s` 换成 `s_K+s_V`。量化的 scale、zero point、对齐和 metadata 还需额外计入。

一种教学逻辑 shape 是 `[L,2,B,H_kv,T,d_h]`。真实框架可能分开 K/V，用分块布局、padding 或特殊 stride；应以所选内核的实际布局为准。

### 4.2 容量计算示例

取 `L=32, H_kv=32, d_h=128, s=2`：

```text
每 token = 2 × 32 × 32 × 128 × 2 = 524,288 bytes = 512 KiB
32K token = 32,768 token
单请求 KV = 524,288 × 32,768 = 17,179,869,184 bytes
         = 16 GiB ≈ 17.18 GB
5 个独立且同长度的活跃序列 = 80 GiB KV
```

这是与 Llama-2-7B 类 MHA 结构相符的**容量假设**，不是说原版模型的上下文窗口天然支持 32K。总显存还包括权重、激活/工作区、CUDA graph 或运行时池等，80 GiB KV 不可能自动“刚好放进 80GB 卡”。

若保持其它条件而 `H_kv=8`，每 token 128 KiB，32K 为 4 GiB。说明参数量相近的模型也可能有很不一样的 KV 压力，不能只按“7B/8B”选择模型。

单位严格区分：`GB=10^9 bytes`，`GiB=2^30 bytes`，`Gbps` 是比特每秒，理想字节带宽先除以 8，再考虑协议开销。

### 4.3 分块、共享与实际占用

若 block 包含 `q` 个 token，无共享时：

\[
M_{blocks}=S_{token}\sum_iq\lceil T_i/q\rceil.
\]

尾块可能未填满；共享前缀时应数**独立物理块**，不能把每个请求的逻辑长度简单相加。还有块表、索引、hash、引用计数、allocator 与 staging buffers。

“总 GPU reserved memory 没下降”不一定意味着卸载无效：框架可能预先分配固定池。至少同时观察逻辑活跃 KV、可用块数、实际分配、reserved pool、峰值进程占用和可承载并发。

### 4.4 MHA、GQA、MQA、MLA

- **MHA**：通常 Q/K/V 各有相同 head 数。
- **GQA**：一组 Q heads 共享一个 KV head；容量公式用 `H_kv`，不用 `H_q`。
- **MQA**：KV head 数为 1，是更极端的共享形式。
- **MLA 等特殊结构**：可能缓存压缩 latent 与位置相关分量，不能直接套普通 KV 公式；要读模型配置和内核实现。

GQA 是模型结构层面的机制，不能把既有 MHA 模型在运行时随意删掉大部分 KV heads 而声称输出等价。[GQA 论文](https://arxiv.org/abs/2305.13245)

张量并行下每 GPU 的 KV 未必等于总 KV 除以 TP 数：KV heads 的分片/复制、TP 大于 KV head 数时的处理、布局和 padding 都会影响结果。先测单卡，再分析多卡。

**自测**：能从一个普通模型配置中的层数、KV heads、head dim 与 KV dtype 算出字节/token，再解释与实测池占用的差值。

## 5. Prefix Caching、分页与缓存有效性

### 5.1 只有兼容前缀才能直接复用

例如：

```text
请求 A：[系统提示 S][文档 D][问题 Q1]
请求 B：[系统提示 S][文档 D][问题 Q2]
```

共同前缀 `[S][D]` 有直接复用的机会。但如果 B 在 D 前插入一个不同的 S，D 中 token 的位置与历史依赖都可能改变；“文档文本相同”不保证 KV 相同。跨不连续片段复用是额外问题，可能需要重算或专门的方法。

除了 token ID，还要考虑模型权重/revision、adapter、位置/RoPE 配置、attention mask、模态输入、KV dtype 与布局，以及租户隔离。清理过期缓存属于正确性的一部分。

### 5.2 页表与 Copy-on-Write

分页将逻辑上的连续 token 序列映射到一组物理 KV 块。块无需物理连续，多个请求可以引用共同前缀。当共享尾部需要被写入时，通常需要独立可写空间或 copy-on-write，避免请求相互污染。

分页解决的是管理和碎片问题。是否支持 CPU/SSD、怎样交换、何时预取，需要额外的框架和后端实现。[PagedAttention 论文](https://arxiv.org/abs/2309.06180)

### 5.3 Hash 与 Radix Tree

一种块标识思路是把“前一块的标识、当前 token 块、额外语义标识”一同 hash。这保留前缀依赖，避免只根据当前片段文本命中错误上下文。实现仍需考虑碰撞与隔离。[vLLM Prefix Caching 设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)

Radix tree 用共享前缀组织 token 序列，便于找最长前缀和管理共享；它属于缓存索引与复用管理，不是 GPU attention 的数学替代品。[SGLang 论文](https://arxiv.org/abs/2312.07104)

**练习**：画出 `[1,2,3,4]` 与 `[1,2,5]` 的共享部分；再说明 `[9,2,3,4]` 为什么不能因为后面三个 token 相同就直接使用前者的全部对应 KV。

## 6. 内存层级、带宽与性能模型

### 6.1 容量、带宽、延迟分别是什么

容量是能存多少；带宽是单位时间能搬多少；延迟是一次操作从提交到完成需要多久。一个设备可以容量很大、顺序带宽不错，却因小随机请求延迟或低队列深度表现很差。

最简单模型是：

\[
T_{move}\approx\alpha+\frac{S}{BW_{effective}}.
\]

`α` 包含固定开销，`S` 是字节数。有效带宽必须测，不能直接用设备包装上的峰值。并发、NUMA、方向、大小、竞争、内存是否 pinned 都会改变它。

### 6.2 SSD 到 GPU 是多段路径

串行经过 SSD 与 H2D 时：

\[
T_{serial}\gtrsim\frac{S}{BW_{ssd}}+\frac{S}{BW_{h2d}}.
\]

若分块流水化且可独立并行，长流的理想吞吐下界接近最慢一级：

\[
T_{pipeline,ideal}\gtrsim\max(S/BW_{ssd},S/BW_{h2d}),
\]

但还有管线填充、尾部、每块开销和竞争，实际必然需要验证。

**假设计算，非实测**：16 GiB 若以 25 GB/s 做一次 H2D，数据项约 0.687 秒；以 7 GB/s 从 SSD 读约 2.454 秒。两段串行约 3.141 秒，不含额外开销。即使理想流水化，也不能把 SSD 的 2.454 秒变没。

如果这 16 GiB 是活跃 dense decode 每步都要重新取回的工作集，这个下界会约束每个 token；如果是恢复会话只读一次，则可能通过避免长 prefill 获益。工作负载定义决定结论。

### 6.3 取回与重算的盈亏

缓存复用有利的一阶条件：

\[
T_{lookup}+T_{retrieve}+T_{inject}<T_{recompute,saved}.
\]

这比较的是相同前缀在相同执行条件下的节省量，不是“所有未命中请求的完整 prefill 时间”。如果已有其它 GPU 前缀命中，新增 SSD 命中实际能多省多少，也要按边际收益计算。

写回不在当前响应关键路径，也不是免费：它消耗 PCIe、CPU、DRAM、SSD 写带宽，可能使后续请求排队。可进一步用预期重用次数乘以节省量，减去写入、占用容量和干扰代价，设计 admission 策略。

### 6.4 Roofline、Amdahl 与排队

- **Roofline**：算术强度是 FLOPs/byte。计算量少而读取量大时，再提升计算峰值可能帮助有限。
- **Amdahl**：若 I/O 只占原服务时间的 10%，即使把它消掉，整体加速也最多约 `1/0.9=1.11` 倍，且这里尚未考虑新开销。
- **排队**：资源利用率接近饱和时，少量额外工作可能显著扩大尾延迟。只测单请求的组件延迟，不足以预测高并发服务。

用这些模型筛选方向，再用实测修正模型；不要用模型本身代替成果。

## 7. GPU 执行与异步搬运

### 7.1 Host、Device、DMA 与 Pinned Memory

Host 通常指 CPU 与主机内存，Device 指 GPU 及其显存。H2D/D2H 是相应传输方向。DMA 允许设备在相对少的 CPU 数据搬运参与下访问注册/固定的缓冲。

普通 pageable host memory 可能需要中间 staging；pinned memory 的物理页固定，更适于 DMA。但 pin/unpin 有成本，占太多会损害整个主机的内存管理，因此通常建立有限且复用的缓冲池。

### 7.2 Stream 与 Event

同一 stream 内操作有序；不同 stream 可以存在并行机会。Event 用来表达“某次拷贝已经完成，计算可以使用这块数据”等依赖。

`non_blocking=True` 主要说明主机调用不必等待某些完成；不自动保证 GPU copy 与 compute 已重叠。还需要合适内存、硬件资源、独立执行队列与正确依赖。D2H 完成前在 CPU 读目标，或 H2D 完成前修改源缓冲，都会产生错误。[PyTorch pinned memory 与 non_blocking 指南](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)

### 7.3 双缓冲与预取

缓冲 A 正被 GPU 消费时，把后续层或块加载到缓冲 B；交换角色后继续。对一个阶段，若本次可用计算窗口是 `C`，下次加载时间为 `D`，理想暴露等待约 `max(0,D-C)`，第一块和尾部仍有代价。

双缓冲必须回答：数据何时可写、何时可读、何时可回收？至少区分 `FREE → LOADING → READY → IN_USE → FREE`；写回路径还要有脏/保存中状态。不能一边写回同一个缓冲，一边覆盖它。

预取过早会占内存、淘汰热点；过晚 GPU 等待；预取错请求会浪费带宽。固定深度不是在所有并发下都合适。

### 7.4 怎样看是否真的重叠

用 timeline 同时观察 I/O、H2D/D2H 与 kernel，并记录依赖等待和 GPU idle。CUDA event 可测 GPU 队列上的阶段，CPU wall clock 测完整请求；不要把异步提交耗时当完成耗时，也不要在每个操作后全局 synchronize，把原本的重叠人为清空。

**自测**：能画出两块缓冲在三个连续层中的所有权，解释哪一个 event 允许它们复用。

## 8. 操作系统与 SSD I/O

### 8.1 普通文件读取经过什么

简化路径：应用调用 → 虚拟文件系统/具体文件系统 → 页缓存命中或缺页 → 块层/驱动 → NVMe SSD → 主存缓冲。实际路径随 OS、文件系统和 API 变化。

同一个文件第二遍读得很快，可能因为数据已在 DRAM page cache 中；这不是 SSD 物理带宽突然变大。应用自己的 DRAM cache 与内核 page cache 又是两个缓存，必须分别说明。

### 8.2 Buffered、O_DIRECT、io_uring、SPDK

| 路径/接口 | 主要改变 | 仍需考虑 |
|---|---|---|
| Buffered I/O | 利用内核 page cache、常见文件 API | 页缓存状态、拷贝、写回语义 |
| O_DIRECT | 通常绕开文件数据 page cache | 对齐限制、文件系统与内核路径、并非自动更快 |
| io_uring | 提交与完成队列式异步接口，可减少部分提交开销 | 可配 buffered/direct；不是自动内核绕过；实际完成和队列深度 |
| SPDK 用户态 NVMe 驱动 | 用户态管理 NVMe 队列与轮询等 | 独立设备权限、hugepages、CPU 核占用、工程成本 |
| GPU Direct Storage 类路径 | 在支持环境中改善存储到 GPU 的数据路径 | 硬件/驱动/文件系统支持及实际走到哪条路径 |

O_DIRECT 的 buffer 地址、长度和偏移可能有对齐要求，随文件系统、内核和设备变化；不要硬编码“任何机器都必须是 4K”。直接 I/O 也不等于数据已持久化，缓存可恢复性与 crash durability 是不同要求。[open(2) 的 O_DIRECT 说明](https://man7.org/linux/man-pages/man2/open.2.html)

SPDK 可以很有价值，但需要先证明当前开销在适合它优化的位置。它的轮询会使用 CPU，吞吐提高的同时应报告 CPU 成本。[SPDK NVMe 文档](https://spdk.io/doc/nvme.html)

### 8.3 块大小、IOPS 与队列深度

每个 I/O 如果固定开销很大，小块操作可能 IOPS 受限；块更大通常更容易获得带宽，但可能多读无用数据、占用更长的服务时间。队列深度增加能隐藏单次设备延迟，过高则可能伤害 p99。

考虑：按 token block 管理 KV，SSD 却按较大 extent 批量读取。应明确逻辑块到文件偏移的映射，区分请求字节、实际读字节和被使用字节，计算读放大与预取浪费。

### 8.4 单机拓扑与 Linux 工程基础

需要会用 SSH、tmux、虚拟环境/容器、Git、日志与基本进程管理。拿到 Linux 服务器后先认识 `nvidia-smi`、`lscpu`、`numactl --hardware`、`lsblk`、`findmnt`；工具是否可用取决于环境。

NUMA 节点代表 CPU/内存局部性。若 GPU 靠近一个 socket，缓冲却在另一个 socket 的内存上，可能发生额外跨 socket 访问。容器中的文件目录也可能实际来自网络存储，必须检查挂载来源。

在共享机器上先测普通授权目录中的测试文件。不能为了试 SPDK 随意接管现有磁盘，也不应为了制造 cold cache 影响其他人的全部页缓存。

**自测**：解释“第二遍 read 快”“磁盘 utilization 高但带宽低”“CPU 满而 SSD 未满”各可能意味着什么，以及需要什么证据区分。

## 9. 缓存策略与调度

### 9.1 工作集、局部性、命中率

工作集是当前及近未来会使用的数据集合。时间局部性是刚访问的数据更可能再被访问，空间局部性是邻近数据可能一起使用。

LRU 基于最近使用时间，简单但可能被一次性扫描污染；LFU 基于频率，可能对热点变化反应慢。多轮会话的空闲时间、公共系统前缀的频率、调度器的下一批信息都是更有针对性的信号。

命中率至少区分请求命中、token 命中、byte 命中与省下的计算量。命中一个昂贵长前缀和命中一个很短片段，价值不同。

### 9.2 Placement、Admission、Eviction、Prefetch

- **Placement**：数据放哪一级。
- **Admission**：是否值得进入缓存；可只保存可能复用的 KV。
- **Eviction**：空间不足时移走谁；从 HBM 移到 DRAM 不等于从系统彻底丢弃。
- **Prefetch**：需求到来之前，提前把哪份数据提升到较快层。

请求到达、块释放、写回完成、队列切换都会触发策略。策略的计算、锁争用和索引查询本身也占时间，不能在 Python 中加入很复杂的“智能决策”却只报告 I/O 减少。

### 9.3 状态与一致性

对象至少需要明确身份、位置、大小、完成状态、引用和生命期。多并发下同一 key 可能重复计算、重复保存、在取回时遭遇淘汰。即使只做单机，这些都是正确性问题。

数据面搬 payload；控制面管理位置、目录和调度。如果 master 只提供对象位置，不应把所有字节都算作经 master 中转。理解控制面/数据面的区别，有助于读 Mooncake 代码。

## 10. 多机扩展所需的前置知识

### 10.1 PD 分离为什么需要搬 KV

Prefill 与 decode 使用相同模型语义，但执行形态不同。如果分别放在不同 GPU/节点运行，prefill 产生的 KV 必须交给 decode，模型分片布局也必须兼容。分离有机会独立扩容和减少干扰，同时增加数据转移与调度成本。[DistServe](https://arxiv.org/abs/2401.09670)

### 10.2 TCP、RDMA 与注册内存

TCP 普遍可用，适合先验证功能。RDMA 在兼容的 NIC、驱动、网络和注册内存环境中，让设备直接访问指定内存区域，减少部分 CPU 参与和拷贝开销。RDMA 不是“没有延迟”，也不是“任意 Python 张量都能直接传”。

GPUDirect RDMA 面向 NIC 与 GPU 内存的数据路径，GPU Direct Storage 面向存储路径，两者不能混为一谈。是否真正使用了这些路径需要日志、配置和实测证据。

### 10.3 多机实验新增了哪些变量

节点带宽与往返延迟、NIC/GPU/NUMA 拓扑、链路拥塞、模型副本与分片、远端对象生命周期、失败与重试、身份与端口配置。两台 GPU 相同并不意味着网络足够好，也不意味着独立节点的模型状态相容。

现在先学“远端 buffer 的注册—发现—读写—完成—注销”这条基本链路。多 NIC 聚合、复杂容错、跨 TP reshaping 都可后置。

## 11. 测量、统计与科学表达

### 11.1 基本指标

对请求到达 `t_a`、首 token `t_1`、最后 token `t_N`：

\[
TTFT=t_1-t_a,\quad E2E=t_N-t_a,\quad TPOT=\frac{t_N-t_1}{N-1}\quad(N>1).
\]

ITL 是相邻 token 的时间差；TPOT 是该请求 decode 阶段的平均值，不能用它代表所有瞬时停顿。`N=1` 的 TPOT 不定义，应记录为空。

吞吐需指明 input tokens/s、output tokens/s 或 requests/s，以及测量窗口。goodput 在本项目中定义为测量窗口内成功且满足预先声明 SLO 的请求数除以窗口时长；与其它论文对照时核对其具体定义。

### 11.2 最容易出现的错误

客户端发送时间包括网络和排队；引擎内部 prefill 时间不包括所有这些。GPU asynchronous timing 只测 enqueue，会低估真实完成。只统计成功请求、丢掉 OOM/超时，会把坏系统画成好系统。

闭环并发测试在服务变慢时自动降低请求到达率；开环按计划持续发请求，更容易观察排队。二者回答不同问题，报告要明确，不能交叉比较。

### 11.3 统计不确定性

先进行若干独立重复并保存每次记录，再报告分布和重复间差异；少量样本的 p99 很不稳定，不要画出看似精确的尾延迟。用置信区间或 bootstrap 时写清抽样单位，不能把同一请求的强相关 token 当成独立实验重复。

先定义要回答的问题和停止条件，再看结果。出现负面结果时检查正确性与控制变量；如果仍成立，它也可以是很有价值的适用边界。

## 12. 知识依赖索引

| 主题 | 核心概念 | 相邻进阶内容 |
|---|---|---|
| 模型与缓存 | attention/KV 依赖；容量公式；prefill/decode；复用与活跃卸载区别 | CUDA attention kernel、Transformer 训练 |
| 引擎与内存 | 引擎执行路径；块与生命周期；正确性；H2D/D2H 测量 | 分布式调度 |
| 存储与重叠 | page cache/direct I/O；有效带宽；预取与依赖；端到端测量 | SPDK、存储侧计算 |
| 跨设备传输 | 网络传输链路；PD 成本；真实跨机对照 | 多 NIC、弹性容错、异构分片 |

学习不是“先把计算机系统课程都补完”。每学一个知识点，就用一张数据路径图、一段小实验或一个可解释的测量来固定它。
