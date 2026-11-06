# 作业2：双 chip 张量并行实验

本目录沿用作业1的 BF16、greedy、纯文本 eager 路径。实验在同一模型内部
切分注意力、GDN、MLP 和词表权重，attention/dense TP 均等于 world size，
DP=PP=1。四个模型依次运行，不同时驻留。结果与完成状态见 RESULTS.md。

## 本次适配

- logits 汇总分支仅对 CUDA tensor 查询 CUDA graph 状态；NPU 的分布式
  argmax 状态申请返回 None，回退到完整词表集合通信。
- HCCL 使用 PyTorch 公开的 `all_gather_into_tensor` 和
  `reduce_scatter_tensor` 接口，兼容实验环境的 torch 2.9。
- 模型切分、进程启动及其余算子复用已有 TokenSpeed 实现。

## 典型层的切分与通信

以下具体数值来自 Qwen3.5-0.8B config，T 为本轮打包 token 数，B 为活动请求数。
权重形状均按 `[out_features, in_features]`，r=0/1。
两个 rank 的隐藏输入与残差 `[T,1024]` 复制，每个 rank 只计算自己的头或通道。

| 结构 | 完整权重/状态 | 每个 rank | 通信位置 |
| --- | --- | --- | --- |
| GDN Q/K/V/Z 投影 | 四段各 `[2048,1024]` | 每段 `[1024,1024]`，各8个头 | 列并行投影不通信 |
| GDN b/a 投影 | 两段各 `[16,1024]` | 每段 `[8,1024]` | 不通信 |
| GDN 深度卷积 | `[6144,1,4]` | Q/K/V 各取本地头，合为 `[3072,1,4]` | 不通信；不是直接取拼接权重前/后半 |
| GDN A_log/dt_bias | 各 `[16]` FP32 | 各 `[8]` | 不通信 |
| GDN 递归状态 | 每请求 `[16,128,128]` FP32 | `[8,128,128]`，布局 `[Hv,V,K]` | 各 rank 独立更新本地头 |
| GDN 卷积历史 | 每请求 `[6144,3]` BF16 | `[3072,3]` | 本地保留 |
| GDN 输出投影 | `[1024,2048]` | `[1024,1024]`，得到部分和 `[T,1024]` | post-attention 边界 AllReduce(SUM) |
| MHA Q+gate 投影 | `[4096,1024]` | `[2048,1024]`，4个Q头及对应gate | 不通信 |
| MHA K/V 投影 | 各 `[512,1024]` | 各 `[256,1024]`，1个KV头 | 不通信 |
| MHA Q/K/V 激活 | Q: `[T,8,256]`；K/V: `[T,2,256]` | Q: `[T,4,256]`；K/V: `[T,1,256]` | 注意力在本地头上计算 |
| MHA KV 缓存 | 每页 K/V 各 `[128,2,256]` | 各 `[128,1,256]` | 历史KV按头切分，无逐步全量交换 |
| MHA 输出投影 | `[1024,2048]` | `[1024,1024]`，得到部分和 `[T,1024]` | post-attention 边界 AllReduce(SUM) |
| MLP gate/up | 各 `[3584,1024]` | 各 `[1792,1024]` | 中间激活 `[T,1792]` 本地计算 |
| MLP down | `[1024,3584]` | `[1024,1792]` | AllReduce(SUM) 合并 `[T,1024]` 部分和 |
| 词嵌入/LM head | `[248320,1024]`，权重共享 | `[124160,1024]` | 嵌入局部结果归约；输出 logits 按词表 AllGather |

隐藏 RMSNorm 参数与残差复制；GDN gated RMSNorm 按每头128维作用，
长度128的缩放参数复制，不能对整个分片统一归一化。
输入 token IDs、位置及请求调度元数据复制，KV/SSM/卷积历史按上述方式切分；
slot ID 在 rank 间保持对应，每个 rank 访问自己的状态张量。

`RowParallelLinear(reduce_results=False)` 不代表遗漏归约：GDN/MHA 输出由
`comm_manager` 在层边界归约。相同 attention TP 与 dense TP 使用 AllReduce，
本实验未把 reduce-scatter/all-gather 的其他并行布局混入基线。
0.8B 有24层，每个完整前向通常含24次注意力边界和24次MLP边界归约；
还存在嵌入、词表汇总和采样同步，最终数量以 trace 为准。
BF16 `[T,1024]` 的归约逻辑张量大小是 `2048*T` 字节；
这不是实际链路流量或带宽测量值。词表汇总每 rank 输入 `[R,124160]`，
R 为本轮需要采样的位置数（通常为完成 prefill 或 decode 的请求数）。

## 复现

从服务器私人 checkout 执行。先确认设备拓扑、其他任务和剩余 HBM；
本次 6/7 是同一张卡的两个 chip，SIO 相连。

```bash
# 模型尺寸、TP、可见chip、输出标签、输入长度、并发、生成长度、正式重复次数
bash examples/homework2/run.sh 0.8B 1 6 matrix-08b-tp1 '128 512' '1 4' 32 3
bash examples/homework2/run.sh 0.8B 2 6,7 matrix-08b-tp2 '128 512' '1 4' 32 3
python examples/homework2/summarize.py --compare \
  artifacts/hw2/matrix-08b-tp1.jsonl artifacts/hw2/matrix-08b-tp2.jsonl
```

每组先预热一次，再正式测量；输出文件拒绝覆盖。输入 token IDs 和输出 IDs
完整保存，TTFT 以首个实际 token 为准，TPOT 按首尾 token 时间差计算。
吞吐为整批输出 token 数/整批墙钟时间，不平均请求各自的吞吐。
`npu-smi` 是包含其他任务的整 chip 快照，不是本模型峰值；profiling 的
allocator 峰值按 rank 另存，两个口径不得混用。

```bash
# 在 run.sh 相同环境中执行真实双rank回归测试
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  examples/homework2/test_tp_collectives.py
# 独立诊断，绝不能把 profiler 耗时填入正式基线
mkdir -p artifacts/hw2/profile-new
HW2_DIAGNOSTICS="$PWD/artifacts/hw2/profile-new" \
  bash examples/homework2/run.sh 0.8B 2 6,7 profile-new 128 1 4 1
```

诊断在预热后采集每 rank 的三个前向，记录 CPU/NPU 活动、通信范围和内存。
并发大于1时，前向可能包含多个 prefill 或混合请求，阶段标签由实际 op 决定，
不假设三个前向一定是“一次prefill两次decode”。旧目录的 capture-enabled
不得复用，否则会提前采集。

## 通信实验设计

假设：提高提交并发可让更多请求共享集合通信启动成本，但会增加张量体积、
请求排队和HBM占用；最终收益由实际调度批量和主机提交开销共同决定。
固定模型、TP、输入长度128、精度、缓存及调度器容量，仅改变提交并发1/4，
用正式矩阵比较吞吐/延迟，用独立 profiler 比较通信时长及占比。
并发不等于每个前向的实际批量，以每次 op 的 token 数和调度日志核验。
计算、通信可能重叠；时间区间取并集并报告交集，不把CPU嵌套范围、
设备kernel累计和端到端耗时相加。设备空闲区间只能称为无已记录设备活动，
不能全部解释成通信等待。

## 数值差异诊断

`compare_scores.py` 从已有JSONL读取相同输入和Engine配置，运行两次；
`HW2_SCORES` 指定新的目录，用rank0完整词表截取生成被消费token对应的logits。
`analyze_logits.py` 按真实request ID对齐，只比较共同生成历史。
`--request-index` 可将原并发case中的某一个请求单独提交，其他配置保持不变。
这些运行包含设备同步和磁盘写入，禁止作为性能数据。

```bash
HW2_SCORES="$PWD/artifacts/hw2/logits-new-tp2" \
python examples/homework2/compare_scores.py \
  --source artifacts/hw2/correctness-9b-tp1.jsonl --tp 2 \
  --length 128 --concurrency 4 --request-index 1 \
  --output artifacts/hw2/logits-new-tp2.jsonl
```

可额外显式设置 `HW2_GDN_REFERENCE=1`，仅在上述诊断进程内用FP32状态
逐token参考扫描替换原生GDN decode，以检验误差是否仅来自原生decode。
不修改正常运行时，不与原生decode性能混合。该替换没有消除9B反例；
因此不能据此把全部差异归因于BF16原生state暂存。正式配置中的SSM pool
虽然是FP32，但原生decode会将状态复制到BF16 scratch，再写回FP32 pool。

## 权重下载与复现边界

`download_models.py` 固定官方模型revision；`download_cdn.py` 获取相同revision
的官方文件元数据，通过官方CDN下载大文件并验证SHA256，支持断点续传。
代理仅用于元数据和小文件，大文件传输不经过该代理。所有文件均校验后才
声明下载完成；实验不自动删除模型或其他任务的缓存。
