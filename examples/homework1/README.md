# Qwen3.5-0.8B Ascend validation

本目录保存单张 Ascend 910C 上的作业验证脚本。模型与算子适配来自
[NekoThm/tokenspeed](https://github.com/NekoThm/tokenspeed/commit/75e4e13638c275c1abfdd5ab971b526afae2a9fe)
的原始提交；本次提交增加验证和实验材料，没有改动核心模型或算子实现。
课程上游为 [lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed)。

## 环境与运行

已验证环境：CANN 9.0.0，PyTorch 2.9.0，torch_npu 2.9.0.post6，
Triton 3.5.0 / Triton Ascend 3.2.1，Transformers 5.12.0。
模型为 Qwen/Qwen3.5-0.8B，revision `2fc06364715b967f1860aea9cf38778875588b17`。
先按项目说明安装运行时与 scheduler，加载本机 CANN 环境并激活虚拟环境。
从仓库根目录运行，`MODEL_PATH` 指向已下载的模型，选择可用的单张芯片：

```bash
export MODEL_PATH=/path/to/Qwen3.5-0.8B
export ASCEND_RT_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python"
mkdir -p outputs/homework1
python -m pytest -q tokenspeed-kernel-npu/test/test_hw1_validation.py
python examples/homework1/calibrate_decode.py > outputs/homework1/calibration.jsonl
python examples/homework1/benchmark_engine.py --model "$MODEL_PATH" --mode baseline \
  --output outputs/homework1/benchmark.jsonl
```

输出文件采用独占创建，重新运行时使用新的文件名。基准每组预热一次，测量三次；
另比较四个不同长度的并发请求与独立运行结果。

端到端独立参考使用 Transformers CPU FP32。先在另一个终端启动本地服务：

```bash
python -m tokenspeed.cli serve "$MODEL_PATH" --language-model-only \
  --device npu --dtype bfloat16 --kv-cache-dtype auto --attention-backend mha \
  --sampling-backend greedy --disable-prefill-graph --disable-pdl \
  --max-model-len 4096 --max-num-seqs 4 --max-total-tokens 16384 \
  --chunked-prefill-size 4096 --prefix-granularity 128 --enforce-eager \
  --disable-autotune --host 127.0.0.1 --port 26426
python examples/homework1/e2e_reference.py --model "$MODEL_PATH" \
  --output outputs/homework1/reference.jsonl
```

参考脚本访问内部原生 generate 端口 `127.0.0.1:26427`。性能基准直接调用
Engine，避免 HTTP 网关差异。运行性能实验前先停止参考服务，确保芯片独占。

诊断与性能基准分开运行，使用新建的空目录：

```bash
mkdir -p outputs/homework1/profile
HW1_DIAGNOSTICS="$PWD/outputs/homework1/profile" \
  python examples/homework1/benchmark_engine.py --model "$MODEL_PATH" \
  --mode profile --output outputs/homework1/profile.jsonl
python examples/homework1/summarize_profile.py outputs/homework1/profile --device-id 5
```

## 已记录结果

9 项算子测试通过；自然文本及 1、128、2048 token 输入的 16 token greedy
输出与独立 FP32 参考逐项一致，teacher-forced argmax 一致。
A-B-A 请求复用和四请求并发隔离检查通过。三个随机种子的 32 步递归误差
与独立 BF16 舍入参考处于相同量级。这些结果覆盖上述样例，不代表所有输入。

Eager BF16，关闭 prefix cache 与 KVStore，每组 3 次测量均值：

| 输入 token | 输出 token | TTFT ms | TPOT ms | 总耗时 ms | HBM MB |
| --- | --- | --- | --- | --- | --- |
| 128 | 32 | 548.163 | 55.817 | 2278.536 | 5420 |
| 128 | 128 | 547.024 | 57.667 | 7870.852 | 5420 |
| 2048 | 32 | 9196.883 | 56.389 | 10944.983 | 5566–5568 |
| 2048 | 128 | 9143.767 | 57.188 | 16406.805 | 5569 |

详细统计见 [benchmark-summary.json](benchmark-summary.json)。TTFT/TPOT 是 Engine
流式墙钟时间，包含调度与 IPC；HBM 是驱动快照占用，包含进程外基线，不是峰值。
原始日志包含机器信息，未收入版本库。

独立 profiler 诊断中，128-token prefill 的 GDN 占设备 kernel 时间约 89.8%；
decode 的 36 次状态复制 kernel 占约 68–70%。采样有 profiler 开销，不能替代
上述基准。没有采集实际 HBM 带宽计数器，因此不据此断言带宽饱和。
