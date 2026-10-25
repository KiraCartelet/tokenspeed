# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Direct Engine measurements; cold prefix cache, one request at a time.

Diagnostic mode wraps the unmodified executor only in this experiment process.
It synchronizes and profiles forwards separately from the uninstrumented baseline.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def install_diagnostics():
    import torch
    import torch_npu
    from tokenspeed_kernel.ops.attention.gdn import ascend

    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    def scoped(fn, name):
        def call(*args, **kwargs):
            with torch.profiler.record_function(name):
                return fn(*args, **kwargs)

        return call

    ascend.gdn_prefill = scoped(ascend.gdn_prefill, "HW1_GDN_PREFILL")
    ascend.gdn_decode = scoped(ascend.gdn_decode, "HW1_GDN_DECODE")
    root = Path(os.environ["HW1_DIAGNOSTICS"])
    original = ModelExecutor.execute_forward_op
    captured = [0]

    def measured(self, op, *args, **kwargs):
        if not (root / "capture-enabled").exists() or captured[0] >= 3:
            return original(self, op, *args, **kwargs)
        index = captured[0]
        captured[0] += 1
        phase = "prefill" if op.num_extends() else "decode"
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        before = torch.npu.memory_allocated()
        begin = time.perf_counter()
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(root / f"{index}-{phase}")
            ),
        ) as profiler:
            result = original(self, op, *args, **kwargs)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - begin
            profiler.step()
        data = {
            "phase": phase,
            "tokens": sum(op.input_lengths),
            "synchronized_profiled_wall_ms": elapsed * 1000,
            "allocated_before_bytes": before,
            "allocated_after_bytes": torch.npu.memory_allocated(),
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        }
        (root / f"{index}-{phase}.json").write_text(json.dumps(data, indent=2))
        return result

    ModelExecutor.execute_forward_op = measured


if os.environ.get("HW1_DIAGNOSTICS"):
    install_diagnostics()


def main():
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["baseline", "profile"], required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    engine = Engine(
        model=args.model,
        device="npu",
        dtype="bfloat16",
        language_model_only=True,
        attention_backend="mha",
        sampling_backend="greedy",
        disable_prefill_graph=True,
        disable_pdl=True,
        enforce_eager=True,
        disable_autotune=True,
        max_model_len=4096,
        max_num_seqs=4,
        max_total_tokens=16384,
        chunked_prefill_size=4096,
        prefix_granularity=128,
        enable_prefix_caching=False,
        disable_kvstore=True,
        seed=42,
        log_level="info",
        enable_log_request_stats=True,
    )
    seed = tokenizer.encode(
        "请解释计算机如何执行矩阵乘法，并给出一个简单例子。", add_special_tokens=False
    )

    def inputs(length):
        return (seed * ((length + len(seed) - 1) // len(seed)))[:length]

    def run(length, count):
        start = time.perf_counter()
        events = []
        for event in engine.generate(
            input_ids=inputs(length),
            sampling_params={
                "max_new_tokens": count,
                "temperature": 0,
                "ignore_eos": True,
            },
            stream=True,
        ):
            events.append({"seconds": time.perf_counter() - start, "event": event})
        nonempty = [x for x in events if x["event"].get("output_ids")]
        ids = [token for x in nonempty for token in x["event"]["output_ids"]]
        meta = events[-1]["event"]["meta_info"]
        assert len(ids) == count, (len(ids), count, events)
        assert meta["prompt_tokens"] == length and meta["completion_tokens"] == count
        assert meta.get("cached_tokens", 0) == 0, meta
        return {
            "input_tokens": length,
            "output_tokens": count,
            "ttft_ms": nonempty[0]["seconds"] * 1000,
            "tpot_ms": (nonempty[-1]["seconds"] - nonempty[0]["seconds"])
            * 1000
            / (count - 1),
            "total_ms": (time.perf_counter() - start) * 1000,
            "output_ids": ids,
            "events": events,
            "timing_scope": "direct Engine streaming wall time; includes host scheduling and IPC",
        }

    try:
        with Path(args.output).open("x") as output:
            if args.mode == "profile":
                run(128, 4)
                root = Path(os.environ["HW1_DIAGNOSTICS"])
                (root / "capture-enabled").touch()
                result = run(128, 4)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                for length in (128, 2048):
                    for count in (32, 128):
                        for iteration in range(4):
                            before = subprocess.check_output(
                                ["npu-smi", "info"], text=True
                            )
                            result = run(length, count)
                            result.update(
                                iteration=iteration,
                                warmup=iteration == 0,
                                npu_smi_before=before,
                                npu_smi_after=subprocess.check_output(
                                    ["npu-smi", "info"], text=True
                                ),
                            )
                            output.write(json.dumps(result, ensure_ascii=False) + "\n")
                            output.flush()
                            print(
                                "RESULT",
                                length,
                                count,
                                iteration,
                                result["ttft_ms"],
                                result["tpot_ms"],
                                flush=True,
                            )
                # Compare packed concurrent requests with isolated references.
                batch_ids = [inputs(1), inputs(17), inputs(129), inputs(65)]
                params = {"max_new_tokens": 16, "temperature": 0, "ignore_eos": True}
                isolated = [
                    engine.generate(input_ids=ids, sampling_params=params)["output_ids"]
                    for ids in batch_ids
                ]
                packed = engine.generate(input_ids=batch_ids, sampling_params=params)
                record = {
                    "case": "concurrent_isolation",
                    "isolated": isolated,
                    "packed": [x["output_ids"] for x in packed],
                }
                record["equal"] = record["isolated"] == record["packed"]
                output.write(json.dumps(record) + "\n")
                print("CONCURRENT_ISOLATION", record["equal"], flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
