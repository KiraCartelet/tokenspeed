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

"""Reproducible TP comparison; each invocation owns one Engine."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

if os.environ.get("HW2_DIAGNOSTICS"):
    from diagnostics import install

    install()


def measure(engine, prompts, count):
    start = time.perf_counter()
    rows = [{"output_ids": [], "token_times_s": []} for _ in prompts]
    for event in engine.generate(
        input_ids=prompts,
        sampling_params={
            "max_new_tokens": count,
            "temperature": 0,
            "ignore_eos": True,
        },
        stream=True,
    ):
        now = time.perf_counter() - start
        index = event.get("index", 0)
        row = rows[index]
        ids = event.get("output_ids", [])
        row["output_ids"].extend(ids)
        row["token_times_s"].extend([now] * len(ids))
        row["meta_info"] = event["meta_info"]
    elapsed = time.perf_counter() - start
    for prompt, row in zip(prompts, rows):
        assert len(row["output_ids"]) == count, row
        assert row["meta_info"]["prompt_tokens"] == len(prompt), row
        assert row["meta_info"].get("cached_tokens", 0) == 0, row
        times = row["token_times_s"]
        row["ttft_ms"] = times[0] * 1000
        row["tpot_ms"] = (times[-1] - times[0]) * 1000 / (count - 1)
        row["latency_ms"] = times[-1] * 1000
    return {
        "requests": rows,
        "wall_s": elapsed,
        "output_tokens_per_s": len(prompts) * count / elapsed,
    }


def main():
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", required=True, type=int, choices=[1, 2])
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", required=True, type=int, nargs="+")
    parser.add_argument("--concurrencies", required=True, type=int, nargs="+")
    parser.add_argument("--new-tokens", required=True, type=int)
    parser.add_argument("--repeats", required=True, type=int)
    args = parser.parse_args()
    if args.new_tokens < 2 or args.repeats < 1:
        parser.error("new-tokens >= 2 and repeats >= 1 required")
    if min(args.lengths + args.concurrencies) < 1 or max(args.concurrencies) > 4:
        parser.error("positive lengths and concurrency 1..4 required")
    if max(args.lengths) + args.new_tokens > 4096:
        parser.error("context exceeds fixed 4096-token capacity")
    output = Path(args.output).open("x")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seeds = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in (
            "请解释计算机如何执行矩阵乘法，并给出一个简单例子。",
            "描述张量并行通信的作用。",
            "解释缓存和内存的区别。",
            "说明向量加法。",
        )
    ]
    config = dict(
        model=args.model,
        device="npu",
        dtype="bfloat16",
        world_size=args.tp,
        nprocs_per_node=args.tp,
        attn_tp_size=args.tp,
        dense_tp_size=args.tp,
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

    def emit(record):
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()

    emit(
        {
            "kind": "config",
            "engine": config,
            "args": vars(args),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "config_sha256": hashlib.sha256(
                (Path(args.model) / "config.json").read_bytes()
            ).hexdigest(),
            "hbm_scope": "whole-chip snapshots including other users; not peak",
        }
    )
    engine = Engine(**config)
    try:
        for length in args.lengths:
            for concurrency in args.concurrencies:
                prompts = [
                    (seed * ((length + len(seed) - 1) // len(seed)))[:length]
                    for seed in seeds[:concurrency]
                ]
                for iteration in range(args.repeats + 1):
                    before = subprocess.check_output(["npu-smi", "info"], text=True)
                    if os.environ.get("HW2_DIAGNOSTICS") and iteration == 1:
                        (
                            Path(os.environ["HW2_DIAGNOSTICS"]) / "capture-enabled"
                        ).touch()
                    result = measure(engine, prompts, args.new_tokens)
                    emit(
                        dict(
                            kind="sample",
                            tp=args.tp,
                            input_ids=prompts,
                            input_tokens=length,
                            concurrency=concurrency,
                            iteration=iteration,
                            warmup=iteration == 0,
                            npu_smi_before=before,
                            npu_smi_after=subprocess.check_output(
                                ["npu-smi", "info"], text=True
                            ),
                            **result
                        )
                    )
                    print(
                        "RESULT",
                        args.tp,
                        length,
                        concurrency,
                        iteration,
                        result["output_tokens_per_s"],
                        flush=True,
                    )
    finally:
        engine.shutdown()
        output.close()


if __name__ == "__main__":
    main()
