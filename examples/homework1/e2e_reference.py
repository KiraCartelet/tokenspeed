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

"""Fixed-input TokenSpeed NPU vs independent Transformers CPU FP32 checks.

Records raw token IDs. Exact greedy agreement is reported, not assumed to be
required for near-tied logits. Teacher-forced scores explain disagreements.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


def generate(ids, count):
    body = {
        "input_ids": ids,
        "sampling_params": {
            "max_new_tokens": count,
            "temperature": 0,
            "ignore_eos": True,
        },
    }
    req = urllib.request.Request(
        "http://127.0.0.1:26427/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seed = tokenizer.encode(
        "The capital of France is Paris. Explain how matrix multiplication works. ",
        add_special_tokens=False,
    )
    cases = [
        (
            "natural",
            tokenizer.encode("The capital of France is", add_special_tokens=False),
        )
    ]
    for length in (1, 128, 2048):
        cases.append(
            (str(length), (seed * ((length + len(seed) - 1) // len(seed)))[:length])
        )
    print("Loading independent Transformers CPU FP32 reference", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    with Path(args.output).open("x") as output, torch.inference_mode():
        for label, ids in cases:
            start = time.perf_counter()
            actual = generate(ids, 16)
            actual_ids = actual["output_ids"]
            x = torch.tensor([ids], dtype=torch.long)
            ref = model.generate(
                x,
                max_new_tokens=16,
                min_new_tokens=16,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            reference_ids = ref[0, len(ids) :].tolist()
            # Score the same NPU continuation against the independent reference.
            # Keep only its 16 prediction positions to bound vocabulary memory.
            full = torch.tensor([ids + actual_ids[:-1]], dtype=torch.long)
            logits = (
                model(full, use_cache=False, logits_to_keep=len(actual_ids))
                .logits[0]
                .float()
            )
            top = logits.topk(2, dim=-1)
            selected = logits.gather(1, torch.tensor(actual_ids)[:, None]).squeeze(1)
            gaps = top.values[:, 0] - selected
            record = {
                "case": label,
                "input_ids": ids,
                "npu": actual,
                "reference_ids": reference_ids,
                "reference_text": tokenizer.decode(reference_ids),
                "exact_greedy_match": actual_ids == reference_ids,
                "teacher_forced_argmax": top.indices[:, 0].tolist(),
                "reference_top1_top2_margin": (
                    top.values[:, 0] - top.values[:, 1]
                ).tolist(),
                "npu_token_reference_logit_gap": gaps.tolist(),
                "reference_dtype": "float32",
                "reference_device": "cpu",
                "seconds": time.perf_counter() - start,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                label,
                "exact",
                record["exact_greedy_match"],
                "max_gap",
                gaps.max().item(),
                flush=True,
            )
        # A-B-A exercises release and reuse after intervening distinct requests.
        a = cases[1][1]
        first = generate(a, 16)
        generate(cases[2][1], 16)
        repeat = generate(a, 16)
        record = {
            "case": "request_reuse_A_B_A",
            "first": first,
            "repeat": repeat,
            "equal": first["output_ids"] == repeat["output_ids"],
        }
        output.write(json.dumps(record) + "\n")
        print("A-B-A", record["equal"], flush=True)


if __name__ == "__main__":
    main()
