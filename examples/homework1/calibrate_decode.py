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

"""Measure BF16 recurrent rounding scale independently of NPU implementation."""

import json
import math

import torch
import torch_npu
from tokenspeed_kernel_npu.ops.gdn import gdn_decode


def step(q, k, v, state, rounded):
    q = q.float() / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    if rounded:
        q, k, state = (
            q.bfloat16().float(),
            k.bfloat16().float(),
            state.bfloat16().float(),
        )
    state = state * torch.exp(-torch.nn.functional.softplus(torch.tensor(-3.0)))
    prediction = torch.bmm(k[:, None, :], state).squeeze(1)
    state = state + torch.bmm(
        k[:, :, None], (0.5 * (v.float() - prediction))[:, None, :]
    )
    out = torch.bmm(q[:, None, :], state).squeeze(1) / math.sqrt(128)
    return (
        (out.bfloat16().float(), state.bfloat16().float()) if rounded else (out, state)
    )


def metrics(x, y):
    d = x.float() - y.float()
    return {"max_abs": d.abs().max().item(), "rmse": d.square().mean().sqrt().item()}


def main():
    torch.set_num_threads(2)
    for seed in (42, 123, 2026):
        torch.manual_seed(seed)
        state = torch.randn(16, 128, 128) * 0.01  # independent K,V layout
        rounded = state.clone()
        pool = state.transpose(-1, -2).contiguous()[None].npu()
        outputs, refs, roundrefs = [], [], []
        for _ in range(32):
            q, k, v = [torch.randn(16, 128).bfloat16() for _ in range(3)]
            expected, state = step(q, k, v, state, False)
            rough, rounded = step(q, k, v, rounded, True)
            actual = (
                gdn_decode(
                    q[None, None].npu(),
                    k[None, None].npu(),
                    v[None, None].npu(),
                    A_log=torch.zeros(16, device="npu"),
                    a=torch.full((1, 1, 16), -3.0, device="npu"),
                    dt_bias=torch.zeros(16, device="npu"),
                    b=torch.zeros(1, 1, 16, device="npu"),
                    initial_state=pool,
                    initial_state_indices=torch.tensor(
                        [0], dtype=torch.int32, device="npu"
                    ),
                    output_state_indices=None,
                    scale=None,
                    use_qk_l2norm=True,
                )[0, 0]
                .cpu()
                .float()
            )
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.03)
            outputs.append(actual)
            refs.append(expected)
            roundrefs.append(rough)
        actual_state = pool[0].cpu().transpose(-1, -2).float()
        torch.testing.assert_close(actual_state, state, atol=0.03, rtol=0.05)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "steps": 32,
                    "npu_vs_fp32_output": metrics(
                        torch.stack(outputs), torch.stack(refs)
                    ),
                    "bf16_rounding_vs_fp32_output": metrics(
                        torch.stack(roundrefs), torch.stack(refs)
                    ),
                    "npu_vs_fp32_state": metrics(actual_state, state),
                    "bf16_rounding_vs_fp32_state": metrics(rounded, state),
                    "npu_vs_rounding_state": metrics(actual_state, rounded),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
