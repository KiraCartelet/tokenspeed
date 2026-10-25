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

"""Homework NPU validation against CPU FP32 mathematical references."""

import json
import math
import unittest

import torch
import torch_npu
from tokenspeed_kernel_npu.ops.gdn import gdn_decode, gdn_prefill
from tokenspeed_kernel_npu.ops.mha import mha_decode_with_kvcache, mha_prefill


def check(actual, expected, label, *, atol, rtol):
    actual = actual.detach().cpu().float()
    expected = expected.float()
    diff = actual - expected
    print(
        json.dumps(
            {
                "case": label,
                "max_abs": diff.abs().max().item(),
                "rmse": diff.square().mean().sqrt().item(),
                "atol": atol,
                "rtol": rtol,
            }
        ),
        flush=True,
    )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def reference(q, k, v, g, beta, initial):
    # Independent CPU implementation uses [H,K,V] matrix multiplications.
    q, k, v = q.float(), k.float(), v.float()
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    state = initial.float().transpose(-1, -2).clone()
    outputs = []
    for t in range(q.shape[0]):
        state = state * g[t].float().exp()[:, None, None]
        prediction = torch.bmm(k[t, :, None, :], state).squeeze(1)
        update = beta[t].float()[:, None] * (v[t] - prediction)
        state = state + torch.bmm(k[t, :, :, None], update[:, None, :])
        outputs.append(
            torch.bmm(q[t, :, None, :], state).squeeze(1) / math.sqrt(q.shape[-1])
        )
    return torch.stack(outputs), state.transpose(-1, -2)


class HomeworkValidation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.assertEqual(torch.npu.device_count(), 1)

    def test_gdn_prefill_reference_and_chunking(self):
        for length in (1, 4, 63, 64, 65, 129):
            with self.subTest(length=length):
                q, k, v = [torch.randn(1, length, 16, 128).bfloat16() for _ in range(3)]
                g = -torch.rand(1, length, 16) * 0.1
                beta = torch.rand(1, length, 16)
                initial = torch.randn(1, 16, 128, 128) * 0.01
                expected, final = reference(q[0], k[0], v[0], g[0], beta[0], initial[0])

                def run(start, end, state):
                    return gdn_prefill(
                        q[:, start:end].npu(),
                        k[:, start:end].npu(),
                        v[:, start:end].npu(),
                        g[:, start:end].npu(),
                        beta[:, start:end].npu(),
                        scale=None,
                        initial_state=state,
                        cu_seqlens=torch.tensor(
                            [0, end - start], dtype=torch.int32, device="npu"
                        ),
                        qk_l2norm=True,
                        output_final_state=True,
                    )

                pool = initial.npu()
                output, state = run(0, length, pool)
                check(
                    output[0], expected, f"gdn_prefill_{length}", atol=0.002, rtol=0.01
                )
                check(state[0], final, f"gdn_state_{length}", atol=0.0001, rtol=0.001)
                torch.testing.assert_close(pool.cpu(), initial, atol=0, rtol=0)
                split = length // 2
                left, saved = run(0, split, pool)
                right, saved = run(split, length, saved)
                check(
                    torch.cat([left, right], dim=1),
                    output.cpu(),
                    f"chunking_{length}",
                    atol=0.002,
                    rtol=0.01,
                )
                check(
                    saved, state.cpu(), f"chunk_state_{length}", atol=0.0001, rtol=0.001
                )

    def test_gdn_decode_multistep_and_slots(self):
        pool = (torch.randn(4, 16, 128, 128) * 0.01).npu()
        untouched = pool.cpu().clone()
        state = untouched[1].clone()
        A = torch.zeros(16)
        bias = torch.zeros(16)
        for t in range(32):
            q, k, v = [torch.randn(1, 1, 16, 128).bfloat16() for _ in range(3)]
            a = torch.full((1, 1, 16), -3.0)
            b = torch.zeros_like(a)
            g = -torch.nn.functional.softplus(a[0])
            expected, state = reference(q[0], k[0], v[0], g, b[0].sigmoid(), state)
            output = gdn_decode(
                q.npu(),
                k.npu(),
                v.npu(),
                A_log=A.npu(),
                a=a.npu(),
                dt_bias=bias.npu(),
                b=b.npu(),
                initial_state=pool,
                initial_state_indices=torch.tensor(
                    [1], device="npu", dtype=torch.int32
                ),
                output_state_indices=None,
                scale=None,
                use_qk_l2norm=True,
            )
            check(output[0], expected, f"decode_{t}", atol=0.01, rtol=0.03)
        check(pool[1], state, "decode_final_state", atol=0.03, rtol=0.05)
        torch.testing.assert_close(
            pool.cpu()[[0, 2, 3]], untouched[[0, 2, 3]], atol=0, rtol=0
        )
        before = pool.cpu().clone()
        output = gdn_decode(
            q.npu(),
            k.npu(),
            v.npu(),
            A_log=A.npu(),
            a=a.npu(),
            dt_bias=bias.npu(),
            b=b.npu(),
            initial_state=pool,
            initial_state_indices=torch.tensor([-1], device="npu", dtype=torch.int32),
            output_state_indices=None,
            scale=None,
            use_qk_l2norm=True,
        )
        self.assertEqual(output.abs().sum().item(), 0)
        torch.testing.assert_close(pool.cpu(), before, atol=0, rtol=0)

    def test_mha_256_varlen(self):
        for lengths in ([1], [3, 5], [127, 129]):
            with self.subTest(lengths=lengths):
                total = sum(lengths)
                q = torch.randn(total, 8, 256).bfloat16()
                k, v = [torch.randn(total, 2, 256).bfloat16() for _ in range(2)]
                bounds = [0]
                expected = []
                for n in lengths:
                    start = bounds[-1]
                    qq = q[start : start + n].float().transpose(0, 1)
                    kk = (
                        k[start : start + n]
                        .float()
                        .repeat_interleave(4, 1)
                        .transpose(0, 1)
                    )
                    vv = (
                        v[start : start + n]
                        .float()
                        .repeat_interleave(4, 1)
                        .transpose(0, 1)
                    )
                    expected.append(
                        torch.nn.functional.scaled_dot_product_attention(
                            qq, kk, vv, is_causal=True
                        ).transpose(0, 1)
                    )
                    bounds.append(start + n)
                out = mha_prefill(
                    q.npu(),
                    k.npu(),
                    v.npu(),
                    torch.tensor(bounds, dtype=torch.int32, device="npu"),
                    bounds,
                    max(lengths),
                )
                check(
                    out, torch.cat(expected), f"mha256_{lengths}", atol=0.02, rtol=0.02
                )

    def test_mha_256_paged_decode(self):
        q = torch.randn(2, 8, 256).bfloat16()
        k, v = [torch.randn(4, 128, 2, 256).bfloat16() for _ in range(2)]
        table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        lengths = torch.tensor([127, 129], dtype=torch.int32)
        expected = []
        for row, length in enumerate(lengths.tolist()):
            kk = (
                k[table[row].long()]
                .reshape(-1, 2, 256)[:length]
                .float()
                .repeat_interleave(4, 1)
                .transpose(0, 1)
            )
            vv = (
                v[table[row].long()]
                .reshape(-1, 2, 256)[:length]
                .float()
                .repeat_interleave(4, 1)
                .transpose(0, 1)
            )
            expected.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[row : row + 1].float().transpose(0, 1), kk, vv
                ).transpose(0, 1)
            )
        out = mha_decode_with_kvcache(
            q.npu(), k.npu(), v.npu(), table.npu(), lengths.npu(), max_seqlen_k=256
        )
        check(out, torch.cat(expected), "mha256_paged_decode", atol=0.02, rtol=0.02)

    def test_gemma_norm_leaf(self):
        from tokenspeed_kernel_npu.ops.layernorm import rmsnorm

        for rows in (1, 17, 128):
            x = torch.randn(rows, 1024).bfloat16()
            weight = (torch.randn(1024) * 0.1).bfloat16()
            effective = (weight + 1).bfloat16()
            expected = (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                * effective.float()
            )
            out = rmsnorm(x.npu(), effective.npu(), 1e-6)
            check(out, expected, f"gemma_leaf_{rows}", atol=0.03, rtol=0.02)
            residual = torch.randn_like(x)
            summed = x.float() + residual.float()
            expected = (
                summed
                * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
                * effective.float()
            )
            out, res = rmsnorm(x.npu(), effective.npu(), 1e-6, residual=residual.npu())
            check(out, expected, f"gemma_residual_leaf_{rows}", atol=0.03, rtol=0.02)
            torch.testing.assert_close(res.cpu(), summed.bfloat16(), atol=0, rtol=0)

    def test_gdn_packed_request_isolation(self):
        lengths = [0, 3, 0, 5]
        bounds = torch.tensor([0, 0, 3, 3, 8], dtype=torch.int32, device="npu")
        q, k, v = [torch.randn(1, 8, 16, 128).bfloat16().npu() for _ in range(3)]
        g = (-torch.rand(1, 8, 16) * 0.1).npu()
        beta = torch.rand(1, 8, 16).npu()
        initial = (torch.randn(4, 16, 128, 128) * 0.01).npu()
        out, final = gdn_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=initial,
            cu_seqlens=bounds,
            qk_l2norm=True,
            output_final_state=True,
        )
        start = 0
        for row, n in enumerate(lengths):
            if not n:
                torch.testing.assert_close(final[row], initial[row], atol=0, rtol=0)
                continue
            expected, state = reference(
                q[0, start : start + n].cpu(),
                k[0, start : start + n].cpu(),
                v[0, start : start + n].cpu(),
                g[0, start : start + n].cpu(),
                beta[0, start : start + n].cpu(),
                initial[row].cpu(),
            )
            check(
                out[0, start : start + n],
                expected,
                f"packed_request_{row}",
                atol=0.002,
                rtol=0.01,
            )
            check(final[row], state, f"packed_state_{row}", atol=0.0001, rtol=0.001)
            start += n

    def test_gdn_decode_distinct_output_slot(self):
        q, k, v = [torch.randn(1, 1, 16, 128).bfloat16().npu() for _ in range(3)]
        pool = (torch.randn(4, 16, 128, 128) * 0.01).npu()
        before = pool.cpu().clone()
        a = torch.full((1, 1, 16), -3.0, device="npu")
        b = torch.zeros_like(a)
        expected, state = reference(
            q[0].cpu(),
            k[0].cpu(),
            v[0].cpu(),
            -torch.nn.functional.softplus(a[0].cpu()),
            b[0].cpu().sigmoid(),
            before[1],
        )
        out = gdn_decode(
            q,
            k,
            v,
            A_log=torch.zeros(16, device="npu"),
            a=a,
            dt_bias=torch.zeros(16, device="npu"),
            b=b,
            initial_state=pool,
            initial_state_indices=torch.tensor([1], dtype=torch.int32, device="npu"),
            output_state_indices=torch.tensor([2], dtype=torch.int32, device="npu"),
            scale=None,
            use_qk_l2norm=True,
        )
        check(out[0], expected, "distinct_slot_output", atol=0.001, rtol=0.01)
        check(pool[2], state, "distinct_slot_state", atol=0.005, rtol=0.02)
        torch.testing.assert_close(
            pool.cpu()[[0, 1, 3]], before[[0, 1, 3]], atol=0, rtol=0
        )

    def test_gated_rmsnorm_runtime(self):
        from tokenspeed.runtime.layers.attention.linear.layernorm_gated import (
            rmsnorm_fn,
        )

        for rows in (1, 5, 65):
            x = torch.randn(rows, 2048).bfloat16()
            z = torch.randn(rows, 4096).bfloat16()[:, :2048]
            w = torch.randn(2048).bfloat16()
            grouped = x.float().reshape(rows, 16, 128)
            expected = (
                grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
            ).reshape(rows, 2048)
            expected = expected * w.float() * torch.nn.functional.silu(z.float())
            # Keep a row-strided NPU gate view, as produced by projections.
            z_device = torch.cat([z, z], dim=1).npu()[:, :2048]
            out = rmsnorm_fn(
                x.npu(), w.npu(), z_device, group_size=128, weights_independent=True
            )
            check(out, expected, f"gated_norm_{rows}", atol=0.04, rtol=0.02)

    def test_conv_decode_history(self):
        from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
            causal_conv1d_update,
        )

        channels = 128
        weight = torch.randn(channels, 4).bfloat16()
        history = torch.randn(3, channels, 3).bfloat16()
        pool = history.npu()
        expected_state = history[1].clone()
        for step in range(8):
            x = torch.randn(1, channels).bfloat16()
            window = torch.cat([expected_state, x[0, :, None]], dim=-1)
            expected = torch.nn.functional.silu(
                (window.float() * weight.float()).sum(-1)
            )
            out = causal_conv1d_update(
                x.npu(),
                pool,
                weight.npu(),
                activation="silu",
                conv_state_indices=torch.tensor([1], dtype=torch.int32, device="npu"),
                parent_indices=None,
            )
            check(out[0], expected, f"conv_decode_{step}", atol=0.03, rtol=0.02)
            expected_state = window[:, 1:]
            torch.testing.assert_close(pool[1].cpu(), expected_state, atol=0, rtol=0)
        torch.testing.assert_close(pool.cpu()[[0, 2]], history[[0, 2]], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
