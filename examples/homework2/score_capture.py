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

"""Capture actual full-vocabulary logits with forward request IDs on rank zero."""

import json
import os
from pathlib import Path


def install():
    import torch

    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessor

    if os.environ.get("HW2_GDN_REFERENCE") == "1":
        # Diagnostic only: isolate BF16 native decode state rounding.
        from tokenspeed_kernel.ops.attention.gdn import ascend
        from tokenspeed_kernel_npu.ops.gdn import gdn_prefill

        def reference_decode(
            q,
            k,
            v,
            *,
            A_log,
            a,
            dt_bias,
            b,
            initial_state,
            initial_state_indices,
            output_state_indices,
            scale,
            use_qk_l2norm,
        ):
            batch = q.shape[0]
            reads = initial_state_indices.long()
            writes = (
                reads if output_state_indices is None else output_state_indices.long()
            )
            state = initial_state[reads.clamp_min(0)].clone()
            state[reads < 0] = 0
            g = -A_log.float().exp() * torch.nn.functional.softplus(
                a[:, 0].float() + dt_bias.float()
            )
            out, final = gdn_prefill(
                q.transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1),
                g.unsqueeze(0),
                b[:, 0].float().sigmoid().unsqueeze(0),
                scale=scale,
                initial_state=state,
                cu_seqlens=torch.arange(batch + 1, device=q.device, dtype=torch.int32),
                qk_l2norm=use_qk_l2norm,
                output_final_state=True,
            )
            valid = (reads >= 0) & (writes >= 0)
            initial_state.index_copy_(0, writes[valid], final[valid])
            return out.transpose(0, 1).masked_fill((reads < 0)[:, None, None, None], 0)

        ascend.gdn_decode = reference_decode

    root = Path(os.environ["HW2_SCORES"])
    root.mkdir(parents=True, exist_ok=True)
    context = None
    counter = 0
    original_forward = ModelExecutor.execute_forward_op
    original_logits = LogitsProcessor._get_logits

    def forward(self, op, *args, **kwargs):
        nonlocal context
        context = op
        try:
            return original_forward(self, op, *args, **kwargs)
        finally:
            context = None

    def logits(self, *args, **kwargs):
        nonlocal counter
        result = original_logits(self, *args, **kwargs)
        if (
            self.tp_rank == 0
            and context is not None
            and (root / "capture-enabled").exists()
        ):
            assert result.shape[0] == len(context.request_ids)
            cpu = result.detach().float().cpu()
            values, ids = cpu.topk(10, dim=-1)
            data = dict(
                step=counter,
                request_ids=list(context.request_ids),
                input_ids=list(context.input_ids),
                input_lengths=list(context.input_lengths),
                num_extends=context.num_extends(),
                top_ids=ids.tolist(),
                top_logits=values.tolist(),
            )
            torch.save(cpu, root / f"{counter}.pt")
            (root / f"{counter}.json").write_text(json.dumps(data))
            counter += 1
        return result

    ModelExecutor.execute_forward_op = forward
    LogitsProcessor._get_logits = logits
