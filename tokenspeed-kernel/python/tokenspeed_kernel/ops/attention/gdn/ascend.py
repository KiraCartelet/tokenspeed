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

"""Ascend GDN registrations."""

import torch
from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.gdn import gdn_decode, gdn_prefill

    @register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="ascend_gdn_chunk_prefill",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.PORTABLE,
        traits={
            "head_dim": frozenset({128}),
            "value_head_dim": frozenset({128}),
            "output_h": frozenset({False}),
            "qk_l2norm": frozenset({False, True}),
        },
    )
    def ascend_gdn_chunk_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        scale: float | None,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        qk_l2norm: bool,
        output_final_state: bool,
        output_h: bool,
    ) -> GdnChunkPrefillResult:
        """Apply packed GDN prefill with the shared K-last state contract.

        Arguments and return follow gdn_chunk_prefill; intermediate checkpoints
        (output_h=True) are unsupported. The runtime owns checkpoint splitting.
        """
        if output_h:
            raise NotImplementedError(
                "Ascend prefill does not return internal checkpoints"
            )
        out, final = gdn_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=qk_l2norm,
            output_final_state=output_final_state,
        )
        return GdnChunkPrefillResult(out=out, final_state=final)

    @register_kernel(
        "attention",
        "gdn_decode_step",
        name="ascend_gdn_decode_step",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.PERFORMANT,
        traits={"head_dim": frozenset({128})},
    )
    def ascend_gdn_decode_step(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        output_state_indices: torch.Tensor | None,
        scale: float | None,
        use_qk_l2norm: bool,
    ) -> torch.Tensor:
        """Run native Ascend decode with gdn_decode_step's indexed state contract."""
        return gdn_decode(
            q,
            k,
            v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
        )
