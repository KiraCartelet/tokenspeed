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

"""Ascend GDN prefill and native recurrent decode adapters."""

import torch
import torch_npu

from tokenspeed_kernel_npu._triton import tl, triton


def gdn_prefill(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor, *, scale: float | None,
    initial_state: torch.Tensor, cu_seqlens: torch.Tensor,
    qk_l2norm: bool, output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scan packed sequences and return outputs and optional final K-last states.

    Q/K are [1, tokens, heads, key_dim]; V is [1, tokens, value_heads,
    value_dim]. G contains log decay, beta the update gate. Initial states are
    [sequences, value_heads, value_dim, key_dim]. Device cu_seqlens partitions
    tokens; empty sequences preserve their initial state. Scale=None means
    key_dim**-0.5. The initial states are never modified.
    """
    _, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    sequences = cu_seqlens.numel() - 1
    if q.shape[0] != 1 or k.shape != q.shape or v.shape[:2] != (1, tokens):
        raise ValueError("GDN prefill requires packed Q/K/V with leading dimension 1")
    if value_heads % heads:
        raise ValueError("Value heads must be divisible by query heads")
    if initial_state.shape != (sequences, value_heads, value_dim, key_dim):
        raise ValueError("Invalid K-last initial state shape")
    if g.shape != (1, tokens, value_heads) or beta.shape != g.shape:
        raise ValueError("Invalid GDN gate shapes")
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    state = initial_state.float().clone()
    q_work, k_work = q[0].float(), k[0].float()
    if qk_l2norm:
        q_work = q_work * torch.rsqrt(q_work.square().sum(-1, keepdim=True) + 1e-6)
        k_work = k_work * torch.rsqrt(k_work.square().sum(-1, keepdim=True) + 1e-6)
    q_work = q_work.repeat_interleave(value_heads // heads, dim=1)
    k_work = k_work.repeat_interleave(value_heads // heads, dim=1)
    q_work = q_work * (key_dim**-0.5 if scale is None else scale)
    values = v[0].float()
    decay = g[0].float().exp()
    gates = beta[0].float()
    # Right insertion skips empty sequences, including repeated leading bounds.
    # All indexing stays on device; no sequence lengths are copied to the host.
    sequence_ids = torch.searchsorted(
        cu_seqlens[1:].contiguous(),
        torch.arange(tokens, device=q.device, dtype=cu_seqlens.dtype),
        right=True,
    )
    for token in range(tokens):
        index = sequence_ids[token : token + 1]
        current = state.index_select(0, index)[0] * decay[token, :, None, None]
        key = k_work[token]
        delta = (values[token] - (current * key[:, None, :]).sum(-1))
        delta = delta * gates[token, :, None]
        current = current + delta[:, :, None] * key[:, None, :]
        out[0, token] = (current * q_work[token, :, None, :]).sum(-1)
        state.index_copy_(0, index, current.unsqueeze(0))
    final = state.to(initial_state.dtype) if output_final_state else None
    return out, final


@triton.jit(do_not_specialize=["s0", "s1", "s2", "s3"])
def _copy_decode_state(Pool, Scratch, Reads, Writes, s0, s1, s2, s3,
                       H: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
                       GATHER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    read = tl.load(Reads + row).to(tl.int64)
    page = read if GATHER else tl.load(Writes + row).to(tl.int64)
    pool_offsets = (page * s0 + offsets // (V * K) * s1
                    + offsets // K % V * s2 + offsets % K * s3)
    scratch_offsets = row * (H * V * K) + offsets
    valid = (offsets < H * V * K) & (read >= 0) & (page >= 0)
    if GATHER:
        values = tl.load(Pool + pool_offsets, valid, other=0)
        tl.store(Scratch + scratch_offsets, values, offsets < H * V * K)
    else:
        values = tl.load(Scratch + scratch_offsets, offsets < H * V * K, other=0)
        tl.store(Pool + pool_offsets, values, valid)


def gdn_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
    A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor,
    initial_state: torch.Tensor, initial_state_indices: torch.Tensor,
    output_state_indices: torch.Tensor | None, scale: float | None,
    use_qk_l2norm: bool,
) -> torch.Tensor:
    """Run native single-token decode and update the selected K-last pages.

    Q/K/V are BF16 [batch, 1, heads, width]; a/b are raw gate logits.
    State is [pages, value_heads, value_width, key_width], gathered into
    BF16 scratch for CANN and cast back to the pool dtype on writeback.
    Negative read rows produce zero output and never write; negative write
    rows suppress writeback. None writes to the read rows. Scale=None uses
    key_width**-0.5. Returns [batch, 1, value_heads, value_width] BF16 output.
    """
    batch, steps, heads, width = q.shape
    value_heads, value_width = v.shape[-2:]
    if steps != 1 or k.shape != q.shape or v.shape[:2] != (batch, 1):
        raise ValueError("Ascend GDN decode requires single-token Q/K/V")
    if value_heads % heads or max(heads, value_heads, width, value_width) > 256:
        raise ValueError("Unsupported Ascend GDN head geometry")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Native Ascend GDN decode requires BF16 Q/K/V")
    if initial_state.shape[1:] != (value_heads, value_width, width):
        raise ValueError("Invalid K-last state pool shape")
    if batch == 0:
        return torch.empty_like(v)
    writes = initial_state_indices if output_state_indices is None else output_state_indices
    scratch = torch.empty((batch, value_heads, value_width, width),
                          dtype=torch.bfloat16, device=q.device)
    grid = (batch, triton.cdiv(value_heads * value_width * width, 1024))
    _copy_decode_state[grid](initial_state, scratch, initial_state_indices, writes,
                             *initial_state.stride(), value_heads, value_width, width,
                             True, 1024)
    query, key = q[:, 0].float(), k[:, 0].float()
    if use_qk_l2norm:
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    g = -A_log.float().exp() * torch.nn.functional.softplus(a[:, 0].float() + dt_bias.float())
    beta = b[:, 0].float().sigmoid().bfloat16()
    out = torch_npu.npu_recurrent_gated_delta_rule(
        query.bfloat16(), key.bfloat16(), v[:, 0].contiguous(), scratch,
        beta=beta, scale=width**-0.5 if scale is None else scale,
        actual_seq_lengths=torch.ones(batch, dtype=torch.int32, device=q.device),
        ssm_state_indices=torch.arange(batch, dtype=torch.int32, device=q.device),
        num_accepted_tokens=None, g=g.contiguous(), gk=None,
    )
    _copy_decode_state[grid](initial_state, scratch, initial_state_indices, writes,
                             *initial_state.stride(), value_heads, value_width, width,
                             False, 1024)
    return out.masked_fill((initial_state_indices < 0)[:, None, None], 0).unsqueeze(1)
