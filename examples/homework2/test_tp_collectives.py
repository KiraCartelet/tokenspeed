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

"""Run with torchrun --standalone --nproc-per-node=2 on two visible NPUs."""

import json
import os

import torch
import torch.distributed as dist
import torch_npu
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.comm_backend import set_global_backend
from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.utils.env import global_server_args_dict


def main():
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    group = (0, 1)
    process_group_manager.register_process_group("hccl", group, dist.group.WORLD)
    backend = HcclBackend()
    set_global_backend(backend)
    global_server_args_dict.update(numerics="auto", logprob_order="torch")
    try:
        local = torch.full((2, 3), rank + 1, device="npu", dtype=torch.float32)
        reduced = backend.all_reduce(local.clone(), group)
        torch.testing.assert_close(
            reduced.cpu(), torch.full((2, 3), 3.0), rtol=0, atol=0
        )
        gathered = backend.all_gather(local, group, dim=1)
        expected = torch.cat([torch.ones(2, 3), torch.full((2, 3), 2.0)], dim=1)
        torch.testing.assert_close(gathered.cpu(), expected, rtol=0, atol=0)
        scattered = backend.reduce_scatter(local, group)
        torch.testing.assert_close(
            scattered.cpu(), torch.full((1, 3), 3.0), rtol=0, atol=0
        )
        counts = [1, 2]
        ragged = torch.full(
            (counts[rank], 3), rank + 1, device="npu", dtype=torch.float32
        )
        assembled = backend.token_all_gather(ragged, group, counts)
        expected = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        torch.testing.assert_close(assembled.cpu(), expected, rtol=0, atol=0)
        sliced = backend.token_reduce_scatter(assembled, group, counts)
        offset = sum(counts[:rank])
        torch.testing.assert_close(
            sliced.cpu(), expected[offset : offset + counts[rank]] * 2, rtol=0, atol=0
        )
        full_weight = torch.arange(32).reshape(8, 4).to(torch.bfloat16)
        head = torch.nn.Linear(4, 4, bias=False, device="npu", dtype=torch.bfloat16)
        head.weight.data.copy_(full_weight[rank * 4 : (rank + 1) * 4].npu())
        hidden = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=torch.bfloat16
        )
        for argmax in (False, True):
            processor = LogitsProcessor(
                PretrainedConfig(vocab_size=8),
                do_argmax=argmax,
                tp_rank=rank,
                tp_size=2,
                tp_group=group,
                dp_lm_head_tp=False,
            )
            assert (
                processor.acquire_dist_argmax_state(
                    head, max_M=2, skip_ping_pong=True, dtype=torch.bfloat16
                )
                is None
            )
            for _ in range(2):
                actual = processor._get_logits(
                    hidden.npu(), head, None, require_full_vocab=False
                )
                torch.testing.assert_close(
                    actual.cpu(), hidden @ full_weight.T, rtol=0, atol=0
                )
        print(
            json.dumps(
                {
                    "rank": rank,
                    "pid": os.getpid(),
                    "device": torch.npu.current_device(),
                    "all_reduce": "pass",
                    "all_gather": "pass",
                    "reduce_scatter": "pass",
                    "ragged_collectives": "pass",
                    "tp_logits": "pass",
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
