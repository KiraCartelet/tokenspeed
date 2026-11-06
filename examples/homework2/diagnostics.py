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

"""Opt-in continuous per-rank traces, separate from performance samples."""

import json
import os
from pathlib import Path


def install():
    import torch
    import torch.distributed as dist
    import torch_npu

    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    root = Path(os.environ["HW2_DIAGNOSTICS"])
    root.mkdir(parents=True, exist_ok=True)
    for name in ("all_reduce", "all_gather_single", "reduce_scatter"):
        original_comm = getattr(HcclBackend, name)

        def scoped(fn, label):
            def call(*args, **kwargs):
                with torch.profiler.record_function(f"HW2_HCCL_{label}"):
                    return fn(*args, **kwargs)

            return call

        setattr(HcclBackend, name, scoped(original_comm, name))
    original = ModelExecutor.execute_forward_op
    captured = 0
    profiler = None

    def forward(self, op, *args, **kwargs):
        nonlocal captured, profiler
        if not (root / "capture-enabled").exists() or captured >= 3:
            return original(self, op, *args, **kwargs)
        rank = self.config.global_rank
        if profiler is None:
            torch.npu.synchronize()
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.barrier()
            profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=3, repeat=1
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(root / f"rank{rank}")
                ),
            )
            profiler.__enter__()
        index = captured
        captured += 1
        phase = "prefill" if op.num_extends() else "decode"
        torch.npu.reset_peak_memory_stats()
        with torch.profiler.record_function(f"HW2_STEP{index}_{phase.upper()}"):
            result = original(self, op, *args, **kwargs)
            torch.npu.synchronize()
        record = dict(
            rank=rank,
            step=index,
            pid=os.getpid(),
            logical_device=torch.npu.current_device(),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            phase=phase,
            tokens=sum(op.input_lengths),
            allocated_bytes=torch.npu.memory_allocated(),
            peak_allocated_bytes=torch.npu.max_memory_allocated(),
            peak_reserved_bytes=torch.npu.max_memory_reserved(),
        )
        (root / f"rank{rank}-{index}-{phase}.json").write_text(
            json.dumps(record, indent=2)
        )
        # Export only after every captured forward; per-step exports make one rank
        # wait for another rank's CPU-side trace conversion in its next collective.
        profiler.step()
        if captured == 3:
            profiler.__exit__(None, None, None)
        return result

    ModelExecutor.execute_forward_op = forward
