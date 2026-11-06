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

"""Use Ascend profiler's overlap-aware step accounting; retain raw CSVs."""

import argparse
import collections
import csv
import json
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def union_duration(intervals):
    end = float("-inf")
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def task_intervals(target, memory):
    events = json.loads(next(target.rglob("trace_view.json")).read_text())
    marker = next(
        e
        for e in events
        if e.get("name") == f"HW2_STEP{memory['step']}_{memory['phase'].upper()}"
    )
    start = float(marker["ts"])
    stop = start + float(marker["dur"])
    groups = collections.defaultdict(list)
    for task in rows(next(target.rglob("task_time.csv"))):
        left = max(start, float(task["task_start(us)"]))
        right = min(stop, float(task["task_stop(us)"]))
        if right <= left:
            continue
        kind = task["kernel_type"]
        if "WAIT" in kind:
            groups["sync_wait"].append((left, right))
        elif kind.startswith("KERNEL"):
            groups["compute"].append((left, right))
        elif kind.startswith("SDMA"):
            groups["sdma"].append((left, right))
    return {k: union_duration(v) for k, v in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summaries = []
    for metadata in sorted(args.root.glob("rank*-*.json")):
        memory = json.loads(metadata.read_text())
        target = args.root / f"rank{memory['rank']}"
        continuous = target.exists()
        if not continuous:
            target = args.root / metadata.stem
        steps = list(target.rglob("step_trace_time.csv"))
        kernels = list(target.rglob("kernel_details.csv"))
        operators = list(target.rglob("operator_details.csv"))
        if len(steps) != 1 or len(kernels) != 1 or len(operators) != 1:
            raise ValueError(f"incomplete or duplicate trace: {target}")
        kernel_rows, op_rows = rows(kernels[0]), rows(operators[0])
        if continuous:
            kernel_rows = [
                r for r in kernel_rows if r["Step Id"] == str(memory["step"])
            ]
        devices = sorted({r["Device_id"] for r in kernel_rows})
        counts, durations = collections.Counter(), collections.Counter()
        for r in kernel_rows:
            counts[r["Name"]] += 1
            durations[r["Name"]] += float(r["Duration(us)"])
        comm = [r for r in op_rows if r["Name"].startswith("HW2_HCCL_")]
        steps_parsed = []
        for step in rows(steps[0]):
            if continuous and step["Step"] != str(memory["step"]):
                continue
            values = {
                k: float(v) for k, v in step.items() if k not in ("Device_id", "Step")
            }
            values.update(device=step["Device_id"], step=step["Step"])
            steps_parsed.append(values)
        summaries.append(
            dict(
                label=metadata.stem,
                memory=memory,
                device_ids=devices,
                step_trace_us=steps_parsed,
                task_union_us_within_host_forward=(
                    task_intervals(target, memory) if continuous else None
                ),
                kernel_count=len(kernel_rows),
                kernel_duration_sum_us=sum(durations.values()),
                communication_calls_trace_total=dict(
                    collections.Counter(r["Name"] for r in comm)
                ),
                communication_host_trace_total_us=sum(
                    float(r["Host Total Duration(us)"]) for r in comm
                ),
                top_kernels=[
                    dict(name=n, count=counts[n], duration_us=d)
                    for n, d in durations.most_common(10)
                ],
            )
        )
    if not summaries:
        raise ValueError("no per-rank metadata")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
