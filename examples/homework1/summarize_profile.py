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

"""Aggregate NPU profiler CSVs; nested operator totals are not summed globally."""

import argparse
import collections
import csv
import json
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--device-id", required=True)
    args = parser.parse_args()
    result = []
    for label in ("0-prefill", "1-decode", "2-decode"):
        root = args.root / label
        kernels = rows(next(root.rglob("kernel_details.csv")))
        operators = rows(next(root.rglob("operator_details.csv")))
        steps = rows(next(root.rglob("step_trace_time.csv")))
        counts = collections.Counter()
        durations = collections.Counter()
        for kernel in kernels:
            counts[kernel["Name"]] += 1
            durations[kernel["Name"]] += float(kernel["Duration(us)"])
        scopes = [r for r in operators if r["Name"].startswith("HW1_GDN_")]
        assert len(scopes) == 18, (label, len(scopes))
        assert {r["Device_id"] for r in kernels} == {args.device_id}
        data = {
            "phase": label,
            "kernel_count": len(kernels),
            "kernel_duration_sum_us": sum(durations.values()),
            "gdn_scope_count": len(scopes),
            "gdn_host_total_us": sum(
                float(r["Host Total Duration(us)"]) for r in scopes
            ),
            "gdn_device_total_us": sum(
                float(r["Device Total Duration(us)"]) for r in scopes
            ),
            "top_kernel_duration": [
                {"name": n, "count": counts[n], "duration_us": d}
                for n, d in durations.most_common(12)
            ],
            "top_kernel_count": [
                {"name": n, "count": c, "duration_us": durations[n]}
                for n, c in counts.most_common(12)
            ],
            "step_trace": steps,
            "memory": json.loads((args.root / f"{label}.json").read_text()),
        }
        result.append(data)
    output = args.root / "summary.json"
    output.write_text(json.dumps(result, indent=2))
    print(output)
    for r in result:
        print(
            r["phase"],
            "kernels",
            r["kernel_count"],
            "kernel_ms",
            r["kernel_duration_sum_us"] / 1000,
            "gdn_device_ms",
            r["gdn_device_total_us"] / 1000,
        )


if __name__ == "__main__":
    main()
