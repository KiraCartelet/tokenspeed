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

"""Summarize repeated samples and compare exact generated token sequences."""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def load(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        if row.get("kind") == "sample" and not row["warmup"]:
            groups[(row["tp"], row["input_tokens"], row["concurrency"])].append(row)
    result = []
    for (tp, length, concurrency), rows in sorted(groups.items()):
        item = dict(
            tp=tp, input_tokens=length, concurrency=concurrency, repeats=len(rows)
        )
        for key in ("ttft_ms", "tpot_ms", "latency_ms"):
            means = [
                statistics.mean(req[key] for req in row["requests"]) for row in rows
            ]
            item[key] = statistics.mean(means)
            item[key + "_sample_sd"] = (
                statistics.stdev(means) if len(means) > 1 else None
            )
        item["output_tokens_per_s"] = statistics.mean(
            row["output_tokens_per_s"] for row in rows
        )
        hbm = defaultdict(list)
        for row in rows:
            for snapshot in (row["npu_smi_before"], row["npu_smi_after"]):
                for line in snapshot.splitlines():
                    match = re.match(
                        r"\|\s+\d+\s+(\d+)\s+\|\s+[0-9A-Fa-f:]+\.\d+\s+\|.*?([0-9]+)\s*/\s*65536\s*\|",
                        line,
                    )
                    if match:
                        hbm[match[1]].append(int(match[2]))
        item["whole_chip_hbm_mb_range"] = {
            chip: [min(values), max(values)] for chip, values in hbm.items()
        }
        result.append(item)
    return result


def compare(left, right):
    a = next(row for row in left if row.get("kind") == "config")
    b = next(row for row in right if row.get("kind") == "config")
    assert a["config_sha256"] == b["config_sha256"], "model configs differ"
    ignored = {"world_size", "nprocs_per_node", "attn_tp_size", "dense_tp_size"}
    assert {k: v for k, v in a["engine"].items() if k not in ignored} == {
        k: v for k, v in b["engine"].items() if k not in ignored
    }, "execution configs differ"

    def index(records):
        return {
            (row["input_tokens"], row["concurrency"], row["iteration"]): row
            for row in records
            if row.get("kind") == "sample"
        }

    l, r = index(left), index(right)
    assert l.keys() == r.keys(), "missing comparison cases"
    comparisons = []
    for key in sorted(l):
        assert l[key]["input_ids"] == r[key]["input_ids"], "input IDs differ"
        for i, (x, y) in enumerate(zip(l[key]["requests"], r[key]["requests"])):
            u, v = x["output_ids"], y["output_ids"]
            first = next((j for j, (m, n) in enumerate(zip(u, v)) if m != n), None)
            comparisons.append(
                dict(
                    input_tokens=key[0],
                    concurrency=key[1],
                    iteration=key[2],
                    request=i,
                    equal=u == v,
                    first_difference=first,
                    tp1_output_ids=u,
                    tp2_output_ids=v,
                )
            )
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    records = [load(p) for p in args.files]
    output = {"performance": summarize([r for rows in records for r in rows])}
    if args.compare:
        assert len(records) == 2
        output["correctness"] = compare(*records)
        output["all_exact"] = all(row["equal"] for row in output["correctness"])
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
