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

"""Compare full distributions only while generated histories are identical."""

import argparse
import json
from pathlib import Path

import torch


def captures(root):
    result = {}
    for path in sorted(root.glob("[0-9]*.json"), key=lambda p: int(p.stem)):
        data = json.loads(path.read_text())
        for row, rid in enumerate(data["request_ids"]):
            result.setdefault(rid, []).append((path.with_suffix(".pt"), row))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    a = p.parse_args()
    left = [
        json.loads(s) for s in a.left.with_suffix(".jsonl").read_text().splitlines()
    ]
    right = [
        json.loads(s) for s in a.right.with_suffix(".jsonl").read_text().splitlines()
    ]
    assert len(left) == len(right) == 2
    lc, rc = captures(a.left), captures(a.right)
    records = []
    for lrun, rrun in zip(left, right):
        assert lrun["input_ids"] == rrun["input_ids"]
        for req, (lres, rres) in enumerate(zip(lrun["results"], rrun["results"])):
            lsteps = lc[lres["meta_info"]["id"]]
            rsteps = rc[rres["meta_info"]["id"]]
            lid, rid = lres["output_ids"], rres["output_ids"]
            assert len(lid) == len(rid)
            assert len(lsteps) >= len(lid) and len(rsteps) >= len(rid)
            lsteps, rsteps = lsteps[: len(lid)], rsteps[: len(rid)]
            metrics = []
            for step, ((lp, li), (rp, ri)) in enumerate(zip(lsteps, rsteps)):
                x = torch.load(lp, map_location="cpu", weights_only=True)[li]
                y = torch.load(rp, map_location="cpu", weights_only=True)[ri]
                assert x[lid[step]] == x.max(), ("left history misaligned", step)
                assert y[rid[step]] == y.max(), ("right history misaligned", step)
                px, py = x.softmax(-1), y.softmax(-1)
                top = sorted(
                    set(x.topk(5).indices.tolist() + y.topk(5).indices.tolist())
                )
                metric = dict(
                    step=step,
                    max_abs_logit_error=(x - y).abs().max().item(),
                    logit_rmse=(x - y).square().mean().sqrt().item(),
                    cosine=torch.nn.functional.cosine_similarity(x, y, dim=0).item(),
                    total_variation=(px - py).abs().sum().item() / 2,
                    tp1_token=lid[step],
                    tp2_token=rid[step],
                    candidates=[
                        dict(
                            id=t,
                            tp1_logit=x[t].item(),
                            tp2_logit=y[t].item(),
                            tp1_probability=px[t].item(),
                            tp2_probability=py[t].item(),
                        )
                        for t in top
                    ],
                )
                metrics.append(metric)
                if lid[step] != rid[step]:
                    break
            records.append(
                dict(
                    iteration=lrun["iteration"],
                    request=req,
                    tokens_equal=lid == rid,
                    shared_history_steps=len(metrics),
                    max_total_variation=max(m["total_variation"] for m in metrics),
                    max_logit_rmse=max(m["logit_rmse"] for m in metrics),
                    first_difference=None if lid == rid else metrics[-1],
                    steps=metrics,
                )
            )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
