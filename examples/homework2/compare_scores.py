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

"""Record top-token scores on the original generation histories."""

import argparse
import json
import os
from pathlib import Path

if os.environ.get("HW2_SCORES"):
    from score_capture import install

    install()

from tokenspeed.runtime.entrypoints.engine import Engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--tp", required=True, type=int)
    p.add_argument("--length", required=True, type=int)
    p.add_argument("--concurrency", required=True, type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--request-index", type=int, help="Run one original request alone")
    a = p.parse_args()
    records = [json.loads(s) for s in Path(a.source).read_text().splitlines()]
    config = dict(records[0]["engine"])
    for key in ("world_size", "nprocs_per_node", "attn_tp_size", "dense_tp_size"):
        config[key] = a.tp
    source = next(
        r
        for r in records
        if r.get("kind") == "sample"
        and r["input_tokens"] == a.length
        and r["concurrency"] == a.concurrency
    )
    count = len(source["requests"][0]["output_ids"])
    inputs = source["input_ids"]
    if a.request_index is not None:
        inputs = [inputs[a.request_index]]
    with Path(a.output).open("x") as out:
        engine = Engine(**config)
        try:
            (Path(os.environ["HW2_SCORES"]) / "capture-enabled").touch()
            for iteration in range(2):
                result = engine.generate(
                    input_ids=inputs,
                    sampling_params={
                        "max_new_tokens": count,
                        "temperature": 0,
                        "ignore_eos": True,
                    },
                )
                out.write(
                    json.dumps(
                        dict(
                            tp=a.tp,
                            gdn_reference=os.environ.get("HW2_GDN_REFERENCE") == "1",
                            iteration=iteration,
                            input_ids=inputs,
                            results=result,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()
        finally:
            engine.shutdown()


if __name__ == "__main__":
    main()
