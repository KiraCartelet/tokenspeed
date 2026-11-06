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

"""CPU-only tests for experiment aggregation and evidence matching."""

import unittest

from summarize import compare, summarize
from summarize_traces import union_duration


class SummaryTest(unittest.TestCase):
    def test_overlapping_device_tasks_are_not_double_counted(self):
        self.assertEqual(union_duration([(1, 4), (3, 5), (8, 9)]), 5)
        self.assertEqual(union_duration([(1, 5), (2, 3), (5, 7)]), 6)
        self.assertEqual(union_duration([]), 0)

    def test_throughput_is_batch_wall_time_metric(self):
        rows = [
            dict(
                kind="sample",
                tp=2,
                input_tokens=128,
                concurrency=2,
                npu_smi_before="",
                npu_smi_after="",
                iteration=i,
                warmup=i == 0,
                output_tokens_per_s=rate,
                requests=[
                    dict(ttft_ms=2, tpot_ms=3, latency_ms=5),
                    dict(ttft_ms=4, tpot_ms=5, latency_ms=7),
                ],
            )
            for i, rate in enumerate([999, 10, 20])
        ]
        actual = summarize(rows)[0]
        self.assertEqual(actual["repeats"], 2)
        self.assertEqual(actual["output_tokens_per_s"], 15)
        self.assertEqual(actual["ttft_ms"], 3)

    def test_missing_case_is_not_a_pass(self):
        config = dict(kind="config", config_sha256="same", engine={"dtype": "bfloat16"})
        sample = dict(
            kind="sample",
            input_tokens=1,
            concurrency=1,
            iteration=0,
            input_ids=[[1]],
            requests=[{"output_ids": [2, 3]}],
        )
        with self.assertRaisesRegex(AssertionError, "missing comparison"):
            compare([config, sample], [config])

    def test_difference_is_reported(self):
        config = dict(kind="config", config_sha256="same", engine={"dtype": "bfloat16"})
        sample = dict(
            kind="sample",
            input_tokens=1,
            concurrency=1,
            iteration=0,
            input_ids=[[1]],
            requests=[{"output_ids": [2, 3]}],
        )
        other = dict(sample, requests=[{"output_ids": [2, 4]}])
        result = compare([config, sample], [config, other])[0]
        self.assertFalse(result["equal"])
        self.assertEqual(result["first_difference"], 1)


if __name__ == "__main__":
    unittest.main()
