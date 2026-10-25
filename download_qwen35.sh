#!/usr/bin/env bash
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

# Download only: no CANN setup, NPU allocation, or serving processes.
set -eo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate tokenspeed-npu

# These settings affect only this script and its child processes.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_PROGRESS_BARS=0
export HF_HUB_DOWNLOAD_TIMEOUT=60

python -u - <<'PY'
from huggingface_hub import constants, snapshot_download
from huggingface_hub.utils import enable_progress_bars

enable_progress_bars()
print("Model: Qwen/Qwen3.5-0.8B", flush=True)
print(f"Cache: {constants.HF_HUB_CACHE}", flush=True)
print("HTTP download: progress shows bytes, rate, and estimated time.", flush=True)
print("Stop your serve.sh first if it is downloading this model, to avoid cache-lock waits.", flush=True)
print("Ctrl+C interrupts the download; completed cache files are retained.", flush=True)

try:
    path = snapshot_download(
        repo_id="Qwen/Qwen3.5-0.8B",
        max_workers=1,
    )
except KeyboardInterrupt:
    print("\nDownload interrupted; run this script again to retry.", flush=True)
    raise SystemExit(130)

print(f"\nModel download complete: {path}", flush=True)
print("You can now start serve.sh in the homework NPU environment.", flush=True)
PY
