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

"""Resolve official HF CDN URLs via proxy, transfer directly, verify SHA256."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sizes", required=True, nargs="+")
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    direct_env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    for size in args.sizes:
        root = args.root / f"Qwen3.5-{size}"
        pin = json.loads((root / "download-revision.json").read_text())
        info = HfApi().model_info(
            pin["repo"], revision=pin["revision"], files_metadata=True
        )
        weights = [
            dict(name=f.rfilename, size=f.size, sha256=f.lfs.sha256)
            for f in info.siblings
            if f.rfilename.endswith(".safetensors")
        ]
        (root / "weight-manifest.json").write_text(json.dumps(weights, indent=2))
        snapshot_download(
            repo_id=pin["repo"],
            revision=pin["revision"],
            local_dir=root,
            allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
            max_workers=2,
        )

        def fetch(item):
            target = root / item["name"]
            partial = root / (item["name"] + ".cdn-part")
            if not target.exists():
                if not partial.exists():
                    candidates = list(
                        (root / ".cache/huggingface/download").glob(
                            f"*.{item['sha256']}.*.incomplete"
                        )
                    )
                    if candidates:
                        max(candidates, key=lambda p: p.stat().st_size).rename(partial)
                metadata = get_hf_file_metadata(
                    hf_hub_url(pin["repo"], item["name"], revision=pin["revision"])
                )
                assert metadata.etag == item["sha256"]
                print("FETCH", size, item["name"], flush=True)
                subprocess.run(
                    [
                        "curl",
                        "-fL",
                        "--retry",
                        "5",
                        "--retry-delay",
                        "3",
                        "--connect-timeout",
                        "15",
                        "--speed-limit",
                        "102400",
                        "--speed-time",
                        "60",
                        "-C",
                        "-",
                        "-o",
                        str(partial),
                        "-sS",
                        metadata.location,
                    ],
                    env=direct_env,
                    check=True,
                )
                assert partial.stat().st_size == item["size"], item
                assert digest(partial) == item["sha256"], item
                partial.rename(target)
            else:
                assert target.stat().st_size == item["size"], item
                assert digest(target) == item["sha256"], item
            print("VERIFIED", size, item["name"], flush=True)
            return item

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            verified = list(pool.map(fetch, weights))
        (root / "verified-weights.json").write_text(json.dumps(verified, indent=2))
        print("COMPLETE_VERIFIED", size, flush=True)


if __name__ == "__main__":
    main()
