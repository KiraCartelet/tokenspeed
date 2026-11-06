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

"""Pin model revisions before downloading the assignment checkpoints."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--sizes", nargs="+", required=True, choices=["4B", "9B", "27B"]
    )
    args = parser.parse_args()
    for size in args.sizes:
        repo = f"Qwen/Qwen3.5-{size}"
        target = Path(args.root) / f"Qwen3.5-{size}"
        target.mkdir(parents=True, exist_ok=True)
        manifest = target / "download-revision.json"
        if manifest.exists():
            revision = json.loads(manifest.read_text())["revision"]
        else:
            revision = HfApi().model_info(repo).sha
            manifest.write_text(
                json.dumps({"repo": repo, "revision": revision}, indent=2)
            )
        print("DOWNLOAD", repo, revision, flush=True)
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=target,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
            max_workers=2,
        )
        print("COMPLETE", repo, flush=True)


if __name__ == "__main__":
    main()
