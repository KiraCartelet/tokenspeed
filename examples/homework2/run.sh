#!/usr/bin/env bash
set -euo pipefail
# Usage: run.sh MODEL_SIZE TP DEVICES OUTPUT_TAG LENGTHS CONCURRENCIES TOKENS REPEATS
size=$1; tp=$2; devices=$3; tag=$4; lengths=$5; concurrencies=$6; tokens=$7; repeats=$8
cd "$(dirname "$0")/../.."
set +u
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source .venv/bin/activate
set -u
export ASCEND_RT_VISIBLE_DEVICES="$devices" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python"
mkdir -p artifacts/hw2
read -ra length_args <<< "$lengths"
read -ra concurrency_args <<< "$concurrencies"
python examples/homework2/benchmark.py --model "$PWD/models/Qwen3.5-$size" --tp "$tp" \
  --output "artifacts/hw2/$tag.jsonl" --lengths "${length_args[@]}" \
  --concurrencies "${concurrency_args[@]}" --new-tokens "$tokens" --repeats "$repeats"
