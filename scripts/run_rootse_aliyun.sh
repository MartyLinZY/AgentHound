#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
key_path="${1:?Usage: bash scripts/run_rootse_aliyun.sh /path/to/aliyun_maas_qwen38.key}"
if [[ "$key_path" != /* ]]; then key_path="$PWD/$key_path"; fi
cd "$repo_dir"
mkdir -p outputs
bash scripts/reproduce_rootse_suite.sh "$key_path"   "$repo_dir/outputs/rootse-suite-aliyun-qwen38-flash-thinking-32k-v1"   "$repo_dir/experiments/rootse_suite/config_aliyun_qwen38_flash.json"   2>&1 | tee -a "$repo_dir/outputs/rootse-aliyun-console.log"
