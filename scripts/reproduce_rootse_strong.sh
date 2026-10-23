#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
key_path="${1:?Usage: bash scripts/reproduce_rootse_strong.sh /path/to/moma_ai.key}"
if [[ "$key_path" != /* ]]; then key_path="$PWD/$key_path"; fi
baseline_dir="$repo_dir/outputs/rootse-full-qwen25-14b-1m-v1"
candidate_dir="$repo_dir/outputs/rootse-full-qwen35-397b-a17b-v1"
if [[ ! -f "$baseline_dir/metrics.json" ]]; then
  bash "$repo_dir/scripts/reproduce_rootse_full.sh" "$key_path" "$baseline_dir"
fi
bash "$repo_dir/scripts/reproduce_rootse_full.sh" "$key_path" "$candidate_dir" "$repo_dir/experiments/rootse_full/config_qwen35_397b.json"
"$repo_dir/.venv/bin/python" "$repo_dir/scripts/compare_rootse_models.py" --baseline "$baseline_dir" --candidate "$candidate_dir"
