#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
key_path="${1:?Usage: bash scripts/run_rootse_deepseek.sh /path/to/deepseek.key [--workers 16] [--limit 2 --methods all_at_once]}"
shift
if [[ "$key_path" != /* ]]; then key_path="$PWD/$key_path"; fi
cd "$repo_dir"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing .venv/bin/python; use the existing project environment.' >&2
  exit 1
fi
exec .venv/bin/python scripts/run_rootse_deepseek.py --api-key-file "$key_path" "$@"
