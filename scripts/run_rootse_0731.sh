#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Compatibility entry: the user-selected request model is now deepseek-v4-flash.
exec bash "$repo_dir/scripts/run_rootse_flash.sh" "$@"
