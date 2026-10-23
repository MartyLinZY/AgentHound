#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
key_path="${1:?Usage: bash scripts/reproduce_rootse_migration.sh /path/to/moma_ai.key [output-directory]}"
output_dir="${2:-$repo_dir/outputs/rootse-migration-state-v2-deepseek-v41-flash-32k}"
if [[ "$key_path" != /* ]]; then key_path="$PWD/$key_path"; fi
if [[ "$output_dir" != /* ]]; then output_dir="$PWD/$output_dir"; fi
cd "$repo_dir"
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
  uv pip install --python .venv/bin/python -r experiments/rootse_suite/requirements.lock.txt -e .
  if [[ ! -x .venv-rootse-analysis/bin/python ]]; then uv venv --python 3.12 .venv-rootse-analysis; fi
  uv pip install --python .venv-rootse-analysis/bin/python -r experiments/rootse_suite/analysis-requirements.lock.txt
else
  if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
  if [[ ! -x .venv-rootse-analysis/bin/python ]]; then python3 -m venv .venv-rootse-analysis; fi
  .venv/bin/python -m pip install -r experiments/rootse_suite/requirements.lock.txt -e .
  .venv-rootse-analysis/bin/python -m pip install -r experiments/rootse_suite/analysis-requirements.lock.txt
fi
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_rootse_migration.py --out "$output_dir" --prepare-only
.venv/bin/python scripts/run_rootse_migration.py --out "$output_dir" --pilot --api-key-file "$key_path"
.venv/bin/python scripts/run_rootse_migration.py --out "$output_dir" --api-key-file "$key_path"
.venv/bin/python scripts/report_rootse_migration.py --out "$output_dir"
.venv-rootse-analysis/bin/python scripts/plot_rootse_migration.py --out "$output_dir"
