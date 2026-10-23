#!/usr/bin/env bash
# One command prepares the environment, fetches pinned sources if absent,
# resumes inference without repeating completed cases, and rebuilds the report.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
key_path="${1:?Usage: bash scripts/reproduce_rootse_full.sh /path/to/moma_ai.key [output-directory]}"
output_dir="${2:-$repo_dir/outputs/rootse-full-qwen25-14b-1m-v1}"
config_path="${3:-$repo_dir/experiments/rootse_full/config.json}"
if [[ "$key_path" != /* ]]; then key_path="$PWD/$key_path"; fi
if [[ "$output_dir" != /* ]]; then output_dir="$PWD/$output_dir"; fi
if [[ "$config_path" != /* ]]; then config_path="$PWD/$config_path"; fi
cd "$repo_dir"
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
  uv pip install --python .venv/bin/python -r experiments/rootse_full/requirements.lock.txt -e .
else
  if [[ ! -x .venv/bin/python ]]; then
    if command -v python3.12 >/dev/null 2>&1; then
      python3.12 -m venv .venv
    else
      python3 -m venv .venv
    fi
  fi
  .venv/bin/python -m ensurepip --upgrade
  .venv/bin/python -m pip install -r experiments/rootse_full/requirements.lock.txt -e .
fi
.venv/bin/python -m pytest -q
.venv/bin/python - "$output_dir" <<'PY'
import json, platform, sys
from importlib.metadata import distributions
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
path = out / 'environment.json'
if not path.exists():
    path.write_text(json.dumps({'python': sys.version, 'platform': platform.platform(),
        'packages': {d.metadata['Name']: d.version for d in distributions()}}, indent=2) + '\n')
PY
.venv/bin/python scripts/run_rootse_full_baseline.py --api-key-file "$key_path" --out "$output_dir" --config "$config_path"
