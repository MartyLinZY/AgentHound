#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
key_path="${1:?Usage: bash scripts/reproduce_rootse_suite.sh /path/to/moma_ai.key [output-directory] [config-file]}"
output_dir="${2:-$repo_dir/outputs/rootse-suite-deepseek-v41-flash-32k-v1}"
config_path="${3:-$repo_dir/experiments/rootse_suite/config.json}"
if [[ "$config_path" != /* ]]; then config_path="$PWD/$config_path"; fi
if [[ "$key_path" != /* ]]; then key_path="$PWD/$key_path"; fi
if [[ "$output_dir" != /* ]]; then output_dir="$PWD/$output_dir"; fi
cd "$repo_dir"
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
  uv pip install --python .venv/bin/python -r experiments/rootse_suite/requirements.lock.txt -e .
else
  if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
  .venv/bin/python -m ensurepip --upgrade
  .venv/bin/python -m pip install -r experiments/rootse_suite/requirements.lock.txt -e .
fi
.venv/bin/python scripts/run_rootse_suite.py --prepare-only --out "$output_dir" --config "$config_path"
.venv/bin/python -m pytest -q
.venv/bin/python - "$output_dir" <<'PY'
from pathlib import Path
import json,sys,platform
from importlib.metadata import distributions
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
p=out/'environment.json'
if not p.exists():p.write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in distributions()}},indent=2)+'\n')
PY
.venv/bin/python scripts/run_rootse_suite.py --api-key-file "$key_path" --out "$output_dir" --config "$config_path"
