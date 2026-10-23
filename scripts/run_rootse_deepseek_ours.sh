#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
key_path="${1:?Usage: bash scripts/run_rootse_deepseek_ours.sh /path/to/deepseek.key [workers=16] [output-directory]}"
workers="${2:-16}"
output_dir="${3:-$repo_dir/outputs/rootse-migration-state-v2-deepseek-official-flash-thinking-32k-v1}"
if [[ "$key_path" != /* ]]; then key_path="$PWD/$key_path"; fi
if [[ "$output_dir" != /* ]]; then output_dir="$PWD/$output_dir"; fi
cd "$repo_dir"
[[ -x .venv/bin/python ]] || { echo 'Missing existing .venv/bin/python' >&2; exit 1; }
# Scoring is offline and reuses the completed official-DeepSeek baseline responses.
baselines="$repo_dir/outputs/rootse-suite-deepseek-official-flash-thinking-32k-v1"
.venv/bin/python - "$baselines" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);m=json.loads((out/'manifest.json').read_text())
if m['status']!='complete' or m['n']!=612:
    raise SystemExit('Expected the completed official DeepSeek six-baseline run')
for name in ['metrics.json','evaluation.json']:
    if not (out/name).is_file():
        raise SystemExit('Missing baseline offline evaluation: '+name)
PY
.venv/bin/python scripts/run_rootse_deepseek_migration.py --out "$output_dir" --workers "$workers" --prepare-only
.venv/bin/python scripts/run_rootse_deepseek_migration.py --out "$output_dir" --workers "$workers" --pilot --api-key-file "$key_path"
.venv/bin/python scripts/run_rootse_deepseek_migration.py --out "$output_dir" --workers "$workers" --api-key-file "$key_path"
.venv/bin/python scripts/report_rootse_deepseek_migration.py --out "$output_dir" --baselines "$baselines"
