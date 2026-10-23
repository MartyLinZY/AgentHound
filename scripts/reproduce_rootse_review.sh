#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
key_path="${1:?Usage: bash scripts/reproduce_rootse_review.sh /path/to/moma_ai.key}"
if [[ "$key_path" != /* ]]; then key_path="$PWD/$key_path"; fi
cd "$repo_dir"
# Keep figure dependencies separate from the frozen inference environment.
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x .venv-rootse-analysis/bin/python ]]; then uv venv --python 3.12 .venv-rootse-analysis; fi
  uv pip install --python .venv-rootse-analysis/bin/python -r experiments/rootse_suite/analysis-requirements.lock.txt
else
  if [[ ! -x .venv-rootse-analysis/bin/python ]]; then python3 -m venv .venv-rootse-analysis; fi
  .venv-rootse-analysis/bin/python -m pip install -r experiments/rootse_suite/analysis-requirements.lock.txt
fi
bash scripts/reproduce_rootse_suite.sh "$key_path"
.venv-rootse-analysis/bin/python scripts/analyze_rootse_suite.py --source ../TrajAudit --out outputs/rootse-suite-deepseek-v41-flash-32k-v1
