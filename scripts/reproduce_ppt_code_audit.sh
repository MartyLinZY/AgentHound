#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
frozen_repo="${1:-$repo_dir/../AgentHound}"
if [[ "$frozen_repo" != /* ]]; then frozen_repo="$PWD/$frozen_repo"; fi
cd "$repo_dir"
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
  uv pip install --python .venv/bin/python -r experiments/rootse_suite/requirements.lock.txt -e .
else
  if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
  .venv/bin/python -m pip install -r experiments/rootse_suite/requirements.lock.txt -e .
fi
export PYTHONPATH="$repo_dir/src"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_ppt_cleanup.py --frozen-repo "$frozen_repo"
