#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
semantic_python="${ATTENTION_SEMANTIC_PYTHON:-.venv-attention-semantic/bin/python}"
semantic_out="${1:-outputs/attentionmap-semantic-v2}"
"$semantic_python" scripts/run_attentionmap_semantic.py --out "$semantic_out" --stage all
"$semantic_python" scripts/analyze_attentionmap_semantic.py --out "$semantic_out"
