#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
attention_python="${ATTENTION_PYTHON:-.venv-attention/bin/python}"
attention_out="${1:-outputs/attentionmap-v1}"
"$attention_python" scripts/run_attentionmap.py --out "$attention_out" --stage all
"$attention_python" scripts/analyze_attentionmap.py --out "$attention_out"
