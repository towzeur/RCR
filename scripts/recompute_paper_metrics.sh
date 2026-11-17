#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${1:-$ROOT/artifacts/paper_runs}"
OUT="${2:-$ROOT/artifacts/evaluations/paper_runs}"
PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" "$ROOT/scripts/evaluate_artifacts.py" "$RUNS"/* \
  --heads etf rcr \
  --out-dir "$OUT"
