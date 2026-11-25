#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PROJECT_PYTHON="${RLVR_PYTHON:-python3}"
PROJECT_ROOT="$PWD"
exec "$PROJECT_PYTHON" -B -S code/scripts/rescore_saved.py \
    --archive "$PROJECT_ROOT/archive/gsm8k/run_20261031/code" \
    --output "$PROJECT_ROOT/results/recomputed_gsm8k" "$@"
