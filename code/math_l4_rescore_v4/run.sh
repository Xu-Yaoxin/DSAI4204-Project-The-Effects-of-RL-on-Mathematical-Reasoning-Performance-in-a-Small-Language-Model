#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
python3 -B -S tests/test_scorer_v4.py
python3 -B -S tests/test_rescore_pipeline.py
exec python3 -B -S rescore.py "$@"
