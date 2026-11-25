#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PROJECT_PYTHON="${RLVR_PYTHON:-python3}"
exec "$PROJECT_PYTHON" -B -S code/math_l4_rescore_v4/rescore.py \
    --experiment-root archive/math_l4/experiment \
    --output results/recomputed_v4 "$@"
