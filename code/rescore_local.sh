#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
RLVR_LOCAL_PYTHON="${RLVR_PYTHON:-python3}"
export PYTHONUNBUFFERED=1
printf '[LOCAL 1/2] 0%% | MODEL-FREE REGRESSION CHECKS\n'
"$RLVR_LOCAL_PYTHON" -S tests/test_data_verifier.py
"$RLVR_LOCAL_PYTHON" -S tests/test_verifier_regressions.py
"$RLVR_LOCAL_PYTHON" -S tests/test_analysis.py
"$RLVR_LOCAL_PYTHON" -S tests/test_training.py
"$RLVR_LOCAL_PYTHON" -S tests/test_rescore_saved.py
printf '[LOCAL 2/2] 50%% | OFFLINE RESCORING AND PAIRED ANALYSIS\n'
exec "$RLVR_LOCAL_PYTHON" -S scripts/rescore_saved.py "$@"
